# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import time

from typing import Any, Literal

import torch
from omegaconf import DictConfig

from tensordict import TensorDict
from torch import device as torch_device, dtype as torch_dtype

from torchrl._utils import logger as torchrl_logger
from torchrl.collectors.llm.weight_update.vllm import vLLMUpdater
from torchrl.envs.llm import RetrieveLogProb
from torchrl.envs.llm.datasets.ifeval import IFEvalEnv
from torchrl.modules.llm import TransformersWrapper, vLLMWrapper
from transformers.models.auto.modeling_auto import AutoModelForCausalLM
from transformers.tokenization_utils import PreTrainedTokenizer

try:
    import ray
except ImportError:
    ray = None


def get_tokenizer(cfg: DictConfig) -> PreTrainedTokenizer:
    from transformers import AutoTokenizer

    model_name = cfg.model.name
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # tokenizer.eos_token = "<|im_end|>"
    if tokenizer.pad_token == tokenizer.eos_token:
        tokenizer.pad_token = "PAD"
    tokenizer.padding_side = "left"
    return tokenizer


def make_env(cfg: DictConfig, devices: list[int] | None = None):
    """Create the environment with proper device allocation.

    Args:
        cfg: The configuration object
        devices: The devices to use for the reference model

    Returns:
        The configured environment
    """
    # Create reference model with proper device allocation
    # For the collector actor, we want inference_model devices first, then ref_model devices
    train_tokenizer = get_tokenizer(cfg)

    # Get device information
    num_inf_devices = cfg.inference_model.num_devices
    num_ref_devices = cfg.ref_model.num_devices
    num_inf_devices + num_ref_devices

    # Create a new config with adjusted device assignments
    ref_cfg = DictConfig(dict(cfg))
    ref_model = get_ref_model(ref_cfg, train_tokenizer, devices=devices)

    # Setup environment
    if cfg.env.dataset == "gsm8k":
        from torchrl.envs.llm import GSM8KEnv

        env = GSM8KEnv(
            repeats=cfg.env.repeats,
            tokenizer=train_tokenizer,
            num_envs=cfg.env.num_envs,
            device=torch.device("cpu"),
        )
    else:  # ifeval
        env = IFEvalEnv(
            repeats=cfg.env.repeats,
            tokenizer=train_tokenizer,
            num_envs=cfg.env.num_envs,
            device=torch.device("cpu"),
        )

    # Pass device directly to RetrieveLogProb - Since, for Ray, the local device is always 0
    # we can just use 0 here.
    device = torch.device("cuda:0")
    env = env.append_transform(
        RetrieveLogProb(
            model=ref_model,
            assistant_only=True,
            tokenizer_kwargs={"chat_template_name": "qwen"},
            device=device,
            log_probs_full_key=("ref_log_probs", "full"),
        )
    )
    return env


def get_train_model(
    cfg: DictConfig,
    devices: list[int] | None = None,
    chat_template_name: str | None = None,
) -> tuple[TransformersWrapper, PreTrainedTokenizer]:
    """Creates and configures the training model with LoRA adapters.

    This function initializes the main training model with LoRA adapters and other
    training-specific configurations like gradient checkpointing. The model is wrapped
    in a TransformersWrapper for policy training.

    Args:
        cfg (DictConfig): The hydra configuration object containing model and training settings.
            Expected to have train_model section with LoRA, quantization, and other
            training-specific parameters.
        devices (list[int] | None, optional): The devices to use for the training model. Defaults to `None`.
        chat_template_name (str | None, optional): The name of the chat template to use. Defaults to `None`.

    Returns:
        tuple[TransformersWrapper, PreTrainedTokenizer]:
            - policy_training: The wrapped training model
            - train_tokenizer: The tokenizer for the model

    Raises:
        RuntimeError: If CUDA is not available or if device allocation fails
    """
    torchrl_logger.info("Creating train model")

    # Set model dtype explicitly
    model_dtype = getattr(torch, cfg.train_model.torch_dtype)

    # Get configured devices or default to [0]
    train_devices = devices if devices is not None else [0]

    # Create max_memory dict - set 0 memory for GPUs we don't want to use
    max_memory = {}
    for i in range(torch.cuda.device_count()):
        if i in train_devices:
            max_memory[i] = "24GiB"  # Allow max memory for devices we want to use
        else:
            max_memory[i] = "0GiB"  # No memory for other devices
    max_memory["cpu"] = "24GiB"  # Allow CPU memory as fallback

    # Let HF handle distribution with max_memory
    device_map = "balanced" if len(train_devices) > 1 else f"cuda:{train_devices[0]}"
    train_model, train_tokenizer = get_hf_model(
        cfg.model.name,
        device_map=device_map,
        max_memory=max_memory,
        lora=cfg.train_model.lora.enabled,
        lora_r=cfg.train_model.lora.r,
        lora_alpha=cfg.train_model.lora.alpha,
        lora_dropout=cfg.train_model.lora.dropout,
        gradient_checkpointing=cfg.train_model.gradient_checkpointing,
        quantize=cfg.train_model.quantization.enabled,
        torch_dtype=model_dtype,
        attn_implementation=cfg.train_model.attn_implementation,
        compile=cfg.model.compile,
        eval_mode=cfg.train_model.eval,
    )

    # Force all model parameters to the same dtype
    for param in train_model.parameters():
        param.data = param.data.to(model_dtype)

    policy_training = TransformersWrapper(
        train_model,
        tokenizer=train_tokenizer,
        input_mode="history",
        generate=False,
        return_log_probs=True,
        pad_output=False,
        device=torch.device("cuda:0"),
    )
    # Ensure model stays in eval mode after wrapping
    policy_training.model.eval()
    policy_training.model.train(False)
    return policy_training, train_tokenizer


def get_inference_model(
    cfg: DictConfig,
    devices: list[int] | None = None,
    make_ray_worker: bool = True,
    tokenizer: PreTrainedTokenizer | None = None,
) -> vLLMWrapper:
    """Creates the vLLM-based inference model for fast generation.

    This function initializes a vLLM model server for efficient inference and wraps
    it in a vLLMWrapper for policy inference. vLLM provides optimized generation
    with better throughput than standard HuggingFace generation.

    Args:
        cfg (DictConfig): The hydra configuration object containing model settings.
            Expected to have inference_model section with vLLM-specific parameters
            like gpu_memory_utilization and generation settings.
        devices (list[int], optional): The devices to use for the inference model. Default: `None`.
        make_ray_worker (bool, optional): Whether to make a ray worker. Default: `True`.
        tokenizer (PreTrainedTokenizer | None, optional): The tokenizer to use. Default: None

    Returns:
        vLLMWrapper: The wrapped vLLM model ready for inference.

    Raises:
        AssertionError: If the vLLM server or model initialization fails
    """
    from torchrl.modules.llm.backends.vllm import make_vllm_worker

    num_devices = cfg.inference_model.num_devices
    if num_devices is None:
        vllm_devices = devices if devices is not None else [1]
    else:
        vllm_devices = None
    torchrl_logger.info(
        f"Creating inference model with num_devices={num_devices}, devices={vllm_devices}"
    )

    model_name = cfg.model.name

    if tokenizer is None:
        tokenizer = get_tokenizer(cfg)

    # vLLM handles device mapping internally
    inference_server = make_vllm_worker(
        model_name=model_name,
        gpu_memory_utilization=cfg.inference_model.gpu_memory_utilization,
        num_devices=num_devices,
        devices=list(vllm_devices)
        if vllm_devices is not None
        else None,  # Convert to list for type compatibility
        make_ray_worker=make_ray_worker,
        enforce_eager=cfg.inference_model.enforce_eager,
    )
    assert inference_server is not None
    policy = vLLMWrapper(
        inference_server,
        input_mode="history",
        chat_template_name="qwen",
        return_log_probs=True,
        tokenizer=tokenizer,
        pad_output=False,
        generate_kwargs={
            "max_tokens": cfg.inference_model.max_tokens,
            "include_stop_str_in_output": cfg.inference_model.include_stop_str_in_output,
            "temperature": cfg.inference_model.temperature,
        },
    )
    assert policy.model is not None
    return policy


def get_ref_model(
    cfg: DictConfig, tokenizer: PreTrainedTokenizer, devices: list[int] | None = None
) -> TransformersWrapper:
    """Creates the reference model for KL penalty computation.

    This function initializes a frozen copy of the base model to serve as the
    reference model for KL divergence computation. The reference model is typically
    quantized and does not require gradient computation.

    Args:
        cfg (DictConfig): The hydra configuration object containing model settings.
            Expected to have ref_model section with quantization and attention settings.
        tokenizer (PreTrainedTokenizer): The tokenizer to use with the reference model.

    Returns:
        TransformersWrapper: The wrapped reference model in eval mode with detached weights.
    """
    torchrl_logger.info("Creating ref model")

    # Get configured devices or default to [2]
    if cfg.ref_model.num_devices is None:
        ref_devices = devices if devices is not None else [2]
    else:
        ref_devices = list(range(cfg.ref_model.num_devices))

    # Create max_memory dict - set 0 memory for GPUs we don't want to use
    max_memory = {}
    for i in range(torch.cuda.device_count()):
        if i in ref_devices:
            max_memory[i] = "24GiB"  # Allow max memory for devices we want to use
        else:
            max_memory[i] = "0GiB"  # No memory for other devices
    max_memory["cpu"] = "24GiB"  # Allow CPU memory as fallback

    # Let HF handle distribution with max_memory
    device_map = "balanced" if len(ref_devices) > 1 else f"cuda:{ref_devices[0]}"
    model_name = cfg.model.name

    ref_model = get_hf_model(
        model_name,
        device_map=device_map,
        max_memory=max_memory,
        torch_dtype=getattr(torch, cfg.ref_model.torch_dtype),
        quantize=cfg.ref_model.quantization.enabled,
        gradient_checkpointing=cfg.ref_model.gradient_checkpointing,
        attn_implementation=cfg.ref_model.attn_implementation,
        lora=False,  # Reference model doesn't need LoRA
        requires_grad=False,
        eval_mode=True,
        lora_dropout=0.0,
    )[0]
    # Detach weights
    TensorDict.from_module(ref_model).data.to_module(ref_model)
    ref_model = TransformersWrapper(
        ref_model,
        tokenizer=tokenizer,
        input_mode="history",
        generate=False,
        return_log_probs=True,
        pad_output=False,
        device=torch.device("cuda:0"),
        chat_template_name="qwen",
    )
    return ref_model


def get_hf_model(
    model_name: str,
    torch_dtype: torch_dtype = torch.float32,
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.1,
    quantize: bool = False,
    fsdp: str = "",
    fsdp_config: Any = None,
    gradient_checkpointing: bool = True,
    device_map: str
    | dict[str, int | str | torch_device]
    | int
    | torch_device
    | None = None,
    lora: bool = True,
    attn_implementation: Literal["flash_attention_2", "flex_attention", "sdpa"]
    | None = "flex_attention",
    requires_grad: bool = True,
    compile: bool = False,
    max_memory: dict[str, str] | None = None,
    eval_mode: bool = False,
) -> tuple[AutoModelForCausalLM, PreTrainedTokenizer]:
    """Creates and configures a HuggingFace model with optional optimizations.

    Args:
        model_name (str): HuggingFace model identifier (e.g., "Qwen/Qwen2.5-3B")
        torch_dtype (torch.dtype, optional): Model precision. Default: torch.float32
        lora_r (int, optional): LoRA rank - controls capacity of adaptations. Default: 8
        lora_alpha (int, optional): LoRA alpha - scales the adaptations. Default: 16
        lora_dropout (float, optional): Dropout probability for LoRA layers. Default: 0.1
        quantize (bool, optional): Whether to enable 4-bit quantization. Default: False
        fsdp (str, optional): Fully Sharded Data Parallel configuration. Default: ""
        fsdp_config (Any, optional): Additional FSDP configurations. Default: None
        gradient_checkpointing (bool, optional): Whether to use gradient checkpointing. Default: True
        device_map (str | dict | int | torch.device | None, optional): Device placement strategy. Default: None
        lora (bool, optional): Whether to apply LoRA adapters. Default: True
        attn_implementation (Literal["flash_attention_2", "flex_attention", "sdpa"] | None, optional):
            Attention implementation to use. Default: "flex_attention"
        requires_grad (bool, optional): Whether to enable gradient computation. Default: True
        compile (bool, optional): Whether to enable model compilation. Default: False
        max_memory (dict[str, str], optional): Memory configuration for distributed training. Default: {}
        eval_mode (bool, optional): Whether to use the model in eval mode. Default: False

    Returns:
        tuple[AutoModelForCausalLM, PreTrainedTokenizer]:
            - model: The configured HuggingFace model
            - tokenizer: The associated tokenizer

    Raises:
        ImportError: If required dependencies are not installed
        RuntimeError: If model initialization fails
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if max_memory is None:
        max_memory = {}

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # tokenizer.eos_token = "<|im_end|>"
    if tokenizer.pad_token == tokenizer.eos_token:
        tokenizer.pad_token = "PAD"
    tokenizer.padding_side = "left"

    # Configure model settings for mixed precision
    # Store original dtype to restore it later
    original_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch_dtype)

    model_configs = {
        "torch_dtype": torch_dtype,
        "device_map": device_map if device_map is not None else "auto",
        "max_memory": max_memory,
    }
    if torch.cuda.is_available() and attn_implementation:
        torchrl_logger.info(f"{attn_implementation} init")
        model_configs["attn_implementation"] = attn_implementation

    try:
        # Configure training settings based on FSDP usage
        if fsdp != "" and fsdp_config is not None:
            torchrl_logger.info("Configurations for FSDP")
        else:
            pass

        # Enable Quantization
        if quantize:
            try:
                from transformers.utils.quantization_config import BitsAndBytesConfig
            except ImportError:
                raise ImportError(
                    "Please install transformers with bitsandbytes support"
                )

            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch_dtype,
            )
            model_configs["quantization_config"] = bnb_config

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            use_cache=not gradient_checkpointing,
            cache_dir="/tmp/.cache",
            **model_configs,
        )

        # Configure gradient checkpointing based on FSDP usage
        if fsdp == "" and fsdp_config is None:
            if gradient_checkpointing:
                torchrl_logger.info("gradient_checkpointing enabled")
                model.gradient_checkpointing_enable()
        else:
            if gradient_checkpointing:
                torchrl_logger.info("gradient_checkpointing enabled")
                model.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False}
                )

        if lora:
            try:
                from peft import get_peft_model, LoraConfig
            except ImportError:
                raise ImportError("Please install peft: pip install peft")

            # Create LoRA config with explicit dtype setting
            lora_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules="all-linear",
                lora_dropout=lora_dropout,  # Standard dropout for regularization
                bias="none",
                task_type="CAUSAL_LM",
                inference_mode=not eval_mode,  # CRITICAL: Must be False for training
                init_lora_weights=True,  # Good practice
            )

            # Initialize LoRA model
            model = get_peft_model(
                model,
                lora_config,
                autocast_adapter_dtype=False,  # Prevent automatic casting of adapter layers
            )

            # Force LoRA layers to correct dtype and eval mode
            for n, p in model.named_parameters():
                if "lora_" in n:  # Only convert LoRA parameters
                    p.data = p.data.to(torch_dtype)
        if eval_mode:
            model.eval()  # Ensure model is in eval mode
        else:
            model.train(True)
        if requires_grad:
            model.requires_grad_(True)
        else:
            model.requires_grad_(False)
        return model, tokenizer

    finally:
        # Restore original dtype
        torch.set_default_dtype(original_dtype)


def make_weight_updater(
    policy_training=None,
    master_address=None,
    master_port=None,
    model_metadata=None,
    vllm_tp_size=None,
) -> vLLMUpdater:
    """Creates a vLLM weight updater for the policy.

    This function can be used in two ways:
    1. Synchronous mode (expert-iteration-sync.py): Pass policy_training to get an initialized updater with metadata
    2. Async mode (expert-iteration-async.py): Pass master_address, master_port, model_metadata, and remote_actor

    Args:
        policy_training (Optional[TransformersWrapper]): The training policy model. Required for sync mode.
        master_address (Optional[str]): Ray master address for async mode.
        master_port (Optional[int]): Ray master port for async mode.
        model_metadata (Optional[dict]): Model metadata for async mode. If not provided but policy_training is,
            it will be extracted from the policy.
        vllm_tp_size (Optional[int]): vLLM tensor parallel size. If not provided, will be set to 1.

    Returns:
        vLLMUpdater: An instance of the weight updater configured to update
            the vLLM worker's weights.
    """
    if model_metadata is None and policy_training is not None:
        # Extract metadata from training policy
        model_metadata = {
            k: (v.dtype, v.shape) for k, v in policy_training.model.state_dict().items()
        }

    return vLLMUpdater(
        master_address=master_address,
        master_port=master_port,
        model_metadata=model_metadata,
        vllm_tp_size=vllm_tp_size,
    )


def compute_device_allocation(cfg):
    """Compute device allocation for different model components.

    Args:
        cfg: The configuration object

    Returns:
        A dictionary containing device allocations for different components
    """
    train_devices = cfg.train_model.num_devices
    inf_devices = cfg.inference_model.num_devices
    ref_devices = cfg.ref_model.num_devices

    # So we need all GPUs for Ray
    train_start = 0
    train_end = train_devices
    inference_start = 0
    inference_end = inf_devices
    ref_start = inference_end
    ref_end = ref_start + ref_devices
    ray_num_gpus = train_devices + inf_devices + ref_devices

    # Create device lists
    train_model_devices = list(range(train_start, train_end))
    inference_model_devices = list(range(inference_start, inference_end))
    ref_model_devices = list(range(ref_start, ref_end))

    # Get total unique devices for CUDA_VISIBLE_DEVICES
    all_devices = sorted(
        set(train_model_devices + inference_model_devices + ref_model_devices)
    )
    cuda_visible_devices = ",".join(map(str, all_devices))

    return {
        "train_model_devices": train_model_devices,
        "inference_model_devices": inference_model_devices,
        "ref_model_devices": ref_model_devices,
        "ray_num_gpus": ray_num_gpus,
        "cuda_visible_devices": cuda_visible_devices,
    }


def create_cosine_scheduler_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 0.5,
) -> torch.optim.lr_scheduler.LRScheduler:
    """Create a cosine scheduler with warmup using PyTorch's built-in schedulers.

    This function creates a learning rate scheduler that:
    1. Linearly increases the learning rate from 0 to the base learning rate during warmup
    2. Follows a cosine curve from the base learning rate to 0 after warmup

    Args:
        optimizer: The optimizer to schedule learning rates for
        num_warmup_steps: Number of warmup steps
        num_training_steps: Total number of training steps
        num_cycles: Number of cosine cycles (default: 0.5 for half a cycle)

    Returns:
        A PyTorch learning rate scheduler
    """
    from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

    # Create warmup scheduler (linear increase from 0 to base LR)
    warmup_scheduler = LinearLR(
        optimizer, start_factor=0.01, end_factor=1.0, total_iters=num_warmup_steps
    )

    # Create cosine decay scheduler (from base LR to 0)
    cosine_scheduler = CosineAnnealingLR(
        optimizer, T_max=num_training_steps - num_warmup_steps, eta_min=0.0
    )

    # Combine warmup and cosine decay
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[num_warmup_steps],
    )

    return scheduler


def get_wandb_run_id(wandb_logger):
    """Get the wandb run ID from a WandbLogger instance.

    Args:
        wandb_logger: The WandbLogger instance

    Returns:
        str: The wandb run ID, or None if not available
    """
    try:
        # Wait a bit for wandb to initialize
        import time

        max_attempts = 10
        for attempt in range(max_attempts):
            if hasattr(wandb_logger, "experiment") and wandb_logger.experiment:
                run_id = wandb_logger.experiment.id
                if run_id:
                    torchrl_logger.info(f"Got wandb run ID: {run_id}")
                    return run_id
            if attempt < max_attempts - 1:
                time.sleep(0.5)
                torchrl_logger.info(
                    f"Waiting for wandb run ID, attempt {attempt + 1}/{max_attempts}"
                )

        torchrl_logger.warning("Could not get wandb run ID after multiple attempts")
        return None
    except Exception as e:
        torchrl_logger.error(f"Error getting wandb run ID: {e}")
        return None


def log_training_metrics(
    wandb_logger,
    replay_buffer,
    batch,
    loss,
    grad_norm,
    global_step,
    data_read_count,
    collector,
    start_time,
    gradient_accumulation_steps,
    history_str=None,
):
    """Log training metrics to wandb.

    Args:
        wandb_logger: The wandb logger instance
        replay_buffer: The replay buffer containing collected data
        batch: The current training batch
        loss: The computed loss object
        grad_norm: The gradient norm value
        global_step: Current global training step
        data_read_count: Total data read count
        collector: The collector instance
        start_time: Training start time
        gradient_accumulation_steps: Number of gradient accumulation steps
        history_str: Optional history string for logging
    """
    with torch.no_grad():
        rb_content = replay_buffer[:]
        batch_policy_version = batch["next", "policy_version"].view(-1).min()
        batch_policy_age = collector.policy_version - batch_policy_version

        metrics = {
            "reward from buffer": float(
                torch.cat(rb_content.get(("next", "reward"), as_list=True)).mean()
            ),
            "reward from batch": float(batch["next", "reward"].mean()),
            "seq_length from buffer": float(
                torch.tensor(
                    [
                        t.numel()
                        for t in rb_content.get(("tokens", "response"), as_list=True)
                    ],
                    dtype=torch.float,
                ).mean()
            ),
            "loss_sft, from loss": float(loss.loss_sft),
            "loss_kl_to_ref, from loss": float(loss.loss_kl_to_ref),
            "kl_to_ref, from loss": float(loss.kl_to_ref),
            "grad_norm": float(grad_norm)
            if global_step % gradient_accumulation_steps == 0
            else 0.0,
            "write_count, from buffer": int(replay_buffer.write_count),
            # how many gradient steps per write
            "gradient_step_throughput (gradient step per write)": float(
                global_step / replay_buffer.write_count
            ),
            # how many optim steps per write
            "optim_step_throughput (optim step per write)": float(
                (global_step // gradient_accumulation_steps) / replay_buffer.write_count
            ),
            "data_read_count (total)": data_read_count,
            "current_policy_version (collector)": collector.policy_version,
            # FIXME: Assume batch is a single trajectory
            # FIXME: The addition of the transform after the env instantiation + _shuttle creation
            #  is messed up - we need the next data
            "batch_policy_version (sampled batch)": batch_policy_version,
            "batch_policy_age (sampled batch)": batch_policy_age,
            "throughput (steps per second)": float(
                global_step / (time.time() - start_time)
            ),
        }

        for name, value in metrics.items():
            wandb_logger.log_scalar(name, value, step=global_step)

        if history_str is not None:
            wandb_logger.log_str("history", history_str, step=global_step)


class RemoteDataLogger:
    """A remote post-processing function that sends logging data to the main process via Ray for centralized logging."""

    def __init__(self, log_queue):
        """Initialize RemoteDataLogger with a Ray actor reference for logging.

        Args:
            log_queue: Ray queue for logging data.
        """
        self.log_queue = log_queue
        self.last_time = None

    def __call__(self, data: TensorDict):
        self.log_data(data)
        return data

    def log_data(self, data: TensorDict):
        logs = {}
        if self.last_time is None:
            self.last_time = time.time()
        else:
            t = time.time()
            elapsed = t - self.last_time
            logs["collector/time/elapsed"] = elapsed
            self.last_time = t

        # Prepare logging data
        logs["collector/rewards/mean"] = float(data["next", "reward"].mean())
        logs["collector/rewards/std"] = float(data["next", "reward"].std())
        logs["collector/rewards/min"] = float(data["next", "reward"].min())
        logs["collector/rewards/max"] = float(data["next", "reward"].max())

        # Response length
        lengths = []
        responses = data["text", "response"]
        for r in responses:
            lengths.append(len(r))
        lengths = torch.tensor(lengths, dtype=torch.float32)
        logs["collector/response_length/mean"] = float(lengths.mean())
        logs["collector/response_length/std"] = float(lengths.std())
        logs["collector/response_length/min"] = float(lengths.min())
        logs["collector/response_length/max"] = float(lengths.max())

        policy_versions = data.get(("next", "policy_version"))
        if isinstance(policy_versions, torch.Tensor):
            policy_versions = policy_versions.float()
            logs["collector/policy_version/mean"] = float(policy_versions.mean())
            logs["collector/policy_version/min"] = float(policy_versions.min())
            logs["collector/policy_version/max"] = float(policy_versions.max())

        # Send to main process via Ray actor
        try:
            self.log_queue.put(logs)
        except Exception as e:
            torchrl_logger.error(f"Failed to send logs to main process: {e}")
