# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import functools

import time
import warnings
from typing import Any, Literal

import torch
from omegaconf import DictConfig
from torch import device as torch_device, dtype as torch_dtype

from torchrl._utils import logger as torchrl_logger, timeit
from torchrl.collectors.llm.weight_update.vllm_v2 import vLLMUpdaterV2
from torchrl.envs.llm import AddThinkingPrompt, GSM8KEnv, KLRewardTransform, RetrieveKL
from torchrl.envs.llm.datasets.ifeval import IFEvalEnv
from torchrl.modules.llm import TransformersWrapper, vLLMWrapper
from transformers.models.auto.modeling_auto import AutoModelForCausalLM
from transformers.tokenization_utils import PreTrainedTokenizer


def check_grpo_dependencies() -> None:
    """Check for required GRPO dependencies and provide helpful error messages.

    This function checks for critical dependencies needed for GRPO training and
    provides installation instructions for missing packages.
    """
    missing_packages = []
    missing_optional = []

    # Core required packages
    required_packages = {
        "datasets": "pip install datasets",
        "peft": "pip install peft",
        "wandb": "pip install wandb",
        "vllm": "pip install vllm",
        "transformers": "pip install transformers",
        "accelerate": "pip install accelerate",
        "ray": "pip install ray",
        "tqdm": "pip install tqdm",
    }

    # Optional but recommended packages
    optional_packages = {
        "flash_attn": "pip install flash-attn",
        "bitsandbytes": "pip install bitsandbytes",
        "xformers": "pip install xformers",
    }

    # Check required packages
    for package, install_cmd in required_packages.items():
        try:
            __import__(package)
        except ImportError:
            missing_packages.append((package, install_cmd))

    # Check optional packages
    for package, install_cmd in optional_packages.items():
        try:
            __import__(package)
        except ImportError:
            missing_optional.append((package, install_cmd))

    # Report missing required packages
    if missing_packages:
        error_msg = (
            "Missing required packages for GRPO training:\n"
            + "\n".join(f"  - {pkg}: {cmd}" for pkg, cmd in missing_packages)
            + "\n\nYou can install all GRPO dependencies with:\n"
            + "  pip install torchrl[grpo]\n"
            + "or install individual packages as shown above."
        )
        raise ImportError(error_msg)

    # Report missing optional packages as warnings
    if missing_optional:
        warning_msg = (
            "Missing optional packages that may improve GRPO performance:\n"
            + "\n".join(f"  - {pkg}: {cmd}" for pkg, cmd in missing_optional)
            + "\n\nThese packages are optional but recommended for optimal performance."
        )
        warnings.warn(warning_msg, UserWarning, stacklevel=2)

    torchrl_logger.info("✓ All required GRPO dependencies are available")


def get_tokenizer(cfg: DictConfig) -> PreTrainedTokenizer:
    from transformers import AutoTokenizer

    model_name = cfg.model.name
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # tokenizer.eos_token = "<|im_end|>"
    if tokenizer.pad_token == tokenizer.eos_token:
        tokenizer.pad_token = "PAD"
    tokenizer.padding_side = "left"
    return tokenizer


def get_train_model(
    cfg: DictConfig,
    devices: list[int] | None = None,
) -> tuple[TransformersWrapper, PreTrainedTokenizer]:
    """Creates and configures the training model with LoRA adapters.

    This function initializes the main training model with LoRA adapters and other
    training-specific configurations like gradient checkpointing. The model is wrapped
    in a TransformersWrapper for policy training.

    Args:
        cfg (DictConfig): The hydra configuration object containing model and training settings.
            Expected to have train_model section with LoRA, quantization, and other
            training-specific parameters.

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
    )

    # Force all model parameters to the same dtype
    for param in train_model.parameters():
        param.data = param.data.to(model_dtype)

    policy_training = TransformersWrapper(
        train_model,
        tokenizer=train_tokenizer,
        input_mode="tokens" if not cfg.env.reasoning else "history",
        generate=False,
        return_log_probs=True,
        pad_output=False,
        device=torch.device("cuda:0"),
        # Enable packing when cfg.train.packing=True by disabling padding
        pad_model_input=not cfg.train.packing,
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
        tokenizer (PreTrainedTokenizer, optional): The tokenizer to use with the inference model. Default: `None`.

    Returns:
        vLLMWrapper: The wrapped vLLM model ready for inference.

    Raises:
        AssertionError: If the vLLM server or model initialization fails
    """
    from torchrl.modules.llm.backends.vllm import AsyncVLLM

    num_devices = cfg.inference_model.num_devices
    if num_devices is None:
        vllm_devices = devices if devices is not None else [1]
        num_devices = len(vllm_devices)
    else:
        vllm_devices = None
    torchrl_logger.info(
        f"Creating AsyncVLLM inference model with num_devices={num_devices}, devices={vllm_devices}"
    )

    model_name = cfg.model.name

    # Use AsyncVLLM for better performance and async processing
    verbose = getattr(cfg.inference_model, "verbose", True)
    compile_model = getattr(
        cfg.inference_model, "compile", False
    )  # Disabled by default for GRPO

    # Build parameters dict for AsyncVLLM with all config options
    inference_params = {
        "model_name": model_name,
        "num_devices": 1,
        "num_replicas": num_devices,
        "gpu_memory_utilization": cfg.inference_model.gpu_memory_utilization,
        "enforce_eager": cfg.inference_model.enforce_eager,
        "verbose": verbose,
        "compile": compile_model,
    }

    # CRITICAL FIX: Configure attention implementation to prevent Flash Attention errors
    # vLLM doesn't accept attn_implementation directly through AsyncEngineArgs
    # Instead, we set the VLLM_ATTENTION_BACKEND environment variable
    if hasattr(cfg.inference_model, "attn_implementation"):
        import os

        attn_impl = cfg.inference_model.attn_implementation

        # Map common attention implementations to vLLM backend names
        attn_backend_map = {
            "flash_attention_2": "FLASH_ATTN",
            "flash_attn": "FLASH_ATTN",
            "sdpa": "TORCH_SDPA",
            "torch_sdpa": "TORCH_SDPA",
            "xformers": "XFORMERS",
        }

        vllm_backend = attn_backend_map.get(attn_impl, attn_impl.upper())
        os.environ["VLLM_ATTENTION_BACKEND"] = vllm_backend

        torchrl_logger.info(
            f"Setting VLLM_ATTENTION_BACKEND={vllm_backend} (from config: {attn_impl})"
        )

    # Add other common vLLM parameters from config if present
    optional_vllm_params = [
        "max_model_len",
        "dtype",
        "trust_remote_code",
        "seed",
        "swap_space",
        "cpu_offload_gb",
        "enable_prefix_caching",
        "tensor_parallel_size",
        "pipeline_parallel_size",
    ]

    for param in optional_vllm_params:
        if hasattr(cfg.inference_model, param):
            value = getattr(cfg.inference_model, param)
            if value is not None:
                inference_params[param] = value

    # Handle torch_dtype specifically (convert string to torch dtype)
    if hasattr(cfg.inference_model, "torch_dtype"):
        dtype_str = cfg.inference_model.torch_dtype
        if dtype_str is not None:
            if isinstance(dtype_str, str):
                inference_params["dtype"] = getattr(torch, dtype_str)
            else:
                inference_params["dtype"] = dtype_str

    inference_server = AsyncVLLM.from_pretrained(**inference_params)
    assert inference_server is not None
    if tokenizer is None:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token == tokenizer.eos_token:
            tokenizer.pad_token = "PAD"
        tokenizer.padding_side = "left"
    policy = vLLMWrapper(
        inference_server,
        input_mode="history",
        chat_template_name="qwen",
        return_log_probs=not cfg.env.reasoning,
        tokenizer=tokenizer,
        pad_output=False,
        generate_kwargs={
            "max_tokens": cfg.inference_model.max_tokens,
            "include_stop_str_in_output": cfg.inference_model.include_stop_str_in_output,
            "temperature": cfg.inference_model.temperature,
            "top_p": cfg.inference_model.top_p,
        },
    )
    assert policy.model is not None
    return policy


def get_ref_model(
    cfg: DictConfig,
    tokenizer: PreTrainedTokenizer,
    devices: list[int] | None = None,
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
    from tensordict import TensorDict

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
    )[0].eval()
    # Detach weights
    TensorDict.from_module(ref_model).data.to_module(ref_model)
    ref_model = TransformersWrapper(
        ref_model,
        input_mode="tokens" if not cfg.env.reasoning else "history",
        tokenizer=tokenizer,
        generate=False,
        return_log_probs=True,
        pad_output=False,
        device=torch.device("cuda:0"),
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
            bnb_config_params = {"bnb_4bit_quant_storage": torch_dtype}
        else:
            bnb_config_params = {}

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
                **bnb_config_params,
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
                lora_dropout=0.0,  # Disable dropout for RL training
                bias="none",
                task_type="CAUSAL_LM",
                inference_mode=True,  # Force inference mode for consistent behavior
                init_lora_weights=True,  # This ensures weights are initialized
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

        model.eval()  # Ensure model is in eval mode
        if requires_grad:
            model.requires_grad_(True)

        return model, tokenizer

    finally:
        # Restore original dtype
        torch.set_default_dtype(original_dtype)


def make_weight_updater(
    vllm_engine,
) -> vLLMUpdaterV2:
    """Creates a vLLM weight updater for the policy using the new V2 API.

    The V2 updater is much simpler - it just needs a vLLM engine that implements
    the RLvLLMEngine interface (like RayLLMWorker, LocalLLMWrapper, or AsyncVLLM).

    Args:
        vllm_engine: A vLLM engine implementing the RLvLLMEngine interface.
            This is typically obtained from the inference policy's model attribute.

    Returns:
        vLLMUpdaterV2: An instance of the weight updater configured to update
            the vLLM worker's weights through the engine's own methods.
    """
    return vLLMUpdaterV2(vllm_engine=vllm_engine)


def compute_device_allocation(cfg):
    """Compute device allocations and Ray GPU config.

    Args:
        cfg: The configuration object

    Returns:
        dict: Updated device configuration containing:
            - train_model_devices: list of devices for training
            - inference_model_devices: list of devices for inference
            - ray_num_gpus: number of GPUs to tell Ray about
            - cuda_visible_devices: string for CUDA_VISIBLE_DEVICES
    """
    train_devices = cfg.train_model.num_devices
    inf_devices = cfg.inference_model.num_devices

    train_start = 0
    train_end = train_devices
    inference_start = 0
    inference_end = inf_devices

    ref_devices = cfg.ref_model.num_devices if cfg.train.use_kl_to_ref else 0
    ray_num_gpus = train_devices + inf_devices + ref_devices

    train_model_devices = list(range(train_start, train_end))
    inference_model_devices = list(range(inference_start, inference_end))

    all_devices = sorted(set(train_model_devices + inference_model_devices))
    if cfg.train.use_kl_to_ref:
        ref_device_start = max(all_devices) + 1 if all_devices else 0
        ref_devices_list = list(range(ref_device_start, ref_device_start + ref_devices))
        all_devices.extend(ref_devices_list)
    cuda_visible_devices = ",".join(map(str, all_devices))

    return {
        "train_model_devices": train_model_devices,
        "inference_model_devices": inference_model_devices,
        "ray_num_gpus": ray_num_gpus,
        "cuda_visible_devices": cuda_visible_devices,
    }


def make_env(cfg: DictConfig, single_env: bool = False):
    """Create the environment.

    Args:
        cfg: The configuration object

    Returns:
        The configured environment
    """
    train_tokenizer = get_tokenizer(cfg)

    # Setup environment
    max_steps = cfg.env.max_steps if cfg.env.reasoning else 1
    if cfg.env.dataset == "gsm8k":
        # Reward scale is 0.0 to 100
        reward_threshold = 20
        env = GSM8KEnv(
            repeats=cfg.env.repeats,
            tokenizer=train_tokenizer,
            num_envs=cfg.env.num_envs if not single_env else 1,
            max_steps=max_steps,
            device=torch.device("cpu"),
            ray_backend=True,
        )
    elif cfg.env.dataset == "ifeval":  # ifeval
        # Reward scale is 0.0 to 2.2
        reward_threshold = 1.0
        env = IFEvalEnv(
            repeats=cfg.env.repeats,
            tokenizer=train_tokenizer,
            num_envs=cfg.env.num_envs if not single_env else 1,
            max_steps=max_steps,
            device=torch.device("cpu"),
            ray_backend=True,
        )
    else:
        raise NotImplementedError(f"Dataset {cfg.env.dataset} not implemented")

    if cfg.env.reasoning:
        env = env.append_transform(
            AddThinkingPrompt(
                cond=lambda td, reward_threshol=reward_threshold, max_steps=max_steps: td[
                    "reward"
                ]
                <= reward_threshold
                and td["step_count"] < max_steps,
                role="user",
                edit_last_turn=False,
                zero_reward=False,
                undo_done=True,
                random_prompt=True,
            ),
        )
    return env


def make_ref_model_factory(cfg: DictConfig) -> functools.partial | None:
    """Create a factory for the reference model if KL to ref is enabled.

    Args:
        cfg: The configuration object

    Returns:
        A partial function that creates the reference model, or None if KL to ref is disabled
    """
    if not cfg.train.use_kl_to_ref:
        return None

    train_tokenizer = get_tokenizer(cfg)
    ref_cfg = DictConfig(dict(cfg))
    ref_model_factory = functools.partial(
        get_ref_model,
        ref_cfg,
        train_tokenizer,
        devices=[0],
    )
    return ref_model_factory


def add_kl_transforms_to_replay_buffer(replay_buffer, cfg: DictConfig):
    """Add KL transforms to replay buffer.

    Args:
        replay_buffer: The replay buffer to add transforms to
        cfg: The configuration object
    """
    if not cfg.train.use_kl_to_ref:
        return

    ref_model_factory = make_ref_model_factory(cfg)
    if ref_model_factory is None:
        return

    if cfg.env.reasoning:
        kl_transform = RetrieveKL(
            ref_model_factory=ref_model_factory,
            add_to_reward=not cfg.train.kl_coef_in_loss,
            coeff=cfg.train.kl_to_ref_coeff,
            use_ray_service=True,
        )
    else:
        kl_transform = KLRewardTransform(
            ref_model_factory=ref_model_factory,
            coef=cfg.train.kl_to_ref_coeff,
            add_to_reward=not cfg.train.kl_coef_in_loss,
            device=torch.device("cuda:0"),
            use_ray_service=True,
        )

    replay_buffer.append_transform(kl_transform, invert=True)


@timeit("Logging metrics")
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
    use_kl_to_ref=True,
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
        step_count = rb_content.get(("next", "step_count")).view(-1).float().mean()
        batch_policy_version = batch["next", "policy_version"].view(-1).min()
        batch_policy_age = collector.policy_version - batch_policy_version

        metrics = {
            "step_count from buffer": float(step_count),
            "reward from buffer": float(
                torch.cat(rb_content.get(("next", "reward"), as_list=True)).mean()
            ),
            "seq_length from buffer": float(
                torch.tensor(
                    [
                        t.numel()
                        for t in rb_content.get(("tokens", "response"), as_list=True)
                    ],
                    dtype=torch.float,
                ).mean()
            ),
            "ESS, from loss": float(loss.ESS),
            "loss_objective, from loss": float(loss.loss_objective),
            "clip_fraction, from loss": float(loss.clip_fraction),
            "kl_approx (train to inference), from loss": float(loss.kl_approx),
            "kl_to_inference (train to inference - differentiable), from loss": float(
                loss.kl_to_inference.mean()
            ),
            "loss_kl_to_inference, from loss": float(loss.loss_kl_to_inference.mean()),
            "entropy loss, from loss": float(loss.loss_entropy.mean()),
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
        if use_kl_to_ref:
            metrics["kl_penalty (inference to ref) from buffer"] = float(
                torch.cat(rb_content.get(("next", "kl_penalty"), as_list=True)).mean()
            )
            metrics["kl_to_ref, from loss"] = float(loss.kl_to_ref.mean())
            metrics["loss_kl_to_ref, from loss"] = float(loss.loss_kl_to_ref.mean())

        for name, value in metrics.items():
            wandb_logger.log_scalar(name, value, step=global_step)

        if history_str is not None:
            wandb_logger.log_str("history", history_str, step=global_step)
