# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import contextlib
import os
from typing import Any, Literal, Sequence

import torch
from omegaconf import DictConfig
from torch import device as torch_device, dtype as torch_dtype

from torchrl import logger as torchrl_logger
from torchrl.modules.llm import TransformersWrapper, vLLMWrapper
from transformers.models.auto.modeling_auto import AutoModelForCausalLM
from transformers.tokenization_utils import PreTrainedTokenizer


@contextlib.contextmanager
def cuda_visible_devices(devices: Sequence[int]):
    """Context manager for temporarily setting CUDA_VISIBLE_DEVICES.

    This utility function allows temporary modification of CUDA device visibility,
    useful for controlling which GPUs are accessible to different model components.

    Args:
        devices (Sequence[int]): List of CUDA device indices to make visible

    Yields:
        None: Use as a context manager

    Example:
        >>> with cuda_visible_devices([0, 1]):
        ...     # Only GPUs 0 and 1 will be visible here
        ...     model = create_model()
    """
    CUDA_VISIBLE_DEVICES = os.getenv("CUDA_VISIBLE_DEVICES")
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, devices))
    yield
    if CUDA_VISIBLE_DEVICES:
        os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES
    else:
        os.unsetenv("CUDA_VISIBLE_DEVICES")


def get_train_model(
    cfg: DictConfig,
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
    train_devices = cfg.train_model.get("devices", [0])

    # Use cuda_visible_devices to restrict visible GPUs and let HF handle distribution
    with cuda_visible_devices(train_devices):
        device_map = "balanced" if len(train_devices) > 1 else f"cuda:0"
        train_model, train_tokenizer = get_hf_model(
            cfg.model.name,
            device_map=device_map,
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
            from_text=False,
            generate=False,
            return_log_probs=True,
        )
    return policy_training, train_tokenizer


def get_inference_model(cfg: DictConfig) -> vLLMWrapper:
    """Creates the vLLM-based inference model for fast generation.

    This function initializes a vLLM model server for efficient inference and wraps
    it in a vLLMWrapper for policy inference. vLLM provides optimized generation
    with better throughput than standard HuggingFace generation.

    Args:
        cfg (DictConfig): The hydra configuration object containing model settings.
            Expected to have inference_model section with vLLM-specific parameters
            like gpu_memory_utilization and generation settings.

    Returns:
        vLLMWrapper: The wrapped vLLM model ready for inference.

    Raises:
        AssertionError: If the vLLM server or model initialization fails
    """
    from torchrl.modules.llm.backends.vllm import make_vllm_worker

    vllm_devices = cfg.inference_model.get("devices", [1])
    torchrl_logger.info(f"Creating inference model on devices {vllm_devices}")

    model_name = cfg.model.name

    # vLLM handles device mapping internally
    inference_server = make_vllm_worker(
        model_name,
        gpu_memory_utilization=cfg.inference_model.gpu_memory_utilization,
        devices=list(vllm_devices),  # Convert to list for type compatibility
        make_ray_worker=True,
    )
    assert inference_server is not None
    policy = vLLMWrapper(
        inference_server,
        from_text=True,
        return_log_probs=True,
        generate_kwargs={
            "max_tokens": cfg.inference_model.max_tokens,
            "include_stop_str_in_output": cfg.inference_model.include_stop_str_in_output,
            "temperature": cfg.inference_model.temperature,
        },
    )
    assert policy.model is not None
    return policy


def get_ref_model(
    cfg: DictConfig, tokenizer: PreTrainedTokenizer
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
    ref_devices = cfg.ref_model.get("devices", [2])

    # Use cuda_visible_devices to restrict to reference device
    with cuda_visible_devices(ref_devices):
        device_map = "balanced" if len(ref_devices) > 1 else f"cuda:0"
        model_name = cfg.model.name

        ref_model = get_hf_model(
            model_name,
            device_map=device_map,
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
            tokenizer=tokenizer,
            from_text=False,
            generate=False,
            return_log_probs=True,
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

    Returns:
        tuple[AutoModelForCausalLM, PreTrainedTokenizer]:
            - model: The configured HuggingFace model
            - tokenizer: The associated tokenizer

    Raises:
        ImportError: If required dependencies are not installed
        RuntimeError: If model initialization fails
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
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
                lora_dropout=lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
                inference_mode=False,
                init_lora_weights=True,  # This ensures weights are initialized
            )

            # Initialize LoRA model
            model = get_peft_model(
                model,
                lora_config,
                autocast_adapter_dtype=False,  # Prevent automatic casting of adapter layers
            )

            # Force LoRA layers to correct dtype
            for n, p in model.named_parameters():
                if "lora_" in n:  # Only convert LoRA parameters
                    p.data = p.data.to(torch_dtype)

        if requires_grad:
            model.requires_grad_(True)

        return model, tokenizer

    finally:
        # Restore original dtype
        torch.set_default_dtype(original_dtype)
