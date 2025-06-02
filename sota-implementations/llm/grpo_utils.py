# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import contextlib
import os
from typing import Literal

import torch

from tensordict import TensorDict

from torchrl import logger as torchrl_logger

from torchrl.modules.llm import TransformersWrapper, vLLMWrapper


@contextlib.contextmanager
def cuda_visible_devices(devices: list[int]):
    CUDA_VISIBLE_DEVICES = os.getenv("CUDA_VISIBLE_DEVICES")
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, devices))
    yield
    if CUDA_VISIBLE_DEVICES:
        os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES
    else:
        os.unsetenv("CUDA_VISIBLE_DEVICES")


def get_train_model(args, train_devices):
    torchrl_logger.info("Creating train model")

    with torch.device(f"cuda:{train_devices[0]}"):
        train_model, train_tokenizer = get_hf_model(
            args.model_name, device_map=train_devices[0]
        )
    policy_training = TransformersWrapper(
        train_model,
        # train_model.eval(),
        tokenizer=train_tokenizer,
        # We have the tokens, let's just use them
        from_text=False,
        generate=False,
        return_log_probs=True,
    )
    return policy_training, train_tokenizer


def get_train_inference_model(args, train_devices):
    torchrl_logger.info("Creating train model")

    with torch.device(
        f"cuda:{train_devices[0]}"
    ) if train_devices else contextlib.nullcontext():
        train_model, train_tokenizer = get_hf_model(
            args.model_name,
            device_map=train_devices[0] if train_devices else [],
            lora=args.lora,
            gradient_checkpointing=args.gradient_checkpointing,
            quantize=args.quantize,
        )
    policy_training = TransformersWrapper(
        train_model,
        # train_model.eval(),
        tokenizer=train_tokenizer,
        # We have the tokens, let's just use them
        from_text=False,
        generate=False,
        return_log_probs=True,
    )
    policy_inference = TransformersWrapper(
        train_model,
        # train_model.eval(),
        tokenizer=train_tokenizer,
        from_text=True,
        generate=True,
        return_log_probs=True,
        generate_kwargs={"max_new_tokens": 1024},
    )
    return policy_training, policy_inference, train_tokenizer


def get_inference_model(args, vllm_devices):
    from torchrl.modules.llm.backends.vllm import make_vllm_worker

    torchrl_logger.info(f"Creating inference model on devices {vllm_devices}")

    model_name = args.model_name

    inference_server = make_vllm_worker(
        model_name,
        gpu_memory_utilization=args.gpu_memory_utilization,
        devices=vllm_devices,
        make_ray_worker=True,
    )
    assert inference_server is not None
    policy = vLLMWrapper(
        inference_server,
        from_text=True,
        return_log_probs=True,
        generate_kwargs={
            "max_tokens": 1024,
            "include_stop_str_in_output": True,
            "temperature": 0.8,
        },
    )
    assert policy.model is not None
    return policy


def get_ref_model(args, tokenizer, ref_device):
    torchrl_logger.info("Creating ref model")
    with torch.device(f"cuda:{ref_device}"):
        model_name = args.model_name
        from transformers import AutoModelForCausalLM

        ref_model = get_hf_model(
            model_name, device_map=ref_device, torch_dtype=torch.bfloat16, quantize=True, gradient_checkpointing=False,
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
    model_name,
    torch_dtype=torch.bfloat16,
    lora_r=8,  # LoRA rank - controls capacity of adaptations
    lora_alpha=16,  # LoRA alpha - scales the adaptations
    lora_dropout=0.1,  # Dropout probability for LoRA layers
    quantize: bool = False,
    fsdp="",  # Fully Sharded Data Parallel configuration
    fsdp_config=None,  # Additional FSDP configurations
    gradient_checkpointing=True,  # Whether to use gradient checkpointing
    # merge_weights=False,  # Whether to merge LoRA weights with base model
    # seed=42,  # Random seed for reproducibility
    device_map: str
    | dict[str, int | str | torch.device]
    | int
    | torch.device
    | None = None,
    lora: bool = True,
    attn_implementation: Literal["flash_attention_2", "flex_attention", "sdpa"]
    | None = "flex_attention",
):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token == tokenizer.eos_token:
        tokenizer.pad_token = "PAD"
    tokenizer.padding_side = "left"

    # Configure model settings for bfloat16 precision
    # Setup flash_attention_2 for memory-efficient attention computation
    if torch_dtype == torch.bfloat16:

        model_configs = {
            "torch_dtype": torch_dtype,
        }
        if torch.cuda.is_available() and attn_implementation:
            torchrl_logger.info(f"{attn_implementation} init")
            model_configs["attn_implementation"] = attn_implementation
    else:
        model_configs = {}

    if device_map:
        model_configs["device_map"] = device_map
    # Configure training settings based on FSDP usage
    # Set up trainer configurations for FSDP or standard training
    if fsdp != "" and fsdp_config is not None:
        torchrl_logger.info("Configurations for FSDP")

        bnb_config_params = {"bnb_4bit_quant_storage": torch_dtype}
    else:
        bnb_config_params = {}

    # Enable Quantization
    if quantize:
        from transformers import BitsAndBytesConfig

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
        # print("Prepare model for quantization")
        # model = prepare_model_for_kbit_training(
        #     model, use_gradient_checkpointing=gradient_checkpointing
        # )

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
        from peft import get_peft_model, LoraConfig

        config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules="all-linear",
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            inference_mode=False,
        )

        model = get_peft_model(model, config).eval()

    model.requires_grad_(True)

    return model, tokenizer
