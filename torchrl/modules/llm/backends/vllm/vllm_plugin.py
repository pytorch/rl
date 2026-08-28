# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import os

from torchrl._utils import logger

# Env var that opts a vLLM process into torchrl's FP32 model overrides. torchrl's
# vLLM backend sets it (enable_fp32_output=True) before the engine and its
# subprocesses start, so the overrides register in every vLLM process torchrl
# owns -- and in none that it does not.
FP32_OVERRIDES_ENV_VAR = "TORCHRL_VLLM_FP32_OVERRIDES"

# Architecture name -> "module.path:ClassName" overrides registered with vLLM.
# Each path must stay importable; test_vllm_plugin.py guards against drift.
FP32_MODEL_OVERRIDES: dict[str, str] = {
    "Qwen3ForCausalLM": "torchrl.modules.llm.backends.vllm._models:Qwen3ForCausalLMFP32",
}


def fp32_overrides_enabled() -> bool:
    """Whether this process opted into torchrl's vLLM FP32 model overrides."""
    return os.environ.get(FP32_OVERRIDES_ENV_VAR, "0").lower() in ("1", "true", "yes")


def register_fp32_overrides() -> None:
    """Register torchrl's FP32 vLLM model overrides -- only when opted in.

    vLLM auto-loads this through the ``vllm.general_plugins`` entry point in
    *every* vLLM process, so it must do nothing unless this process explicitly
    asked for torchrl's overrides via ``TORCHRL_VLLM_FP32_OVERRIDES``. Otherwise
    merely *installing* torchrl would mutate an unrelated project's vLLM
    ``ModelRegistry`` -- replacing its model classes with torchrl's, which track
    a newer vLLM API and would break an older host vLLM at logits time.
    """
    if not fp32_overrides_enabled():
        return

    from vllm.model_executor.models.registry import ModelRegistry

    for arch, model_cls_path in FP32_MODEL_OVERRIDES.items():
        ModelRegistry.register_model(arch, model_cls_path)

    logger.info("Registered torchrl FP32 vLLM model overrides")


def enable_fp32_overrides() -> None:
    """Opt this process and its child vLLM processes into torchrl's overrides.

    Sets ``TORCHRL_VLLM_FP32_OVERRIDES`` so spawned vLLM workers and the registry
    subprocess inherit the opt-in, then registers in-process. Call before
    constructing a vLLM engine when you want torchrl's FP32 model overrides.
    """
    os.environ[FP32_OVERRIDES_ENV_VAR] = "1"
    register_fp32_overrides()
