# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from torchrl._utils import logger

# Architecture name -> "module.path:ClassName" overrides registered with vLLM.
# Each path must stay importable; test_vllm_plugin.py guards against drift.
FP32_MODEL_OVERRIDES: dict[str, str] = {
    "Qwen3ForCausalLM": "torchrl.modules.llm.backends.vllm._models:Qwen3ForCausalLMFP32",
}


def register_fp32_overrides() -> None:
    """Register FP32 overrides for vLLM models."""
    from vllm.model_executor.models.registry import ModelRegistry

    for arch, model_cls_path in FP32_MODEL_OVERRIDES.items():
        ModelRegistry.register_model(arch, model_cls_path)

    logger.info("Registered Qwen3 FP32 model overrides")
