# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from torchrl._utils import logger


def register_fp32_overrides() -> None:
    """Register FP32 overrides for vLLM models."""
    from vllm.model_executor.models.registry import ModelRegistry

    # ======= Register models here =======
    # Register Qwen3 models with FP32 override
    ModelRegistry.register_model(
        "Qwen3ForCausalLM",
        "torchrl.modules.llm.backends._models:Qwen3ForCausalLMFP32",
    )

    logger.info("Registered Qwen3 FP32 model overrides")
