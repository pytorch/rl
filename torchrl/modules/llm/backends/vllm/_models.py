# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Override the last layers of your models here."""

from __future__ import annotations

import os

import torch

try:
    from vllm.config import VllmConfig
    from vllm.model_executor.models.qwen3 import Qwen3ForCausalLM
except ImportError:

    class VllmConfig:
        """Placeholder for VllmConfig class when vLLM is not installed."""

    class Qwen3ForCausalLM:
        """Placeholder for Qwen3ForCausalLM class when vLLM is not installed."""


def is_fp32_output_enabled() -> bool:
    """Check if FP32 output is enabled."""
    return os.getenv("VLLM_ENABLE_FP32_OUTPUT", "0") == "1"


class Qwen3ForCausalLMFP32(Qwen3ForCausalLM):
    """Qwen3ForCausalLM with FP32 output."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        if is_fp32_output_enabled():
            self.lm_head.float()

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        if is_fp32_output_enabled():
            hidden_states = hidden_states.float()
        logits = self.logits_processor(self.lm_head, hidden_states)
        return logits
