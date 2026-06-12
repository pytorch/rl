# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import importlib

import pytest

from torchrl.modules.llm.backends.vllm.vllm_plugin import FP32_MODEL_OVERRIDES


@pytest.mark.parametrize("arch", sorted(FP32_MODEL_OVERRIDES))
def test_fp32_override_paths_importable(arch):
    """Every registered override must point at an importable class.

    vLLM resolves these "module.path:ClassName" strings lazily, so a stale
    path is only discovered at server startup when vLLM inspects the
    architecture. This test does not require vLLM: ``_models`` falls back to
    placeholder classes when vLLM is absent, keeping the import path valid.
    """
    module_path, _, class_name = FP32_MODEL_OVERRIDES[arch].partition(":")
    module = importlib.import_module(module_path)
    assert hasattr(module, class_name), FP32_MODEL_OVERRIDES[arch]
