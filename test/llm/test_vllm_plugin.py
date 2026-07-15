# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import importlib

import pytest

from torchrl.modules.llm.backends.vllm.vllm_plugin import (
    FP32_MODEL_OVERRIDES,
    fp32_overrides_enabled,
    FP32_OVERRIDES_ENV_VAR,
    register_fp32_overrides,
)


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


@pytest.mark.parametrize(
    "value,expected",
    [
        (None, False),
        ("0", False),
        ("", False),
        ("no", False),
        ("1", True),
        ("true", True),
        ("True", True),
        ("yes", True),
    ],
)
def test_fp32_overrides_enabled_reads_env(monkeypatch, value, expected):
    if value is None:
        monkeypatch.delenv(FP32_OVERRIDES_ENV_VAR, raising=False)
    else:
        monkeypatch.setenv(FP32_OVERRIDES_ENV_VAR, value)
    assert fp32_overrides_enabled() is expected


def test_register_fp32_overrides_is_noop_without_optin(monkeypatch):
    """Without the opt-in, registration must do nothing -- and must not even
    import vLLM. This is what lets another project install torchrl without its
    vLLM ``ModelRegistry`` being mutated. Returning before the vLLM import keeps
    the no-op path safe on machines with no vLLM at all.
    """
    monkeypatch.delenv(FP32_OVERRIDES_ENV_VAR, raising=False)
    # Must not raise even where vLLM is absent (early return precedes the import).
    register_fp32_overrides()
