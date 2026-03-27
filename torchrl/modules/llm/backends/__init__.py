# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""LLM backends.

These backends can be optional and may rely on native extensions. We avoid
importing them at module import time and lazily load on attribute access.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    # Base classes - vLLM
    "RLvLLMEngine",
    # Sync vLLM
    "make_vllm_worker",
    "RayLLMWorker",
    "LocalLLMWrapper",
    # Async vLLM
    "_AsyncvLLMWorker",
    "_AsyncLLMEngine",
    "AsyncVLLM",
    "make_async_vllm_engine",
    # Utilities - vLLM
    "stateless_init_process_group",
    "stateless_init_process_group_async",
    # Base classes - SGLang
    "RLSGLangEngine",
    # Async SGLang
    "AsyncSGLang",
]

_LAZY_ATTRS: dict[str, tuple[str, str]] = {
    # Base classes and interfaces - vLLM
    "RLvLLMEngine": ("torchrl.modules.llm.backends.vllm", "RLvLLMEngine"),
    # Sync vLLM
    "make_vllm_worker": ("torchrl.modules.llm.backends.vllm", "make_vllm_worker"),
    "RayLLMWorker": ("torchrl.modules.llm.backends.vllm", "RayLLMWorker"),
    "LocalLLMWrapper": ("torchrl.modules.llm.backends.vllm", "LocalLLMWrapper"),
    # Async vLLM
    "_AsyncvLLMWorker": ("torchrl.modules.llm.backends.vllm", "_AsyncvLLMWorker"),
    "_AsyncLLMEngine": ("torchrl.modules.llm.backends.vllm", "_AsyncLLMEngine"),
    "AsyncVLLM": ("torchrl.modules.llm.backends.vllm", "AsyncVLLM"),
    "make_async_vllm_engine": (
        "torchrl.modules.llm.backends.vllm",
        "make_async_vllm_engine",
    ),
    # Utilities - vLLM
    "stateless_init_process_group": (
        "torchrl.modules.llm.backends.vllm",
        "stateless_init_process_group",
    ),
    "stateless_init_process_group_async": (
        "torchrl.modules.llm.backends.vllm",
        "stateless_init_process_group_async",
    ),
    # Base classes and interfaces - SGLang
    "RLSGLangEngine": ("torchrl.modules.llm.backends.sglang", "RLSGLangEngine"),
    # Async SGLang
    "AsyncSGLang": ("torchrl.modules.llm.backends.sglang", "AsyncSGLang"),
}


def __getattr__(name: str) -> Any:  # noqa: ANN401
    target = _LAZY_ATTRS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = target
    module = __import__(module_name, fromlist=[attr_name])
    return getattr(module, attr_name)
