# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""vLLM backends for TorchRL.

This module provides comprehensive vLLM integration including:
- Base classes and interfaces
- Synchronous vLLM workers
- Asynchronous vLLM services
- Shared utilities

Examples:
    >>> # Create an async vLLM service (recommended)
    >>> from torchrl.modules.llm.backends.vllm import AsyncVLLM
    >>> service = AsyncVLLM.from_pretrained("Qwen/Qwen2.5-3B")

    >>> # Create a sync Ray worker
    >>> from torchrl.modules.llm.backends.vllm import make_vllm_worker
    >>> worker = make_vllm_worker("Qwen/Qwen2.5-3B", make_ray_worker=True)

    >>> # All engines implement the same interface
    >>> from torchrl.modules.llm.backends.vllm import RLvLLMEngine
    >>> updater = vLLMUpdaterV2(any_engine)  # Works with any RLvLLMEngine
"""

from __future__ import annotations

from typing import Any

__all__ = [
    # Base classes and interfaces
    "RLvLLMEngine",
    # Synchronous vLLM
    "make_vllm_worker",
    "RayLLMWorker",
    "LocalLLMWrapper",
    # Asynchronous vLLM
    "AsyncVLLM",
    "make_async_vllm_engine",
    "_AsyncLLMEngine",
    "_AsyncvLLMWorker",
    # Utilities
    "stateless_init_process_group",
    "stateless_init_process_group_async",
]

_LAZY_ATTRS: dict[str, tuple[str, str]] = {
    # Base
    "RLvLLMEngine": ("torchrl.modules.llm.backends.vllm.base", "RLvLLMEngine"),
    # Sync
    "make_vllm_worker": (
        "torchrl.modules.llm.backends.vllm.vllm_sync",
        "make_vllm_worker",
    ),
    "RayLLMWorker": ("torchrl.modules.llm.backends.vllm.vllm_sync", "RayLLMWorker"),
    "LocalLLMWrapper": (
        "torchrl.modules.llm.backends.vllm.vllm_sync",
        "LocalLLMWrapper",
    ),
    # Async
    "_AsyncLLMEngine": (
        "torchrl.modules.llm.backends.vllm.vllm_async",
        "_AsyncLLMEngine",
    ),
    "_AsyncvLLMWorker": (
        "torchrl.modules.llm.backends.vllm.vllm_async",
        "_AsyncvLLMWorker",
    ),
    "AsyncVLLM": ("torchrl.modules.llm.backends.vllm.vllm_async", "AsyncVLLM"),
    "make_async_vllm_engine": (
        "torchrl.modules.llm.backends.vllm.vllm_async",
        "make_async_vllm_engine",
    ),
    # Utils
    "stateless_init_process_group": (
        "torchrl.modules.llm.backends.vllm.vllm_utils",
        "stateless_init_process_group",
    ),
    "stateless_init_process_group_async": (
        "torchrl.modules.llm.backends.vllm.vllm_utils",
        "stateless_init_process_group_async",
    ),
}


def __getattr__(name: str) -> Any:  # noqa: ANN401
    target = _LAZY_ATTRS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = target
    module = __import__(module_name, fromlist=[attr_name])
    return getattr(module, attr_name)
