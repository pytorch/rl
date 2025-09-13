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

# Base classes and interfaces
from .base import RLvLLMEngine

# Asynchronous vLLM
from .vllm_async import (
    _AsyncLLMEngine,
    _AsyncvLLMWorker,
    AsyncVLLM,
    make_async_vllm_engine,
)

# Synchronous vLLM
from .vllm_sync import LocalLLMWrapper, make_vllm_worker, RayLLMWorker

# Shared utilities
from .vllm_utils import stateless_init_process_group, stateless_init_process_group_async

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
