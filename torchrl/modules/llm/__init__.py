# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""LLM utilities for TorchRL.

Note:
    This package contains optional integrations (e.g. vLLM) that may rely on native
    extensions. To keep `import torchrl` / `import torchrl.envs` lightweight and
    robust, we **avoid importing optional backends at module import time** and
    instead only import those backends on demand.
"""

from __future__ import annotations

from typing import Any

from .policies.common import ChatHistory, LLMWrapperBase, LogProbs, Masks, Text, Tokens
from .policies.sglang_wrapper import SGLangWrapper
from .policies.transformers_wrapper import (
    RemoteTransformersWrapper,
    TransformersWrapper,
)
from .policies.vllm_wrapper import vLLMWrapper

__all__ = [
    # Data structures
    "ChatHistory",
    "LogProbs",
    "Masks",
    "Text",
    "Tokens",
    # Wrapper base class
    "LLMWrapperBase",
    # Local wrappers
    "TransformersWrapper",
    "vLLMWrapper",
    "SGLangWrapper",
    # Remote wrappers
    "RemoteTransformersWrapper",
    # Async vLLM (recommended)
    "AsyncVLLM",
    "make_async_vllm_engine",
    "stateless_init_process_group_async",
    # Sync vLLM utilities
    "make_vllm_worker",
    "stateless_init_process_group",
    # Async SGLang
    "AsyncSGLang",
    "RLSGLangEngine",
]


def __getattr__(name: str) -> Any:  # noqa: ANN401
    # Keep backends optional and on-demand to avoid importing vLLM/SGLang native extensions
    # as a side-effect of importing torchrl.
    if name in {
        "AsyncVLLM",
        "make_async_vllm_engine",
        "make_vllm_worker",
        "stateless_init_process_group",
        "stateless_init_process_group_async",
        "AsyncSGLang",
        "RLSGLangEngine",
    }:
        from . import backends  # local import is intentional / required

        return getattr(backends, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
