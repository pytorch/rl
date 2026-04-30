# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""LLM policy wrappers.

This subpackage includes optional wrappers that may rely on native extensions
(e.g. vLLM). To avoid importing optional dependencies at module import time,
we avoid importing those dependencies at module import time.
"""

from __future__ import annotations

from .common import ChatHistory, LLMWrapperBase, LogProbs, Masks, Text, Tokens
from .sglang_wrapper import SGLangWrapper
from .transformers_wrapper import RemoteTransformersWrapper, TransformersWrapper
from .vllm_wrapper import vLLMWrapper

__all__ = [
    "TransformersWrapper",
    "RemoteTransformersWrapper",
    "vLLMWrapper",
    "SGLangWrapper",
    "LLMWrapperBase",
    "Text",
    "LogProbs",
    "Masks",
    "Tokens",
    "ChatHistory",
]
