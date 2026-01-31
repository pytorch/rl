# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""SGLang backends for TorchRL.

This module provides comprehensive SGLang integration including:
- Base classes and interfaces
- Asynchronous SGLang server services
- Shared utilities

Examples:
    >>> # Connect to an existing SGLang server
    >>> from torchrl.modules.llm.backends.sglang import AsyncSGLang
    >>> service = AsyncSGLang.connect("http://localhost:30000")

    >>> # Launch a managed SGLang server
    >>> from torchrl.modules.llm.backends.sglang import AsyncSGLang
    >>> service = AsyncSGLang.from_pretrained("Qwen/Qwen2.5-3B")

    >>> # All engines implement the same interface
    >>> from torchrl.modules.llm.backends.sglang import RLSGLangEngine
"""

from __future__ import annotations

from typing import Any

__all__ = [
    # Base classes and interfaces
    "RLSGLangEngine",
    # Asynchronous SGLang
    "AsyncSGLang",
    # Utilities
    "get_open_port",
    "wait_for_server",
]

_LAZY_ATTRS: dict[str, tuple[str, str]] = {
    # Base
    "RLSGLangEngine": ("torchrl.modules.llm.backends.sglang.base", "RLSGLangEngine"),
    # Async
    "AsyncSGLang": (
        "torchrl.modules.llm.backends.sglang.sglang_server",
        "AsyncSGLang",
    ),
    # Utils
    "get_open_port": (
        "torchrl.modules.llm.backends.sglang.sglang_utils",
        "get_open_port",
    ),
    "wait_for_server": (
        "torchrl.modules.llm.backends.sglang.sglang_utils",
        "wait_for_server",
    ),
}


def __getattr__(name: str) -> Any:  # noqa: ANN401
    target = _LAZY_ATTRS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = target
    module = __import__(module_name, fromlist=[attr_name])
    return getattr(module, attr_name)
