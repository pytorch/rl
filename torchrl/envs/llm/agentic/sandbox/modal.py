# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Modal hosted-sandbox backend (stub).

Tracked in the agentic ``__init__.py`` TODO list. Stub kept so the import
surface is stable.
"""
from __future__ import annotations

import importlib.util
from collections.abc import Sequence
from typing import ClassVar

from .base import ResourceLimits, Sandbox, SandboxError, SandboxResult

_has_modal = importlib.util.find_spec("modal") is not None


class ModalSandbox:
    """Modal-hosted sandbox (not yet implemented)."""

    name: ClassVar[str] = "modal"

    def __init__(self, limits: ResourceLimits | None = None) -> None:
        self.limits = limits or ResourceLimits()

    async def open(self) -> None:
        raise NotImplementedError(
            "ModalSandbox is not yet implemented. See the TODO list in "
            "torchrl/envs/llm/agentic/__init__.py."
        )

    async def close(self) -> None:
        pass

    async def __aenter__(self) -> ModalSandbox:  # pragma: no cover
        await self.open()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:  # pragma: no cover
        await self.close()

    async def run(
        self,
        argv: Sequence[str],
        *,
        stdin: bytes | None = None,
        cwd: str | None = None,
        limits: ResourceLimits | None = None,
    ) -> SandboxResult:  # pragma: no cover
        raise NotImplementedError

    async def write_file(self, path: str, data: bytes) -> None:  # pragma: no cover
        raise NotImplementedError

    async def read_file(
        self, path: str, max_bytes: int | None = None
    ) -> bytes:  # pragma: no cover
        raise NotImplementedError


__all__ = ["ModalSandbox"]
