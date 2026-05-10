# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Docker / Podman sandbox backend (stub).

Tracked in the agentic ``__init__.py`` TODO list as
"E2B / Modal / Docker real implementations." This file exists so the
import surface is stable from day one and downstream code can reference
``DockerSandbox`` even before the implementation lands.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import ClassVar

from .base import ResourceLimits, Sandbox, SandboxError, SandboxResult


class DockerSandbox:
    """Container-based sandbox (not yet implemented)."""

    name: ClassVar[str] = "docker"

    def __init__(
        self,
        limits: ResourceLimits | None = None,
        *,
        image: str = "python:3.11-slim",
    ) -> None:
        self.limits = limits or ResourceLimits()
        self.image = image

    async def open(self) -> None:
        raise NotImplementedError(
            "DockerSandbox is not yet implemented. See the TODO list in "
            "torchrl/envs/llm/agentic/__init__.py and contribute! For now "
            "use BubblewrapSandbox (Linux) or SeatbeltSandbox (macOS)."
        )

    async def close(self) -> None:
        pass

    async def __aenter__(self) -> DockerSandbox:  # pragma: no cover
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


__all__ = ["DockerSandbox"]
