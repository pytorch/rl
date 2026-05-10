# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Unsandboxed subprocess backend (testing / fallback only).

Runs argv directly via :func:`asyncio.create_subprocess_exec` with no
isolation. Emits ``UserWarning`` on every :meth:`open` call so the lack of
containment is impossible to miss in a real deployment.
"""
from __future__ import annotations

import asyncio
import os
import shutil
import time
import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import ClassVar

from torchrl._utils import logger as torchrl_logger

from .base import ResourceLimits, Sandbox, SandboxError, SandboxResult

_OUTPUT_CAP = 1 << 20  # 1 MiB per stream


class UnsafeSubprocessSandbox:
    """Bare ``asyncio.create_subprocess_exec`` with no isolation.

    Useful for unit tests and environments where neither bubblewrap nor
    sandbox-exec is available. **Not a security boundary.** Emits a
    :class:`UserWarning` on every :meth:`open` so this is loud.

    The ``limits.fs_write_roots`` and ``limits.network`` policies are *not
    enforced* by this backend; pass them anyway so tests can switch to
    :class:`BubblewrapSandbox` or :class:`SeatbeltSandbox` without code
    changes.

    Examples:
        >>> import asyncio
        >>> async def go():
        ...     async with UnsafeSubprocessSandbox() as s:
        ...         r = await s.run(["echo", "hi"])
        ...         return r.stdout.strip()
        >>> asyncio.run(go())  # doctest: +SKIP
        'hi'
    """

    name: ClassVar[str] = "unsafe-subprocess"

    def __init__(self, limits: ResourceLimits | None = None) -> None:
        self.limits = limits or ResourceLimits()
        self._opened = False

    async def open(self) -> None:
        if self._opened:
            return
        warnings.warn(
            "UnsafeSubprocessSandbox provides NO isolation. Do not use it "
            "with untrusted model output in production. Switch to "
            "BubblewrapSandbox (Linux) or SeatbeltSandbox (macOS).",
            UserWarning,
            stacklevel=2,
        )
        self._opened = True

    async def close(self) -> None:
        self._opened = False

    async def __aenter__(self) -> UnsafeSubprocessSandbox:
        await self.open()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.close()

    def _build_env(self, limits: ResourceLimits) -> dict[str, str]:
        if limits.env is None:
            base = {
                "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
                "HOME": os.environ.get("HOME", "/tmp"),
                "LANG": os.environ.get("LANG", "C.UTF-8"),
            }
            return base
        return dict(limits.env)

    async def run(
        self,
        argv: Sequence[str],
        *,
        stdin: bytes | None = None,
        cwd: str | None = None,
        limits: ResourceLimits | None = None,
    ) -> SandboxResult:
        if not self._opened:
            raise SandboxError("sandbox is not open; call open() first")
        eff = self.limits.narrow(limits)
        env = self._build_env(eff)
        t0 = time.monotonic()
        try:
            proc = await asyncio.create_subprocess_exec(
                *argv,
                stdin=asyncio.subprocess.PIPE if stdin is not None else None,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd,
                env=env,
            )
        except FileNotFoundError as e:
            raise SandboxError(f"could not launch subprocess: {e}") from e
        try:
            out_b, err_b = await asyncio.wait_for(
                proc.communicate(stdin),
                timeout=eff.wall_seconds,
            )
            timed_out = False
        except asyncio.TimeoutError:
            proc.kill()
            try:
                out_b, err_b = await proc.communicate()
            except Exception:  # pragma: no cover -- defensive
                out_b, err_b = b"", b""
            timed_out = True
        wall = time.monotonic() - t0
        truncated = len(out_b) > _OUTPUT_CAP or len(err_b) > _OUTPUT_CAP
        if truncated:
            torchrl_logger.warning(
                "UnsafeSubprocessSandbox truncated subprocess output (cap=%d)",
                _OUTPUT_CAP,
            )
        return SandboxResult(
            stdout=out_b[:_OUTPUT_CAP].decode("utf-8", errors="replace"),
            stderr=err_b[:_OUTPUT_CAP].decode("utf-8", errors="replace"),
            exit_code=proc.returncode if proc.returncode is not None else -1,
            wall_seconds=wall,
            timed_out=timed_out,
            truncated=truncated,
        )

    async def write_file(self, path: str, data: bytes) -> None:
        if not self._opened:
            raise SandboxError("sandbox is not open; call open() first")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_bytes(data)

    async def read_file(
        self, path: str, max_bytes: int | None = None
    ) -> bytes:
        if not self._opened:
            raise SandboxError("sandbox is not open; call open() first")
        b = Path(path).read_bytes()
        if max_bytes is not None and len(b) > max_bytes:
            return b[:max_bytes]
        return b


def _shutil_which(name: str) -> str | None:
    return shutil.which(name)


__all__ = ["UnsafeSubprocessSandbox"]
