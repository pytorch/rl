# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""macOS sandbox-exec backend (seatbelt).

Generates a small Scheme profile from the :class:`ResourceLimits` and runs
the target command via ``sandbox-exec -p <profile> --``. Matches the
isolation guarantees of bubblewrap to the extent macOS allows: filesystem
read/write restrictions and full network deny.

.. note::
   Apple has officially deprecated ``sandbox-exec``, but the binary still
   ships with macOS 14+ and works for our purposes. Where stronger
   guarantees are needed (or for portability across CI platforms) prefer a
   container backend (Docker stub today).
"""
from __future__ import annotations

import asyncio
import os
import shutil
import time
from collections.abc import Sequence
from pathlib import Path
from typing import ClassVar

from .base import _path_is_within_roots, ResourceLimits, SandboxError, SandboxResult

_OUTPUT_CAP = 1 << 20

_has_sandbox_exec = shutil.which("sandbox-exec") is not None


def _profile(limits: ResourceLimits) -> str:
    """Build a sandbox-exec Scheme profile from ``limits``."""
    if limits.network == "allowlist":
        raise SandboxError(
            "SeatbeltSandbox cannot enforce network='allowlist'. Use "
            "network='none' or provide an externally enforced network policy."
        )
    lines: list[str] = [
        "(version 1)",
        "(deny default)",
        "(allow process-fork)",
        "(allow process-exec)",
        "(allow signal (target self))",
        "(allow sysctl-read)",
        "(allow mach-lookup)",
        "(allow ipc-posix-shm)",
    ]
    if limits.fs_read_roots:
        for root in limits.fs_read_roots:
            if not Path(root).is_absolute():
                raise SandboxError(
                    f"sandbox filesystem roots must be absolute: {root!r}"
                )
            lines.append(f'(allow file-read* (subpath "{root}"))')
    else:
        lines.append("(allow file-read*)")  # documented backend default
    if limits.network in ("none", "loopback"):
        lines.append("(deny network*)")
    else:
        lines.append("(allow network*)")
    if limits.fs_write_roots:
        # Allow writes only under the named roots.
        for root in limits.fs_write_roots:
            lines.append(f'(allow file-write* (subpath "{root}"))')
    # /private/var, /tmp need write for many runtimes; allow only if user
    # explicitly listed them.
    return "\n".join(lines)


class SeatbeltSandbox:
    """macOS sandbox backed by ``sandbox-exec``.

    Args:
        limits: Construction-time resource limits.

    Raises:
        SandboxError: at :meth:`open` if ``sandbox-exec`` is not available.
    """

    name: ClassVar[str] = "seatbelt"

    def __init__(self, limits: ResourceLimits | None = None) -> None:
        self.limits = limits if limits is not None else ResourceLimits()
        self._exec = shutil.which("sandbox-exec")
        self._opened = False

    async def open(self) -> None:
        if self._opened:
            return
        if not self._exec:
            raise SandboxError(
                "sandbox-exec not found. SeatbeltSandbox requires macOS."
            )
        self._opened = True

    async def close(self) -> None:
        self._opened = False

    async def __aenter__(self) -> SeatbeltSandbox:
        await self.open()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.close()

    def _build_argv(
        self,
        argv: Sequence[str],
        limits: ResourceLimits,
        cwd: str | None = None,
    ) -> list[str]:
        return [self._exec or "sandbox-exec", "-p", _profile(limits), "--", *argv]

    def _build_env(self, limits: ResourceLimits) -> dict[str, str]:
        if limits.env is None:
            return {
                "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
                "HOME": os.environ.get("HOME", "/tmp"),
                "LANG": os.environ.get("LANG", "C.UTF-8"),
            }
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
        sb_argv = self._build_argv(argv, eff)
        t0 = time.monotonic()
        try:
            proc = await asyncio.create_subprocess_exec(
                *sb_argv,
                stdin=asyncio.subprocess.PIPE if stdin is not None else None,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd,
                env=self._build_env(eff),
            )
        except FileNotFoundError as e:
            raise SandboxError(f"could not launch sandbox-exec: {e}") from e
        try:
            out_b, err_b = await asyncio.wait_for(
                proc.communicate(stdin), timeout=eff.wall_seconds
            )
            timed_out = False
        except TimeoutError:
            proc.kill()
            try:
                out_b, err_b = await proc.communicate()
            except Exception:  # pragma: no cover
                out_b, err_b = b"", b""
            timed_out = True
        wall = time.monotonic() - t0
        truncated = len(out_b) > _OUTPUT_CAP or len(err_b) > _OUTPUT_CAP
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
        if not _path_is_within_roots(path, self.limits.fs_write_roots):
            raise SandboxError(
                f"refusing to write to {path!r}: outside fs_write_roots "
                f"{self.limits.fs_write_roots!r}"
            )
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_bytes(data)

    async def read_file(self, path: str, max_bytes: int | None = None) -> bytes:
        if not self._opened:
            raise SandboxError("sandbox is not open; call open() first")
        if self.limits.fs_read_roots and not _path_is_within_roots(
            path, self.limits.fs_read_roots
        ):
            raise SandboxError(
                f"refusing to read {path!r}: outside fs_read_roots "
                f"{self.limits.fs_read_roots!r}"
            )
        b = Path(path).read_bytes()
        if max_bytes is not None and len(b) > max_bytes:
            return b[:max_bytes]
        return b


__all__ = ["SeatbeltSandbox"]
