# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Bubblewrap-backed sandbox (Linux default).

Builds a ``bwrap`` argv prefix from the :class:`ResourceLimits` and runs the
target command inside the resulting unprivileged user namespace. This gives
us:

- a private mount namespace (write_roots are bind-mounted RW, the rest is RO)
- a private network namespace by default (``--unshare-net``); ``"allowlist"``
  and ``"full"`` keep the host network namespace and rely on the caller to
  ensure connections only succeed where allowed.
- a private PID namespace (``--unshare-pid``)
- ``--die-with-parent`` so the sandbox dies with the parent process.

The implementation is best-effort: bubblewrap's API is large, and edge
cases (rootless overlays, nested user namespaces in some kernels) are
documented but not silently papered over.
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

_has_bwrap = shutil.which("bwrap") is not None


class BubblewrapSandbox:
    """Linux sandbox backed by ``bwrap`` (bubblewrap).

    Args:
        limits: Construction-time resource limits.
        bwrap_path: Override the ``bwrap`` executable path. Default uses
            :func:`shutil.which`.

    Raises:
        SandboxError: at :meth:`open` time if ``bwrap`` is not on ``PATH``
            and ``bwrap_path`` was not supplied.

    Example:
        >>> import asyncio  # doctest: +SKIP
        >>> from torchrl.envs.llm.agentic.sandbox import (
        ...     BubblewrapSandbox, ResourceLimits,
        ... )
        >>> async def go():
        ...     async with BubblewrapSandbox(
        ...         limits=ResourceLimits(network="none",
        ...                               fs_write_roots=("/tmp/work",))
        ...     ) as s:
        ...         r = await s.run(["python3", "-c", "print('hi')"])
        ...         return r.stdout.strip()
    """

    name: ClassVar[str] = "bubblewrap"

    def __init__(
        self,
        limits: ResourceLimits | None = None,
        *,
        bwrap_path: str | None = None,
    ) -> None:
        self.limits = limits if limits is not None else ResourceLimits()
        self._bwrap = bwrap_path or shutil.which("bwrap")
        self._opened = False

    async def open(self) -> None:
        if self._opened:
            return
        if not self._bwrap:
            raise SandboxError(
                "bwrap not found on PATH. Install bubblewrap "
                "(apt-get install bubblewrap / dnf install bubblewrap) or "
                "use UnsafeSubprocessSandbox for testing."
            )
        self._opened = True

    async def close(self) -> None:
        self._opened = False

    async def __aenter__(self) -> BubblewrapSandbox:
        await self.open()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.close()

    def _build_argv(
        self, argv: Sequence[str], limits: ResourceLimits, cwd: str | None
    ) -> list[str]:
        if limits.network == "allowlist":
            raise SandboxError(
                "BubblewrapSandbox cannot enforce network='allowlist'. Use "
                "network='none' or provide an externally enforced network policy."
            )
        bw: list[str] = [self._bwrap or "bwrap"]
        bw += ["--die-with-parent", "--unshare-user", "--unshare-pid", "--unshare-ipc"]
        if limits.network in ("none", "loopback"):
            bw += ["--unshare-net"]
        # Empty read roots retain the documented backend default of a read-only
        # host root. When roots are supplied, start from an empty filesystem and
        # expose exactly those roots plus explicitly writable roots.
        if limits.fs_read_roots:
            bw += ["--tmpfs", "/"]
            roots = (*limits.fs_read_roots, *limits.fs_write_roots)
            parent_dirs: set[str] = {"/proc", "/dev"}
            for root in roots:
                root_path = Path(root)
                if not root_path.is_absolute():
                    raise SandboxError(
                        f"sandbox filesystem roots must be absolute: {root!r}"
                    )
                if root in limits.fs_read_roots and not root_path.exists():
                    raise SandboxError(f"filesystem read root does not exist: {root!r}")
                parent_dirs.update(str(parent) for parent in root_path.parents)
            for directory in sorted(parent_dirs, key=lambda value: value.count("/")):
                if directory != "/":
                    bw += ["--dir", directory]
            for root in limits.fs_read_roots:
                bw += ["--ro-bind", root, root]
        else:
            bw += ["--ro-bind", "/", "/"]
        bw += ["--proc", "/proc", "--dev", "/dev"]
        for root in limits.fs_write_roots:
            Path(root).mkdir(parents=True, exist_ok=True)
            bw += ["--bind", root, root]
        if cwd:
            bw += ["--chdir", cwd]
        # Clear env, then re-add only what we want.
        bw += ["--clearenv"]
        env = limits.env or {
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "HOME": "/tmp",
            "LANG": os.environ.get("LANG", "C.UTF-8"),
        }
        for k, v in env.items():
            bw += ["--setenv", k, v]
        # prlimit guards memory; CPU seconds we leave to wall_seconds + ulimit
        # via shell-out only when memory_bytes is set.
        if limits.memory_bytes is not None and shutil.which("prlimit"):
            bw += [
                "prlimit",
                f"--as={limits.memory_bytes}",
                "--",
            ]
        bw += list(argv)
        return bw

    def _build_env(self) -> dict[str, str]:
        # bwrap sees the parent env only for argv expansion; --clearenv +
        # --setenv handle the child env. Keep the parent-side env minimal.
        return {
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        }

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
        bw_argv = self._build_argv(argv, eff, cwd)
        t0 = time.monotonic()
        try:
            proc = await asyncio.create_subprocess_exec(
                *bw_argv,
                stdin=asyncio.subprocess.PIPE if stdin is not None else None,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=self._build_env(),
            )
        except FileNotFoundError as e:
            raise SandboxError(f"could not launch bwrap: {e}") from e
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
        # Writes happen on the host side at a path that will be bind-mounted
        # RW into the sandbox. Verify the target lies under a write root.
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


__all__ = ["BubblewrapSandbox"]
