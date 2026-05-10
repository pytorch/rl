# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Sandbox protocol and value types.

A :class:`Sandbox` is an async context manager owning an isolated execution
environment. :meth:`Sandbox.run` launches a subprocess inside it,
:meth:`Sandbox.write_file` and :meth:`Sandbox.read_file` mediate I/O. The
default backends -- :class:`BubblewrapSandbox` (Linux) and
:class:`SeatbeltSandbox` (macOS) -- enforce filesystem and network isolation
via OS-bundled tools.

For environments where neither is available,
:class:`UnsafeSubprocessSandbox` provides a no-op fallback that runs a bare
subprocess with no isolation. It emits a ``UserWarning`` on every
:meth:`open` call so the lack of containment is impossible to miss.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, ClassVar, Literal, Protocol, runtime_checkable


_NetworkPolicy = Literal["none", "loopback", "allowlist", "full"]


class SandboxError(RuntimeError):
    """Raised on sandbox infrastructure failures (launch, kernel error, etc.).

    Tool processes that exit non-zero do *not* raise; the non-zero status is
    surfaced via :attr:`SandboxResult.exit_code`.
    """


@dataclass(frozen=True, slots=True)
class ResourceLimits:
    """Per-sandbox or per-call resource limits.

    Attributes:
        cpu_seconds: Soft CPU budget. ``None`` means unlimited.
        wall_seconds: Wall-clock timeout. ``None`` means unlimited.
        memory_bytes: Address-space cap. ``None`` means unlimited.
        network: Policy for outbound network. ``"none"`` blocks all sockets,
            ``"loopback"`` allows 127.0.0.0/8 only, ``"allowlist"`` consults
            :attr:`network_allowlist`, ``"full"`` is unrestricted.
        network_allowlist: ``host:port`` strings, used only when
            ``network == "allowlist"``.
        fs_read_roots: Absolute paths the sandbox may read from. Empty means
            backend default (typically ``/`` read-only on Linux/macOS).
        fs_write_roots: Absolute paths the sandbox may write to. Empty means
            no writes allowed.
        max_processes: Cap on concurrent subprocesses. ``None`` for unlimited.
        env: Environment-variable allowlist. ``None`` means a clean env with
            only ``PATH``, ``HOME``, ``LANG``.
    """

    cpu_seconds: float | None = 30.0
    wall_seconds: float | None = 60.0
    memory_bytes: int | None = 512 * 1024 * 1024
    network: _NetworkPolicy = "none"
    network_allowlist: tuple[str, ...] = ()
    fs_read_roots: tuple[str, ...] = ()
    fs_write_roots: tuple[str, ...] = ()
    max_processes: int | None = 32
    env: Mapping[str, str] | None = None

    def narrow(self, other: ResourceLimits | None) -> ResourceLimits:
        """Return a new :class:`ResourceLimits` that is at most as permissive
        as both ``self`` and ``other``. Used by :meth:`Sandbox.run` to apply a
        per-call override that may only narrow the construction limits.
        """
        if other is None:
            return self

        def _min_or(a: float | None, b: float | None) -> float | None:
            if a is None:
                return b
            if b is None:
                return a
            return min(a, b)

        # Tighten network: choose the strictest.
        rank: dict[_NetworkPolicy, int] = {
            "none": 0,
            "loopback": 1,
            "allowlist": 2,
            "full": 3,
        }
        net = self.network if rank[self.network] <= rank[other.network] else other.network
        return ResourceLimits(
            cpu_seconds=_min_or(self.cpu_seconds, other.cpu_seconds),
            wall_seconds=_min_or(self.wall_seconds, other.wall_seconds),
            memory_bytes=_min_or(self.memory_bytes, other.memory_bytes),
            network=net,
            network_allowlist=(
                tuple(set(self.network_allowlist) & set(other.network_allowlist))
                if self.network_allowlist or other.network_allowlist
                else ()
            ),
            fs_read_roots=tuple(
                sorted(set(self.fs_read_roots) & set(other.fs_read_roots))
            )
            if self.fs_read_roots and other.fs_read_roots
            else (self.fs_read_roots or other.fs_read_roots),
            fs_write_roots=tuple(
                sorted(set(self.fs_write_roots) & set(other.fs_write_roots))
            )
            if self.fs_write_roots and other.fs_write_roots
            else (self.fs_write_roots or other.fs_write_roots),
            max_processes=_min_or(self.max_processes, other.max_processes),
            env=other.env if other.env is not None else self.env,
        )


@dataclass(frozen=True, slots=True)
class SandboxResult:
    """Outcome of a single :meth:`Sandbox.run` invocation.

    Attributes:
        stdout: Captured standard output (may be truncated).
        stderr: Captured standard error (may be truncated).
        exit_code: Subprocess exit status. Negative on signal.
        wall_seconds: Observed wall-clock duration.
        timed_out: ``True`` if the subprocess hit
            :attr:`ResourceLimits.wall_seconds` before exiting.
        truncated: ``True`` if stdout/stderr were truncated by an output cap.
        artifacts: File contents emitted under
            :attr:`ResourceLimits.fs_write_roots`, keyed by relative path.
            Populated lazily by backends that support it; default empty.
    """

    stdout: str
    stderr: str
    exit_code: int
    wall_seconds: float
    timed_out: bool = False
    truncated: bool = False
    artifacts: Mapping[str, bytes] = field(default_factory=dict)


@runtime_checkable
class Sandbox(Protocol):
    """An async context manager owning an isolated execution environment.

    Lifecycle: ``open()`` is idempotent and required before ``run()``;
    ``close()`` releases all OS resources. Use as ``async with sandbox:`` to
    bracket lifecycle automatically.

    :meth:`run` does *not* raise on tool exit codes. It raises
    :class:`SandboxError` only on infrastructure failures (sandbox launch,
    host kernel error). Per-call ``limits`` may only narrow construction
    ``limits``; widening attempts are silently clamped.

    All paths in :meth:`write_file` / :meth:`read_file` are sandbox-virtual;
    the backend is responsible for translating to host paths.
    """

    name: ClassVar[str]
    limits: ResourceLimits

    async def open(self) -> None: ...

    async def close(self) -> None: ...

    async def __aenter__(self) -> Sandbox: ...

    async def __aexit__(self, exc_type, exc, tb) -> None: ...

    async def run(
        self,
        argv: Sequence[str],
        *,
        stdin: bytes | None = None,
        cwd: str | None = None,
        limits: ResourceLimits | None = None,
    ) -> SandboxResult: ...

    async def write_file(self, path: str, data: bytes) -> None: ...

    async def read_file(
        self, path: str, max_bytes: int | None = None
    ) -> bytes: ...


__all__ = [
    "ResourceLimits",
    "Sandbox",
    "SandboxError",
    "SandboxResult",
]
