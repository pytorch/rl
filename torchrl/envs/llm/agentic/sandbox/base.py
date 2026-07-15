# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Sandbox protocol and value types.

A :class:`Sandbox` is an async context manager owning an isolated
execution environment. :meth:`Sandbox.run` launches a subprocess inside
it, :meth:`Sandbox.write_file` and :meth:`Sandbox.read_file` mediate
I/O. The default backends -- :class:`BubblewrapSandbox` (Linux) and
:class:`SeatbeltSandbox` (macOS) -- enforce filesystem and network
isolation via OS-bundled tools.

For environments where neither is available,
:class:`UnsafeSubprocessSandbox` provides a no-op fallback that runs a
bare subprocess with no isolation. It emits a ``UserWarning`` on every
:meth:`open` call so the lack of containment is impossible to miss.

Value types (:class:`ResourceLimits`, :class:`SandboxResult`) are
:class:`tensordict.TensorClass` subclasses so they stack across batch
dims and compose with TorchRL's batched envs.
"""
from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import ClassVar, Literal, Protocol, runtime_checkable

from tensordict import TensorClass


_NetworkPolicy = Literal["none", "loopback", "allowlist", "full"]


def _normalise_root(path: str) -> str:
    """Return an absolute, normalised root without requiring it to exist."""
    return os.path.abspath(os.path.normpath(path))


def _path_is_within_roots(path: str, roots: Sequence[str]) -> bool:
    """Return whether ``path`` resolves inside one of ``roots``."""
    target = Path(path).resolve(strict=False)
    for root in roots:
        resolved_root = Path(root).resolve(strict=False)
        if target == resolved_root or resolved_root in target.parents:
            return True
    return False


def _intersect_roots(
    first: Sequence[str],
    second: Sequence[str],
    *,
    empty_is_unrestricted: bool,
) -> tuple[str, ...]:
    """Intersect two unions of filesystem subtrees."""
    if not first or not second:
        if empty_is_unrestricted:
            return tuple(second or first)
        return ()
    intersection: set[str] = set()
    for left in map(_normalise_root, first):
        for right in map(_normalise_root, second):
            try:
                common = os.path.commonpath((left, right))
            except ValueError:
                continue
            if common == left:
                intersection.add(right)
            elif common == right:
                intersection.add(left)
    return tuple(sorted(intersection))


class SandboxError(RuntimeError):
    """Raised on sandbox infrastructure failures.

    Covers launch failures, kernel errors, etc. Tool processes that
    exit non-zero do *not* raise; the non-zero status is surfaced via
    :attr:`SandboxResult.exit_code`.
    """


class ResourceLimits(TensorClass["nocast"]):
    """Per-sandbox or per-call resource limits.

    Attributes:
        cpu_seconds: Soft CPU budget. ``None`` means unlimited.
        wall_seconds: Wall-clock timeout. ``None`` means unlimited.
        memory_bytes: Address-space cap. ``None`` means unlimited.
        network: Policy for outbound network. ``"none"`` blocks all
            sockets, ``"loopback"`` allows 127.0.0.0/8 only,
            ``"allowlist"`` consults :attr:`network_allowlist`,
            ``"full"`` is unrestricted.
        network_allowlist: ``host:port`` strings, used only when
            ``network == "allowlist"``.
        fs_read_roots: Absolute paths the sandbox may read from.
            Empty means backend default (typically ``/`` read-only on
            Linux/macOS).
        fs_write_roots: Absolute paths the sandbox may write to.
            Empty means no writes allowed.
        max_processes: Cap on concurrent subprocesses. ``None`` for
            unlimited.
        env: Environment-variable allowlist. ``None`` means a clean
            env with only ``PATH``, ``HOME``, ``LANG``.
    """

    cpu_seconds: float | None = 30.0
    wall_seconds: float | None = 60.0
    memory_bytes: int | None = 512 * 1024 * 1024
    network: str = "none"
    network_allowlist: tuple = ()
    fs_read_roots: tuple = ()
    fs_write_roots: tuple = ()
    max_processes: int | None = 32
    env: Mapping[str, str] | None = None

    def narrow(self, other: ResourceLimits | None) -> ResourceLimits:
        """Return the tightest combination of ``self`` and ``other``.

        The result is at most as permissive as either input. Used by
        :meth:`Sandbox.run` to apply a per-call override that may only
        narrow the construction limits.
        """
        if other is None:
            return self

        def _min_or(a: float | None, b: float | None) -> float | None:
            if a is None:
                return b
            if b is None:
                return a
            return min(a, b)

        rank: dict[str, int] = {
            "none": 0,
            "loopback": 1,
            "allowlist": 2,
            "full": 3,
        }
        net = (
            self.network if rank[self.network] <= rank[other.network] else other.network
        )
        if self.network_allowlist or other.network_allowlist:
            allow = tuple(
                sorted(set(self.network_allowlist) & set(other.network_allowlist))
            )
        else:
            allow = ()
        # An empty read-root set means the backend default (unrestricted
        # read-only host access), whereas an empty write-root set means no
        # writes. Their intersections therefore have different identities.
        read_roots = _intersect_roots(
            self.fs_read_roots,
            other.fs_read_roots,
            empty_is_unrestricted=True,
        )
        write_roots = _intersect_roots(
            self.fs_write_roots,
            other.fs_write_roots,
            empty_is_unrestricted=False,
        )
        return ResourceLimits(
            cpu_seconds=_min_or(self.cpu_seconds, other.cpu_seconds),
            wall_seconds=_min_or(self.wall_seconds, other.wall_seconds),
            memory_bytes=_min_or(self.memory_bytes, other.memory_bytes),
            network=net,
            network_allowlist=allow,
            fs_read_roots=read_roots,
            fs_write_roots=write_roots,
            max_processes=_min_or(self.max_processes, other.max_processes),
            env=other.env if other.env is not None else self.env,
        )


class SandboxResult(TensorClass["nocast"]):
    """Outcome of a single :meth:`Sandbox.run` invocation.

    Attributes:
        stdout: Captured standard output (may be truncated).
        stderr: Captured standard error (may be truncated).
        exit_code: Subprocess exit status. Negative on signal.
        wall_seconds: Observed wall-clock duration.
        timed_out: ``True`` if the subprocess hit
            :attr:`ResourceLimits.wall_seconds` before exiting.
        truncated: ``True`` if stdout/stderr were truncated by an
            output cap.
        artifacts: File contents emitted under
            :attr:`ResourceLimits.fs_write_roots`, keyed by relative
            path. Populated lazily by backends that support it;
            default empty.
    """

    stdout: str
    stderr: str
    exit_code: int
    wall_seconds: float
    timed_out: bool = False
    truncated: bool = False
    artifacts: Mapping[str, bytes] | None = None


@runtime_checkable
class Sandbox(Protocol):
    """Async context manager owning an isolated execution environment.

    Lifecycle: ``open()`` is idempotent and required before
    ``run()``; ``close()`` releases all OS resources. Use as
    ``async with sandbox:`` to bracket lifecycle automatically.

    :meth:`run` does *not* raise on tool exit codes. It raises
    :class:`SandboxError` only on infrastructure failures (sandbox
    launch, host kernel error). Per-call ``limits`` may only narrow
    construction ``limits``; widening attempts are silently clamped.

    All paths in :meth:`write_file` / :meth:`read_file` are
    sandbox-virtual; the backend is responsible for translating to
    host paths.
    """

    name: ClassVar[str]
    limits: ResourceLimits

    async def open(self) -> None:
        ...

    async def close(self) -> None:
        ...

    async def __aenter__(self) -> Sandbox:
        ...

    async def __aexit__(self, exc_type, exc, tb) -> None:
        ...

    async def run(
        self,
        argv: Sequence[str],
        *,
        stdin: bytes | None = None,
        cwd: str | None = None,
        limits: ResourceLimits | None = None,
    ) -> SandboxResult:
        ...

    async def write_file(self, path: str, data: bytes) -> None:
        ...

    async def read_file(self, path: str, max_bytes: int | None = None) -> bytes:
        ...


__all__ = [
    "ResourceLimits",
    "Sandbox",
    "SandboxError",
    "SandboxResult",
]
