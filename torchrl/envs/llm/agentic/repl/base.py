# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""REPL protocol and value types.

A :class:`Repl` runs stateful code inside a :class:`Sandbox`. State persists
across :meth:`execute` calls until :meth:`restart`. :meth:`interrupt`
preserves state but cancels the current execution. Timeouts surface as
``ReplResult.timed_out=True`` rather than raising.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, ClassVar, Literal, Protocol, runtime_checkable

from ..sandbox.base import Sandbox


@dataclass(frozen=True, slots=True)
class ReplDisplay:
    """A rich output (image, JSON, HTML) emitted via Jupyter's display
    protocol. Subprocess REPLs emit nothing here.
    """

    media_type: str
    data: Any


@dataclass(frozen=True, slots=True)
class ReplError:
    """Structured error from the kernel (exception name, value, traceback)."""

    ename: str
    evalue: str
    traceback: str = ""


@dataclass(frozen=True, slots=True)
class ReplResult:
    """Outcome of one :meth:`Repl.execute` invocation.

    Attributes:
        stdout: Captured stdout.
        stderr: Captured stderr.
        display: Rich outputs in emit order.
        error: Structured error, if any.
        timed_out: ``True`` if execution hit the timeout.
        execution_count: Monotonic counter (Jupyter); ``-1`` for subprocess.
    """

    stdout: str = ""
    stderr: str = ""
    display: tuple[ReplDisplay, ...] = ()
    error: ReplError | None = None
    timed_out: bool = False
    execution_count: int = -1

    @property
    def text(self) -> str:
        """Convenience: stdout + stderr + (error.evalue if error)."""
        out: list[str] = []
        if self.stdout:
            out.append(self.stdout)
        if self.stderr:
            out.append(self.stderr)
        if self.error:
            out.append(f"{self.error.ename}: {self.error.evalue}")
        return "\n".join(out).strip()


@runtime_checkable
class Repl(Protocol):
    """Stateful code-execution session.

    Lifecycle: ``open()`` is idempotent and required before ``execute()``;
    ``close()`` releases the kernel. Use as ``async with repl:`` to bracket.

    Invariants:

    - :meth:`execute` is stateful (variables persist) until :meth:`restart`.
    - :meth:`interrupt` does not lose state.
    - :meth:`execute` never raises on user-code errors; errors surface in
      :attr:`ReplResult.error`. Infrastructure failures raise.
    """

    name: ClassVar[str]
    sandbox: Sandbox

    async def open(self) -> None: ...

    async def close(self) -> None: ...

    async def __aenter__(self) -> Repl: ...

    async def __aexit__(self, exc_type, exc, tb) -> None: ...

    async def execute(
        self, code: str, *, timeout: float | None = None
    ) -> ReplResult: ...

    async def interrupt(self) -> None: ...

    async def restart(self) -> None: ...


__all__ = ["Repl", "ReplDisplay", "ReplError", "ReplResult"]
