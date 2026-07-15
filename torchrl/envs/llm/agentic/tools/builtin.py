# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Built-in tools.

- :class:`PythonTool` -- run code in a :class:`Repl` (state persists
  across calls in the same episode).
- :class:`ShellTool` -- run argv inside a :class:`Sandbox`.
- :class:`FileReadTool` -- read a file from inside a :class:`Sandbox`.
- :class:`StopTool` -- explicit episode terminator. Raises
  :class:`StopSignal` so the dispatcher can mark the episode done.
"""
from __future__ import annotations

import shlex
from collections.abc import Mapping
from typing import Any, ClassVar

from ..protocols import TextPart, ToolContext, ToolError, ToolResult
from ..repl.base import Repl
from ..sandbox.base import Sandbox


class StopSignal(Exception):
    """Raised by :class:`StopTool` to terminate the agent loop.

    :class:`~torchrl.envs.llm.agentic.ToolCompose` catches this and sets
    the corresponding episode-end flag in the step output.
    """


class PythonTool:
    """Execute Python code in a stateful :class:`Repl`.

    Args:
        repl: The REPL backend. Must be opened (or used as a context
            manager) before dispatch. ``ToolCompose`` opens/closes it
            on env reset/close when it owns the repl.
        timeout: Default per-call timeout (seconds). Per-call ``ctx`` may
            override.
        output_max_chars: Cap on the returned text (longer is truncated
            with a marker).

    Examples:
        >>> from torchrl.envs.llm.agentic.sandbox import UnsafeSubprocessSandbox
        >>> from torchrl.envs.llm.agentic.repl import SubprocessRepl
        >>> tool = PythonTool(repl=SubprocessRepl(UnsafeSubprocessSandbox()))
    """

    name: ClassVar[str] = "python"
    description: ClassVar[str] = "Execute Python code; state persists across calls."
    input_schema: ClassVar[Mapping[str, Any]] = {
        "type": "object",
        "properties": {"code": {"type": "string"}},
        "required": ["code"],
    }
    output_schema: ClassVar[Mapping[str, Any] | None] = None
    wants_state: ClassVar[bool] = False

    def __init__(
        self,
        repl: Repl,
        *,
        timeout: float | None = 30.0,
        output_max_chars: int = 8192,
    ) -> None:
        self.repl = repl
        self.timeout = timeout
        self.output_max_chars = output_max_chars

    async def setup(self) -> None:
        await self.repl.sandbox.open()
        try:
            await self.repl.open()
        except Exception:
            await self.repl.sandbox.close()
            raise

    async def teardown(self) -> None:
        try:
            await self.repl.close()
        finally:
            await self.repl.sandbox.close()

    async def run(self, args: Mapping[str, Any], ctx: ToolContext) -> ToolResult:
        code = args.get("code", "")
        if not isinstance(code, str):
            raise ToolError("'code' must be a string")
        result = await self.repl.execute(code, timeout=self.timeout)
        text = result.text
        truncated = False
        if len(text) > self.output_max_chars:
            text = text[: self.output_max_chars] + "\n... [truncated]"
            truncated = True
        return ToolResult(
            parts=(TextPart(text=text),),
            is_error=result.error is not None or result.timed_out,
            meta={
                "execution_count": result.execution_count,
                "timed_out": result.timed_out,
                "truncated": truncated,
            },
        )


class ShellTool:
    """Execute a shell command inside a :class:`Sandbox`.

    Accepts either ``argv: list[str]`` or ``command: str``. ``command``
    is split with :func:`shlex.split` -- callers needing pipes should
    use ``argv=["sh", "-c", "..."]`` explicitly.
    """

    name: ClassVar[str] = "shell"
    description: ClassVar[str] = "Execute a shell command in a sandbox."
    input_schema: ClassVar[Mapping[str, Any]] = {
        "type": "object",
        "properties": {
            "command": {"type": "string"},
            "argv": {"type": "array", "items": {"type": "string"}},
            "cwd": {"type": "string"},
        },
    }
    output_schema: ClassVar[Mapping[str, Any] | None] = None
    wants_state: ClassVar[bool] = False

    def __init__(self, sandbox: Sandbox) -> None:
        self.sandbox = sandbox

    async def setup(self) -> None:
        await self.sandbox.open()

    async def teardown(self) -> None:
        await self.sandbox.close()

    async def run(self, args: Mapping[str, Any], ctx: ToolContext) -> ToolResult:
        argv = args.get("argv")
        command = args.get("command")
        if argv is None and command is None:
            raise ToolError("ShellTool requires 'argv' or 'command'")
        if argv is None:
            argv = shlex.split(str(command))
        result = await self.sandbox.run(list(argv), cwd=args.get("cwd"))
        body_lines: list[str] = []
        if result.stdout:
            body_lines.append(result.stdout)
        if result.stderr:
            body_lines.append(f"[stderr]\n{result.stderr}")
        body_lines.append(f"[exit {result.exit_code}]")
        return ToolResult(
            parts=(TextPart(text="\n".join(body_lines).rstrip()),),
            is_error=result.exit_code != 0 or result.timed_out,
            meta={
                "exit_code": result.exit_code,
                "timed_out": result.timed_out,
                "wall_seconds": result.wall_seconds,
            },
        )


class FileReadTool:
    """Read a file from inside a :class:`Sandbox`."""

    name: ClassVar[str] = "file_read"
    description: ClassVar[str] = "Read a file from the sandbox filesystem."
    input_schema: ClassVar[Mapping[str, Any]] = {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "max_bytes": {"type": "integer"},
        },
        "required": ["path"],
    }
    output_schema: ClassVar[Mapping[str, Any] | None] = None
    wants_state: ClassVar[bool] = False

    def __init__(self, sandbox: Sandbox) -> None:
        self.sandbox = sandbox

    async def setup(self) -> None:
        await self.sandbox.open()

    async def teardown(self) -> None:
        await self.sandbox.close()

    async def run(self, args: Mapping[str, Any], ctx: ToolContext) -> ToolResult:
        path = args["path"]
        max_bytes = args.get("max_bytes")
        try:
            data = await self.sandbox.read_file(path, max_bytes=max_bytes)
        except FileNotFoundError as e:
            raise ToolError(f"file not found: {path}") from e
        return ToolResult.from_text(
            data.decode("utf-8", errors="replace"),
            meta={"bytes": len(data)},
        )


class StopTool:
    """Zero-arg tool that ends the agent episode.

    Raises :class:`StopSignal` from :meth:`run`. The dispatcher catches
    this and sets the corresponding flag in the step output so the env
    can terminate.
    """

    name: ClassVar[str] = "stop"
    description: ClassVar[str] = "Signal that the agent has finished its task."
    input_schema: ClassVar[Mapping[str, Any]] = {
        "type": "object",
        "properties": {"reason": {"type": "string"}},
    }
    output_schema: ClassVar[Mapping[str, Any] | None] = None
    wants_state: ClassVar[bool] = False

    async def setup(self) -> None:
        pass

    async def teardown(self) -> None:
        pass

    async def run(self, args: Mapping[str, Any], ctx: ToolContext) -> ToolResult:
        reason = str(args.get("reason", "done"))
        raise StopSignal(reason)


__all__ = [
    "FileReadTool",
    "PythonTool",
    "ShellTool",
    "StopSignal",
    "StopTool",
]
