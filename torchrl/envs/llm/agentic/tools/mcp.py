# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Model Context Protocol (MCP) adapter.

Connects to an MCP server over stdio and exposes each remote tool as a
native :class:`~torchrl.envs.llm.agentic.Tool`. Unlike the legacy
:class:`~torchrl.envs.llm.transforms.MCPToolTransform`, no background
thread is needed -- our new dispatcher is already async, so we drive
the MCP client coroutines directly.

Optional dependency: install ``mcp`` (the official Python SDK) to use.
"""
from __future__ import annotations

import asyncio
import importlib.util
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from torchrl._utils import logger as torchrl_logger

from ..protocols import TextPart, ToolContext, ToolError, ToolResult

_has_mcp = importlib.util.find_spec("mcp") is not None


@dataclass(frozen=True, slots=True)
class MCPServerConfig:
    """How to launch an MCP server over stdio.

    Attributes:
        command: Executable, typically ``"npx"`` or ``"uvx"``.
        args: Arguments passed to ``command``.
        env: Optional environment-variable overrides.
    """

    command: str
    args: tuple[str, ...] = ()
    env: Mapping[str, str] | None = None


class MCPToolset:
    """Pool of :class:`Tool` instances backed by one MCP server.

    Connect once at :meth:`open` time, the server's ``tools/list`` is
    queried and each remote tool becomes a :class:`_MCPTool` exposing
    its native schema. Tools share the underlying MCP session for
    efficiency.

    Args:
        config: How to launch the server.
        name_prefix: Optional prefix prepended to every discovered tool
            name (e.g. ``"browser_"``). Useful when stacking multiple
            servers under one :class:`ToolCompose`.
        request_timeout: Default per-call timeout (seconds) forwarded
            to the MCP client.

    Example:
        >>> import asyncio  # doctest: +SKIP
        >>> from torchrl.envs.llm.agentic.tools.mcp import (
        ...     MCPServerConfig, MCPToolset,
        ... )
        >>> async def go():
        ...     pool = MCPToolset(
        ...         MCPServerConfig(command="npx",
        ...                         args=("@browsermcp/mcp@latest",))
        ...     )
        ...     await pool.open()
        ...     for tool in pool.tools:
        ...         print(tool.name, tool.description)
        ...     await pool.close()
    """

    def __init__(
        self,
        config: MCPServerConfig,
        *,
        name_prefix: str = "",
        request_timeout: float = 30.0,
    ) -> None:
        if not _has_mcp:
            raise ImportError(
                "MCPToolset requires the 'mcp' package. "
                "Install with `pip install mcp`."
            )
        self.config = config
        self.name_prefix = name_prefix
        self.request_timeout = request_timeout
        self._exit_stack: Any = None
        self._session: Any = None
        self._tools: list[_MCPTool] = []

    async def open(self) -> None:
        if self._session is not None:
            return
        from contextlib import AsyncExitStack

        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        self._exit_stack = AsyncExitStack()
        params = StdioServerParameters(
            command=self.config.command,
            args=list(self.config.args),
            env=dict(self.config.env) if self.config.env else None,
        )
        read, write = await self._exit_stack.enter_async_context(
            stdio_client(params)
        )
        session = await self._exit_stack.enter_async_context(
            ClientSession(read, write)
        )
        await session.initialize()
        listed = await session.list_tools()
        self._session = session
        self._tools = [
            _MCPTool(
                session=session,
                remote_name=t.name,
                description=t.description or "",
                input_schema=dict(t.inputSchema or {"type": "object"}),
                exposed_name=f"{self.name_prefix}{t.name}",
                request_timeout=self.request_timeout,
            )
            for t in listed.tools
        ]
        torchrl_logger.info(
            "MCPToolset connected to %s with %d tools",
            self.config.command,
            len(self._tools),
        )

    async def close(self) -> None:
        if self._exit_stack is not None:
            try:
                await self._exit_stack.aclose()
            except Exception:  # pragma: no cover -- defensive
                torchrl_logger.exception("MCPToolset close raised; continuing")
            self._exit_stack = None
            self._session = None
            self._tools = []

    @property
    def tools(self) -> tuple[Any, ...]:
        """Tuple of :class:`Tool` instances after :meth:`open`."""
        return tuple(self._tools)


class _MCPTool:
    """Single tool backed by an MCP session."""

    output_schema = None
    wants_state = False

    def __init__(
        self,
        *,
        session: Any,
        remote_name: str,
        description: str,
        input_schema: Mapping[str, Any],
        exposed_name: str,
        request_timeout: float,
    ) -> None:
        self._session = session
        self._remote_name = remote_name
        self.name = exposed_name
        self.description = description
        self.input_schema = dict(input_schema)
        self._timeout = request_timeout

    async def setup(self) -> None:
        # Session is opened by MCPToolset; nothing per-tool.
        pass

    async def teardown(self) -> None:
        pass

    async def run(
        self, args: Mapping[str, Any], ctx: ToolContext
    ) -> ToolResult:
        try:
            response = await asyncio.wait_for(
                self._session.call_tool(
                    self._remote_name, dict(args)
                ),
                timeout=self._timeout,
            )
        except asyncio.TimeoutError as e:
            raise ToolError(
                f"MCP call {self._remote_name!r} timed out after "
                f"{self._timeout}s"
            ) from e
        except Exception as e:  # noqa: BLE001
            raise ToolError(f"MCP call {self._remote_name!r} failed: {e}") from e
        # The MCP response is a list of content items; we coerce to text.
        content = getattr(response, "content", None) or []
        text_parts: list[str] = []
        for item in content:
            text = getattr(item, "text", None)
            if text is not None:
                text_parts.append(text)
            else:
                # JSON-serialise structured content fragments.
                text_parts.append(
                    json.dumps(
                        getattr(item, "model_dump", lambda: str(item))(),
                        ensure_ascii=False,
                    )
                )
        return ToolResult(
            parts=(TextPart(text="\n".join(text_parts)),),
            is_error=bool(getattr(response, "isError", False)),
        )


__all__ = ["MCPServerConfig", "MCPToolset"]
