# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Model Context Protocol (MCP) adapter.

Connects to an MCP server over stdio and exposes each remote tool as a
native :class:`~torchrl.envs.llm.agentic.Tool`. The MCP client coroutines
run directly on the persistent async loop owned by
:class:`~torchrl.envs.llm.agentic.ToolCompose`, without a separate
MCP-specific runner.

Optional dependency: install ``mcp`` (the official Python SDK) to use.
"""
from __future__ import annotations

import asyncio
import importlib.util
import json
from collections.abc import Mapping
from dataclasses import dataclass
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
        ...     async with pool:
        ...         return [(tool.name, tool.description) for tool in pool.tools]
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
        self._loop: asyncio.AbstractEventLoop | None = None
        self._tools: list[_MCPTool] = []

    async def open(self) -> None:
        if self._session is not None:
            if asyncio.get_running_loop() is not self._loop:
                raise RuntimeError("MCPToolset is already open on another event loop")
            return
        from contextlib import AsyncExitStack

        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        exit_stack = AsyncExitStack()
        params = StdioServerParameters(
            command=self.config.command,
            args=list(self.config.args),
            env=dict(self.config.env) if self.config.env else None,
        )
        try:
            read, write = await exit_stack.enter_async_context(stdio_client(params))
            session = await exit_stack.enter_async_context(ClientSession(read, write))
            await session.initialize()
            listed = await session.list_tools()
        except Exception:
            await exit_stack.aclose()
            raise
        self._exit_stack = exit_stack
        self._session = session
        self._loop = asyncio.get_running_loop()
        discovered = {t.name: t for t in listed.tools}
        if self._tools:
            existing = {tool._remote_name: tool for tool in self._tools}
            if existing.keys() != discovered.keys():
                await self.close()
                raise RuntimeError("MCP server tool list changed while reopening")
            for remote_name, tool in existing.items():
                spec = discovered[remote_name]
                tool.description = spec.description or ""
                tool.input_schema = dict(spec.inputSchema or {"type": "object"})
        else:
            self._tools = [
                _MCPTool(
                    toolset=self,
                    remote_name=t.name,
                    description=t.description or "",
                    input_schema=dict(t.inputSchema or {"type": "object"}),
                    exposed_name=f"{self.name_prefix}{t.name}",
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
            self._loop = None

    async def __aenter__(self) -> MCPToolset:
        await self.open()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: Any,
    ) -> None:
        await self.close()

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
        toolset: MCPToolset,
        remote_name: str,
        description: str,
        input_schema: Mapping[str, Any],
        exposed_name: str,
    ) -> None:
        self._toolset = toolset
        self._remote_name = remote_name
        self.name = exposed_name
        self.description = description
        self.input_schema = dict(input_schema)

    async def setup(self) -> None:
        # Session is opened by MCPToolset; nothing per-tool.
        pass

    async def teardown(self) -> None:
        pass

    async def run(self, args: Mapping[str, Any], ctx: ToolContext) -> ToolResult:
        session = self._toolset._session
        if session is None:
            raise ToolError("MCPToolset is not open")
        if asyncio.get_running_loop() is not self._toolset._loop:
            raise ToolError(
                "MCP tools must run on the event loop that opened their toolset; "
                "use ToolCompose.add_toolset(pool)"
            )
        try:
            response = await asyncio.wait_for(
                session.call_tool(self._remote_name, dict(args)),
                timeout=self._toolset.request_timeout,
            )
        except TimeoutError as e:
            raise ToolError(
                f"MCP call {self._remote_name!r} timed out after "
                f"{self._toolset.request_timeout}s"
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
                model_dump = getattr(item, "model_dump", None)
                fragment = model_dump() if model_dump is not None else str(item)
                text_parts.append(
                    json.dumps(
                        fragment,
                        ensure_ascii=False,
                    )
                )
        return ToolResult(
            parts=(TextPart(text="\n".join(text_parts)),),
            is_error=bool(getattr(response, "isError", False)),
        )


__all__ = ["MCPServerConfig", "MCPToolset"]
