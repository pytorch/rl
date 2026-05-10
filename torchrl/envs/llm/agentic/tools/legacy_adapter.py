# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Adapter lifting a legacy ``ToolTransformBase`` into a
:class:`~torchrl.envs.llm.agentic.Tool`.

Existing user code -- ``PythonInterpreter``, ``BrowserTransform``,
``MCPToolTransform``, ``SimpleToolTransform`` -- can drop into
:class:`~torchrl.envs.llm.agentic.ToolCompose` without rewriting.

Bridge semantics
~~~~~~~~~~~~~~~~

The legacy classes implement ``_process_batch_item(content, index)`` which
takes the *raw assistant message string* and returns a list of result
strings. The agentic dispatcher already parsed the response and called us
with ``args``, so the adapter synthesises a single-call message in the
shape the legacy class expects (XML by default, configurable), runs the
legacy ``_process_batch_item``, and returns the joined output string.

This means the legacy class re-parses the synthesised message internally;
that's the cost of bridging two different protocols. For tools that don't
need this round-trip (i.e. anything new), write a native
:class:`~torchrl.envs.llm.agentic.Tool` instead.
"""
from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping
from typing import Any, Callable

from ..protocols import TextPart, ToolContext, ToolError, ToolResult


def _default_render(name: str, args: Mapping[str, Any]) -> str:
    body = json.dumps(dict(args), ensure_ascii=False)
    return f'<tool name="{name}">{body}</tool>'


def as_tool(
    transform: Any,
    *,
    name: str,
    input_schema: Mapping[str, Any] | None = None,
    description: str = "",
    wants_state: bool = False,
    render_call: Callable[[str, Mapping[str, Any]], str] | None = None,
):
    """Wrap a legacy tool transform as a new-style
    :class:`~torchrl.envs.llm.agentic.Tool`.

    Args:
        transform: An instance of a legacy
            :class:`~torchrl.envs.llm.transforms.tools.ToolTransformBase`
            subclass (or anything with a compatible
            ``_process_batch_item(content: str, index: int) -> list[str] | None``).
        name: Tool name to expose to the LLM. Should match the name the
            legacy transform expects in the synthesised XML envelope (i.e.
            the same string the model would write in
            ``<tool name="...">``).
        input_schema: JSON Schema dict for the LLM. If ``None``, a
            permissive ``{"type": "object"}`` is used.
        description: Tool description for the LLM.
        wants_state: Set to ``True`` to receive a filtered TensorDict view
            via ``ctx.state`` (mirrors the legacy ``pass_state_to_tools``
            knob).
        render_call: Custom function to format ``(name, args)`` into the
            string the legacy transform parses. Defaults to
            ``<tool name="X">{json args}</tool>``.

    Returns:
        A :class:`Tool`-conforming object.

    Examples:
        >>> from torchrl.envs.llm.transforms import PythonInterpreter  # doctest: +SKIP
        >>> from torchrl.envs.llm.agentic.tools import as_tool
        >>> tool = as_tool(
        ...     PythonInterpreter(persistent=True),
        ...     name="python",
        ...     input_schema={"type": "object",
        ...                   "properties": {"code": {"type": "string"}},
        ...                   "required": ["code"]},
        ... )
    """
    return _LegacyToolAdapter(
        transform,
        name=name,
        input_schema=input_schema or {"type": "object"},
        description=description,
        wants_state=wants_state,
        render_call=render_call or _default_render,
    )


class _LegacyToolAdapter:
    """Tool that delegates to a legacy ``ToolTransformBase``-style object."""

    output_schema = None

    def __init__(
        self,
        transform: Any,
        *,
        name: str,
        input_schema: Mapping[str, Any],
        description: str,
        wants_state: bool,
        render_call: Callable[[str, Mapping[str, Any]], str],
    ) -> None:
        # Per-instance attrs deliberately (rather than ClassVar) -- the
        # adapter is a factory and each invocation produces a distinct
        # tool with its own name/schema.
        self.name = name
        self.input_schema = dict(input_schema)
        self.description = description
        self.wants_state = wants_state
        self._transform = transform
        self._render = render_call

    async def setup(self) -> None:
        # Legacy transforms don't have a uniform setup hook; honor it if
        # the duck-typed object provides one.
        hook = getattr(self._transform, "setup", None)
        if callable(hook):
            res = hook()
            if asyncio.iscoroutine(res):
                await res

    async def teardown(self) -> None:
        hook = getattr(self._transform, "teardown", None)
        if callable(hook):
            res = hook()
            if asyncio.iscoroutine(res):
                await res
        # The legacy PythonInterpreter has its own close/shutdown patterns;
        # if exposed, call them.
        for method_name in ("close", "shutdown"):
            method = getattr(self._transform, method_name, None)
            if callable(method):
                try:
                    res = method()
                    if asyncio.iscoroutine(res):
                        await res
                except Exception:  # pragma: no cover -- defensive
                    pass
                break

    async def run(
        self, args: Mapping[str, Any], ctx: ToolContext
    ) -> ToolResult:
        rendered = self._render(self.name, args)
        process = getattr(self._transform, "_process_batch_item", None)
        if not callable(process):
            raise ToolError(
                f"legacy transform {type(self._transform).__name__!r} has "
                "no _process_batch_item; not adaptable"
            )
        # Legacy tools are sync; offload so we don't block the event loop.
        try:
            results = await asyncio.to_thread(process, rendered, 0)
        except Exception as e:  # pragma: no cover -- depends on legacy impl
            raise ToolError(str(e)) from e
        if not results:
            return ToolResult.from_text("", meta={"adapter": "legacy"})
        text = "\n".join(str(r) for r in results)
        return ToolResult(
            parts=(TextPart(text=text),),
            is_error=False,
            meta={"adapter": "legacy", "count": len(results)},
        )


__all__ = ["as_tool"]
