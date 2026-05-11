# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""OpenAI-compatible tool-call parser.

Reads structured tool calls from the assistant message envelope
(``message.tool_calls`` or top-level ``tool_calls``), as produced by the
OpenAI Chat Completions API and any compatible server (vLLM with
``--enable-auto-tool-choice``, etc.).
"""
from __future__ import annotations

import json
import uuid
from collections.abc import Mapping
from typing import Any, ClassVar

from ..protocols import ParsedCall, ParseResult, ToolResult


class OpenAIToolCallParser:
    """Parses OpenAI-style ``tool_calls`` from an assistant message.

    Accepts any of these shapes -- the full message dict:

    .. code-block:: json

        {"role": "assistant", "content": "...", "tool_calls": [...]}

    the choice dict:

    .. code-block:: json

        {"message": {"role": "assistant", "tool_calls": [...]}}

    or a bare list under ``tool_calls`` at the top level.

    Each call's ``id`` is preserved as :attr:`ParsedCall.call_id`. Arguments
    are JSON-decoded from the ``function.arguments`` string.

    Examples:
        >>> p = OpenAIToolCallParser()
        >>> resp = {
        ...     "role": "assistant",
        ...     "content": "thinking...",
        ...     "tool_calls": [{
        ...         "id": "call_a",
        ...         "type": "function",
        ...         "function": {"name": "echo",
        ...                      "arguments": '{"text": "hi"}'},
        ...     }],
        ... }
        >>> r = p.parse(resp)
        >>> r.calls[0].tool, r.calls[0].args, r.calls[0].call_id
        ('echo', {'text': 'hi'}, 'call_a')
    """

    name: ClassVar[str] = "openai"

    def parse(self, response: str | Mapping[str, Any]) -> ParseResult:
        if isinstance(response, str):
            try:
                data: Any = json.loads(response)
            except json.JSONDecodeError:
                return ParseResult(text=response, calls=(), raw=response)
        else:
            data = response
        if isinstance(data, Mapping):
            message = data.get("message", data)
            content = message.get("content") or ""
            tool_calls = message.get("tool_calls") or data.get("tool_calls") or ()
        else:
            content = ""
            tool_calls = data or ()
        calls: list[ParsedCall] = []
        for tc in tool_calls:
            if not isinstance(tc, Mapping):
                continue
            fn = tc.get("function") or {}
            raw_args = fn.get("arguments")
            if isinstance(raw_args, str):
                try:
                    args = json.loads(raw_args) if raw_args else {}
                except json.JSONDecodeError:
                    args = {"raw": raw_args}
            else:
                args = dict(raw_args or {})
            calls.append(
                ParsedCall(
                    tool=str(fn.get("name", "")),
                    args=args,
                    call_id=str(tc.get("id") or uuid.uuid4().hex),
                    tag=None,
                )
            )
        return ParseResult(
            text=str(content) if isinstance(content, str) else "",
            calls=tuple(calls),
            raw=response,
        )

    def render_call(self, call: ParsedCall) -> str:
        return json.dumps(
            {
                "id": call.call_id,
                "type": "function",
                "function": {
                    "name": call.tool,
                    "arguments": json.dumps(dict(call.args), ensure_ascii=False),
                },
            },
            ensure_ascii=False,
        )

    def render_result(self, call_id: str, result: ToolResult) -> Mapping[str, Any]:
        # OpenAI shape: a "tool" role message with tool_call_id correlation.
        return {
            "role": "tool",
            "tool_call_id": call_id,
            "content": result.text,
            "is_error": result.is_error,
        }
