# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""JSON-block parser: top-level ``{"message": ..., "tools": [...]}``."""
from __future__ import annotations

import json
import uuid
from collections.abc import Mapping
from typing import Any, ClassVar

from ..protocols import ParsedCall, ParseResult, ToolResult


class JSONToolCallParser:
    """Parses LLM responses formatted as a single JSON object.

    Expected shape::

        {
          "message": "Let me search.",
          "tools": [
            {"tool": "search", "args": {"query": "x"}, "id": "c1"},
            {"tool": "summarize", "args": {"text": "..."}}
          ]
        }

    The optional ``id`` field on each call is used as the stable
    ``call_id``; when absent a uuid4 hex is assigned. Successor to
    :class:`~torchrl.envs.llm.transforms.JSONCallParser`.

    Examples:
        >>> p = JSONToolCallParser()
        >>> resp = '{"message": "ok", "tools": [{"tool": "echo", "args": {"x": 1}}]}'
        >>> r = p.parse(resp)
        >>> r.text, r.calls[0].tool, r.calls[0].args
        ('ok', 'echo', {'x': 1})
    """

    name: ClassVar[str] = "json_block"

    def parse(self, response: str | Mapping[str, Any]) -> ParseResult:
        if isinstance(response, str):
            try:
                data = json.loads(response)
            except json.JSONDecodeError:
                return ParseResult(text=response, calls=(), raw=response)
        else:
            data = response
        if not isinstance(data, Mapping):
            return ParseResult(text=str(data), calls=(), raw=response)
        tools_data = data.get("tools") or ()
        calls = tuple(
            ParsedCall(
                tool=str(c["tool"]),
                args=dict(c.get("args") or {}),
                call_id=str(c.get("id") or c.get("call_id") or uuid.uuid4().hex),
                tag=c.get("tag"),
            )
            for c in tools_data
        )
        return ParseResult(
            text=str(data.get("message", "")),
            calls=calls,
            raw=response,
        )

    def render_call(self, call: ParsedCall) -> str:
        return json.dumps(
            {"tool": call.tool, "args": dict(call.args), "id": call.call_id},
            ensure_ascii=False,
        )

    def render_result(
        self, call_id: str, result: ToolResult
    ) -> Mapping[str, Any]:
        return {
            "role": "tool",
            "content": json.dumps(
                {
                    "id": call_id,
                    "is_error": result.is_error,
                    "output": result.text,
                },
                ensure_ascii=False,
            ),
        }
