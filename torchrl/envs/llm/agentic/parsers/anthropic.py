# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Anthropic ``tool_use`` parser.

Reads ``tool_use`` content blocks from an assistant message and emits a
``tool_result`` block per call when rendering. Matches the Messages API
shape used by Claude.
"""
from __future__ import annotations

import json
import uuid
from collections.abc import Mapping
from typing import Any, ClassVar

from ..protocols import ParsedCall, ParseResult, ToolResult


class AnthropicToolUseParser:
    """Parses Anthropic-style ``tool_use`` content blocks.

    Accepts either the full assistant message::

        {
          "role": "assistant",
          "content": [
            {"type": "text", "text": "Let me search."},
            {"type": "tool_use", "id": "toolu_1",
             "name": "search", "input": {"q": "x"}}
          ]
        }

    or a bare ``content`` list. Each block's ``id`` is preserved as
    :attr:`ParsedCall.call_id`.

    Examples:
        >>> p = AnthropicToolUseParser()
        >>> resp = {"role": "assistant", "content": [
        ...     {"type": "text", "text": "ok"},
        ...     {"type": "tool_use", "id": "u1",
        ...      "name": "echo", "input": {"text": "hi"}},
        ... ]}
        >>> r = p.parse(resp)
        >>> r.text, r.calls[0].tool, r.calls[0].args, r.calls[0].call_id
        ('ok', 'echo', {'text': 'hi'}, 'u1')
    """

    name: ClassVar[str] = "anthropic"

    def parse(self, response: str | Mapping[str, Any]) -> ParseResult:
        if isinstance(response, str):
            try:
                data: Any = json.loads(response)
            except json.JSONDecodeError:
                return ParseResult(text=response, calls=(), raw=response)
        else:
            data = response
        if isinstance(data, Mapping):
            content = data.get("content")
        else:
            content = data
        if isinstance(content, str):
            return ParseResult(text=content, calls=(), raw=response)
        if not isinstance(content, list):
            return ParseResult(text="", calls=(), raw=response)
        text_parts: list[str] = []
        calls: list[ParsedCall] = []
        for block in content:
            if not isinstance(block, Mapping):
                continue
            btype = block.get("type")
            if btype == "text":
                text_parts.append(str(block.get("text", "")))
            elif btype == "tool_use":
                calls.append(
                    ParsedCall(
                        tool=str(block.get("name", "")),
                        args=dict(block.get("input") or {}),
                        call_id=str(block.get("id") or uuid.uuid4().hex),
                        tag=None,
                    )
                )
        return ParseResult(
            text="\n".join(text_parts).strip(),
            calls=tuple(calls),
            raw=response,
        )

    def render_call(self, call: ParsedCall) -> str:
        return json.dumps(
            {
                "type": "tool_use",
                "id": call.call_id,
                "name": call.tool,
                "input": dict(call.args),
            },
            ensure_ascii=False,
        )

    def render_result(
        self, call_id: str, result: ToolResult
    ) -> Mapping[str, Any]:
        return {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": call_id,
                    "content": result.text,
                    "is_error": result.is_error,
                }
            ],
        }
