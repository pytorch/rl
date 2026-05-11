# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""XML-block parser: ``<tool name="x" tag="y">{json}</tool>``."""
from __future__ import annotations

import json
import re
import uuid
from collections.abc import Mapping
from typing import Any, ClassVar

from ..protocols import ParsedCall, ParseResult, ToolResult


class XMLToolCallParser:
    r"""Parses XML-style tool blocks embedded in the assistant message.

    Format:

        <tool name="search" tag="A">{"query": "torchrl"}</tool>

    or, for argless tools:

        <tool name="stop"></tool>

    Successor to :class:`~torchrl.envs.llm.transforms.XMLBlockParser`. Differs
    in that every :class:`ParsedCall` is given a stable ``call_id`` (the
    ``tag`` if present, else a uuid4) so results can be correlated across
    the dispatch boundary.

    Examples:
        >>> p = XMLToolCallParser()
        >>> r = p.parse('<tool name="echo" tag="1">{"text": "hi"}</tool>ok')
        >>> r.calls[0].tool, r.calls[0].args, r.calls[0].call_id, r.text
        ('echo', {'text': 'hi'}, '1', 'ok')
    """

    name: ClassVar[str] = "xml"

    _re = re.compile(
        r'<tool\s+name="(?P<name>[^"]+)"'
        r'(?:\s+tag="(?P<tag>[^"]+)")?\s*>\s*'
        r"(?P<body>.*?)\s*</tool>",
        re.DOTALL,
    )

    def parse(self, response: str | Mapping[str, Any]) -> ParseResult:
        text = response if isinstance(response, str) else str(response.get("text", ""))
        calls: list[ParsedCall] = []

        def repl(m: re.Match) -> str:
            tag = m.group("tag")
            body = m.group("body")
            try:
                args = json.loads(body) if body.strip() else {}
            except json.JSONDecodeError:
                args = {"raw": body}
            calls.append(
                ParsedCall(
                    tool=m.group("name"),
                    args=args,
                    call_id=tag if tag else uuid.uuid4().hex,
                    tag=tag,
                )
            )
            return ""

        cleaned = self._re.sub(repl, text).strip()
        return ParseResult(text=cleaned, calls=tuple(calls), raw=response)

    def render_call(self, call: ParsedCall) -> str:
        tag = f' tag="{call.tag}"' if call.tag else ""
        body = json.dumps(dict(call.args), ensure_ascii=False)
        return f'<tool name="{call.tool}"{tag}>{body}</tool>'

    def render_result(self, call_id: str, result: ToolResult) -> Mapping[str, Any]:
        body = result.text
        prefix = "[error] " if result.is_error else ""
        return {
            "role": "tool",
            "content": (
                f'<tool_result call_id="{call_id}">{prefix}{body}</tool_result>'
            ),
        }
