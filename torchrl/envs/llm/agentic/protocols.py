# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Core protocols and value types for the agentic toolkit.

Three layered concerns:

- :class:`Tool` -- a unit an LLM can invoke by name. Async-first.
- :class:`ToolCallParser` -- turns an LLM response into structured
  :class:`ParsedCall` items and renders results back into the family's
  message shape.
- (See :class:`~torchrl.envs.llm.agentic.sandbox.Sandbox` and
  :class:`~torchrl.envs.llm.agentic.repl.Repl` for isolation and state.)

Stable ``call_id`` invariant: every :class:`ParsedCall` carries a
``call_id`` (parser-supplied if the family provides one -- OpenAI ``id``,
Anthropic ``tool_use_id`` -- else a parser-assigned uuid4). Round-trips
through :meth:`ToolCallParser.render_result` so downstream consumers can
correlate calls and results.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, ClassVar, Literal, Protocol, runtime_checkable

from tensordict import TensorDictBase


# ----- result parts -----

@dataclass(frozen=True, slots=True)
class TextPart:
    """A text fragment of a :class:`ToolResult`."""

    text: str
    kind: Literal["text"] = "text"


@dataclass(frozen=True, slots=True)
class JsonPart:
    """A JSON-serialisable structured fragment of a :class:`ToolResult`."""

    data: Any
    kind: Literal["json"] = "json"


@dataclass(frozen=True, slots=True)
class ImagePart:
    """An image fragment of a :class:`ToolResult` (raw bytes + media type)."""

    data: bytes
    media_type: str = "image/png"
    kind: Literal["image"] = "image"


@dataclass(frozen=True, slots=True)
class FileRefPart:
    """A reference to a file produced by a tool (path inside the sandbox)."""

    path: str
    media_type: str | None = None
    kind: Literal["file_ref"] = "file_ref"


ToolResultPart = TextPart | JsonPart | ImagePart | FileRefPart


@dataclass(frozen=True, slots=True)
class ToolResult:
    """The output of a single :meth:`Tool.run` invocation.

    Attributes:
        parts: Ordered tuple of result fragments. ``parts[0]`` is conventionally
            text. Most call sites only need ``result.text``.
        is_error: Whether the tool raised or otherwise produced an error.
            ``parts[0]`` should describe the error when ``True``.
        meta: Free-form metadata (timing, tokens used, raw provider payload).
    """

    parts: tuple[ToolResultPart, ...] = ()
    is_error: bool = False
    meta: Mapping[str, Any] = field(default_factory=dict)

    @property
    def text(self) -> str:
        """Concatenation of all :class:`TextPart` and stringified
        :class:`JsonPart` content. Convenience for the common case."""
        out: list[str] = []
        for p in self.parts:
            if isinstance(p, TextPart):
                out.append(p.text)
            elif isinstance(p, JsonPart):
                import json as _json

                out.append(_json.dumps(p.data, ensure_ascii=False))
            elif isinstance(p, FileRefPart):
                out.append(f"<file:{p.path}>")
            elif isinstance(p, ImagePart):
                out.append(f"<image:{p.media_type}:{len(p.data)} bytes>")
        return "\n".join(out)

    @classmethod
    def from_text(
        cls,
        text: str,
        *,
        is_error: bool = False,
        meta: Mapping[str, Any] | None = None,
    ) -> ToolResult:
        """Shorthand for the common single-text-part result."""
        return cls(
            parts=(TextPart(text=text),),
            is_error=is_error,
            meta=dict(meta or {}),
        )


@dataclass
class ToolError(Exception):
    """Raised by tools to signal a structured failure.

    Catching this in :class:`ToolCompose` produces a
    :class:`ToolResult` with ``is_error=True``. Anything else surfaces as
    an unstructured error (still wrapped, but flagged in ``meta``).
    """

    message: str
    detail: Mapping[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return self.message


# ----- call / parse types -----

@dataclass(frozen=True, slots=True)
class ParsedCall:
    """A single tool invocation parsed out of an LLM response.

    Attributes:
        tool: The name of the tool to invoke.
        args: Already-decoded keyword arguments. Validation against
            :attr:`Tool.input_schema` happens in :class:`ToolCompose`.
        call_id: Stable identifier (parser-assigned if not present in the
            source). Round-trips through :meth:`ToolCallParser.render_result`.
        tag: Optional human-visible label (back-compat with
            ``ExecuteToolsInOrder``).
    """

    tool: str
    args: Mapping[str, Any]
    call_id: str
    tag: str | None = None


@dataclass(frozen=True, slots=True)
class ParseResult:
    """Output of :meth:`ToolCallParser.parse`.

    Attributes:
        text: Cleaned message body with tool-call syntax stripped (when the
            family embeds calls in the text -- XML, JSON-block). Empty for
            providers where calls live in a structured field (OpenAI,
            Anthropic).
        calls: Calls in the order the model emitted them.
        raw: The original response, for round-trip and debugging.
    """

    text: str
    calls: tuple[ParsedCall, ...]
    raw: Any = None


# ----- context passed to a Tool -----

@dataclass
class ToolContext:
    """Per-call context handed to :meth:`Tool.run`.

    Attributes:
        call_id: The :attr:`ParsedCall.call_id`. Stable across this turn.
        tag: Optional :attr:`ParsedCall.tag`.
        state: Read-only filtered view of the env state. Only populated when
            the owning :class:`ToolCompose` has ``pass_state_to_tools=True``
            *and* the tool has ``wants_state=True``.
        sandbox: The compose-level sandbox, if any. Tools may also hold
            their own sandbox by reference.
        repl: The compose-level REPL, if any.
        compose: Back-reference to the owning :class:`ToolCompose` for
            tool-to-tool dispatch from inside a tool body.
    """

    call_id: str
    tag: str | None = None
    state: TensorDictBase | None = None
    sandbox: Any | None = None
    repl: Any | None = None
    compose: Any | None = None


# ----- protocols -----

@runtime_checkable
class Tool(Protocol):
    """A unit invoked by name from an LLM response.

    Subclasses (or duck-typed equivalents) declare ``name``, ``description``,
    and ``input_schema`` (JSON Schema dict) at the class level, and implement
    an async :meth:`run`.

    A tool may opt in to receiving env state via the ``wants_state`` class
    attribute -- :class:`ToolCompose` will populate ``ctx.state`` when both
    sides agree.

    Example:
        >>> from torchrl.envs.llm.agentic import Tool, ToolContext, ToolResult
        >>> class EchoTool:
        ...     name = "echo"
        ...     description = "Returns its input."
        ...     input_schema = {"type": "object",
        ...                     "properties": {"text": {"type": "string"}},
        ...                     "required": ["text"]}
        ...     output_schema = None
        ...     wants_state = False
        ...     async def run(self, args, ctx):
        ...         return ToolResult.from_text(args["text"])
        ...     async def setup(self): pass
        ...     async def teardown(self): pass
    """

    name: ClassVar[str]
    description: ClassVar[str]
    input_schema: ClassVar[Mapping[str, Any]]
    output_schema: ClassVar[Mapping[str, Any] | None]
    wants_state: ClassVar[bool]

    async def run(
        self, args: Mapping[str, Any], ctx: ToolContext
    ) -> ToolResult: ...

    async def setup(self) -> None: ...

    async def teardown(self) -> None: ...


@runtime_checkable
class ToolCallParser(Protocol):
    """Parses an LLM response into :class:`ParsedCall` items and renders
    results back into the family's message shape.

    Implementations must guarantee:

    1. :meth:`parse` is pure and synchronous.
    2. Every returned :class:`ParsedCall` has a non-empty :attr:`call_id`.
    3. ``parse -> render_call`` round-trips for calls produced by
       :meth:`parse` (within the same parser family).
    4. :meth:`render_result` produces a mapping suitable for one new
       message in :class:`~torchrl.data.llm.History` (keys at minimum:
       ``role``, ``content``).
    """

    name: ClassVar[str]

    def parse(self, response: str | Mapping[str, Any]) -> ParseResult: ...

    def render_call(self, call: ParsedCall) -> str: ...

    def render_result(
        self, call_id: str, result: ToolResult
    ) -> Mapping[str, Any]: ...
