# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Agentic toolkit for ChatEnv.

A first-class, async-first stack for LLM tool use. Drop a
:class:`~torchrl.envs.llm.agentic.ToolCompose` into a ``TransformedEnv``
wrapping an unmodified :class:`~torchrl.envs.llm.ChatEnv`, register a few
:class:`~torchrl.envs.llm.agentic.Tool` instances, pick a parser, and you
have an agent loop with parallel dispatch, sandboxed execution, and a
stateful REPL.

See ``docs/source/reference/llms_envs.rst`` and
``docs/source/tutorials/llm_agentic.rst`` for a walkthrough.
"""
# TODO: contributors please update as items are picked up.
# - streaming tool results (AsyncIterator[ToolEvent] from Tool.run)
# - per-tool token-budget accounting
# - E2B / Modal real implementations (stubs land first)
# - harmony parser (gpt-oss / o1-style)
# - Ray dispatcher (ToolCompose(parallel="ray"))
# - multimodal tool outputs (image / audio in ToolResult.parts)
# - structured-output validation against Tool.output_schema
# - per-tool retry / circuit breaker
# - tool-result caching (content-addressed) for replay
# - formal deprecation of legacy tool transforms once the new API soaks
from __future__ import annotations

from .protocols import (
    FileRefPart,
    ImagePart,
    JsonPart,
    ParsedCall,
    ParseResult,
    Tool,
    ToolCallParser,
    ToolContext,
    ToolError,
    ToolResult,
    ToolResultPart,
    TextPart,
)
from .schema import json_schema_from_pydantic, validate_args

__all__ = [
    "FileRefPart",
    "ImagePart",
    "JsonPart",
    "ParseResult",
    "ParsedCall",
    "TextPart",
    "Tool",
    "ToolCallParser",
    "ToolContext",
    "ToolError",
    "ToolResult",
    "ToolResultPart",
    "json_schema_from_pydantic",
    "validate_args",
]
