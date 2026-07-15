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
from __future__ import annotations

from .compose import DispatchResult, ToolCompose
from .protocols import (
    FileRefPart,
    ImagePart,
    JsonPart,
    ParsedCall,
    ParseResult,
    TextPart,
    Tool,
    ToolCallParser,
    ToolContext,
    ToolError,
    ToolResult,
    ToolResultPart,
)
from .rate_limit import RateLimiter
from .schema import json_schema_from_pydantic, validate_args
from .tools import (
    as_tool,
    FileReadTool,
    HttpTool,
    MCPServerConfig,
    MCPToolset,
    PythonTool,
    ShellTool,
    StopSignal,
    StopTool,
)

__all__ = [
    "DispatchResult",
    "FileReadTool",
    "FileRefPart",
    "HttpTool",
    "ImagePart",
    "JsonPart",
    "MCPServerConfig",
    "MCPToolset",
    "ParseResult",
    "ParsedCall",
    "PythonTool",
    "RateLimiter",
    "ShellTool",
    "StopSignal",
    "StopTool",
    "TextPart",
    "Tool",
    "ToolCallParser",
    "ToolCompose",
    "ToolContext",
    "ToolError",
    "ToolResult",
    "ToolResultPart",
    "as_tool",
    "json_schema_from_pydantic",
    "validate_args",
]
