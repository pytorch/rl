# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Built-in tools and the legacy-transform adapter.

- :class:`PythonTool`, :class:`ShellTool`, :class:`FileReadTool`,
  :class:`StopTool` -- ready-to-use tools wired to the agentic
  Sandbox/Repl primitives.
- :func:`as_tool` -- adapter lifting any legacy
  :class:`~torchrl.envs.llm.transforms.tools.ToolTransformBase` subclass
  into a :class:`~torchrl.envs.llm.agentic.Tool` so existing user code
  keeps working inside :class:`~torchrl.envs.llm.agentic.ToolCompose`.
"""
from __future__ import annotations

from .builtin import FileReadTool, PythonTool, ShellTool, StopSignal, StopTool
from .http import HttpTool
from .legacy_adapter import as_tool
from .mcp import MCPServerConfig, MCPToolset

__all__ = [
    "FileReadTool",
    "HttpTool",
    "MCPServerConfig",
    "MCPToolset",
    "PythonTool",
    "ShellTool",
    "StopSignal",
    "StopTool",
    "as_tool",
]
