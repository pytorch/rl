# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Pluggable parsers turning LLM responses into :class:`ParsedCall` items.

Available parsers:

- :class:`XMLToolCallParser` -- ``<tool name="x">{...}</tool>`` blocks
  embedded in the message body. Successor to the legacy
  :class:`~torchrl.envs.llm.transforms.XMLBlockParser`.
- :class:`JSONToolCallParser` -- top-level JSON with ``message`` and
  ``tools`` fields. Successor to
  :class:`~torchrl.envs.llm.transforms.JSONCallParser`.
- :class:`OpenAIToolCallParser` -- structured ``tool_calls`` array on the
  assistant message (OpenAI / vLLM-with-tools).
- :class:`AnthropicToolUseParser` -- ``tool_use`` content blocks
  (Anthropic).
"""
from __future__ import annotations

from .anthropic import AnthropicToolUseParser
from .json_block import JSONToolCallParser
from .openai import OpenAIToolCallParser
from .xml import XMLToolCallParser

__all__ = [
    "AnthropicToolUseParser",
    "JSONToolCallParser",
    "OpenAIToolCallParser",
    "XMLToolCallParser",
]
