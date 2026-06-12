# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Backward-compatibility re-exports for conversation containers.

:class:`~tensordict.llm.History`, :class:`~tensordict.llm.ContentBase` and
:func:`~tensordict.llm.add_chat_template` now live in ``tensordict.llm``,
which is their canonical home. This module re-exports them so that existing
``torchrl.data.llm.history`` import paths keep working.

New code should import from :mod:`tensordict.llm` directly.
"""
from __future__ import annotations

from tensordict.llm.history import (  # noqa: F401
    _assistant_content_spans,
    _CHAT_TEMPLATES,
    _CUSTOM_INVERSE_PARSERS,
    _CUSTOM_MODEL_FAMILY_KEYWORDS,
    _fallback_assistant_tokens_mask,
    add_chat_template,
    ContentBase,
    History,
)

__all__ = ["add_chat_template", "ContentBase", "History"]
