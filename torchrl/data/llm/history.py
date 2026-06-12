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

The spec-related functionality stays here: tensordict cannot depend on
torchrl's spec classes, so :func:`history_default_spec` (and the
backward-compatible ``History.default_spec`` classmethod it powers) are
defined in this module.
"""
from __future__ import annotations

import dataclasses

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

__all__ = ["add_chat_template", "ContentBase", "History", "history_default_spec"]


def _history_default_spec(cls, shape=(-1,)):
    """A default spec to use in transforms / envs that return History objects.

    Args:
        cls (type): The :class:`~tensordict.llm.History` class (or subclass) to build the spec for.
        shape (torch.Size, optional): The shape of the returned History spec. Defaults to `(-1)` (variable length
            along the time dimension).

    .. seealso:: :func:`~torchrl.data.llm.history_default_spec`.
    """
    # Composite/NonTensor cannot be imported at module level: torchrl.data is
    # still initializing when this module is first imported.
    from torchrl.data import Composite, NonTensor

    def get_default_value(field):
        if field.default is not dataclasses.MISSING:
            return field.default
        elif field.type in (str, "str"):
            return "foo"
        else:
            return None

    defaults = {
        k: NonTensor(
            example_data=get_default_value(cls.__dataclass_fields__[k]),
            shape=shape,
        )
        for k in cls.__dataclass_fields__
    }

    return Composite(defaults, shape=shape[:-1], data_cls=cls)


def history_default_spec(shape=(-1,)):
    """A default Composite spec for :class:`~tensordict.llm.History` objects, to use in transforms / envs.

    Args:
        shape (torch.Size, optional): The shape of the returned History spec. Defaults to `(-1)` (variable length
            along the time dimension).

    Example:
        >>> import tensordict
        >>> from torchrl.data.llm import history_default_spec
        >>> tensordict.set_list_to_stack(True).set()
        >>>
        >>> spec = history_default_spec()
        >>> print(spec)
        Composite(
            role: NonTensor(
                shape=torch.Size([-1]),
                space=None,
                device=None,
                dtype=None,
                domain=None,
                example_data=foo),
            content: NonTensor(
                shape=torch.Size([-1]),
                space=None,
                device=None,
                dtype=None,
                domain=None,
                example_data=foo),
            device=None,
            shape=torch.Size([-1]))
        >>> print(spec.zero())
        History(
            content=NonTensorData(data=foo, batch_size=torch.Size([1]), device=None),
            role=NonTensorData(data=foo, batch_size=torch.Size([1]), device=None),
            batch_size=torch.Size([1]),
            device=None,
            is_shared=False)

    """
    return _history_default_spec(History, shape)


# `History` lives in tensordict, which cannot depend on torchrl's specs.
# Attach the `default_spec` classmethod here so that the established
# `History.default_spec()` API (mirroring ChatHistory/Text/Tokens) keeps
# working once torchrl is imported.
History.default_spec = classmethod(_history_default_spec)
