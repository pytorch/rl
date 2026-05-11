# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""JSON Schema helpers for :class:`~torchrl.envs.llm.agentic.Tool`.

Tools declare ``input_schema`` as a plain JSON Schema dict (matches OpenAI,
Anthropic, and MCP). A small ``validate_args`` helper enforces required
fields and primitive types without pulling in a full JSON Schema validator.
For users who prefer pydantic, :func:`json_schema_from_pydantic` converts a
``BaseModel`` subclass to the equivalent dict.
"""
from __future__ import annotations

import importlib.util
from collections.abc import Mapping
from typing import Any

_has_pydantic = importlib.util.find_spec("pydantic") is not None


_TYPE_MAP: dict[str, type | tuple[type, ...]] = {
    "string": str,
    "integer": int,
    "number": (int, float),
    "boolean": bool,
    "array": list,
    "object": dict,
    "null": type(None),
}


class SchemaValidationError(ValueError):
    """Raised by :func:`validate_args` on a schema mismatch."""


def validate_args(args: Mapping[str, Any], schema: Mapping[str, Any] | None) -> None:
    """Validate ``args`` against a JSON Schema dict.

    Implements the subset that matters for tool-call dispatch:

    - top-level ``type: object``,
    - ``required`` field presence,
    - per-property ``type`` (single string, not the array form).

    Anything else is permitted. Tools that need richer validation should
    do it inside :meth:`Tool.run` (or use pydantic via
    :func:`json_schema_from_pydantic`).

    Raises:
        SchemaValidationError: on missing required fields or type mismatches.
    """
    if not schema:
        return
    if schema.get("type") not in (None, "object"):
        return
    required = schema.get("required") or ()
    for key in required:
        if key not in args:
            raise SchemaValidationError(f"missing required argument: {key!r}")
    props: Mapping[str, Any] = schema.get("properties") or {}
    for key, sub in props.items():
        if key not in args:
            continue
        expected = sub.get("type")
        if not expected:
            continue
        py_type = _TYPE_MAP.get(expected)
        if py_type is None:
            continue
        if not isinstance(args[key], py_type):
            raise SchemaValidationError(
                f"argument {key!r} expected JSON type {expected!r}, "
                f"got {type(args[key]).__name__}"
            )


def json_schema_from_pydantic(model: Any) -> dict[str, Any]:
    """Return the JSON Schema dict for a ``pydantic.BaseModel`` subclass.

    Equivalent to ``model.model_json_schema()`` (pydantic v2). Raises
    ``ImportError`` if pydantic isn't installed.

    Examples:
        >>> from pydantic import BaseModel  # doctest: +SKIP
        >>> class Args(BaseModel):
        ...     code: str
        >>> json_schema_from_pydantic(Args)  # doctest: +SKIP
        {'type': 'object', 'properties': {'code': {'type': 'string'}}, ...}
    """
    if not _has_pydantic:
        raise ImportError(
            "pydantic is not installed. Install pydantic or pass a JSON "
            "Schema dict directly to your Tool's input_schema."
        )
    if hasattr(model, "model_json_schema"):
        return model.model_json_schema()
    raise TypeError(f"{model!r} is not a pydantic v2 BaseModel subclass.")
