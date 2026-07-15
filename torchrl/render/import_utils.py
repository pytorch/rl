# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import hashlib
import importlib
import importlib.util
import inspect
import sys
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

__all__ = ["call_with_supported_kwargs", "import_from_string"]


def import_from_string(spec: str) -> Any:
    """Imports an object from a ``"module:attribute"`` string.

    Args:
        spec: Import specification.

    Returns:
        The imported Python object.
    """
    if not isinstance(spec, str) or ":" not in spec:
        raise ValueError(
            "Import specs must have the form 'module.submodule:object', "
            f"got {spec!r}."
        )
    module_name, attr_path = spec.rsplit(":", 1)
    if not module_name or not attr_path:
        raise ValueError(
            "Import specs must have the form 'module.submodule:object', "
            f"got {spec!r}."
        )
    module = _import_module_or_file(module_name, spec)
    obj: Any = module
    current = module_name
    for attr in attr_path.split("."):
        current = f"{current}.{attr}"
        if not hasattr(obj, attr):
            raise ImportError(
                f"Could not import {spec!r}. Checked '{current}', but the "
                f"attribute '{attr}' was missing."
            )
        obj = getattr(obj, attr)
    return obj


def _import_module_or_file(module_name: str, spec: str) -> Any:
    path = Path(module_name).expanduser()
    if path.suffix == ".py" or "/" in module_name or "\\" in module_name:
        if not path.exists():
            raise ImportError(
                f"Could not import file '{module_name}' from import spec {spec!r}."
            )
        digest = hashlib.sha1(str(path.resolve()).encode()).hexdigest()
        import_name = f"_torchrl_render_{path.stem}_{digest}"
        module_spec = importlib.util.spec_from_file_location(import_name, path)
        if module_spec is None or module_spec.loader is None:
            raise ImportError(
                f"Could not load Python file '{module_name}' from import spec {spec!r}."
            )
        module = importlib.util.module_from_spec(module_spec)
        sys.modules[import_name] = module
        try:
            module_spec.loader.exec_module(module)
        except Exception as err:
            raise ImportError(
                f"Could not execute Python file '{module_name}' from import spec {spec!r}."
            ) from err
        return module
    try:
        return importlib.import_module(module_name)
    except Exception as err:
        raise ImportError(
            f"Could not import module '{module_name}' from import spec {spec!r}."
        ) from err


def call_with_supported_kwargs(
    factory: Callable[..., Any],
    preferred_arg: Any,
    kwargs: Mapping[str, Any],
) -> Any:
    """Calls a user factory with a spec object or supported keyword arguments.

    Args:
        factory: User callable.
        preferred_arg: Spec object used for the documented one-argument protocol.
        kwargs: Keyword candidates for convenience protocols.

    Returns:
        The factory return value.
    """
    signature = inspect.signature(factory)
    parameters = list(signature.parameters.values())
    if any(param.kind == inspect.Parameter.VAR_POSITIONAL for param in parameters):
        return factory(preferred_arg)
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in parameters):
        return factory(**dict(kwargs))
    positional = [
        param
        for param in parameters
        if param.kind
        in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    required = [
        param
        for param in parameters
        if param.default is inspect.Parameter.empty
        and param.kind
        in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        )
    ]
    supported = {
        name: value
        for name, value in kwargs.items()
        if name in signature.parameters
        and signature.parameters[name].kind
        in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    }
    missing = [param.name for param in required if param.name not in supported]
    if missing:
        if len(positional) == 1 and len(missing) == 1:
            return factory(preferred_arg)
        raise TypeError(
            f"Could not call factory {factory!r}; missing required argument(s) "
            f"{missing}. Use a single spec argument or one of {sorted(kwargs)}."
        )
    return factory(**supported)
