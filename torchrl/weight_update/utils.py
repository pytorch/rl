from __future__ import annotations

from typing import Any


def _resolve_model(context: Any, model_id: str) -> Any:
    """Resolve model_id like 'policy' or 'env.value_net' to actual object.

    Also processes getitem notation like 'env.transform[0]' to actual object.

    Args:
        context: The context object (collector or inner_collector).
        model_id: A string address like "policy" or "env.value_net".

    Returns:
        The object at the specified address.

    Examples:
        _resolve_model(collector, "policy")  # -> collector.policy
        _resolve_model(collector, "env.value_net")  # -> collector.env.value_net
    """
    parts = model_id.split(".")
    obj = context
    for i, part in enumerate(parts):
        if "[" in part:
            key, *indices = part.split("[")
            indices = [int(index[:-1]) for index in indices]
            try:
                obj = getattr(obj, key)
            except AttributeError:
                raise AttributeError(
                    f"Attribute {key} from {parts[:i + 1]} not found in {'.'.join(parts[:i])}={obj}"
                )
            for index in indices:
                obj = obj[index]
        else:
            try:
                obj = getattr(obj, part)
            except AttributeError:
                raise AttributeError(
                    f"Attribute {part} from {parts[:i + 1]} not found in {'.'.join(parts[:i])}={obj}"
                )
    return obj
