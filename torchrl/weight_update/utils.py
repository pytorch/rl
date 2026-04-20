from __future__ import annotations

import re
from typing import Any


def _resolve_attr(context: Any, attr_path: str) -> Any:
    """Resolve an attribute path like 'policy' or 'env.value_net' to actual object.

    Also processes getitem notation like 'env.transform[0]' or '_receiver_schemes["model_id"]'
    to actual object.

    Args:
        context: The context object (collector or inner_collector).
        attr_path: A string address like "policy", "env.value_net", or
            "_receiver_schemes['model_id']".

    Returns:
        The object at the specified address.

    Examples:
        >>> _resolve_attr(collector, "policy")  # -> collector.policy
        >>> _resolve_attr(collector, "env.value_net")  # -> collector.env.value_net
        >>> _resolve_attr(collector, "_receiver_schemes['model_id']")  # -> collector._receiver_schemes['model_id']
    """
    # Pattern to match subscript access: attr[key] or attr["key"] or attr['key'] or attr[0]
    subscript_pattern = re.compile(r"^([^\[]+)(.*)$")

    parts = attr_path.split(".")
    obj = context
    for i, part in enumerate(parts):
        if "[" in part:
            match = subscript_pattern.match(part)
            if match:
                key = match.group(1)
                subscripts_str = match.group(2)

                # Get the base attribute
                if key:
                    try:
                        obj = getattr(obj, key)
                    except AttributeError:
                        raise AttributeError(
                            f"Attribute {key} from {parts[:i + 1]} not found in {'.'.join(parts[:i])}={obj}"
                        )

                # Parse and apply all subscripts
                # Match each [xxx] where xxx can be int, 'string', or "string"
                subscript_matches = re.findall(r"\[([^\]]+)\]", subscripts_str)
                for subscript in subscript_matches:
                    # Try to parse as int first
                    try:
                        index = int(subscript)
                        obj = obj[index]
                    except ValueError:
                        # It's a string key - remove quotes if present
                        if (subscript.startswith("'") and subscript.endswith("'")) or (
                            subscript.startswith('"') and subscript.endswith('"')
                        ):
                            subscript = subscript[1:-1]
                        obj = obj[subscript]
        else:
            try:
                obj = getattr(obj, part)
            except AttributeError:
                raise AttributeError(
                    f"Attribute {part} from {parts[:i + 1]} not found in {'.'.join(parts[:i])}={obj}"
                )
    return obj


# Alias for backwards compatibility
_resolve_model = _resolve_attr
