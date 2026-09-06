# parse_nested_key

torchrl.render.parse_nested_key(*value: NestedKey | str | Sequence[str]*) → NestedKey[[source]](../../_modules/torchrl/render/config.html#parse_nested_key)

Parses dotted strings into TensorDict nested keys.

Parameters:

**value** - Nested key, dotted string, or sequence of key components.

Returns:

A TensorDict nested key.