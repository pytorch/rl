# call_with_supported_kwargs

torchrl.render.call_with_supported_kwargs(*factory: Callable[[...], Any]*, *preferred_arg: Any*, *kwargs: Mapping[str, Any]*) → Any[[source]](../../_modules/torchrl/render/import_utils.html#call_with_supported_kwargs)

Calls a user factory with a spec object or supported keyword arguments.

Parameters:

- **factory** - User callable.
- **preferred_arg** - Spec object used for the documented one-argument protocol.
- **kwargs** - Keyword candidates for convenience protocols.

Returns:

The factory return value.