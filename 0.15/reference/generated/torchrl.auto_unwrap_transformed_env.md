# auto_unwrap_transformed_env

*class*torchrl.auto_unwrap_transformed_env(*allow_none=False*)[[source]](../../_modules/torchrl/_utils.html#auto_unwrap_transformed_env)

Get the current setting for automatically unwrapping TransformedEnv instances.

Parameters:

**allow_none** (*bool**,**optional*) - If True, returns `None` if no setting has been
specified. Otherwise, returns the default setting. Defaults to `False`.

seealso: [`set_auto_unwrap_transformed_env()`](torchrl.set_auto_unwrap_transformed_env.html#torchrl.set_auto_unwrap_transformed_env)

Returns:

The current setting for automatically unwrapping TransformedEnv

instances.

Return type:

bool or None