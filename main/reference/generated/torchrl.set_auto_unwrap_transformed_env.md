# set_auto_unwrap_transformed_env

*class*torchrl.set_auto_unwrap_transformed_env(*mode: bool*)[[source]](../../_modules/torchrl/_utils.html#set_auto_unwrap_transformed_env)

A context manager or decorator to control whether TransformedEnv should automatically unwrap nested TransformedEnv instances.

Parameters:

**mode** (*bool*) - Whether to automatically unwrap nested `TransformedEnv`
instances. If `False`, `TransformedEnv` will not unwrap nested instances.
Defaults to `True`.

Note

Until v0.9, this will raise a warning if `TransformedEnv` are nested
and the value is not set explicitly (auto_unwrap=True default behavior).
You can set the value of `auto_unwrap_transformed_env()`
through:

- The `AUTO_UNWRAP_TRANSFORMED_ENV` environment variable;
- By setting `torchrl.set_auto_unwrap_transformed_env(val: bool).set()` at the
beginning of your script;
- By using `torchrl.set_auto_unwrap_transformed_env(val: bool)` as a context
manager or a decorator.

See also

`TransformedEnv`

Examples

```
>>> with set_auto_unwrap_transformed_env(False):
... env = TransformedEnv(TransformedEnv(env))
... assert not isinstance(env.base_env, TransformedEnv)
>>> @set_auto_unwrap_transformed_env(False)
... def my_function():
... env = TransformedEnv(TransformedEnv(env))
... assert not isinstance(env.base_env, TransformedEnv)
... return env
```