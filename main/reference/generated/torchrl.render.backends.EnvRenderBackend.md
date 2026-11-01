# EnvRenderBackend

*class*torchrl.render.backends.EnvRenderBackend[[source]](../../_modules/torchrl/render/backends/env.html#EnvRenderBackend)

Captures frames by calling `env.render()`.

Examples

```
>>> from torchrl.render.backends import EnvRenderBackend
>>> EnvRenderBackend().name
'env'
```