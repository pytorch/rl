# NullRenderBackend

*class*torchrl.render.backends.NullRenderBackend[[source]](../../_modules/torchrl/render/backends/null.html#NullRenderBackend)

Fallback backend used when no RGB renderer is available.

Examples

```
>>> from torchrl.render.backends import NullRenderBackend
>>> NullRenderBackend().capture(None, None, None, step=0, trajectory_index=0) is None
True
```