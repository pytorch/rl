# RenderBackend

*class*torchrl.render.backends.RenderBackend(**args*, ***kwargs*)[[source]](../../_modules/torchrl/render/backends/base.html#RenderBackend)

Protocol implemented by rlrender frame-capture backends.

Examples

```
>>> from torchrl.render.backends import NullRenderBackend
>>> backend = NullRenderBackend()
>>> backend.name
'null'
```

capture(*env: Any*, *tensordict: [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)*, *config: [RenderConfig](torchrl.render.RenderConfig.html#torchrl.render.RenderConfig)*, ***, *step: int*, *trajectory_index: int*) → [FrameBundle](torchrl.render.FrameBundle.html#torchrl.render.FrameBundle) | None[[source]](../../_modules/torchrl/render/backends/base.html#RenderBackend.capture)

Captures frames for one rollout step.

close() → None[[source]](../../_modules/torchrl/render/backends/base.html#RenderBackend.close)

Closes backend resources.

supports(*env: Any*, *config: [RenderConfig](torchrl.render.RenderConfig.html#torchrl.render.RenderConfig)*) → bool[[source]](../../_modules/torchrl/render/backends/base.html#RenderBackend.supports)

Returns whether the backend can capture from this environment.