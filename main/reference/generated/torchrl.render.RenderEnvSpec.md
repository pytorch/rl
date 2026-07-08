# RenderEnvSpec

*class*torchrl.render.RenderEnvSpec(*device: [device](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)*, *seed: int | None*, *max_steps: int | None*, *from_pixels: bool*, *pixels_only: bool*, *camera: list[str]*, *render_mode: str | None*, *env_kwargs: dict[str, Any]*, *config: [RenderConfig](torchrl.render.RenderConfig.html#torchrl.render.RenderConfig)*, *checkpoint: Any | None = None*)[[source]](../../_modules/torchrl/render/config.html#RenderEnvSpec)

Context object passed to environment factories.

Parameters:

- **device** - Device requested for the environment.
- **seed** - Optional environment seed.
- **max_steps** - Optional rollout step limit.
- **from_pixels** - Whether the factory should request pixel observations.
- **pixels_only** - Whether only pixel observations should be returned.
- **camera** - Requested camera names.
- **render_mode** - Optional render mode, such as `"rgb_array"`.
- **env_kwargs** - Extra keyword arguments supplied by the user.
- **config** - Full render configuration.
- **checkpoint** - Checkpoint payload loaded from `config.ckpt`, when
available. Factories can read checkpointed environment metadata
from it (see [`save_render_checkpoint()`](torchrl.render.save_render_checkpoint.html#torchrl.render.save_render_checkpoint)).

Examples

```
>>> from torchrl.render import RenderConfig, RenderEnvSpec
>>> cfg = RenderConfig("policy.pt", "p:make", "e:make", max_steps=2)
>>> spec = RenderEnvSpec.from_config(cfg)
>>> spec.max_steps
2
```

*classmethod*from_config(*config: [RenderConfig](torchrl.render.RenderConfig.html#torchrl.render.RenderConfig)*, *checkpoint: Any | None = None*) → RenderEnvSpec[[source]](../../_modules/torchrl/render/config.html#RenderEnvSpec.from_config)

Builds an environment spec from a render config.