# make_render_env

torchrl.render.make_render_env(*config: [RenderConfig](torchrl.render.RenderConfig.html#torchrl.render.RenderConfig)*, *checkpoint: Any | None = None*) → Any[[source]](../../_modules/torchrl/render/env.html#make_render_env)

Builds and prepares an environment for rendering.

Parameters:

- **config** - Render configuration.
- **checkpoint** - Optional checkpoint payload loaded from `config.ckpt`.
Exposed to environment factories so render environments can be
rebuilt from checkpointed metadata (see
[`save_render_checkpoint()`](torchrl.render.save_render_checkpoint.html#torchrl.render.save_render_checkpoint)).

Returns:

A TorchRL environment when wrapping is possible, otherwise the factory result.