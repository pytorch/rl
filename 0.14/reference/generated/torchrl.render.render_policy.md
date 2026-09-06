# render_policy

torchrl.render.render_policy(*config: [RenderConfig](torchrl.render.RenderConfig.html#torchrl.render.RenderConfig)*) → [RenderResult](torchrl.render.RenderResult.html#torchrl.render.RenderResult)[[source]](../../_modules/torchrl/render.html#render_policy)

Renders a policy according to `config` and writes the requested artifact.

Parameters:

**config** - Render configuration.

Returns:

The render result with trajectories, metadata, and artifact paths.