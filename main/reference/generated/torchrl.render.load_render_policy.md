# load_render_policy

torchrl.render.load_render_policy(*config: [RenderConfig](torchrl.render.RenderConfig.html#torchrl.render.RenderConfig)*, *env: Any | None = None*, ***, *checkpoint: Any | None = None*, *checkpoint_digest: str | None = None*) → Any[[source]](../../_modules/torchrl/render/policy.html#load_render_policy)

Builds and prepares a policy for rendering.

Parameters:

- **config** - Render configuration.
- **env** - Optional environment used to expose specs to the policy factory.

Keyword Arguments:

- **checkpoint** - Checkpoint payload supplied by the caller. When `None`,
the payload is loaded from `config.ckpt`.
- **checkpoint_digest** - SHA256 digest of the checkpoint file. When `None`,
the digest is computed here.

Returns:

A TensorDict-compatible policy callable or module.