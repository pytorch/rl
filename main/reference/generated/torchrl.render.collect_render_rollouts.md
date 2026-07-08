# collect_render_rollouts

torchrl.render.collect_render_rollouts(*env: Any*, *policy: Any*, *config: [RenderConfig](torchrl.render.RenderConfig.html#torchrl.render.RenderConfig)*) → [RenderResult](torchrl.render.RenderResult.html#torchrl.render.RenderResult)[[source]](../../_modules/torchrl/render/rollout.html#collect_render_rollouts)

Collects sequential render rollouts.

[`EnvBase`](torchrl.envs.EnvBase.html#torchrl.envs.EnvBase) environments are rolled out through
[`rollout()`](torchrl.envs.EnvBase.html#id2); other environments fall back to a
duck-typed reset/step loop. Captured frames span the initial state through
the terminal state of each trajectory.

Parameters:

- **env** - Environment returned by [`torchrl.render.make_render_env()`](torchrl.render.make_render_env.html#torchrl.render.make_render_env).
- **policy** - TensorDict-compatible policy.
- **config** - Render configuration.

Returns:

A render result containing trajectories and in-memory frames.