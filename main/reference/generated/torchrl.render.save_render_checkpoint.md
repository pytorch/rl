# save_render_checkpoint

torchrl.render.save_render_checkpoint(*path: str | Path | None*, *model: Any*, ***, *env_metadata: Mapping[str, Any] | None = None*, *frames: int | None = None*, *metrics: Mapping[str, Any] | None = None*, *config: Mapping[str, Any] | None = None*, *extra: Mapping[str, Any] | None = None*) → Path | None[[source]](../../_modules/torchrl/render/checkpoint.html#save_render_checkpoint)

Writes a checkpoint in the layout expected by rlrender factories.

The model weights are stored under the canonical `"model_state_dict"` key
probed by [`infer_state_dict()`](torchrl.render.infer_state_dict.html#torchrl.render.infer_state_dict), and `env_metadata` entries are merged
at the top level of the payload so environment and policy factories can
rebuild the training setup. Conventional environment metadata keys used by
the sota-implementations factories are `"env_name"`, `"env_backend"`,
`"env_config_overrides"`, `"env_num_envs"`, `"env_batch_mode"`,
`"normalize_observation"`, and `"vecnorm"` (frozen observation
normalization statistics).

Parameters:

- **path** - Destination checkpoint path. `None` or `""` disables
checkpointing.
- **model** - Module exposing `state_dict()`, or a ready state-dict mapping.
- **env_metadata** - Environment metadata merged into the payload.
- **frames** - Number of training frames collected so far.
- **metrics** - Scalar metrics recorded at checkpoint time.
- **config** - JSON-serializable training configuration.
- **extra** - Additional payload entries merged last.

Returns:

The written checkpoint path, or `None` when checkpointing is disabled.

Examples

```
>>> import tempfile
>>> import torch
>>> from torchrl.render import load_checkpoint, save_render_checkpoint
>>> module = torch.nn.Linear(2, 2)
>>> with tempfile.TemporaryDirectory() as tmpdir:
... path = save_render_checkpoint(
... f"{tmpdir}/policy.pt", module, env_metadata={"env_name": "CartPole-v1"}
... )
... payload = load_checkpoint(path)
>>> payload["env_name"]
'CartPole-v1'
```