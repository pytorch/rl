# MJLabWrapper

torchrl.envs.MJLabWrapper(**args*, ***kwargs*)[[source]](../../_modules/torchrl/envs/libs/mjlab.html#MJLabWrapper)

TorchRL wrapper for a pre-built `mjlab.envs.ManagerBasedRlEnv`.

Parameters:

- **env** - The mjlab manager-based RL environment to wrap.
- **device** - Torch device for actions and returned tensors. When `None`,
it is inferred from `env.device`. The TorchRL device must match
the mjlab simulation device; a mismatch raises an error instead of
silently copying tensors across devices in the hot path.
- **batch_size** - Batch size of the wrapper. When `None`, it is inferred as
`[env.num_envs]`. mjlab exposes a flat vectorized batch, so only
one-dimensional batch sizes are accepted.
- **native_autoreset** - If `False` (default), TorchRL drives resets through
the standard `"_reset"` mask by setting `env.cfg.auto_reset` to
`False` and calling `env.reset(env_ids=...)` on done rows. If
`True`, mjlab's native auto-reset is kept and TorchRL marks the
terminal `"next"` observation as invalid while carrying mjlab's
post-reset observation into the next root tensordict, matching the
native-auto-reset contract used by Isaac Lab.
- **from_pixels** - If `True`, append RGB pixels under `pixels_key` at
reset and step time. If the mjlab scene has an RGB
`CameraSensor`, TorchRL uses the sensor's batched output with
shape `[num_envs, H, W, 3]`. Otherwise it falls back to
`render()`, which requires `num_envs == 1` and
`render_mode="rgb_array"` because mjlab viewer rendering returns
one frame for the whole scene rather than one image per
environment row.
- **pixels_only** - If `True`, return only the pixel observation. Requires
`from_pixels=True`.
- **pixels_key** - TensorDict key for pixel observations. Defaults to
`"pixels"`.
- **pixels_sensor** - Name of the mjlab RGB `CameraSensor` to use for
`from_pixels=True`. If `None` and exactly one RGB camera sensor
is present, it is selected automatically.
- **allow_done_after_reset** - Passed to [`EnvBase`](torchrl.envs.EnvBase.html#torchrl.envs.EnvBase).
Defaults to `False`.

mjlab reference: Zakka et al., "mjlab: A Lightweight Framework for
GPU-Accelerated Robot Learning", arXiv:2601.22074.

Examples

```
>>> from mjlab.envs import ManagerBasedRlEnv 
>>> from mjlab.tasks.registry import load_env_cfg 
>>> from torchrl.envs import MJLabWrapper 
>>> cfg = load_env_cfg("Mjlab-Cartpole-Balance") 
>>> cfg.scene.num_envs = 4 
>>> base_env = ManagerBasedRlEnv(cfg, device="cuda:0") 
>>> env = MJLabWrapper(base_env) 
>>> td = env.rollout(10)
```