# MJLabEnv

torchrl.envs.MJLabEnv(**args*, *num_workers: int | None = None*, ***kwargs*)[[source]](../../_modules/torchrl/envs/libs/mjlab.html#MJLabEnv)

Build and wrap an mjlab task from mjlab's task registry.

Parameters:

- **task_id** - Registered mjlab task id, for example
`"Mjlab-Velocity-Flat-Unitree-G1"`.
- **cfg** - Optional mjlab `ManagerBasedRlEnvCfg`. When omitted, `task_id`
is loaded from `mjlab.tasks.registry`. The config is deep-copied
before TorchRL mutates `scene.num_envs` or `auto_reset`.
- **play** - If `True` and `cfg` is omitted, load mjlab's play/evaluation
config for `task_id`.
- **num_envs** - Number of parallel mjlab worlds. Overrides
`cfg.scene.num_envs`.
- **device** - Simulation device. Defaults to `"cuda:0"` when CUDA is
available, otherwise `"cpu"`.
- **batch_size** - TorchRL batch size. Must be `[num_envs]`. If provided and
`num_envs` is omitted, it sets `num_envs`.
- **render_mode** - mjlab render mode. Set to `"rgb_array"` to enable
`render()`. Automatically set when `from_pixels=True` uses
the single-env render fallback. It is not required when pixels come
from an mjlab RGB `CameraSensor`.
- **native_autoreset** - See [`MJLabWrapper`](torchrl.envs.MJLabWrapper.html#torchrl.envs.MJLabWrapper).
- **from_pixels** - See [`MJLabWrapper`](torchrl.envs.MJLabWrapper.html#torchrl.envs.MJLabWrapper).
- **pixels_only** - See [`MJLabWrapper`](torchrl.envs.MJLabWrapper.html#torchrl.envs.MJLabWrapper).
- **pixels_key** - See [`MJLabWrapper`](torchrl.envs.MJLabWrapper.html#torchrl.envs.MJLabWrapper).
- **pixels_sensor** - See [`MJLabWrapper`](torchrl.envs.MJLabWrapper.html#torchrl.envs.MJLabWrapper).
- **num_workers** - If greater than one, return a lazy
[`ParallelEnv`](torchrl.envs.ParallelEnv.html#torchrl.envs.ParallelEnv) with one mjlab env per worker.

See also `MJLabEnvConfig`.

mjlab reference: Zakka et al., "mjlab: A Lightweight Framework for
GPU-Accelerated Robot Learning", arXiv:2601.22074.

Examples

```
>>> from torchrl.envs import MJLabEnv 
>>> env = MJLabEnv( 
... "Mjlab-Velocity-Flat-Unitree-G1", num_envs=1024, device="cuda:0"
... )
>>> td = env.reset()
```