# GenesisEnv

torchrl.envs.GenesisEnv(**args*, *num_workers: int | None = None*, ***kwargs*)[[source]](../../_modules/torchrl/envs/libs/genesis.html#GenesisEnv)

Genesis environment built from a named configuration.

Parameters:

- **env_name** (*str*) - registered environment name. Currently one of
`'franka_reach'` or `'franka_grab'`.
- **task_name** (*str**,**optional*) - task name; unused by the built-in configs.
- **num_workers** (*int**,**optional*) - when `> 1`, returns a lazy
[`ParallelEnv`](torchrl.envs.ParallelEnv.html#torchrl.envs.ParallelEnv) wrapping per-worker Genesis envs.
Defaults to `1`.
- **max_steps** (*int**,**optional*) - truncation horizon. Defaults to `1000`.
- **frame_skip** (*int**,**optional*) - physics steps per env step. Defaults to `1`.
- **from_pixels** (*bool**,**optional*) - if `True`, a default camera is added
to the scene before it is built and a `pixels` entry is added
to the observation. Defaults to `False`.
- **pixels_key** (*str**,**optional*) - key for the rendered frame. Defaults to
`"pixels"`.
- **pixels_res** (*tuple**[**int**,**int**]**,**optional*) - `(H, W)` of the auto-added
camera. Ignored when `from_pixels=False`. Defaults to
`(320, 320)`.
- **device** ([*torch.device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)*,**optional*) - torch device. When `None` (the
default), uses [`torch.get_default_device()`](https://docs.pytorch.org/docs/stable/generated/torch.get_default_device.html#torch.get_default_device) and calls
`genesis.init()` with the matching backend (`gs.cpu`,
`gs.cuda`, or `gs.metal`). If Genesis is already initialized
with a different device, raises `RuntimeError`.
- **batch_size** ([*torch.Size*](https://docs.pytorch.org/docs/stable/size.html#torch.Size)*,**optional*) - env batch size. Defaults to `()`.
- **allow_done_after_reset** (*bool**,**optional*) - Defaults to `False`.

For task-specific obs / reward / action, subclass [`GenesisWrapper`](torchrl.envs.GenesisWrapper.html#torchrl.envs.GenesisWrapper)
directly - see its docstring for the hook list.

Examples

```
>>> from torchrl.envs import GenesisEnv
>>> env = GenesisEnv(env_name="franka_reach")
>>> td = env.rollout(10)
```