# extract_qpos_trajectory

torchrl.render.extract_qpos_trajectory(*rollout: [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)*, *qpos_key: NestedKey | str = 'qpos'*) → list[list[float]][[source]](../../_modules/torchrl/render/mujoco_wasm.html#extract_qpos_trajectory)

Extracts a qpos trajectory from a rollout TensorDict.

Parameters:

- **rollout** - Rollout TensorDict saved by `rlrender`.
- **qpos_key** - TensorDict key containing a `T x nq` qpos tensor.

Returns:

A Python list of qpos waypoints suitable for
[`play_mujoco_wasm_trajectory()`](torchrl.render.play_mujoco_wasm_trajectory.html#torchrl.render.play_mujoco_wasm_trajectory).

Examples

```
>>> import torch
>>> from tensordict import TensorDict
>>> from torchrl.render.mujoco_wasm import extract_qpos_trajectory
>>> rollout = TensorDict({"qpos": torch.zeros(2, 3)}, batch_size=[2])
>>> extract_qpos_trajectory(rollout, "qpos")
[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
```