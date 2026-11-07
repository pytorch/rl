# MujocoStateReader

*class*torchrl.render.backends.MujocoStateReader[[source]](../../_modules/torchrl/render/backends/mujoco.html#MujocoStateReader)

Reads simulator state from TorchRL-native and Gym MuJoCo environments.

The reader keeps simulator state separate from policy observations. It
accepts environments exposing a `get_state()` method that returns a
TensorDict or mapping with a `"qpos"` entry, as well as TorchRL
[`GymWrapper`](torchrl.envs.GymWrapper.html#torchrl.envs.GymWrapper) instances around Gymnasium MuJoCo
environments.

Examples

```
>>> from types import SimpleNamespace
>>> import numpy as np
>>> from torchrl.render.backends import MujocoStateReader
>>> env = SimpleNamespace(
... data=SimpleNamespace(
... qpos=np.array([0.0, 1.0]),
... qvel=np.array([0.5, 0.0]),
... time=0.25,
... )
... )
>>> state = MujocoStateReader().capture(env)
>>> state["qpos"].tolist()
[0.0, 1.0]
```

capture(*env: Any*) → [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)[[source]](../../_modules/torchrl/render/backends/mujoco.html#MujocoStateReader.capture)

Returns a detached snapshot of the environment's MuJoCo state.

Parameters:

**env** - TorchRL-native MuJoCo environment, Gym-backed MuJoCo
environment, or object exposing MuJoCo `data` directly.

Returns:

A TensorDict containing `qpos` and any available `qvel`,
`act`, `ctrl`, `mocap_pos`, `mocap_quat`, and `time`
entries.

Raises:

- **TypeError** - If the environment does not expose a supported state.
- **KeyError** - If the exposed state does not contain `qpos`.

supports(*env: Any*) → bool[[source]](../../_modules/torchrl/render/backends/mujoco.html#MujocoStateReader.supports)

Returns whether `env` exposes a readable MuJoCo state.