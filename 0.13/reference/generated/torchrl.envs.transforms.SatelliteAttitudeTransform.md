# SatelliteAttitudeTransform

*class*torchrl.envs.transforms.SatelliteAttitudeTransform(**args: Any*, *execute: bool = False*, *multi_action_dim: int = 1*, *stack_rewards: bool = True*, *stack_observations: bool = False*, ***kwargs: Any*)[[source]](../../_modules/torchrl/envs/custom/mujoco/_satellite_primitives.html#SatelliteAttitudeTransform)

Expand desired satellite attitudes into CMG gimbal-rate sequences.

This transform is a satellite-specific preset. The policy-facing action is a
desired attitude quaternion, provided either as a raw tensor under
`action_key`, under `(action_key, "target")` / `(action_key,
"attitude")`, or through a [`SatelliteMacroAction`](torchrl.envs.transforms.SatelliteMacroAction.html#torchrl.envs.transforms.SatelliteMacroAction) (which also carries
per-action durations). The transform computes the current attitude error,
applies a small proportional-derivative steering law in body-rate
coordinates, maps it through the instantaneous CMG Jacobian, and delegates
fixed-length interpolation / execution to
[`MacroPrimitiveTransform`](torchrl.envs.transforms.MacroPrimitiveTransform.html#torchrl.envs.transforms.MacroPrimitiveTransform).

Parameters:

- **num_cmgs** - `4` for the pyramid CMG cluster or `6` for the orthogonal
cluster.
- **action_scale** - scale used by [`SatelliteEnv`](torchrl.envs.SatelliteEnv.html#torchrl.envs.SatelliteEnv) to map
normalized actions to physical gimbal rates. If `None`, the
transform tries to read `action_scale` from its parent env and
falls back to `3.0`.
- **attitude_gain** - proportional gain applied to the quaternion log error.
- **angular_rate_gain** - damping gain applied to `bus_omega`.
- **jacobian_rotor_h** - rotor-momentum scale used in the steering Jacobian.

Examples

```
>>> import torch
>>> from tensordict import TensorDict
>>> from torchrl.envs import SatelliteAttitudeTransform
>>> td = TensorDict({
... "action": torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
... "bus_quat": torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
... "bus_omega": torch.zeros(1, 3),
... "gimbal_angles": torch.cat([torch.zeros(1, 4), torch.ones(1, 4)], -1),
... }, batch_size=[1])
>>> SatelliteAttitudeTransform(num_cmgs=4, macro_steps=2, settle_steps=0).inv(td)["action"].shape
torch.Size([1, 2, 4])
```

attitude_action_target(*tensordict: [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)*, *target_quat: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*) → [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)[[source]](../../_modules/torchrl/envs/custom/mujoco/_satellite_primitives.html#SatelliteAttitudeTransform.attitude_action_target)

Compute the normalized gimbal-rate target for `target_quat`.

current_action(*tensordict: [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)*, *batch_shape: [Size](https://docs.pytorch.org/docs/stable/size.html#torch.Size)*, *device: [device](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)*, *dtype: [dtype](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.dtype)*, *action_dim: int*) → [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)[[source]](../../_modules/torchrl/envs/custom/mujoco/_satellite_primitives.html#SatelliteAttitudeTransform.current_action)

Return the low-level action used as the interpolation start.

The base implementation starts every macro from the zero action: in the
inverse path `action_key` carries the incoming macro action (the
*target*), so it must not be read back here as the start. Subclasses that
can read the controlled state from observations (e.g. joint positions)
override this hook.

transform_input_spec(*input_spec: [Composite](torchrl.data.Composite.html#torchrl.data.Composite)*) → [Composite](torchrl.data.Composite.html#torchrl.data.Composite)[[source]](../../_modules/torchrl/envs/custom/mujoco/_satellite_primitives.html#SatelliteAttitudeTransform.transform_input_spec)

Transforms the input spec such that the resulting spec matches transform mapping.

Parameters:

**input_spec** ([*TensorSpec*](torchrl.data.TensorSpec.html#torchrl.data.TensorSpec)) - spec before the transform

Returns:

expected spec after the transform