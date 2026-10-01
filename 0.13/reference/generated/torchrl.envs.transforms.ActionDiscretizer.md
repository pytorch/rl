# ActionDiscretizer

*class*torchrl.envs.transforms.ActionDiscretizer(*num_intervals: int | [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*, *action_key: NestedKey = 'action'*, *out_action_key: NestedKey = None*, *sampling=None*, *categorical: bool = True*)[[source]](../../_modules/torchrl/envs/transforms/_action.html#ActionDiscretizer)

A transform to discretize a continuous action space.

This transform makes it possible to use an algorithm designed for discrete
action spaces such as DQN over environments with a continuous action space.

Parameters:

- **num_intervals** (*int**or*[*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) - the number of discrete values
for each element of the action space. If a single integer is provided,
all action items are sliced with the same number of elements.
If a tensor is provided, it must have the same number of elements
as the action space (ie, the length of the `num_intervals` tensor
must match the last dimension of the action space).
- **action_key** (*NestedKey**,**optional*) - the action key to use. Points to
the action of the parent env (the floating point action).
Defaults to `"action"`.
- **out_action_key** (*NestedKey**,**optional*) - the key where the discrete
action should be written. If `None` is provided, it defaults to
the value of `action_key`. If both keys do not match, the
continuous action_spec is moved from the `full_action_spec`
environment attribute to the `full_state_spec` container,
as only the discrete action should be sampled for an action to
be taken. Providing `out_action_key` can ensure that the
floating point action is available to be recorded.
- **sampling** (*ActionDiscretizer.SamplingStrategy**,**optinoal*) - an element
of the `ActionDiscretizer.SamplingStrategy` `IntEnum` object
(`MEDIAN`, `LOW`, `HIGH` or `RANDOM`). Indicates how the
continuous action should be sampled in the provided interval.
- **categorical** (*bool**,**optional*) - if `False`, one-hot encoding is used.
Defaults to `True`.

Examples

```
>>> from torchrl.envs import GymEnv, check_env_specs
>>> import torch
>>> base_env = GymEnv("HalfCheetah-v4")
>>> num_intervals = torch.arange(5, 11)
>>> categorical = True
>>> sampling = ActionDiscretizer.SamplingStrategy.MEDIAN
>>> t = ActionDiscretizer(
... num_intervals=num_intervals,
... categorical=categorical,
... sampling=sampling,
... out_action_key="action_disc",
... )
>>> env = base_env.append_transform(t)
TransformedEnv(
 env=GymEnv(env=HalfCheetah-v4, batch_size=torch.Size([]), device=cpu),
 transform=ActionDiscretizer(
 num_intervals=tensor([ 5, 6, 7, 8, 9, 10]),
 action_key=action,
 out_action_key=action_disc,,
 sampling=0,
 categorical=True))
>>> check_env_specs(env)
>>> # Produce a rollout
>>> r = env.rollout(4)
>>> print(r)
TensorDict(
 fields={
 action: Tensor(shape=torch.Size([4, 6]), device=cpu, dtype=torch.float32, is_shared=False),
 action_disc: Tensor(shape=torch.Size([4, 6]), device=cpu, dtype=torch.int64, is_shared=False),
 done: Tensor(shape=torch.Size([4, 1]), device=cpu, dtype=torch.bool, is_shared=False),
 next: TensorDict(
 fields={
 done: Tensor(shape=torch.Size([4, 1]), device=cpu, dtype=torch.bool, is_shared=False),
 observation: Tensor(shape=torch.Size([4, 17]), device=cpu, dtype=torch.float64, is_shared=False),
 reward: Tensor(shape=torch.Size([4, 1]), device=cpu, dtype=torch.float32, is_shared=False),
 terminated: Tensor(shape=torch.Size([4, 1]), device=cpu, dtype=torch.bool, is_shared=False),
 truncated: Tensor(shape=torch.Size([4, 1]), device=cpu, dtype=torch.bool, is_shared=False)},
 batch_size=torch.Size([4]),
 device=cpu,
 is_shared=False),
 observation: Tensor(shape=torch.Size([4, 17]), device=cpu, dtype=torch.float64, is_shared=False),
 terminated: Tensor(shape=torch.Size([4, 1]), device=cpu, dtype=torch.bool, is_shared=False),
 truncated: Tensor(shape=torch.Size([4, 1]), device=cpu, dtype=torch.bool, is_shared=False)},
 batch_size=torch.Size([4]),
 device=cpu,
 is_shared=False)
>>> assert r["action"].dtype == torch.float
>>> assert r["action_disc"].dtype == torch.int64
>>> assert (r["action"] < base_env.action_spec.high).all()
>>> assert (r["action"] > base_env.action_spec.low).all()
```

Note

Custom Sampling Strategies

To implement a custom sampling strategy beyond the built-in options
(`MEDIAN`, `LOW`, `HIGH`, `RANDOM`), subclass `ActionDiscretizer`
and override the `custom_arange()` method. This
method computes the normalized interval positions (values in `[0, 1)`)
that determine where each discrete action maps within the continuous
action interval.

Example:

```
>>> class LogSpacedActionDiscretizer(ActionDiscretizer):
... def custom_arange(self, nint, device):
... # Use logarithmic spacing instead of linear
... return torch.logspace(-2, 0, nint, device=device) - 0.01
```

*class*SamplingStrategy(*value*, *names=None*, ***, *module=None*, *qualname=None*, *type=None*, *start=1*, *boundary=None*)[[source]](../../_modules/torchrl/envs/transforms/_action.html#ActionDiscretizer.SamplingStrategy)

The sampling strategies for ActionDiscretizer.

custom_arange(*nint*, *device*)[[source]](../../_modules/torchrl/envs/transforms/_action.html#ActionDiscretizer.custom_arange)

Compute the normalized interval positions for discretization.

This method generates values in the range [0, 1) that determine where
each discrete action maps within the continuous action interval.

Override this method in a subclass to implement custom sampling
strategies beyond the built-in `MEDIAN`, `LOW`, `HIGH`, and
`RANDOM` strategies.

Parameters:

- **nint** (*int*) - the number of intervals (discrete actions) for this
action dimension.
- **device** ([*torch.device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)) - the device on which to create the tensor.

Returns:

a 1D tensor of shape `(nint,)` with values in

`[0, 1)` representing the normalized positions within each
interval.

Return type:

[torch.Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)

Example

```
>>> class CustomActionDiscretizer(ActionDiscretizer):
... def custom_arange(self, nint, device):
... # Custom sampling: use logarithmic spacing
... return torch.logspace(-2, 0, nint, device=device) - 0.01
```

inv(*tensordict*)[[source]](../../_modules/torchrl/envs/transforms/_action.html#ActionDiscretizer.inv)

Reads the input tensordict, and for the selected keys, applies the inverse transform.

By default, this method:

- calls directly `_inv_apply_transform()`.
- does not call `_inv_call()`.

Note

`inv` also works with regular keyword arguments using [`dispatch`](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.nn.dispatch.html#tensordict.nn.dispatch) to cast the args
names to the keys.

Note

`inv` is called by [`extend()`](torchrl.data.ReplayBuffer.html#torchrl.data.ReplayBuffer.extend).

transform_input_spec(*input_spec*)[[source]](../../_modules/torchrl/envs/transforms/_action.html#ActionDiscretizer.transform_input_spec)

Transforms the input spec such that the resulting spec matches transform mapping.

Parameters:

**input_spec** ([*TensorSpec*](torchrl.data.TensorSpec.html#torchrl.data.TensorSpec)) - spec before the transform

Returns:

expected spec after the transform