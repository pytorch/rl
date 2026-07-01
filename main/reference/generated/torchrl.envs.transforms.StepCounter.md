# StepCounter

*class*torchrl.envs.transforms.StepCounter(*max_steps: int | None = None*, *truncated_key: str | None = 'truncated'*, *step_count_key: str | None = 'step_count'*, *update_done: bool = True*)[[source]](../../_modules/torchrl/envs/transforms/_env.html#StepCounter)

Counts the steps from a reset and optionally sets the truncated state to `True` after a certain number of steps.

The `"done"` state is also adapted accordingly (as done is the disjunction
of task completion and early truncation).

Parameters:

- **max_steps** (*int**,**optional*) - a positive integer that indicates the
maximum number of steps to take before setting the `truncated_key`
entry to `True`.
- **truncated_key** (*str**,**optional*) - the key where the truncated entries
should be written. Defaults to `"truncated"`, which is recognised by
data collectors as a reset signal.
This argument can only be a string (not a nested key) as it will be
matched to each of the leaf done key in the parent environment
(eg, a `("agent", "done")` key will be accompanied by a
`("agent", "truncated")` if the `"truncated"` key name is used).
- **step_count_key** (*str**,**optional*) - the key where the step count entries
should be written. Defaults to `"step_count"`.
This argument can only be a string (not a nested key) as it will be
matched to each of the leaf done key in the parent environment
(eg, a `("agent", "done")` key will be accompanied by a
`("agent", "step_count")` if the `"step_count"` key name is used).
- **update_done** (*bool**,**optional*) - if `True`, the `"done"` boolean tensor
at the level of `"truncated"`
will be updated.
This signal indicates that the trajectory has reached its ends,
either because the task is completed (`"completed"` entry is
`True`) or because it has been truncated (`"truncated"` entry
is `True`).
Defaults to `True`.

Note

To ensure compatibility with environments that have multiple
done_key(s), this transform will write a step_count entry for
every done entry within the tensordict.

Examples

```
>>> import gymnasium
>>> from torchrl.envs import GymWrapper
>>> base_env = GymWrapper(gymnasium.make("Pendulum-v1"))
>>> env = TransformedEnv(base_env,
... StepCounter(max_steps=5))
>>> rollout = env.rollout(100)
>>> print(rollout)
TensorDict(
 fields={
 action: Tensor(shape=torch.Size([5, 1]), device=cpu, dtype=torch.float32, is_shared=False),
 done: Tensor(shape=torch.Size([5, 1]), device=cpu, dtype=torch.bool, is_shared=False),
 completed: Tensor(shape=torch.Size([5, 1]), device=cpu, dtype=torch.bool, is_shared=False)},
 next: TensorDict(
 fields={
 done: Tensor(shape=torch.Size([5, 1]), device=cpu, dtype=torch.bool, is_shared=False),
 completed: Tensor(shape=torch.Size([5, 1]), device=cpu, dtype=torch.bool, is_shared=False)},
 observation: Tensor(shape=torch.Size([5, 3]), device=cpu, dtype=torch.float32, is_shared=False),
 reward: Tensor(shape=torch.Size([5, 1]), device=cpu, dtype=torch.float32, is_shared=False),
 step_count: Tensor(shape=torch.Size([5, 1]), device=cpu, dtype=torch.int64, is_shared=False),
 truncated: Tensor(shape=torch.Size([5, 1]), device=cpu, dtype=torch.bool, is_shared=False)},
 batch_size=torch.Size([5]),
 device=cpu,
 is_shared=False),
 observation: Tensor(shape=torch.Size([5, 3]), device=cpu, dtype=torch.float32, is_shared=False),
 step_count: Tensor(shape=torch.Size([5, 1]), device=cpu, dtype=torch.int64, is_shared=False),
 truncated: Tensor(shape=torch.Size([5, 1]), device=cpu, dtype=torch.bool, is_shared=False)},
 batch_size=torch.Size([5]),
 device=cpu,
 is_shared=False)
>>> print(rollout["next", "step_count"])
tensor([[1],
 [2],
 [3],
 [4],
 [5]])
```

*property*all_done_keys*: list[NestedKey]*

Returns done keys for ALL reset keys (including nested ones).

Used for propagating done to nested agent-level keys in MARL envs.

*property*all_truncated_keys*: list[NestedKey]*

Returns truncated keys for ALL reset keys (including nested ones).

Used for propagating truncated to nested agent-level keys in MARL envs.

forward(*tensordict: [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)*) → [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)[[source]](../../_modules/torchrl/envs/transforms/_env.html#StepCounter.forward)

Reads the input tensordict, and for the selected keys, applies the transform.

By default, this method:

- calls directly `_apply_transform()`.
- does not call `_step()` or `_call()`.

This method is not called within env.step at any point. However, is is called within
[`sample()`](torchrl.data.ReplayBuffer.html#torchrl.data.ReplayBuffer.sample).

Note

`forward` also works with regular keyword arguments using [`dispatch`](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.nn.dispatch.html#tensordict.nn.dispatch) to cast the args
names to the keys.

Examples

```
>>> class TransformThatMeasuresBytes(Transform):
... '''Measures the number of bytes in the tensordict, and writes it under `"bytes"`.'''
... def __init__(self):
... super().__init__(in_keys=[], out_keys=["bytes"])
...
... def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
... bytes_in_td = tensordict.bytes()
... tensordict["bytes"] = bytes
... return tensordict
>>> t = TransformThatMeasuresBytes()
>>> env = env.append_transform(t) # works within envs
>>> t(TensorDict(a=0)) # Works offline too.
```

transform_input_spec(*input_spec: [Composite](torchrl.data.Composite.html#torchrl.data.Composite)*) → [Composite](torchrl.data.Composite.html#torchrl.data.Composite)[[source]](../../_modules/torchrl/envs/transforms/_env.html#StepCounter.transform_input_spec)

Transforms the input spec such that the resulting spec matches transform mapping.

Parameters:

**input_spec** ([*TensorSpec*](torchrl.data.TensorSpec.html#torchrl.data.TensorSpec)) - spec before the transform

Returns:

expected spec after the transform

transform_observation_spec(*observation_spec: [Composite](torchrl.data.Composite.html#torchrl.data.Composite)*) → [Composite](torchrl.data.Composite.html#torchrl.data.Composite)[[source]](../../_modules/torchrl/envs/transforms/_env.html#StepCounter.transform_observation_spec)

Transforms the observation spec such that the resulting spec matches transform mapping.

Parameters:

**observation_spec** ([*TensorSpec*](torchrl.data.TensorSpec.html#torchrl.data.TensorSpec)) - spec before the transform

Returns:

expected spec after the transform

transform_output_spec(*output_spec: [Composite](torchrl.data.Composite.html#torchrl.data.Composite)*) → [Composite](torchrl.data.Composite.html#torchrl.data.Composite)[[source]](../../_modules/torchrl/envs/transforms/_env.html#StepCounter.transform_output_spec)

Transforms the output spec such that the resulting spec matches transform mapping.

This method should generally be left untouched. Changes should be implemented using
`transform_observation_spec()`, `transform_reward_spec()` and `transform_full_done_spec()`.
:param output_spec: spec before the transform
:type output_spec: TensorSpec

Returns:

expected spec after the transform