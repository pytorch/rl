# RemoveEmptySpecs

*class*torchrl.envs.transforms.RemoveEmptySpecs(*in_keys: Sequence[NestedKey] | None = None*, *out_keys: Sequence[NestedKey] | None = None*, *in_keys_inv: Sequence[NestedKey] | None = None*, *out_keys_inv: Sequence[NestedKey] | None = None*)[[source]](../../_modules/torchrl/envs/transforms/_keys.html#RemoveEmptySpecs)

Removes empty specs and content from an environment.

Examples

```
>>> import torch
>>> from tensordict import TensorDict
>>> from torchrl.data import Unbounded, Composite, ... Categorical
>>> from torchrl.envs import EnvBase, TransformedEnv, RemoveEmptySpecs
>>>
>>>
>>> class DummyEnv(EnvBase):
... def __init__(self, *args, **kwargs):
... super().__init__(*args, **kwargs)
... self.observation_spec = Composite(
... observation=UnboundedContinuous((*self.batch_size, 3)),
... other=Composite(
... another_other=Composite(shape=self.batch_size),
... shape=self.batch_size,
... ),
... shape=self.batch_size,
... )
... self.action_spec = UnboundedContinuous((*self.batch_size, 3))
... self.done_spec = Categorical(
... 2, (*self.batch_size, 1), dtype=torch.bool
... )
... self.full_done_spec["truncated"] = self.full_done_spec[
... "terminated"].clone()
... self.reward_spec = Composite(
... reward=UnboundedContinuous(*self.batch_size, 1),
... other_reward=Composite(shape=self.batch_size),
... shape=self.batch_size
... )
...
... def _reset(self, tensordict):
... return self.observation_spec.rand().update(self.full_done_spec.zero())
...
... def _step(self, tensordict):
... return TensorDict(
... {},
... batch_size=[]
... ).update(self.observation_spec.rand()).update(
... self.full_done_spec.zero()
... ).update(self.full_reward_spec.rand())
...
... def _set_seed(self, seed) -> None:
... pass
>>>
>>>
>>> base_env = DummyEnv()
>>> print(base_env.rollout(2))
TensorDict(
 fields={
 action: Tensor(shape=torch.Size([2, 3]), device=cpu, dtype=torch.float32, is_shared=False),
 done: Tensor(shape=torch.Size([2, 1]), device=cpu, dtype=torch.bool, is_shared=False),
 next: TensorDict(
 fields={
 done: Tensor(shape=torch.Size([2, 1]), device=cpu, dtype=torch.bool, is_shared=False),
 observation: Tensor(shape=torch.Size([2, 3]), device=cpu, dtype=torch.float32, is_shared=False),
 other: TensorDict(
 fields={
 another_other: TensorDict(
 fields={
 },
 batch_size=torch.Size([2]),
 device=cpu,
 is_shared=False)},
 batch_size=torch.Size([2]),
 device=cpu,
 is_shared=False),
 other_reward: TensorDict(
 fields={
 },
 batch_size=torch.Size([2]),
 device=cpu,
 is_shared=False),
 reward: Tensor(shape=torch.Size([2, 1]), device=cpu, dtype=torch.float32, is_shared=False),
 terminated: Tensor(shape=torch.Size([2, 1]), device=cpu, dtype=torch.bool, is_shared=False),
 truncated: Tensor(shape=torch.Size([2, 1]), device=cpu, dtype=torch.bool, is_shared=False)},
 batch_size=torch.Size([2]),
 device=cpu,
 is_shared=False),
 observation: Tensor(shape=torch.Size([2, 3]), device=cpu, dtype=torch.float32, is_shared=False),
 terminated: Tensor(shape=torch.Size([2, 1]), device=cpu, dtype=torch.bool, is_shared=False),
 truncated: Tensor(shape=torch.Size([2, 1]), device=cpu, dtype=torch.bool, is_shared=False)},
 batch_size=torch.Size([2]),
 device=cpu,
 is_shared=False)
>>> check_env_specs(base_env)
>>> env = TransformedEnv(base_env, RemoveEmptySpecs())
>>> print(env.rollout(2))
TensorDict(
 fields={
 action: Tensor(shape=torch.Size([2, 3]), device=cpu, dtype=torch.float32, is_shared=False),
 done: Tensor(shape=torch.Size([2, 1]), device=cpu, dtype=torch.bool, is_shared=False),
 next: TensorDict(
 fields={
 done: Tensor(shape=torch.Size([2, 1]), device=cpu, dtype=torch.bool, is_shared=False),
 observation: Tensor(shape=torch.Size([2, 3]), device=cpu, dtype=torch.float32, is_shared=False),
 reward: Tensor(shape=torch.Size([2, 1]), device=cpu, dtype=torch.float32, is_shared=False),
 terminated: Tensor(shape=torch.Size([2, 1]), device=cpu, dtype=torch.bool, is_shared=False),
 truncated: Tensor(shape=torch.Size([2, 1]), device=cpu, dtype=torch.bool, is_shared=False)},
 batch_size=torch.Size([2]),
 device=cpu,
 is_shared=False),
 observation: Tensor(shape=torch.Size([2, 3]), device=cpu, dtype=torch.float32, is_shared=False),
 terminated: Tensor(shape=torch.Size([2, 1]), device=cpu, dtype=torch.bool, is_shared=False),
 truncated: Tensor(shape=torch.Size([2, 1]), device=cpu, dtype=torch.bool, is_shared=False)},
 batch_size=torch.Size([2]),
 device=cpu,
 is_shared=False)
check_env_specs(env)
```

forward(*next_tensordict: [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)*) → [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)

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

transform_input_spec(*input_spec: [TensorSpec](torchrl.data.TensorSpec.html#torchrl.data.TensorSpec)*) → [TensorSpec](torchrl.data.TensorSpec.html#torchrl.data.TensorSpec)[[source]](../../_modules/torchrl/envs/transforms/_keys.html#RemoveEmptySpecs.transform_input_spec)

Transforms the input spec such that the resulting spec matches transform mapping.

Parameters:

**input_spec** ([*TensorSpec*](torchrl.data.TensorSpec.html#torchrl.data.TensorSpec)) - spec before the transform

Returns:

expected spec after the transform

transform_output_spec(*output_spec: [Composite](torchrl.data.Composite.html#torchrl.data.Composite)*) → [Composite](torchrl.data.Composite.html#torchrl.data.Composite)[[source]](../../_modules/torchrl/envs/transforms/_keys.html#RemoveEmptySpecs.transform_output_spec)

Transforms the output spec such that the resulting spec matches transform mapping.

This method should generally be left untouched. Changes should be implemented using
`transform_observation_spec()`, `transform_reward_spec()` and `transform_full_done_spec()`.
:param output_spec: spec before the transform
:type output_spec: TensorSpec

Returns:

expected spec after the transform