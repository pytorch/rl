# Stack

*class*torchrl.envs.transforms.Stack(*in_keys: Sequence[NestedKey]*, *out_key: NestedKey*, *in_key_inv: NestedKey | None = None*, *out_keys_inv: Sequence[NestedKey] | None = None*, *dim: int = -1*, *allow_positive_dim: bool = False*, ***, *del_keys: bool = True*)[[source]](../../_modules/torchrl/envs/transforms/_tensor.html#Stack)

Stacks tensors and tensordicts.

Concatenates a sequence of tensors or tensordicts along a new dimension.
The tensordicts or tensors under `in_keys` must all have the same shapes.

This transform only stacks the inputs into one output key. Stacking multiple
groups of input keys into different output keys requires multiple
transforms.

This transform can be useful for environments that have multiple agents with
identical specs under different keys. The specs and tensordicts for the
agents can be stacked together under a shared key, in order to run MARL
algorithms that expect the tensors for observations, rewards, etc. to
contain batched data for all the agents.

Parameters:

- **in_keys** (*sequence**of**NestedKey*) - keys to be stacked.
- **out_key** (*NestedKey*) - key of the resulting stacked entry.
- **in_key_inv** (*NestedKey**,**optional*) - key to unstack during `inv`
calls. Default is `None`.
- **out_keys_inv** (*sequence**of**NestedKey**,**optional*) - keys of the resulting
unstacked entries after `inv` calls. Default is `None`.
- **dim** (*int**,**optional*) - dimension to insert. Default is `-1`.
- **allow_positive_dim** (*bool**,**optional*) - if `True`, positive dimensions
are accepted. Defaults to `False`, ie. non-negative dimensions are
not permitted.

Keyword Arguments:

**del_keys** (*bool**,**optional*) - if `True`, the input values will be deleted
after stacking. Default is `True`.

Examples

```
>>> import torch
>>> from tensordict import TensorDict
>>> from torchrl.envs import Stack
>>> td = TensorDict({"key1": torch.zeros(3), "key2": torch.ones(3)}, [])
>>> td
TensorDict(
 fields={
 key1: Tensor(shape=torch.Size([3]), device=cpu, dtype=torch.float32, is_shared=False),
 key2: Tensor(shape=torch.Size([3]), device=cpu, dtype=torch.float32, is_shared=False)},
 batch_size=torch.Size([]),
 device=None,
 is_shared=False)
>>> transform = Stack(in_keys=["key1", "key2"], out_key="out", dim=-2)
>>> transform(td)
TensorDict(
 fields={
 out: Tensor(shape=torch.Size([2, 3]), device=cpu, dtype=torch.float32, is_shared=False)},
 batch_size=torch.Size([]),
 device=None,
 is_shared=False)
>>> td["out"]
tensor([[0., 0., 0.],
 [1., 1., 1.]])
```

```
>>> agent_0 = TensorDict({"obs": torch.rand(4, 5), "reward": torch.zeros(1)})
>>> agent_1 = TensorDict({"obs": torch.rand(4, 5), "reward": torch.zeros(1)})
>>> td = TensorDict({"agent_0": agent_0, "agent_1": agent_1})
>>> transform = Stack(in_keys=["agent_0", "agent_1"], out_key="agents")
>>> transform(td)
TensorDict(
 fields={
 agents: TensorDict(
 fields={
 obs: Tensor(shape=torch.Size([2, 4, 5]), device=cpu, dtype=torch.float32, is_shared=False),
 reward: Tensor(shape=torch.Size([2, 1]), device=cpu, dtype=torch.float32, is_shared=False)},
 batch_size=torch.Size([2]),
 device=None,
 is_shared=False)},
 batch_size=torch.Size([]),
 device=None,
 is_shared=False)
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

transform_done_spec(*done_spec: [TensorSpec](torchrl.data.TensorSpec.html#torchrl.data.TensorSpec)*) → [TensorSpec](torchrl.data.TensorSpec.html#torchrl.data.TensorSpec)[[source]](../../_modules/torchrl/envs/transforms/_tensor.html#Stack.transform_done_spec)

Transforms the done spec such that the resulting spec matches transform mapping.

Parameters:

**done_spec** ([*TensorSpec*](torchrl.data.TensorSpec.html#torchrl.data.TensorSpec)) - spec before the transform

Returns:

expected spec after the transform

transform_input_spec(*input_spec: [TensorSpec](torchrl.data.TensorSpec.html#torchrl.data.TensorSpec)*) → [TensorSpec](torchrl.data.TensorSpec.html#torchrl.data.TensorSpec)[[source]](../../_modules/torchrl/envs/transforms/_tensor.html#Stack.transform_input_spec)

Transforms the input spec such that the resulting spec matches transform mapping.

Parameters:

**input_spec** ([*TensorSpec*](torchrl.data.TensorSpec.html#torchrl.data.TensorSpec)) - spec before the transform

Returns:

expected spec after the transform

transform_observation_spec(*observation_spec: [TensorSpec](torchrl.data.TensorSpec.html#torchrl.data.TensorSpec)*) → [TensorSpec](torchrl.data.TensorSpec.html#torchrl.data.TensorSpec)[[source]](../../_modules/torchrl/envs/transforms/_tensor.html#Stack.transform_observation_spec)

Transforms the observation spec such that the resulting spec matches transform mapping.

Parameters:

**observation_spec** ([*TensorSpec*](torchrl.data.TensorSpec.html#torchrl.data.TensorSpec)) - spec before the transform

Returns:

expected spec after the transform

transform_reward_spec(*reward_spec: [TensorSpec](torchrl.data.TensorSpec.html#torchrl.data.TensorSpec)*) → [TensorSpec](torchrl.data.TensorSpec.html#torchrl.data.TensorSpec)[[source]](../../_modules/torchrl/envs/transforms/_tensor.html#Stack.transform_reward_spec)

Transforms the reward spec such that the resulting spec matches transform mapping.

Parameters:

**reward_spec** ([*TensorSpec*](torchrl.data.TensorSpec.html#torchrl.data.TensorSpec)) - spec before the transform

Returns:

expected spec after the transform