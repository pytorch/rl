# SelectTransform

*class*torchrl.envs.transforms.SelectTransform(**selected_keys: NestedKey*, *keep_rewards: bool = True*, *keep_dones: bool = True*)[[source]](../../_modules/torchrl/envs/transforms/_keys.html#SelectTransform)

Select keys from the input tensordict.

In general, the [`ExcludeTransform`](torchrl.envs.transforms.ExcludeTransform.html#torchrl.envs.transforms.ExcludeTransform) should be preferred: this transforms also

selects the "action" (or other keys from input_spec), "done" and "reward"
keys but other may be necessary.

Parameters:

***selected_keys** (*iterable**of**NestedKey*) - The name of the keys to select. If the key is
not present, it is simply ignored.

Keyword Arguments:

- **keep_rewards** (*bool**,**optional*) - if `False`, the reward keys must be provided
if they should be kept. Defaults to `True`.
- **keep_dones** (*bool**,**optional*) - if `False`, the done keys must be provided
if they should be kept. Defaults to `True`.

Examples

```
>>> import gymnasium
>>> from torchrl.envs import GymWrapper
>>> env = TransformedEnv(
... GymWrapper(gymnasium.make("Pendulum-v1")),
... SelectTransform("observation", "reward", "done", keep_dones=False), # we leave done behind
... )
>>> env.rollout(3) # the truncated key is now absent
TensorDict(
 fields={
 action: Tensor(shape=torch.Size([3, 1]), device=cpu, dtype=torch.float32, is_shared=False),
 done: Tensor(shape=torch.Size([3, 1]), device=cpu, dtype=torch.bool, is_shared=False),
 next: TensorDict(
 fields={
 done: Tensor(shape=torch.Size([3, 1]), device=cpu, dtype=torch.bool, is_shared=False),
 observation: Tensor(shape=torch.Size([3, 3]), device=cpu, dtype=torch.float32, is_shared=False),
 reward: Tensor(shape=torch.Size([3, 1]), device=cpu, dtype=torch.float32, is_shared=False)},
 batch_size=torch.Size([3]),
 device=cpu,
 is_shared=False),
 observation: Tensor(shape=torch.Size([3, 3]), device=cpu, dtype=torch.float32, is_shared=False)},
 batch_size=torch.Size([3]),
 device=cpu,
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

transform_output_spec(*output_spec: [Composite](torchrl.data.Composite.html#torchrl.data.Composite)*) → [Composite](torchrl.data.Composite.html#torchrl.data.Composite)[[source]](../../_modules/torchrl/envs/transforms/_keys.html#SelectTransform.transform_output_spec)

Transforms the output spec such that the resulting spec matches transform mapping.

This method should generally be left untouched. Changes should be implemented using
`transform_observation_spec()`, `transform_reward_spec()` and `transform_full_done_spec()`.
:param output_spec: spec before the transform
:type output_spec: TensorSpec

Returns:

expected spec after the transform