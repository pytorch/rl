# RenameTransform

*class*torchrl.envs.transforms.RenameTransform(*in_keys*, *out_keys*, *in_keys_inv=None*, *out_keys_inv=None*, *create_copy=False*)[[source]](../../_modules/torchrl/envs/transforms/_keys.html#RenameTransform)

A transform to rename entries in the output tensordict (or input tensordict via the inverse keys).

Parameters:

- **in_keys** (*sequence**of**NestedKey*) - the entries to rename.
- **out_keys** (*sequence**of**NestedKey*) - the name of the entries after renaming.
- **in_keys_inv** (*sequence**of**NestedKey**,**optional*) - the entries to rename
in the input tensordict, which will be passed to `EnvBase._step()`.
- **out_keys_inv** (*sequence**of**NestedKey**,**optional*) - the names of the entries
in the input tensordict after renaming.
- **create_copy** (*bool**,**optional*) - if `True`, the entries will be copied
with a different name rather than being renamed. This allows for
renaming immutable entries such as `"reward"` and `"done"`.

Examples

```
>>> from torchrl.envs.libs.gym import GymEnv
>>> env = TransformedEnv(
... GymEnv("Pendulum-v1"),
... RenameTransform(["observation", ], ["stuff",], create_copy=False),
... )
>>> tensordict = env.rollout(3)
>>> print(tensordict)
TensorDict(
 fields={
 action: Tensor(shape=torch.Size([3, 1]), device=cpu, dtype=torch.float32, is_shared=False),
 done: Tensor(shape=torch.Size([3, 1]), device=cpu, dtype=torch.bool, is_shared=False),
 next: TensorDict(
 fields={
 done: Tensor(shape=torch.Size([3, 1]), device=cpu, dtype=torch.bool, is_shared=False),
 reward: Tensor(shape=torch.Size([3, 1]), device=cpu, dtype=torch.float32, is_shared=False),
 stuff: Tensor(shape=torch.Size([3, 3]), device=cpu, dtype=torch.float32, is_shared=False)},
 batch_size=torch.Size([3]),
 device=cpu,
 is_shared=False),
 stuff: Tensor(shape=torch.Size([3, 3]), device=cpu, dtype=torch.float32, is_shared=False)},
 batch_size=torch.Size([3]),
 device=cpu,
 is_shared=False)
>>> # if the output is also an input, we need to rename if both ways:
>>> from torchrl.envs.libs.brax import BraxEnv
>>> env = TransformedEnv(
... BraxEnv("fast"),
... RenameTransform(["state"], ["newname"], ["state"], ["newname"])
... )
>>> _ = env.set_seed(1)
>>> tensordict = env.rollout(3)
>>> assert "newname" in tensordict.keys()
>>> assert "state" not in tensordict.keys()
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

transform_input_spec(*input_spec: [Composite](torchrl.data.Composite.html#torchrl.data.Composite)*) → [Composite](torchrl.data.Composite.html#torchrl.data.Composite)[[source]](../../_modules/torchrl/envs/transforms/_keys.html#RenameTransform.transform_input_spec)

Transforms the input spec such that the resulting spec matches transform mapping.

Parameters:

**input_spec** ([*TensorSpec*](torchrl.data.TensorSpec.html#torchrl.data.TensorSpec)) - spec before the transform

Returns:

expected spec after the transform

transform_output_spec(*output_spec: [Composite](torchrl.data.Composite.html#torchrl.data.Composite)*) → [Composite](torchrl.data.Composite.html#torchrl.data.Composite)[[source]](../../_modules/torchrl/envs/transforms/_keys.html#RenameTransform.transform_output_spec)

Transforms the output spec such that the resulting spec matches transform mapping.

This method should generally be left untouched. Changes should be implemented using
`transform_observation_spec()`, `transform_reward_spec()` and `transform_full_done_spec()`.
:param output_spec: spec before the transform
:type output_spec: TensorSpec

Returns:

expected spec after the transform