# CatTensors

*class*torchrl.envs.transforms.CatTensors(*in_keys: Sequence[NestedKey] | None = None*, *out_key: NestedKey = 'observation_vector'*, *dim: int = -1*, ***, *del_keys: bool = True*, *unsqueeze_if_oor: bool = False*, *sort: bool = True*)[[source]](../../_modules/torchrl/envs/transforms/_tensor.html#CatTensors)

Concatenates several keys in a single tensor.

This is especially useful if multiple keys describe a single state (e.g.
"observation_position" and
"observation_velocity")

Parameters:

- **in_keys** (*sequence**of**NestedKey*) - keys to be concatenated. If None (or not provided)
the keys will be retrieved from the parent environment the first time
the transform is used. This behavior will only work if a parent is set.
- **out_key** (*NestedKey*) - key of the resulting tensor.
- **dim** (*int**,**optional*) - dimension along which the concatenation will occur.
Default is `-1`.

Keyword Arguments:

- **del_keys** (*bool**,**optional*) - if `True`, the input values will be deleted after
concatenation. Default is `True`.
- **unsqueeze_if_oor** (*bool**,**optional*) - if `True`, CatTensor will check that
the indicated dimension exists for the tensors to concatenate. If not,
the tensors will be unsqueezed along that dimension.
Default is `False`.
- **sort** (*bool**,**optional*) - if `True`, the keys will be sorted in the
transform. Otherwise, the order provided by the user will prevail.
Defaults to `True`.

Examples

```
>>> transform = CatTensors(in_keys=["key1", "key2"])
>>> td = TensorDict({"key1": torch.zeros(1, 1),
... "key2": torch.ones(1, 1)}, [1])
>>> _ = transform(td)
>>> print(td.get("observation_vector"))
tensor([[0., 1.]])
>>> transform = CatTensors(in_keys=["key1", "key2"], dim=-2, unsqueeze_if_oor=True)
>>> td = TensorDict({"key1": torch.zeros(1),
... "key2": torch.ones(1)}, [])
>>> _ = transform(td)
>>> print(td.get("observation_vector").shape)
torch.Size([2, 1])
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

transform_observation_spec(*observation_spec: [TensorSpec](torchrl.data.TensorSpec.html#torchrl.data.TensorSpec)*) → [TensorSpec](torchrl.data.TensorSpec.html#torchrl.data.TensorSpec)[[source]](../../_modules/torchrl/envs/transforms/_tensor.html#CatTensors.transform_observation_spec)

Transforms the observation spec such that the resulting spec matches transform mapping.

Parameters:

**observation_spec** ([*TensorSpec*](torchrl.data.TensorSpec.html#torchrl.data.TensorSpec)) - spec before the transform

Returns:

expected spec after the transform