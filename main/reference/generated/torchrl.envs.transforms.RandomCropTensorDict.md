# RandomCropTensorDict

*class*torchrl.envs.transforms.RandomCropTensorDict(*sub_seq_len: int*, *sample_dim: int = -1*, *mask_key: NestedKey | None = None*)[[source]](../../_modules/torchrl/envs/transforms/_misc.html#RandomCropTensorDict)

A trajectory sub-sampler for ReplayBuffer and modules.

Gathers a sub-sequence of a defined length along the last dimension of the input
tensordict.
This can be used to get cropped trajectories from trajectories sampled
from a ReplayBuffer.

This transform is primarily designed to be used with replay buffers and modules.
Currently, it cannot be used as an environment transform.
Do not hesitate to request for this behavior through an issue if this is
desired.

Parameters:

- **sub_seq_len** (*int*) - the length of the sub-trajectory to sample
- **sample_dim** (*int**,**optional*) - the dimension along which the cropping
should occur. Negative dimensions should be preferred to make
the transform robust to tensordicts of varying batch dimensions.
Defaults to -1 (the default time dimension in TorchRL).
- **mask_key** (*NestedKey*) - If provided, this represents the mask key to be looked
for when doing the sampling. If provided, it only valid elements will
be returned. It is assumed that the mask is a boolean tensor with
first True values and then False values, not mixed together.
`RandomCropTensorDict` will NOT check that this is respected
hence any error caused by an improper mask risks to go unnoticed.
Defaults: None (no mask key).

forward(*tensordict: [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)*) → [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)[[source]](../../_modules/torchrl/envs/transforms/_misc.html#RandomCropTensorDict.forward)

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