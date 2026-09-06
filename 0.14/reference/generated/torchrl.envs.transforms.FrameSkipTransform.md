# FrameSkipTransform

*class*torchrl.envs.transforms.FrameSkipTransform(*frame_skip: int = 1*)[[source]](../../_modules/torchrl/envs/transforms/_env.html#FrameSkipTransform)

A frame-skip transform.

This transform applies the same action repeatedly in the parent environment,
which improves stability on certain training sota-implementations.

Parameters:

**frame_skip** (*int**,**optional*) - a positive integer representing the number
of frames during which the same action must be applied.

forward(*tensordict*)[[source]](../../_modules/torchrl/envs/transforms/_env.html#FrameSkipTransform.forward)

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