# TerminateTransform

*class*torchrl.envs.transforms.TerminateTransform(*stop: Callable[[[TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)], Any]*, ***, *write_done: bool = True*)[[source]](../../_modules/torchrl/envs/transforms/_env.html#TerminateTransform)

Terminate a rollout when a user-supplied predicate becomes true.

After each environment step, `stop(next_tensordict)` is evaluated and its
boolean result is OR-ed into the environment's `terminated` (and, by
default, `done`) entries. Combined with
`rollout(..., break_when_any_done=True)` (the default), this ends the
rollout as soon as the goal condition is reached - without writing a
bespoke stepping loop. It is the natural companion of the
[`rollout()`](torchrl.envs.EnvBase.html#id2) `actions` keyword for scripted,
goal-terminated replays.

Parameters:

**stop** (*callable*) - a callable taking the post-step (`"next"`)
TensorDict and returning a boolean scalar or a boolean tensor
broadcastable to the environment's done entries.

Keyword Arguments:

**write_done** (*bool**,**optional*) - if `True` (default), also OR the flag
into the `done` entries so `break_when_any_done` halts the
rollout. Set to `False` to write only `terminated` entries.

Examples

```
>>> import torch
>>> from torchrl.envs import GymEnv, TransformedEnv
>>> from torchrl.envs.transforms import TerminateTransform
>>> env = TransformedEnv( 
... GymEnv("Pendulum-v1"),
... TerminateTransform(lambda td: td["observation"][..., 0] > 0.99),
... )
>>> rollout = env.rollout(200, break_when_any_done=True)
```

forward(*tensordict: [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)*) → [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)[[source]](../../_modules/torchrl/envs/transforms/_env.html#TerminateTransform.forward)

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