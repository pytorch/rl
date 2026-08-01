# BurnInTransform

*class*torchrl.envs.transforms.BurnInTransform(*modules: Sequence[[TensorDictModuleBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.nn.TensorDictModuleBase.html#tensordict.nn.TensorDictModuleBase)]*, *burn_in: int*, *out_keys: Sequence[NestedKey] | None = None*)[[source]](../../_modules/torchrl/envs/transforms/_env.html#BurnInTransform)

Transform to partially burn-in data sequences.

This transform is useful to obtain up-to-date recurrent states when
they are not available. It burns-in a number of steps along the time dimension
from sampled sequential data slices and returns the remaining data sequence with
the burnt-in data in its initial time step. This transform is intended to be used as a
replay buffer transform, not as an environment transform.

Parameters:

- **modules** (*sequence**of**TensorDictModule*) - A list of modules used to burn-in data sequences.
- **burn_in** (*int*) - The number of time steps to burn in.
- **out_keys** (*sequence**of**NestedKey**,**optional*) - destination keys. Defaults to
- **`** (*all the modules out_keys that point to the next time step**(**e.g. "hidden" if*) -
- **(****"next"** -
- **module****)****.** (*"hidden"**)**` is part**of**the out_keys**of**a*) -

Note

This transform expects as inputs TensorDicts with its last dimension being the
time dimension. It also assumes that all provided modules can process
sequential data.

Examples

```
>>> import torch
>>> from tensordict import TensorDict
>>> from torchrl.envs.transforms import BurnInTransform
>>> from torchrl.modules import GRUModule
>>> gru_module = GRUModule(
... input_size=10,
... hidden_size=10,
... in_keys=["observation", "hidden"],
... out_keys=["intermediate", ("next", "hidden")],
... default_recurrent_mode=True,
... )
>>> burn_in_transform = BurnInTransform(
... modules=[gru_module],
... burn_in=5,
... )
>>> td = TensorDict({
... "observation": torch.randn(2, 10, 10),
... "hidden": torch.randn(2, 10, gru_module.gru.num_layers, 10),
... "is_init": torch.zeros(2, 10, 1),
... }, batch_size=[2, 10])
>>> td = burn_in_transform(td)
>>> td.shape
torch.Size([2, 5])
>>> td.get("hidden").abs().sum()
tensor(86.3008)
```

```
>>> from torchrl.data import LazyMemmapStorage, TensorDictReplayBuffer
>>> buffer = TensorDictReplayBuffer(
... storage=LazyMemmapStorage(2),
... batch_size=1,
... )
>>> buffer.append_transform(burn_in_transform)
>>> td = TensorDict({
... "observation": torch.randn(2, 10, 10),
... "hidden": torch.randn(2, 10, gru_module.gru.num_layers, 10),
... "is_init": torch.zeros(2, 10, 1),
... }, batch_size=[2, 10])
>>> buffer.extend(td)
>>> td = buffer.sample(1)
>>> td.shape
torch.Size([1, 5])
>>> td.get("hidden").abs().sum()
tensor(37.0344)
```

forward(*tensordict: [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)*) → [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)[[source]](../../_modules/torchrl/envs/transforms/_env.html#BurnInTransform.forward)

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