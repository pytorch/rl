# LazyStackStorage

*class*torchrl.data.replay_buffers.LazyStackStorage(*max_size: int | None = None*, ***, *compilable: bool = False*, *stack_dim: int = 0*, *device: [device](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device) | str | int | None = None*)[[source]](../../_modules/torchrl/data/replay_buffers/storages.html#LazyStackStorage)

A ListStorage that returns LazyStackTensorDict instances.

This storage allows for heterougeneous structures to be indexed as a single TensorDict representation.
It uses [`LazyStackedTensorDict`](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.LazyStackedTensorDict.html#tensordict.LazyStackedTensorDict) which operates on non-contiguous lists of tensordicts,
lazily stacking items when queried.
This means that this storage is going to be fast to sample but data access may be slow (as it requires a stack).
Tensors of heterogeneous shapes can also be stored within the storage and stacked together.
Because the storage is represented as a list, the number of tensors to store in memory will grow linearly with
the size of the buffer.

If possible, nested tensors can also be created via [`densify()`](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.LazyStackedTensorDict.html#tensordict.LazyStackedTensorDict.densify)
(see [`nested`](https://docs.pytorch.org/docs/stable/nested.html#module-torch.nested)).

Parameters:

**max_size** (*int**,**optional*) - the maximum number of elements stored in the storage.
If not provided, an unlimited storage is created.

Keyword Arguments:

- **compilable** (*bool**,**optional*) - if `True`, the storage will be made compatible with [`compile()`](https://docs.pytorch.org/docs/stable/generated/torch.compile.html#torch.compile) at
the cost of being executable in multiprocessed settings.
- **stack_dim** (*int**,**optional*) - the stack dimension in terms of TensorDict batch sizes. Defaults to 0.
- **device** (*str**,**optional*) - the device to use for the storage. Defaults to None (inputs are not moved to the device).

Examples

```
>>> import torch
>>> from torchrl.data import ReplayBuffer, LazyStackStorage
>>> from tensordict import TensorDict
>>> _ = torch.manual_seed(0)
>>> rb = ReplayBuffer(storage=LazyStackStorage(max_size=1000, stack_dim=-1))
>>> data0 = TensorDict(a=torch.randn((10,)), b=torch.rand(4), c="a string!")
>>> data1 = TensorDict(a=torch.randn((11,)), b=torch.rand(4), c="another string!")
>>> _ = rb.add(data0)
>>> _ = rb.add(data1)
>>> rb.sample(10)
LazyStackedTensorDict(
 fields={
 a: Tensor(shape=torch.Size([10, -1]), device=cpu, dtype=torch.float32, is_shared=False),
 b: Tensor(shape=torch.Size([10, 4]), device=cpu, dtype=torch.float32, is_shared=False),
 c: NonTensorStack(
 ['another string!', 'another string!', 'another st...,
 batch_size=torch.Size([10]),
 device=None)},
 exclusive_fields={
 },
 batch_size=torch.Size([10]),
 device=None,
 is_shared=False,
 stack_dim=0)
```

attach(*buffer: Any*) → None

This function attaches a sampler to this storage.

Buffers that read from this storage must be included as an attached
entity by calling this method. This guarantees that when data
in the storage changes, components are made aware of changes even if the storage
is shared with other buffers (eg. Priority Samplers).

Parameters:

**buffer** - the object that reads from this storage.

dump(**args*, ***kwargs*)

Alias for `dumps()`.

load(**args*, ***kwargs*)

Alias for `loads()`.

register_load_hook(*hook*)

Register a load hook for this storage.

The hook is forwarded to the checkpointer.

register_save_hook(*hook*)

Register a save hook for this storage.

The hook is forwarded to the checkpointer.

save(**args*, ***kwargs*)

Alias for `dumps()`.