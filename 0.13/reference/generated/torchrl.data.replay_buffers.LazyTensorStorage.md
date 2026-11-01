# LazyTensorStorage

*class*torchrl.data.replay_buffers.LazyTensorStorage(*max_size: int*, ***, *device: [device](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device) | str = 'cpu'*, *ndim: int = 1*, *compilable: bool = False*, *consolidated: bool = False*, *shared_init: bool = False*, *cleanup_memmap: bool = True*)[[source]](../../_modules/torchrl/data/replay_buffers/storages.html#LazyTensorStorage)

A pre-allocated tensor storage for tensors and tensordicts.

Parameters:

**max_size** (*int*) - size of the storage, i.e. maximum number of elements stored
in the buffer.

Keyword Arguments:

- **device** ([*torch.device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)*,**optional*) - device where the sampled tensors will be
stored and sent. Default is `torch.device("cpu")`.
If "auto" is passed, the device is automatically gathered from the
first batch of data passed. This is not enabled by default to avoid
data placed on GPU by mistake, causing OOM issues.
- **ndim** (*int**,**optional*) -

the number of dimensions to be accounted for when
measuring the storage size. For instance, a storage of shape `[3, 4]`
has capacity `3` if `ndim=1` and `12` if `ndim=2`.
Defaults to `1`.

Important

When using a collector with `trajs_per_batch`,
keep the default `ndim=1`. `trajs_per_batch` writes
variable-length trajectories as flat 1-D sequences, which is
incompatible with a storage that expects a fixed second
dimension (`ndim >= 2`).
- **compilable** (*bool**,**optional*) - whether the storage is compilable.
If `True`, the writer cannot be shared between multiple processes.
Defaults to `False`.
- **consolidated** (*bool**,**optional*) - if `True`, the storage will be consolidated after
its first expansion. Defaults to `False`.
- **shared_init** (*bool**,**optional*) - if `True`, enables multiprocess coordination
during storage initialization. First process initializes with memmap,
others wait and load from the shared memmap. Defaults to `False`.
- **cleanup_memmap** (*bool**,**optional*) - if `True` and `shared_init=True`,
the temporary memmap will be deleted after initialization and the
storage will operate in RAM. Defaults to `True`.

Examples

```
>>> data = TensorDict({
... "some data": torch.randn(10, 11),
... ("some", "nested", "data"): torch.randn(10, 11, 12),
... }, batch_size=[10, 11])
>>> storage = LazyTensorStorage(100)
>>> storage.set(range(10), data)
>>> len(storage) # only the first dimension is considered as indexable
10
>>> storage.get(0)
TensorDict(
 fields={
 some data: Tensor(shape=torch.Size([11]), device=cpu, dtype=torch.float32, is_shared=False),
 some: TensorDict(
 fields={
 nested: TensorDict(
 fields={
 data: Tensor(shape=torch.Size([11, 12]), device=cpu, dtype=torch.float32, is_shared=False)},
 batch_size=torch.Size([11]),
 device=cpu,
 is_shared=False)},
 batch_size=torch.Size([11]),
 device=cpu,
 is_shared=False)},
 batch_size=torch.Size([11]),
 device=cpu,
 is_shared=False)
>>> storage.set(0, storage.get(0).zero_()) # zeros the data along index ``0``
```

This class also supports tensorclass data.

Examples

```
>>> from tensordict import tensorclass
>>> @tensorclass
... class MyClass:
... foo: torch.Tensor
... bar: torch.Tensor
>>> data = MyClass(foo=torch.randn(10, 11), bar=torch.randn(10, 11, 12), batch_size=[10, 11])
>>> storage = LazyTensorStorage(10)
>>> storage.set(range(10), data)
>>> storage.get(0)
MyClass(
 bar=Tensor(shape=torch.Size([11, 12]), device=cpu, dtype=torch.float32, is_shared=False),
 foo=Tensor(shape=torch.Size([11]), device=cpu, dtype=torch.float32, is_shared=False),
 batch_size=torch.Size([11]),
 device=cpu,
 is_shared=False)
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