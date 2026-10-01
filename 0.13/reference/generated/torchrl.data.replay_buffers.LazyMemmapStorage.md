# LazyMemmapStorage

*class*torchrl.data.replay_buffers.LazyMemmapStorage(*max_size: int*, ***, *scratch_dir=None*, *device: [device](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device) | str = 'cpu'*, *ndim: int = 1*, *existsok: bool = False*, *compilable: bool = False*, *shared_init: bool = False*, *auto_cleanup: bool | None = None*)[[source]](../../_modules/torchrl/data/replay_buffers/storages.html#LazyMemmapStorage)

A memory-mapped storage for tensors and tensordicts.

Parameters:

**max_size** (*int*) - size of the storage, i.e. maximum number of elements stored
in the buffer.

Keyword Arguments:

- **scratch_dir** (*str**or**path*) - directory where memmap-tensors will be written.
If `shared_init=True` and no `scratch_dir` is provided, a shared
temporary directory will be created automatically.
- **device** ([*torch.device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)*,**optional*) - device where the sampled tensors will be
stored and sent. Default is `torch.device("cpu")`.
If `None` is provided, the device is automatically gathered from the
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
- **existsok** (*bool**,**optional*) - whether an error should be raised if any of the
tensors already exists on disk. Defaults to `True`. If `False`, the
tensor will be opened as is, not overewritten.
- **shared_init** (*bool**,**optional*) - if `True`, enables multiprocess coordination
during storage initialization. First process initializes the memmap,
others wait and load from the shared directory. Defaults to `False`.
- **auto_cleanup** (*bool**,**optional*) - if `True`, automatically registers this
storage for cleanup when the process exits (normally or via Ctrl+C/SIGTERM).
This removes the memmap files from disk when no longer needed.
Defaults to `True` when `scratch_dir` is `None` (using temp directory),
and `False` when a custom `scratch_dir` is provided (preserving user data).

Note

When checkpointing a `LazyMemmapStorage`, one can provide a path identical to where the storage is
already stored to avoid executing long copies of data that is already stored on disk.
This will only work if the default `TensorStorageCheckpointer` checkpointer is used.

Example:

```
>>> from tensordict import TensorDict
>>> from torchrl.data import TensorStorage, LazyMemmapStorage, ReplayBuffer
>>> import tempfile
>>> from pathlib import Path
>>> import time
>>> td = TensorDict(a=0, b=1).expand(1000).clone()
>>> # We pass a path that is <main_ckpt_dir>/storage to LazyMemmapStorage
>>> rb_memmap = ReplayBuffer(storage=LazyMemmapStorage(10_000_000, scratch_dir="dump/storage"))
>>> rb_memmap.extend(td);
>>> # Checkpointing in `dump` is a zero-copy, as the data is already in `dump/storage`
>>> rb_memmap.dumps(Path("./dump"))
```

Examples

```
>>> data = TensorDict({
... "some data": torch.randn(10, 11),
... ("some", "nested", "data"): torch.randn(10, 11, 12),
... }, batch_size=[10, 11])
>>> storage = LazyMemmapStorage(100)
>>> storage.set(range(10), data)
>>> len(storage) # only the first dimension is considered as indexable
10
>>> storage.get(0)
TensorDict(
 fields={
 some data: MemoryMappedTensor(shape=torch.Size([11]), device=cpu, dtype=torch.float32, is_shared=False),
 some: TensorDict(
 fields={
 nested: TensorDict(
 fields={
 data: MemoryMappedTensor(shape=torch.Size([11, 12]), device=cpu, dtype=torch.float32, is_shared=False)},
 batch_size=torch.Size([11]),
 device=cpu,
 is_shared=False)},
 batch_size=torch.Size([11]),
 device=cpu,
 is_shared=False)},
 batch_size=torch.Size([11]),
 device=cpu,
 is_shared=False)
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
>>> storage = LazyMemmapStorage(10)
>>> storage.set(range(10), data)
>>> storage.get(0)
MyClass(
 bar=MemoryMappedTensor(shape=torch.Size([11, 12]), device=cpu, dtype=torch.float32, is_shared=False),
 foo=MemoryMappedTensor(shape=torch.Size([11]), device=cpu, dtype=torch.float32, is_shared=False),
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

cleanup() → bool[[source]](../../_modules/torchrl/data/replay_buffers/storages.html#LazyMemmapStorage.cleanup)

Clean up memmap files from disk.

This method removes the memmap directory and all its contents from disk.
It is automatically called on process exit if `auto_cleanup=True`.

Returns:

`True` if cleanup was performed, `False` if already cleaned up

or no cleanup needed.

Return type:

bool

Note

After cleanup, the storage is no longer usable. Any attempt to access
the storage will result in undefined behavior.

Example

```
>>> storage = LazyMemmapStorage(1000, auto_cleanup=True)
>>> # ... use storage ...
>>> storage.cleanup() # Manually clean up when done
```

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