# ListStorage

*class*torchrl.data.replay_buffers.ListStorage(*max_size: int | None = None*, ***, *compilable: bool = False*, *device: [device](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device) | str | int | None = None*)

A storage stored in a list.

This class cannot be extended with PyTrees, the data provided during calls to
`extend()` should be iterables
(like lists, tuples, tensors or tensordicts with non-empty batch-size).

Parameters:

**max_size** (*int**,**optional*) - the maximum number of elements stored in the storage.
If not provided, an unlimited storage is created.

Keyword Arguments:

- **compilable** (*bool**,**optional*) - if `True`, the storage will be made compatible with [`compile()`](https://docs.pytorch.org/docs/stable/generated/torch.compile.html#torch.compile) at
the cost of being executable in multiprocessed settings.
- **device** (*str**,**optional*) - the device to use for the storage. Defaults to None (inputs are not moved to the device).

as_dataset() → [StorageDataset](torchrl.data.replay_buffers.StorageDataset.html#torchrl.data.replay_buffers.StorageDataset)

Returns a map-style [`torch.utils.data.Dataset`](https://docs.pytorch.org/docs/stable/data.html#torch.utils.data.Dataset) reading this storage.

See `StorageDataset` for the batched fetch and
collation contract. Multi-dimensional storages are read through
`flatten()`.

Examples

```
>>> import torch
>>> from torch.utils.data import DataLoader
>>> from torchrl.data import LazyTensorStorage, ReplayBuffer, tensordict_collate
>>> rb = ReplayBuffer(storage=LazyTensorStorage(100))
>>> _ = rb.extend(torch.arange(100))
>>> loader = DataLoader(
... rb.storage.as_dataset(), batch_size=4, shuffle=True, collate_fn=tensordict_collate
... )
>>> next(iter(loader)).shape
torch.Size([4])
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