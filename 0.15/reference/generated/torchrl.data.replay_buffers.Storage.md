# Storage

*class*torchrl.data.replay_buffers.Storage(*max_size: int*, *checkpointer: [StorageCheckpointerBase](torchrl.data.replay_buffers.StorageCheckpointerBase.html#torchrl.data.replay_buffers.StorageCheckpointerBase) | None = None*, *compilable: bool = False*)

A Storage is the container of a replay buffer.

Every storage must have a set, get and __len__ methods implemented.
Get and set should support integers as well as list of integers.

`as_dataset()` wraps a storage in a map-style
[`torch.utils.data.Dataset`](https://docs.pytorch.org/docs/stable/data.html#torch.utils.data.Dataset) that a [`torch.utils.data.DataLoader`](https://docs.pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader)
reads with any torch sampler, fetching every index batch through a single
`get()` call.

The storage does not need to have a definite size, but if it does one should
make sure that it is compatible with the buffer size.

as_dataset() → [StorageDataset](torchrl.data.replay_buffers.StorageDataset.html#torchrl.data.replay_buffers.StorageDataset)[[source]](../../_modules/torchrl/data/replay_buffers/storages/base.html#Storage.as_dataset)

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

attach(*buffer: Any*) → None[[source]](../../_modules/torchrl/data/replay_buffers/storages/base.html#Storage.attach)

This function attaches a sampler to this storage.

Buffers that read from this storage must be included as an attached
entity by calling this method. This guarantees that when data
in the storage changes, components are made aware of changes even if the storage
is shared with other buffers (eg. Priority Samplers).

Parameters:

**buffer** - the object that reads from this storage.

dump(**args*, ***kwargs*)[[source]](../../_modules/torchrl/data/replay_buffers/storages/base.html#Storage.dump)

Alias for `dumps()`.

load(**args*, ***kwargs*)[[source]](../../_modules/torchrl/data/replay_buffers/storages/base.html#Storage.load)

Alias for `loads()`.

register_load_hook(*hook*)[[source]](../../_modules/torchrl/data/replay_buffers/storages/base.html#Storage.register_load_hook)

Register a load hook for this storage.

The hook is forwarded to the checkpointer.

register_save_hook(*hook*)[[source]](../../_modules/torchrl/data/replay_buffers/storages/base.html#Storage.register_save_hook)

Register a save hook for this storage.

The hook is forwarded to the checkpointer.

save(**args*, ***kwargs*)[[source]](../../_modules/torchrl/data/replay_buffers/storages/base.html#Storage.save)

Alias for `dumps()`.