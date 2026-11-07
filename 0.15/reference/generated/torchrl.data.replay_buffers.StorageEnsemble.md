# StorageEnsemble

*class*torchrl.data.replay_buffers.StorageEnsemble(**storages: [Storage](torchrl.data.replay_buffers.Storage.html#torchrl.data.replay_buffers.Storage)*, *transforms: list[[Transform](torchrl.envs.transforms.Transform.html#torchrl.envs.transforms.Transform)] = None*)

An ensemble of storages.

This class is designed to work with [`ReplayBufferEnsemble`](torchrl.data.ReplayBufferEnsemble.html#torchrl.data.ReplayBufferEnsemble).

Parameters:

**storages** (*sequence**of*[*Storage*](torchrl.data.replay_buffers.Storage.html#torchrl.data.replay_buffers.Storage)) - the storages to make the composite storage.

Keyword Arguments:

**transforms** (list of `Transform`, optional) - a list of
transforms of the same length as storages.

Warning

This class signatures for `get()` does not match other storages, as
it will return a tuple `(buffer_id, samples)` rather than just the samples.

Warning

This class does not support writing (similarly to [`WriterEnsemble`](torchrl.data.replay_buffers.WriterEnsemble.html#torchrl.data.replay_buffers.WriterEnsemble)).
To extend one of the replay buffers, simply index the parent
[`ReplayBufferEnsemble`](torchrl.data.ReplayBufferEnsemble.html#torchrl.data.ReplayBufferEnsemble) object.

See also

[`StorageEnsembleConfig`](torchrl.trainers.algorithms.configs.data.StorageEnsembleConfig.html#torchrl.trainers.algorithms.configs.data.StorageEnsembleConfig)

as_dataset()[[source]](../../_modules/torchrl/data/replay_buffers/storages/ensemble.html#StorageEnsemble.as_dataset)

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