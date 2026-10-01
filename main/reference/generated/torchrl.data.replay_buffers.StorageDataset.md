# StorageDataset

*class*torchrl.data.replay_buffers.StorageDataset(*storage: [Storage](torchrl.data.replay_buffers.Storage.html#torchrl.data.replay_buffers.Storage)*)[[source]](../../_modules/torchrl/data/replay_buffers/dataloader.html#StorageDataset)

A map-style [`torch.utils.data.Dataset`](https://docs.pytorch.org/docs/stable/data.html#torch.utils.data.Dataset) reading a TorchRL storage.

The dataset has one item per storage entry and a
[`torch.utils.data.DataLoader`](https://docs.pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader) reads it with any torch sampler and
worker processes. `__getitems__` fetches an index batch with a single
`get()` call, so the loader
receives one batch rather than a list of items: pass
`tensordict_collate()` as `collate_fn`, which returns such batches
unchanged and stacks lists of items. Storages with more than one dimension
are read through `flatten()`.
Reading the storage directly with a DataLoader, without this adapter,
fetches items one by one and passes the list to the collate function.

DataLoader workers read the storage content live: creating the dataset
moves a CPU tensor storage to shared memory and memory-mapped storages are
read through their files, so rows written after the workers start are
visible to them under every start method. Reads are not synchronized with
writes, and a row written while a worker reads it can come back partially
updated. A [`ListStorage`](torchrl.data.replay_buffers.ListStorage.html#torchrl.data.replay_buffers.ListStorage) cannot be
sent to spawned workers and forked workers read a snapshot of it. Only the
storage is sent to the workers, not the buffers attached to it.

Parameters:

**storage** ([*Storage*](torchrl.data.replay_buffers.Storage.html#torchrl.data.replay_buffers.Storage)) - the storage to read. Must be one-dimensional.

Examples

```
>>> import torch
>>> from tensordict import TensorDict
>>> from torch.utils.data import DataLoader
>>> from torchrl.data import LazyTensorStorage, ReplayBuffer, tensordict_collate
>>> rb = ReplayBuffer(storage=LazyTensorStorage(100))
>>> _ = rb.extend(TensorDict({"obs": torch.arange(100)}, [100]))
>>> dataset = rb.storage.as_dataset()
>>> len(dataset)
100
>>> loader = DataLoader(
... dataset, batch_size=4, shuffle=True, collate_fn=tensordict_collate
... )
>>> next(iter(loader))["obs"].shape
torch.Size([4])
```