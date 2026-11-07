# tensordict_collate

torchrl.data.tensordict_collate(*batch: Any*) → Any[[source]](../../_modules/torchrl/data/replay_buffers/dataloader.html#tensordict_collate)

Collate function for a [`torch.utils.data.DataLoader`](https://docs.pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader) reading TorchRL storages or buffers.

A batch that is already a tensor, a tensor collection or a tuple of them,
as fetched by `StorageDataset` or yielded by
[`ReplayBufferDataset`](torchrl.data.ReplayBufferDataset.html#torchrl.data.ReplayBufferDataset), is returned unchanged. A list of samples, as
produced by per-item storages or by composing datasets with
[`torch.utils.data.ConcatDataset`](https://docs.pytorch.org/docs/stable/data.html#torch.utils.data.ConcatDataset), is stacked: tensor collections
lazily when their shapes differ, tensors densely, and mappings or tuples
element-wise. The default torch collation iterates a tensordict over its
batch dimension and cannot be used.

Parameters:

**batch** (*Tensor**,**TensorDictBase**or**list*) - the fetched batch.

Examples

```
>>> import torch
>>> from tensordict import TensorDict
>>> from torch.utils.data import DataLoader
>>> from torchrl.data import LazyTensorStorage, ReplayBuffer, tensordict_collate
>>> rb = ReplayBuffer(storage=LazyTensorStorage(100))
>>> _ = rb.extend(TensorDict({"obs": torch.arange(100)}, [100]))
>>> loader = DataLoader(
... rb.storage.as_dataset(),
... batch_size=4,
... shuffle=True,
... collate_fn=tensordict_collate,
... )
>>> next(iter(loader))["obs"].shape
torch.Size([4])
```