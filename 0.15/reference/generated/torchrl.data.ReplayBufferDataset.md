# ReplayBufferDataset

*class*torchrl.data.ReplayBufferDataset(*replay_buffer: [ReplayBuffer](torchrl.data.ReplayBuffer.html#torchrl.data.ReplayBuffer)*, ***, *num_batches: int | None = None*)[[source]](../../_modules/torchrl/data/replay_buffers/dataloader.html#ReplayBufferDataset)

A [`torch.utils.data.IterableDataset`](https://docs.pytorch.org/docs/stable/data.html#torch.utils.data.IterableDataset) streaming batches from a replay buffer.

Iterating the dataset iterates the buffer: every item is one batch of
`replay_buffer.batch_size` elements with the buffer transforms applied.
Pass the dataset to a [`torch.utils.data.DataLoader`](https://docs.pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader) with
`batch_size=None` and [`tensordict_collate()`](torchrl.data.tensordict_collate.html#torchrl.data.tensordict_collate) to sample in worker
processes. Each worker holds its own copy of the buffer, so the sampler and
the transforms run in the worker and `num_batches` is split between
workers. The storage content is shared rather than copied and workers
observe later writes as described in `StorageDataset`.

Buffer prefetching is disabled in workers and prefetched batches are never
serialized to them, the DataLoader prefetches instead. A buffer built with
a [`torch.Generator`](https://docs.pytorch.org/docs/stable/generated/torch.Generator.html#torch.Generator) is reseeded once per worker from the worker
seed, so sampling in workers is reproducible when the DataLoader is seeded
(`torch.manual_seed` or `DataLoader(generator=...)`).

Samplers whose
`requires_shared_state` is
`True`, which is every sampler except those that declare their draws
stateless such as [`RandomSampler`](torchrl.data.replay_buffers.RandomSampler.html#torchrl.data.replay_buffers.RandomSampler) and
[`SliceSampler`](torchrl.data.replay_buffers.SliceSampler.html#torchrl.data.replay_buffers.SliceSampler), are rejected when
workers are used. So is a [`RateLimitedReplayBuffer`](torchrl.data.RateLimitedReplayBuffer.html#torchrl.data.RateLimitedReplayBuffer)
that has not been shared with `share()`,
since each worker would otherwise spend its own copy of the sample budget.

Parameters:

**replay_buffer** ([*ReplayBuffer*](torchrl.data.ReplayBuffer.html#torchrl.data.ReplayBuffer)) - the buffer to sample from. Its
`batch_size` must be set.

Keyword Arguments:

**num_batches** (*int**or**None**,**optional*) - number of batches yielded by one
iterator, shared between DataLoader workers. `None` streams
batches until the sampler runs out, which samplers with replacement
never do. Defaults to `None`.

Examples

```
>>> import torch
>>> from tensordict import TensorDict
>>> from torch.utils.data import DataLoader
>>> from torchrl.data import (
... LazyMemmapStorage,
... SliceSampler,
... TensorDictReplayBuffer,
... tensordict_collate,
... )
>>> rb = TensorDictReplayBuffer(
... storage=LazyMemmapStorage(1000),
... sampler=SliceSampler(num_slices=4, traj_key="episode", cache_values=True),
... batch_size=32,
... )
>>> _ = rb.extend(
... TensorDict(
... {"obs": torch.randn(1000, 3), "episode": torch.arange(1000) // 50},
... [1000],
... )
... )
>>> loader = DataLoader(
... rb.as_dataset(num_batches=8),
... batch_size=None,
... num_workers=4,
... persistent_workers=True,
... collate_fn=tensordict_collate,
... )
>>> batches = list(loader)
>>> len(batches), batches[0]["obs"].shape
(8, torch.Size([32, 3]))
```