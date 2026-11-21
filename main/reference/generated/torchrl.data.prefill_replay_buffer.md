# prefill_replay_buffer

torchrl.data.prefill_replay_buffer(*rb: [ReplayBuffer](torchrl.data.ReplayBuffer.html#torchrl.data.ReplayBuffer)*, *dataset: str | [ReplayBuffer](torchrl.data.ReplayBuffer.html#torchrl.data.ReplayBuffer)*, *n_samples: int | None = None*, *chunk_size: int = 1000*) → [ReplayBuffer](torchrl.data.ReplayBuffer.html#torchrl.data.ReplayBuffer)[[source]](../../_modules/torchrl/data/replay_buffers/offline_to_online.html#prefill_replay_buffer)

Copy the transitions of an offline dataset into a mutable replay buffer.

A simpler alternative to [`OfflineToOnlineReplayBuffer`](torchrl.data.OfflineToOnlineReplayBuffer.html#torchrl.data.OfflineToOnlineReplayBuffer) for users
who want a single flat buffer (no per-batch sampling ratio, slightly higher
memory usage since offline data is copied).

For a buffer with a local one-dimensional storage, transitions are read
with `dataset[index]`, so each stored transition is copied exactly once
and the dataset's transforms are applied. They are read in storage order,
starting from the oldest transition once a round-robin storage has
wrapped, so trajectories stay contiguous for slice or sequence samplers on
`rb`. For a multi-dimensional storage, a
[`ReplayBufferEnsemble`](torchrl.data.ReplayBufferEnsemble.html#torchrl.data.ReplayBufferEnsemble), a buffer whose storage is not
local (such as a Ray-backed buffer) or a buffer whose sampler may filter
entries (any sampler other than the random, without-replacement,
prioritized and slice samplers, such as a consuming or staleness-aware
one), chunks are drawn with `dataset.sample()`, which samples with the
dataset's sampler.

Parameters:

- **rb** ([*ReplayBuffer*](torchrl.data.ReplayBuffer.html#torchrl.data.ReplayBuffer)) - a mutable replay buffer to seed.
- **dataset** (*str**or*[*ReplayBuffer*](torchrl.data.ReplayBuffer.html#torchrl.data.ReplayBuffer)) - offline dataset or a prefixed ID string
(`"minari:..."` / `"d4rl:..."`).
- **n_samples** (*int**,**optional*) - number of transitions to copy. When the
dataset is copied in storage order they are the oldest
`n_samples` stored transitions, capped at the storage length;
otherwise they are drawn with `dataset.sample()`. Defaults to
the full dataset.
- **chunk_size** (*int**,**optional*) - number of transitions copied per
iteration. When `dataset` is a string, this is also used as the
dataset constructor batch size. Default: `1000`.

Returns:

`rb` mutated in-place (also returned for chaining).

Return type:

[ReplayBuffer](torchrl.data.ReplayBuffer.html#torchrl.data.ReplayBuffer)

Examples

```
>>> import torch
>>> from tensordict import TensorDict
>>> from torchrl.data import ReplayBuffer, LazyTensorStorage
>>> from torchrl.data.replay_buffers.offline_to_online import (
... prefill_replay_buffer)
>>> dataset = ReplayBuffer(storage=LazyTensorStorage(500))
>>> _ = dataset.extend(TensorDict({"obs": torch.randn(500, 4)}, [500]))
>>> online_rb = ReplayBuffer(storage=LazyTensorStorage(10_000))
>>> _ = prefill_replay_buffer(online_rb, dataset, n_samples=200)
>>> len(online_rb)
200
```