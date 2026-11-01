# prefill_replay_buffer

torchrl.data.prefill_replay_buffer(*rb: [ReplayBuffer](torchrl.data.ReplayBuffer.html#torchrl.data.ReplayBuffer)*, *dataset: str | [ReplayBuffer](torchrl.data.ReplayBuffer.html#torchrl.data.ReplayBuffer)*, *n_samples: int | None = None*, *chunk_size: int = 1000*) → [ReplayBuffer](torchrl.data.ReplayBuffer.html#torchrl.data.ReplayBuffer)[[source]](../../_modules/torchrl/data/replay_buffers/offline_to_online.html#prefill_replay_buffer)

Copy samples from an offline dataset into a mutable replay buffer.

A simpler alternative to [`OfflineToOnlineReplayBuffer`](torchrl.data.OfflineToOnlineReplayBuffer.html#torchrl.data.OfflineToOnlineReplayBuffer) for users
who want a single flat buffer (no per-batch sampling ratio, slightly higher
memory usage since offline data is copied).

Parameters:

- **rb** ([*ReplayBuffer*](torchrl.data.ReplayBuffer.html#torchrl.data.ReplayBuffer)) - a mutable replay buffer to seed.
- **dataset** (*str**or*[*ReplayBuffer*](torchrl.data.ReplayBuffer.html#torchrl.data.ReplayBuffer)) - offline dataset or a prefixed ID string
(`"minari:..."` / `"d4rl:..."`).
- **n_samples** (*int**,**optional*) - maximum number of samples to copy.
Defaults to the full dataset.
- **chunk_size** (*int**,**optional*) - number of samples copied per iteration.
When `dataset` is a string, this is also used as the dataset
constructor batch size. Default: `1000`.

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