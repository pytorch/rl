# OfflineToOnlineReplayBuffer

*class*torchrl.data.OfflineToOnlineReplayBuffer(*offline_dataset*, ***, *online_storage: [Storage](torchrl.data.replay_buffers.Storage.html#torchrl.data.replay_buffers.Storage) | None = None*, *online_capacity: int | None = None*, *offline_fraction: float = 0.5*, *batch_size: int | None = None*, *transform=None*, ***dataset_kwargs*)[[source]](../../_modules/torchrl/data/replay_buffers/offline_to_online.html#OfflineToOnlineReplayBuffer)

A replay buffer combining an immutable offline dataset with a growing online buffer.

`extend()` routes new experience to the online buffer only; the offline
dataset is never modified. `sample()` draws **exactly**
`round(offline_fraction * batch_size)` transitions from the offline
dataset and the remainder from the online buffer, concatenated into a flat
`[batch_size]` TensorDict.

The split is deterministic per batch (not merely correct in expectation),
so `offline_fraction` is honored on every single `sample()` call.

When the online buffer is empty (i.e. before any `extend()` call), or
once `offline_fraction` has been annealed to 0, `sample()` draws from
a single buffer only.

Note

Offline and online data must share a compatible key structure so
the two sampled batches can be concatenated. This is automatic when
both come from the same environment (TED format).

Parameters:

**offline_dataset** (*str**or*[*ReplayBuffer*](torchrl.data.ReplayBuffer.html#torchrl.data.ReplayBuffer)) - an offline dataset object (e.g.
[`MinariExperienceReplay`](torchrl.data.datasets.MinariExperienceReplay.html#torchrl.data.datasets.MinariExperienceReplay)) or a
prefixed ID string such as `"minari:mujoco/hopper/expert-v0"` or
`"d4rl:halfcheetah-medium-v2"` resolved via
[`load_dataset()`](torchrl.data.datasets.load_dataset.html#torchrl.data.datasets.load_dataset).

Keyword Arguments:

- **online_storage** ([*Storage*](torchrl.data.replay_buffers.Storage.html#torchrl.data.replay_buffers.Storage)*,**optional*) - storage backend for the online
buffer. Mutually exclusive with `online_capacity`.
- **online_capacity** (*int**,**optional*) - shorthand that creates a
`LazyTensorStorage` of this size.
Mutually exclusive with `online_storage`.
- **offline_fraction** ([*float*](torchrl.data.llm.TopKRewardSelector.html#torchrl.data.llm.TopKRewardSelector.float)*,**optional*) - fraction of each batch drawn from
the offline dataset. Must be in `(0, 1)`. Default: `0.5`.
- **batch_size** (*int**,**optional*) - default batch size for `sample()`. Required
when `offline_dataset` is a string, and forwarded to the dataset
constructor.
- **transform** (*Callable**,**optional*) - applied to the concatenated sample
batch on the read side.
- ****dataset_kwargs** - forwarded to the dataset constructor when
`offline_dataset` is a string.

Examples

```
>>> import torch
>>> from tensordict import TensorDict
>>> from torchrl.data import (
... OfflineToOnlineReplayBuffer, ReplayBuffer, LazyTensorStorage)
>>> offline = ReplayBuffer(storage=LazyTensorStorage(1000))
>>> _ = offline.extend(TensorDict({"observation": torch.randn(1000, 4)}, [1000]))
>>> rb = OfflineToOnlineReplayBuffer(
... offline_dataset=offline,
... online_capacity=500,
... offline_fraction=0.5,
... batch_size=32,
... )
>>> _ = rb.extend(TensorDict({"observation": torch.randn(10, 4)}, [10]))
>>> rb.sample(32).batch_size
torch.Size([32])
```

anneal(*step: int*, *total_steps: int*) → None[[source]](../../_modules/torchrl/data/replay_buffers/offline_to_online.html#OfflineToOnlineReplayBuffer.anneal)

Linearly decay `offline_fraction` toward 0 over `total_steps`.

Call once per training iteration to gradually shift the sampling
distribution from offline-dominant to purely online. Clamps at 0 for
`step >= total_steps`.

Parameters:

- **step** (*int*) - current training step (0-indexed).
- **total_steps** (*int*) - step at which `offline_fraction` reaches 0.

extend(*data*) → [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)[[source]](../../_modules/torchrl/data/replay_buffers/offline_to_online.html#OfflineToOnlineReplayBuffer.extend)

Add new online experience to the online buffer.

Parameters:

**data** - a TensorDict (or compatible sequence) to add.

Returns:

Indices at which the data was stored in the online buffer.

*property*offline_buffer

The immutable offline dataset.

*property*offline_fraction*: float*

The current offline sampling fraction (after any annealing).

*property*online_buffer*: [ReplayBuffer](torchrl.data.ReplayBuffer.html#torchrl.data.ReplayBuffer)*

The mutable online replay buffer.

sample(*batch_size: int | None = None*) → [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)[[source]](../../_modules/torchrl/data/replay_buffers/offline_to_online.html#OfflineToOnlineReplayBuffer.sample)

Sample a flat `[batch_size]` batch split between the two buffers.

Draws `round(offline_fraction * batch_size)` from the offline dataset
and the rest from the online buffer. Falls back to a single buffer
when the online buffer is empty or the offline split rounds to 0.

Parameters:

**batch_size** (*int**,**optional*) - number of samples to draw. Falls back
to the `batch_size` set in `__init__`.

Returns:

TensorDictBase with batch size `[batch_size]`.