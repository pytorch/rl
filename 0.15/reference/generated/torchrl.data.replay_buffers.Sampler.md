# Sampler

*class*torchrl.data.replay_buffers.Sampler(**args*, ***kwargs*)

A generic sampler base class for composable Replay Buffers.

Variables:

**requires_shared_state** (*bool*) - `True` when sampling mutates state that
every consumer of the buffer must observe, such as
without-replacement bookkeeping, priorities, consumption marks,
staleness counters or streaming queues. Such a sampler cannot be
copied into [`torch.utils.data.DataLoader`](https://docs.pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader) workers.
Defaults to `True`; samplers whose draws depend only on the
storage content and their configuration, such as
[`RandomSampler`](torchrl.data.replay_buffers.RandomSampler.html#torchrl.data.replay_buffers.RandomSampler) and [`SliceSampler`](torchrl.data.replay_buffers.SliceSampler.html#torchrl.data.replay_buffers.SliceSampler), set it to
`False`.

can_sample(*storage: [Storage](torchrl.data.replay_buffers.Storage.html#torchrl.data.replay_buffers.Storage)*, *batch_size: int*) → bool[[source]](../../_modules/torchrl/data/replay_buffers/samplers/base.html#Sampler.can_sample)

Returns whether the sampler can draw the requested batch.