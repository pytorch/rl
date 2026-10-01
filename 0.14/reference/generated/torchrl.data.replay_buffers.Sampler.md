# Sampler

*class*torchrl.data.replay_buffers.Sampler(**args*, ***kwargs*)

A generic sampler base class for composable Replay Buffers.

can_sample(*storage: [Storage](torchrl.data.replay_buffers.Storage.html#torchrl.data.replay_buffers.Storage)*, *batch_size: int*) → bool[[source]](../../_modules/torchrl/data/replay_buffers/samplers/base.html#Sampler.can_sample)

Returns whether the sampler can draw the requested batch.