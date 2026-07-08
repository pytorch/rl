# Sampling Strategies

Samplers control how data is retrieved from the replay buffer storage.

See also

The trajectory-aware samplers ([`SliceSampler`](generated/torchrl.data.replay_buffers.SliceSampler.html#torchrl.data.replay_buffers.SliceSampler) and its variants)
recover episode boundaries from the stored data at sampling time. The
conventions they rely on -- trajectory ids, end flags, circular-storage
wraparound and the write cursor -- are documented in
[Trajectory boundaries](data_layout.html#ref-traj-boundaries).

| [`PrioritizedSampler`](generated/torchrl.data.replay_buffers.PrioritizedSampler.html#torchrl.data.replay_buffers.PrioritizedSampler)(*args, **kwargs) | Prioritized sampler for replay buffer. |
| --- | --- |
| [`PrioritizedSliceSampler`](generated/torchrl.data.replay_buffers.PrioritizedSliceSampler.html#torchrl.data.replay_buffers.PrioritizedSliceSampler)(*args, **kwargs) | Samples slices of data along the first dimension, given start and stop signals, using prioritized sampling. |
| [`ConsumingSampler`](generated/torchrl.data.replay_buffers.ConsumingSampler.html#torchrl.data.replay_buffers.ConsumingSampler)(*args, **kwargs) | A random sampler that consumes entries after they have been sampled. |
| [`RandomSampler`](generated/torchrl.data.replay_buffers.RandomSampler.html#torchrl.data.replay_buffers.RandomSampler)(*args, **kwargs) | A uniformly random sampler for composable replay buffers. |
| [`Sampler`](generated/torchrl.data.replay_buffers.Sampler.html#torchrl.data.replay_buffers.Sampler)(*args, **kwargs) | A generic sampler base class for composable Replay Buffers. |
| [`SamplerEnsemble`](generated/torchrl.data.replay_buffers.SamplerEnsemble.html#torchrl.data.replay_buffers.SamplerEnsemble)(*args, **kwargs) | An ensemble of samplers. |
| [`SamplerWithoutReplacement`](generated/torchrl.data.replay_buffers.SamplerWithoutReplacement.html#torchrl.data.replay_buffers.SamplerWithoutReplacement)(*args, **kwargs) | A data-consuming sampler that ensures that the same sample is not present in consecutive batches. |
| [`SliceSampler`](generated/torchrl.data.replay_buffers.SliceSampler.html#torchrl.data.replay_buffers.SliceSampler)(*args, **kwargs) | Samples slices of data along the first dimension, given start and stop signals. |
| [`SliceSamplerWithoutReplacement`](generated/torchrl.data.replay_buffers.SliceSamplerWithoutReplacement.html#torchrl.data.replay_buffers.SliceSamplerWithoutReplacement)(*args, **kwargs) | Samples slices of data along the first dimension, given start and stop signals, without replacement. |
| [`StalenessAwareSampler`](generated/torchrl.data.replay_buffers.StalenessAwareSampler.html#torchrl.data.replay_buffers.StalenessAwareSampler)(*args, **kwargs) | A sampler that weights entries by freshness and filters stale entries. |

## Writers

Writers control how data is written to the storage.

| [`RoundRobinWriter`](generated/torchrl.data.replay_buffers.RoundRobinWriter.html#torchrl.data.replay_buffers.RoundRobinWriter)([compilable]) | A RoundRobin Writer class for composable replay buffers. |
| --- | --- |
| [`TensorDictMaxValueWriter`](generated/torchrl.data.replay_buffers.TensorDictMaxValueWriter.html#torchrl.data.replay_buffers.TensorDictMaxValueWriter)([rank_key, reduction]) | A Writer class for composable replay buffers that keeps the top elements based on some ranking key. |
| [`TensorDictRoundRobinWriter`](generated/torchrl.data.replay_buffers.TensorDictRoundRobinWriter.html#torchrl.data.replay_buffers.TensorDictRoundRobinWriter)([compilable]) | A RoundRobin Writer class for composable, tensordict-based replay buffers. |
| [`Writer`](generated/torchrl.data.replay_buffers.Writer.html#torchrl.data.replay_buffers.Writer)([compilable]) | A ReplayBuffer base Writer class. |
| [`WriterEnsemble`](generated/torchrl.data.replay_buffers.WriterEnsemble.html#torchrl.data.replay_buffers.WriterEnsemble)(*writers) | An ensemble of writers. |