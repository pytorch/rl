# StalenessAwareSampler

*class*torchrl.data.replay_buffers.StalenessAwareSampler(**args*, ***kwargs*)[[source]](../../_modules/torchrl/data/replay_buffers/samplers.html#StalenessAwareSampler)

A sampler that weights entries by freshness and filters stale entries.

This sampler is designed for asynchronous training setups (e.g., async PPO)
where collected data may come from older policy versions. It reads a
`policy_version` field from the storage and uses it to:

- **Hard gate**: Exclude entries whose staleness exceeds `max_staleness`.
- **Freshness weighting**: Sample proportionally to a weight that decays
with staleness (default: `1 / (staleness + 1)`).

The training loop is responsible for updating `consumer_version`
(typically after each optimizer step or weight update) so the sampler
can compute staleness = `consumer_version - policy_version`.

Parameters:

- **max_staleness** (*int**,**optional*) - Hard cutoff. Entries with
`staleness > max_staleness` are excluded from sampling.
`-1` (default) means no cutoff.
- **staleness_weight_fn** (*callable**,**optional*) - A callable that maps a
staleness tensor (int) to a weight tensor (float). Defaults to
`lambda s: 1.0 / (s.float() + 1.0)`.
- **version_key** (*NestedKey**,**optional*) - The key in the storage holding
the policy version. Defaults to `"policy_version"`.

Examples

```
>>> from torchrl.data import TensorDictReplayBuffer, LazyTensorStorage
>>> from torchrl.data.replay_buffers.samplers import StalenessAwareSampler
>>> sampler = StalenessAwareSampler(max_staleness=5)
>>> buffer = TensorDictReplayBuffer(
... storage=LazyTensorStorage(1000),
... sampler=sampler,
... batch_size=32,
... )
>>> # In training loop:
>>> # sampler.consumer_version = current_training_step
```

Integration with [`Collector`](torchrl.collectors.Collector.html#torchrl.collectors.Collector) and
`PolicyVersion`:

```
from torchrl.collectors import Collector
from torchrl.envs.transforms import PolicyVersion
from torchrl.data import TensorDictReplayBuffer, LazyTensorStorage

sampler = StalenessAwareSampler(max_staleness=10)
buffer = TensorDictReplayBuffer(
 storage=LazyTensorStorage(10_000),
 sampler=sampler,
 batch_size=256,
)
collector = Collector(
 env,
 policy,
 frames_per_batch=1000,
 total_frames=100_000,
 env_transforms=[PolicyVersion(collector)],
)
for step, data in enumerate(collector):
 buffer.extend(data)
 sampler.consumer_version = step
 batch = buffer.sample()
 # ... train on batch ...
```

Note

`StalenessAwareSampler` intentionally does **not** inherit from
[`PrioritizedSampler`](torchrl.data.replay_buffers.PrioritizedSampler.html#torchrl.data.replay_buffers.PrioritizedSampler). `PrioritizedSampler` maintains a
segment-tree over per-transition TD-error priorities that are
updated after each training step. Staleness weighting is
fundamentally different: weights are derived from a single scalar
(`consumer_version`) and per-entry `policy_version` stamps,
and are recomputed on every `sample()` call rather than
maintained incrementally. Sharing the segment-tree machinery
would add complexity without benefit.

*property*consumer_version*: int*

The current training iteration / consumer version.

increment_consumer_version()[[source]](../../_modules/torchrl/data/replay_buffers/samplers.html#StalenessAwareSampler.increment_consumer_version)

Increment the consumer version by 1.

*property*max_staleness*: int*

The maximum allowed staleness. -1 means no limit.