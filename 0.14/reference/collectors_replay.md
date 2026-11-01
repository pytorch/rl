# Collectors and Replay Buffers

See also

For the conceptual story behind the patterns on this page --
contiguous 1-D trajectories, the boundary keys (`is_init`,
`done`, `terminated`, `truncated`), the limits of
`ndim>=2` storages with multi-process collectors, and why
[`split_trajectories()`](generated/torchrl.collectors.utils.split_trajectories.html#torchrl.collectors.utils.split_trajectories) is no longer
recommended -- see [Data layout: contiguous trajectories](data_layout.html#data-layout).

## Collectors and replay buffers interoperability

In the simplest scenario where single transitions have to be sampled
from the replay buffer, little attention has to be given to the way
the collector is built. Flattening the data after collection will
be a sufficient preprocessing step before populating the storage:

```
>>> memory = ReplayBuffer(
... storage=LazyTensorStorage(N),
... transform=lambda data: data.reshape(-1))
>>> for data in collector:
... memory.extend(data)
```

If trajectory slices have to be collected, the recommended way to achieve this is to create
a multidimensional buffer and sample using the [`SliceSampler`](generated/torchrl.data.replay_buffers.SliceSampler.html#torchrl.data.replay_buffers.SliceSampler)
sampler class. One must ensure that the data passed to the buffer is properly shaped, with the
`time` and `batch` dimensions clearly separated. In practice, the following configurations
will work:

```
>>> # Single environment: no need for a multi-dimensional buffer
>>> memory = ReplayBuffer(
... storage=LazyTensorStorage(N),
... sampler=SliceSampler(num_slices=4, trajectory_key=("collector", "traj_ids"))
... )
>>> collector = Collector(env, policy, frames_per_batch=N, total_frames=-1)
>>> for data in collector:
... memory.extend(data)
>>> # Batched environments: a multi-dim buffer is required
>>> memory = ReplayBuffer(
... storage=LazyTensorStorage(N, ndim=2),
... sampler=SliceSampler(num_slices=4, trajectory_key=("collector", "traj_ids"))
... )
>>> env = ParallelEnv(4, make_env)
>>> collector = Collector(env, policy, frames_per_batch=N, total_frames=-1)
>>> for data in collector:
... memory.extend(data)
>>> # Synchronous process collection behaves like ParallelEnv if cat_results="stack"
>>> memory = ReplayBuffer(
... storage=LazyTensorStorage(N, ndim=2),
... sampler=SliceSampler(num_slices=4, trajectory_key=("collector", "traj_ids"))
... )
>>> collector = Collector(make_env, policy,
... num_collectors=4,
... sync=True,
... frames_per_batch=N,
... total_frames=-1,
... cat_results="stack")
>>> for data in collector:
... memory.extend(data)
>>> # Process collection + parallel env: adapt ndim for both batch dimensions
>>> memory = ReplayBuffer(
... storage=LazyTensorStorage(N, ndim=3),
... sampler=SliceSampler(num_slices=4, trajectory_key=("collector", "traj_ids"))
... )
>>> collector = Collector(lambda: ParallelEnv(2, make_env), policy,
... num_collectors=4,
... sync=True,
... frames_per_batch=N,
... total_frames=-1,
... cat_results="stack")
>>> for data in collector:
... memory.extend(data)
```

Important

The `ndim=2` and `ndim=3` examples above apply to **fixed-frame
batches** (the default, without `trajs_per_batch`). When
`trajs_per_batch` is set, each trajectory is written to the buffer as a
**flat 1-D sequence** of variable length. A storage with `ndim >= 2`
expects a fixed second dimension that variable-length trajectories cannot
satisfy. Always use the default `ndim=1` when combining
`trajs_per_batch` with a replay buffer.

## Complete trajectory collection with `trajs_per_batch`

When using `Collector(num_collectors=N)` with fixed-frame batches (the
concrete result is [`MultiSyncCollector`](generated/torchrl.collectors.MultiSyncCollector.html#torchrl.collectors.MultiSyncCollector) or
[`MultiAsyncCollector`](generated/torchrl.collectors.MultiAsyncCollector.html#torchrl.collectors.MultiAsyncCollector))
and a [`SliceSampler`](generated/torchrl.data.replay_buffers.SliceSampler.html#torchrl.data.replay_buffers.SliceSampler), adjacent frames in the buffer can
come from **different workers and different episodes** without an intervening
`done` signal. The sampler has no way to detect these invisible boundaries,
so it may draw slices that straddle unrelated trajectories -- silently
corrupting the training data.

Setting `trajs_per_batch` on the collector solves this. Each worker
assembles **complete trajectories** (episodes whose last step carries
`("next", "done") == True`) before writing them to the buffer as flat 1-D
sequences -- no padding, no artificial boundaries. Every trajectory in the
buffer is guaranteed to be a genuine episode segment, making it directly
compatible with [`SliceSampler`](generated/torchrl.data.replay_buffers.SliceSampler.html#torchrl.data.replay_buffers.SliceSampler).

`trajs_per_batch` is not tied to replay buffers or multi-process
collection: on any collector -- including the single-process
[`Collector`](generated/torchrl.collectors.Collector.html#torchrl.collectors.Collector) -- setting it (without a
`replay_buffer`) switches the iterator from fixed-frame batches to batches
of exactly that many **complete trajectories**, zero-padded along time with
a `("collector", "mask")` entry marking the valid steps. Episodes
spanning internal collection steps are reassembled, and in-flight episodes
are held back for the next batch. This is the natural fit for on-policy
algorithms whose training unit is the episode (e.g. GRPO-style
group-relative advantages):

```
from torchrl.collectors import Collector

collector = Collector(
 env,
 policy,
 frames_per_batch=200, # internal polling granularity only
 total_frames=-1,
 trajs_per_batch=16, # one yield = 16 whole episodes
 traj_format="padded", # default until v0.16, then "cat"
)
for batch in collector: # batch: [16, max_traj_len]
 mask = batch["collector", "mask"]
 returns = (batch["next", "reward"].squeeze(-1) * mask).sum(-1)
 # ...
```

The padded layout is convenient for per-trajectory reductions but
materializes `16 * max_traj_len` frames even when most episodes are short.
With `traj_format="cat"` the same batches come out **flat and unpadded**
instead: trajectories are concatenated along time (shape `[sum_i T_i]`),
contiguous and in completion order, with `("next", "done")` `True` at the
last step of each and `("collector", "traj_ids")` telling them apart -- the
same layout the replay-buffer write path produces. Prefer it when episode
lengths vary a lot or frames are large (images, token sequences):

```
collector = Collector(
 env,
 policy,
 frames_per_batch=200,
 total_frames=-1,
 trajs_per_batch=16,
 traj_format="cat",
)
for batch in collector: # batch: [sum of the 16 episode lengths]
 done = batch["next", "done"].squeeze(-1)
 episode_idx = done.long().cumsum(0) - done.long()
 # per-episode return without any padding
 returns = torch.zeros(16).index_add_(
 0, episode_idx, batch["next", "reward"].squeeze(-1)
 )
 # ...
```

**Synchronous iteration (for-loop)**

```
from torchrl.collectors import Collector
from torchrl.data import ReplayBuffer, LazyTensorStorage, SliceSampler

rb = ReplayBuffer(
 storage=LazyTensorStorage(100_000),
 sampler=SliceSampler(slice_len=32, end_key=("next", "done")),
 batch_size=256,
)
collector = Collector(
 make_env,
 policy,
 num_collectors=4,
 replay_buffer=rb,
 frames_per_batch=200,
 total_frames=500_000,
 trajs_per_batch=8, # each worker writes complete trajectories
 sync=True,
)
for _ in collector: # yields None (data goes straight to rb)
 batch = rb.sample() # contiguous sub-sequences, no cross-episode leaks
 loss = loss_fn(batch)
 # ...
```

**Asynchronous collection (``start()``)**

For off-policy algorithms where data collection and training run
concurrently, use [`start()`](generated/torchrl.collectors.BaseCollector.html#torchrl.collectors.BaseCollector.start):

```
collector = Collector(
 make_env,
 policy,
 num_collectors=4,
 replay_buffer=rb,
 frames_per_batch=200,
 total_frames=-1,
 trajs_per_batch=8,
 sync=False,
)
collector.start() # workers fill rb in background threads/processes
for step in range(train_steps):
 batch = rb.sample()
 loss = loss_fn(batch)
 # ...
 collector.update_policy_weights_()
collector.async_shutdown()
```

This pattern fully decouples data collection from training and is the
recommended way to maximise inference throughput on multi-core machines or
GPU-accelerated environments.

**Direct collectors** also support `trajs_per_batch` with the same
replay-buffer semantics:

```
collector = Collector(
 env, policy,
 backend="direct",
 replay_buffer=rb,
 frames_per_batch=200,
 total_frames=-1,
 trajs_per_batch=8,
)
collector.start()
# ...
```

Warning

Without `trajs_per_batch`, a multi-process collector writes fixed-frame
batches from each worker. If the buffer uses a
[`SliceSampler`](generated/torchrl.data.replay_buffers.SliceSampler.html#torchrl.data.replay_buffers.SliceSampler), the sampler will reconstruct episode
boundaries from `done` signals, but worker batch boundaries are invisible
-- consecutive frames in the buffer may belong to completely different
episodes.

A partial mitigation is `set_truncated=True`, which marks every batch
boundary with a `truncated` (and therefore `done`) signal. This
prevents cross-episode slices but introduces artificial truncations that
value estimators must handle correctly.

`trajs_per_batch` is the recommended solution: it guarantees clean
episode boundaries in the buffer without artificial truncations.

See also

- [`BaseCollector`](generated/torchrl.collectors.BaseCollector.html#torchrl.collectors.BaseCollector) for the full `trajs_per_batch`
API, completeness guarantee, and batched-environment behaviour.
- [`SliceSampler`](generated/torchrl.data.replay_buffers.SliceSampler.html#torchrl.data.replay_buffers.SliceSampler) for configuring sub-sequence sampling
from the buffer.
- [Trajectory boundaries](data_layout.html#ref-traj-boundaries) for the contract the
sampler relies on: which markers delimit trajectories in storage, how
boundaries are recovered at read time
([`find_start_stop_traj()`](generated/torchrl.data.find_start_stop_traj.html#torchrl.data.find_start_stop_traj)), and the blind spot when
neither ids nor end flags are present.
- The trajectory batching section in the
single-node collector docs for the non-replay-buffer usage
(padded `(trajs, max_len)` batches, or flat unpadded batches with
`traj_format="cat"`).

## Helper functions

| [`split_trajectories`](generated/torchrl.collectors.utils.split_trajectories.html#torchrl.collectors.utils.split_trajectories)(rollout_tensordict, *[, ...]) | A util function for trajectory separation. |
| --- | --- |