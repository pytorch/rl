# Single Node Collectors

TorchRL provides several collector classes for single-node data collection, each with different execution strategies.

## Single node data collectors

| [`BaseCollector`](generated/torchrl.collectors.BaseCollector.html#torchrl.collectors.BaseCollector)(*[, pre_collect_hook, ...]) | Base class for data collectors. |
| --- | --- |
| [`Collector`](generated/torchrl.collectors.Collector.html#torchrl.collectors.Collector)(create_env_fn[, policy, ...]) | Generic data collector for RL problems. |
| [`AsyncCollector`](generated/torchrl.collectors.AsyncCollector.html#torchrl.collectors.AsyncCollector)(*args[, sync]) | Runs a single DataCollector on a separate process. |
| [`AsyncBatchedCollector`](generated/torchrl.collectors.AsyncBatchedCollector.html#torchrl.collectors.AsyncBatchedCollector)(create_env_fn, *[, ...]) | Asynchronous collector with env slots and a policy server. |
| [`MultiCollector`](generated/torchrl.collectors.MultiCollector.html#torchrl.collectors.MultiCollector)(*args[, sync]) | Runs a given number of DataCollectors on separate processes. |
| [`MultiSyncCollector`](generated/torchrl.collectors.MultiSyncCollector.html#torchrl.collectors.MultiSyncCollector)(*args[, sync]) | Runs a given number of DataCollectors on separate processes synchronously. |
| [`MultiAsyncCollector`](generated/torchrl.collectors.MultiAsyncCollector.html#torchrl.collectors.MultiAsyncCollector)(*args[, sync]) | Runs a given number of DataCollectors on separate processes asynchronously. |

## Trajectory batching

Pass `trajs_per_batch=N` to any collector to receive batches of exactly *N*
complete, zero-padded trajectories instead of fixed-frame batches.
Trajectories that span multiple internal collection steps are automatically
reassembled. Each yielded [`TensorDict`](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDict.html#tensordict.TensorDict) has shape
`(N, max_traj_len)` and includes a `("collector", "mask")` boolean tensor
marking valid time steps.

`frames_per_batch` still controls how frequently the environment is polled
internally; it does **not** determine the output batch size when
`trajs_per_batch` is set.

```
from torchrl.collectors import Collector
from torchrl.envs import GymEnv

collector = Collector(
 GymEnv("CartPole-v1"),
 policy=my_policy,
 frames_per_batch=200, # controls internal polling frequency
 total_frames=10000,
 trajs_per_batch=4,
 traj_format="padded",
)

for batch in collector:
 # batch.shape == (4, max_traj_len)
 valid = batch[("collector", "mask")] # (4, max_traj_len) bool
 loss = compute_loss(batch, valid)
 collector.update_policy_weights_()
```

**Unpadded batches**: with `traj_format="cat"` the *N* trajectories are
concatenated along time instead of stacked and padded. Each yield is then a
flat `[sum_i T_i]` batch with no mask: trajectories are contiguous, in
completion order, with `("next", "done")` `True` at the last step of each
and `("collector", "traj_ids")` identifying them. Prefer it when episode
lengths vary widely or frames are large -- the padded layout materializes
`N * max_traj_len` frames, the flat one only the steps actually collected.

Note

The current default layout is `"padded"`, but it will change to
`"cat"` in torchrl v0.16. Omitting `traj_format` while yielding
`trajs_per_batch` batches emits a `FutureWarning`; pass the
layout explicitly.

**Replay buffer integration**: when a `replay_buffer` is also provided,
complete trajectories are written to the buffer as **flat 1-D sequences** (no
padding) instead of being yielded. This is the recommended pattern for
off-policy training with [`SliceSampler`](generated/torchrl.data.replay_buffers.SliceSampler.html#torchrl.data.replay_buffers.SliceSampler), especially
with multi-process collectors where fixed-frame batches can silently mix
episodes. See [Complete trajectory collection with trajs_per_batch](collectors_replay.html#collectors-replay-trajs) for full details and examples.

Note

The deprecated collector aliases were removed in v0.13. Use the canonical
classes directly: `BaseCollector`, `Collector`, `AsyncCollector`,
`MultiCollector`, `MultiSyncCollector`, and `MultiAsyncCollector`.

## Using AsyncBatchedCollector

The [`AsyncBatchedCollector`](generated/torchrl.collectors.AsyncBatchedCollector.html#torchrl.collectors.AsyncBatchedCollector) pairs an [`AsyncEnvPool`](generated/torchrl.envs.AsyncEnvPool.html#torchrl.envs.AsyncEnvPool)
with an [`InferenceServer`](generated/torchrl.modules.inference_server.InferenceServer.html#torchrl.modules.inference_server.InferenceServer) to pipeline environment
stepping and batched GPU inference. You only need to supply **env factories**
and a **policy** - all internal wiring is handled automatically:

```
from torchrl.collectors import AsyncBatchedCollector
from torchrl.envs import GymEnv
from tensordict.nn import TensorDictModule
import torch.nn as nn

policy = TensorDictModule(
 nn.Sequential(nn.Linear(4, 64), nn.ReLU(), nn.Linear(64, 2)),
 in_keys=["observation"],
 out_keys=["action"],
)

collector = AsyncBatchedCollector(
 create_env_fn=[lambda: GymEnv("CartPole-v1")] * 8,
 policy=policy,
 frames_per_batch=200,
 total_frames=10000,
 max_batch_size=8,
)

for data in collector:
 # data is a lazy-stacked TensorDict of collected transitions
 pass

collector.shutdown()
```

**Key advantages over** [`Collector`](generated/torchrl.collectors.Collector.html#torchrl.collectors.Collector):

- The inference server automatically **batches policy forward passes** from
all environments, maximising GPU utilisation.
- Environment stepping and inference run in **overlapping fashion**, reducing
idle time.
- Supports `yield_completed_trajectories=True` for episode-level yields.

## Using MultiCollector

The [`MultiCollector`](generated/torchrl.collectors.MultiCollector.html#torchrl.collectors.MultiCollector) class is the recommended way to run parallel data collection.
It uses a `sync` parameter to dispatch to either [`MultiSyncCollector`](generated/torchrl.collectors.MultiSyncCollector.html#torchrl.collectors.MultiSyncCollector) or [`MultiAsyncCollector`](generated/torchrl.collectors.MultiAsyncCollector.html#torchrl.collectors.MultiAsyncCollector):

```
from torchrl.collectors import MultiCollector
from torchrl.envs import GymEnv

def make_env():
 return GymEnv("CartPole-v1")

# Synchronous multi-worker collection (recommended for on-policy algorithms)
sync_collector = MultiCollector(
 create_env_fn=[make_env] * 4, # 4 parallel workers
 policy=my_policy,
 frames_per_batch=1000,
 total_frames=100000,
 sync=True, # ← All workers complete before delivering batch
)

# Asynchronous multi-worker collection (recommended for off-policy algorithms)
async_collector = MultiCollector(
 create_env_fn=[make_env] * 4,
 policy=my_policy,
 frames_per_batch=1000,
 total_frames=100000,
 sync=False, # ← First-come-first-serve delivery
)

# Iterate over collected data
for data in sync_collector:
 # Train on data...
 pass

sync_collector.shutdown()
```

**Comparison:**

| Feature | `sync=True` | `sync=False` |
| --- | --- | --- |
| Batch delivery | All workers complete first | First available worker |
| Policy consistency | All data from same policy version | Data may be from older policy |
| Best for | On-policy (PPO, A2C) | Off-policy (SAC, DQN) |
| Throughput | Limited by slowest worker | Higher throughput |

## Running the Collector Asynchronously

Passing replay buffers to a collector allows us to start the collection and get rid of the iterative nature of the
collector.
If you want to run a data collector in the background, simply run [`start()`](generated/torchrl.collectors.BaseCollector.html#torchrl.collectors.BaseCollector.start):

```
>>> collector = Collector(..., replay_buffer=rb) # pass your replay buffer
>>> collector.start()
>>> # little pause
>>> time.sleep(10)
>>> # Start training
>>> for i in range(optim_steps):
... data = rb.sample() # Sampling from the replay buffer
... # rest of the training loop
```

Single-process collectors ([`Collector`](generated/torchrl.collectors.Collector.html#torchrl.collectors.Collector)) will run the process using multithreading,
so be mindful of Python's GIL and related multithreading restrictions.

Multiprocessed collectors will on the other hand let the child processes handle the filling of the buffer on their own,
which truly decouples the data collection and training.

Data collectors that have been started with start() should be shut down using
[`async_shutdown()`](generated/torchrl.collectors.BaseCollector.html#torchrl.collectors.BaseCollector.async_shutdown).

Tip

For maximum throughput with trajectory-based training (e.g. recurrent
policies, decision transformers), combine `start()` with
`trajs_per_batch` and a [`SliceSampler`](generated/torchrl.data.replay_buffers.SliceSampler.html#torchrl.data.replay_buffers.SliceSampler):

```
rb = ReplayBuffer(
 storage=LazyTensorStorage(100_000),
 sampler=SliceSampler(slice_len=32, end_key=("next", "done")),
 batch_size=256,
 shared=True,
)
collector = MultiCollector(
 [make_env] * 4,
 policy,
 replay_buffer=rb,
 frames_per_batch=200,
 total_frames=-1,
 trajs_per_batch=8,
 sync=False,
)
collector.start()
for step in range(train_steps):
 batch = rb.sample() # clean trajectory slices
 # ...
collector.async_shutdown()
```

Each worker writes only **complete trajectories** to the buffer, so the
sampler never draws slices that cross episode boundaries. See
[Complete trajectory collection with trajs_per_batch](collectors_replay.html#collectors-replay-trajs) for a full discussion.

Warning

Running a collector asynchronously decouples the collection from training, which means that the training
performance may be drastically different depending on the hardware, load and other factors (although it is generally
expected to provide significant speed-ups). Make sure you understand how this may affect your algorithm and if it
is a legitimate thing to do! (For example, on-policy algorithms such as PPO should not be run asynchronously
unless properly benchmarked).