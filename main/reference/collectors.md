# torchrl.collectors package

Data collectors are the bridge between your environments and training loop, managing the process of gathering
experience data using your policy. They handle environment resets, policy execution, and data aggregation,
making it easy to collect high-quality training data efficiently.

TorchRL provides several collector implementations optimized for different scenarios:

- [`Collector`](generated/torchrl.collectors.Collector.html#torchrl.collectors.Collector): Single-process collection on the training worker
- [`AsyncBatchedCollector`](generated/torchrl.collectors.AsyncBatchedCollector.html#torchrl.collectors.AsyncBatchedCollector): Async environments + auto-batching inference server (see [`AsyncBatchedCollector`](generated/torchrl.collectors.AsyncBatchedCollector.html#torchrl.collectors.AsyncBatchedCollector))
- [`MultiCollector`](generated/torchrl.collectors.MultiCollector.html#torchrl.collectors.MultiCollector): Parallel collection across multiple workers (see below)
- [`Evaluator`](generated/torchrl.collectors.Evaluator.html#torchrl.collectors.Evaluator): Sync or async evaluation during training (see [evaluation](collectors_eval.html#collectors-eval))
- **Distributed collectors**: For multi-node setups using Ray, RPC, or distributed backends (see [`DistributedCollector`](generated/torchrl.collectors.distributed.DistributedCollector.html#torchrl.collectors.distributed.DistributedCollector) / [`RPCCollector`](generated/torchrl.collectors.distributed.RPCCollector.html#torchrl.collectors.distributed.RPCCollector))

## MultiCollector API

The [`MultiCollector`](generated/torchrl.collectors.MultiCollector.html#torchrl.collectors.MultiCollector) class provides a unified interface for multi-process data collection.
Use the `sync` parameter to choose between synchronous and asynchronous collection:

```
from torchrl.collectors import MultiCollector

# Synchronous collection: all workers complete before delivering batch
collector = MultiCollector(
 create_env_fn=[make_env] * 4, # 4 parallel workers
 policy=my_policy,
 frames_per_batch=200,
 total_frames=10000,
 sync=True, # synchronized delivery
)

# Asynchronous collection: first-come-first-serve delivery
collector = MultiCollector(
 create_env_fn=[make_env] * 4,
 policy=my_policy,
 frames_per_batch=200,
 total_frames=10000,
 sync=False, # async delivery (faster, but policy may lag)
)
```

**When to use sync vs async:**

- `sync=True`: Use for on-policy algorithms (PPO, A2C) where data must match current policy
- `sync=False`: Use for off-policy algorithms (SAC, DQN) where slight policy lag is acceptable

## Key Features

- **Flexible execution**: Choose between sync, async, and distributed collection
- **Device management**: Control where environments and policies execute
- **Weight synchronization**: Keep inference policies up-to-date with training weights
- **Replay buffer integration**: Seamless compatibility with TorchRL's replay buffers
- **Trajectory assembly**: Collect complete trajectories with `trajs_per_batch` --
padded whole-episode batches for on-policy training, or flat unpadded writes into a
replay buffer for clean [`SliceSampler`](generated/torchrl.data.replay_buffers.SliceSampler.html#torchrl.data.replay_buffers.SliceSampler) sampling --
see [Complete trajectory collection with trajs_per_batch](collectors_replay.html#collectors-replay-trajs)
- **Batching strategies**: Multiple ways to organize collected data
- **Profiler-ready**: Set `TORCHRL_PROFILING=1` to emit named ranges on the
collector, env, and policy hot paths -- see [Profiling collectors and envs](profiling.html#ref-profiling)

## Collection hooks

Collectors accept optional hooks for per-rollout side effects:
`pre_collect_hook` is called before a rollout starts, and
`post_collect_hook` is called with the [`TensorDictBase`](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)
batch that will be yielded by iteration. Hook return values are ignored, and
exceptions raised by hooks propagate to the caller and stop collection.

Hooks are intended for instrumentation and worker-local side effects, such as
stepping a profiler or recording rollout metrics. Use `postproc` when the
collected data itself should be transformed before training.

For [`MultiCollector`](generated/torchrl.collectors.MultiCollector.html#torchrl.collectors.MultiCollector), [`MultiSyncCollector`](generated/torchrl.collectors.MultiSyncCollector.html#torchrl.collectors.MultiSyncCollector), and
[`MultiAsyncCollector`](generated/torchrl.collectors.MultiAsyncCollector.html#torchrl.collectors.MultiAsyncCollector), hooks run in each worker process. The helper
methods [`map_fn()`](generated/torchrl.collectors.BaseCollector.html#torchrl.collectors.BaseCollector.map_fn) and
[`get_distant_attr()`](generated/torchrl.collectors.BaseCollector.html#torchrl.collectors.BaseCollector.get_distant_attr) broadcast to each
worker for multi-process collectors and to each actor for
[`RayCollector`](generated/torchrl.collectors.distributed.RayCollector.html#torchrl.collectors.distributed.RayCollector).

## Quick Example

```
from torchrl.collectors import Collector
from torchrl.envs import GymEnv, ParallelEnv

# Create a batched environment
def make_env():
 return GymEnv("Pendulum-v1")

env = ParallelEnv(4, make_env)

# Create collector
collector = Collector(
 env,
 policy=my_policy,
 frames_per_batch=200,
 total_frames=10000,
)

# Collect data
for data in collector:
 # data is a TensorDict with shape [4, 50] (4 envs, 50 steps each)
 # Use data for training...

 # Update policy weights periodically
 if should_update:
 collector.update_policy_weights_()

collector.shutdown()
```

## Removed legacy names

The deprecated collector aliases were removed in v0.13. Use the canonical
collector classes directly: `Collector`, `MultiCollector`,
`MultiSyncCollector`, `MultiAsyncCollector`, and `BaseCollector`.

## Documentation Sections

- [Collector Basics](collectors_basics.html)

- [Trajectory IDs](collectors_basics.html#trajectory-ids)
- [Collectors and batch size](collectors_basics.html#collectors-and-batch-size)
- [Collectors and policy copies](collectors_basics.html#collectors-and-policy-copies)
- [Single Node Collectors](collectors_single.html)

- [Single node data collectors](collectors_single.html#single-node-data-collectors)
- [Trajectory batching](collectors_single.html#trajectory-batching)
- [Using AsyncBatchedCollector](collectors_single.html#using-asyncbatchedcollector)
- [Using MultiCollector](collectors_single.html#using-multicollector)
- [Running the Collector Asynchronously](collectors_single.html#running-the-collector-asynchronously)
- [Collector Internals](collectors_internals.html)

- [Per-timestep flow](collectors_internals.html#per-timestep-flow)
- [The carrier](collectors_internals.html#the-carrier)
- [Sync points](collectors_internals.html#sync-points)
- [Device casting flags](collectors_internals.html#device-casting-flags)
- [Trajectory IDs](collectors_internals.html#trajectory-ids)
- [Collection hooks](collectors_internals.html#collection-hooks)
- [Where to look in the code](collectors_internals.html#where-to-look-in-the-code)
- [See also](collectors_internals.html#see-also)
- [Evaluation](collectors_eval.html)

- [Why use an Evaluator?](collectors_eval.html#why-use-an-evaluator)
- [Quick example](collectors_eval.html#quick-example)
- [Synchronous usage](collectors_eval.html#synchronous-usage)
- [Asynchronous usage](collectors_eval.html#asynchronous-usage)
- [Device placement and compilation](collectors_eval.html#device-placement-and-compilation)
- [Overlap policy (backpressure)](collectors_eval.html#overlap-policy-backpressure)
- [Result callbacks](collectors_eval.html#result-callbacks)
- [Backends](collectors_eval.html#backends)
- [Custom metrics and callbacks](collectors_eval.html#custom-metrics-and-callbacks)
- [API Reference](collectors_eval.html#api-reference)
- [Distributed Collectors](collectors_distributed.html)

- [DistributedCollector](generated/torchrl.collectors.distributed.DistributedCollector.html)
- [RPCCollector](generated/torchrl.collectors.distributed.RPCCollector.html)
- [DistributedSyncCollector](generated/torchrl.collectors.distributed.DistributedSyncCollector.html)
- [submitit_delayed_launcher](generated/torchrl.collectors.distributed.submitit_delayed_launcher.html)
- [RayCollector](generated/torchrl.collectors.distributed.RayCollector.html)
- [RayEvalWorker](generated/torchrl.collectors.distributed.RayEvalWorker.html)
- [Removed legacy names](collectors_distributed.html#removed-legacy-names)
- [Weight Synchronization](collectors_weightsync.html)

- [Lifecycle of Weight Synchronization](collectors_weightsync.html#lifecycle-of-weight-synchronization)
- [Scheme-Specific Behavior](collectors_weightsync.html#scheme-specific-behavior)
- [Usage Examples](collectors_weightsync.html#usage-examples)
- [Evaluator Weight Sync](collectors_weightsync.html#evaluator-weight-sync)
- [Transports](collectors_weightsync.html#transports)
- [Schemes](collectors_weightsync.html#schemes)
- [Legacy: Weight Updaters](collectors_weightsync.html#legacy-weight-updaters)
- [Collectors and Replay Buffers](collectors_replay.html)

- [Collectors and replay buffers interoperability](collectors_replay.html#collectors-and-replay-buffers-interoperability)
- [Complete trajectory collection with `trajs_per_batch`](collectors_replay.html#complete-trajectory-collection-with-trajs-per-batch)
- [Helper functions](collectors_replay.html#helper-functions)