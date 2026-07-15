# Distributed Collectors

TorchRL provides a set of distributed data collectors. These tools support
multiple backends (`'gloo'`, `'nccl'`, `'mpi'` with the [`DistributedCollector`](generated/torchrl.collectors.distributed.DistributedCollector.html#torchrl.collectors.distributed.DistributedCollector)
or PyTorch RPC with [`RPCCollector`](generated/torchrl.collectors.distributed.RPCCollector.html#torchrl.collectors.distributed.RPCCollector)) and launchers (`'ray'`,
`submitit` or `torch.multiprocessing`).
They can be efficiently used in synchronous or asynchronous mode, on a single
node or across multiple nodes.

`RayCollector` uses Ray to place and control its worker actors, but an
attached inference server or replay buffer selects its own payload transport.
With a replay buffer attached, workers write rollouts directly through the
replay endpoint instead of returning them to the driver. Without one, yielded
rollouts use Ray's object store. See [Choosing a payload transport](services_workflow.html#ref-service-transports) for the
backend/transport compatibility table and the recommended direct-to-replay
topology.

*Resources*: Find examples for these collectors in the
[dedicated folder](https://github.com/pytorch/rl/examples/distributed/collectors).

Note

*Choosing the sub-collector*: All distributed collectors support the various single machine collectors.
One may wonder why using a [`MultiSyncCollector`](generated/torchrl.collectors.MultiSyncCollector.html#torchrl.collectors.MultiSyncCollector) or a [`ParallelEnv`](generated/torchrl.envs.ParallelEnv.html#torchrl.envs.ParallelEnv)
instead. In general, multiprocessed collectors have a lower IO footprint than
parallel environments which need to communicate at each step. Yet, the model specs
play a role in the opposite direction, since using parallel environments will
result in a faster execution of the policy (and/or transforms) since these
operations will be vectorized.

Note

*Choosing the device of a collector (or a parallel environment)*: Sharing data
among processes is achieved via shared-memory buffers with parallel environment
and multiprocessed environments executed on CPU. Depending on the capabilities
of the machine being used, this may be prohibitively slow compared to sharing
data on GPU which is natively supported by cuda drivers.
In practice, this means that using the `device="cpu"` keyword argument when
building a parallel environment or collector can result in a slower collection
than using `device="cuda"` when available.

Note

Given the library's many optional dependencies (eg, Gym, Gymnasium, and many others)
warnings can quickly become quite annoying in multiprocessed / distributed settings.
By default, TorchRL filters out these warnings in sub-processes. If one still wishes to
see these warnings, they can be displayed by setting `torchrl.filter_warnings_subprocess=False`.

Tip

All distributed collectors support `trajs_per_batch` combined with
`replay_buffer`. When set, each remote worker assembles **complete
trajectories** and writes them to the shared buffer as flat 1-D sequences,
which is directly compatible with [`SliceSampler`](generated/torchrl.data.replay_buffers.SliceSampler.html#torchrl.data.replay_buffers.SliceSampler).
See [Complete trajectory collection with trajs_per_batch](collectors_replay.html#collectors-replay-trajs) for examples and best practices.

| [`DistributedCollector`](generated/torchrl.collectors.distributed.DistributedCollector.html#torchrl.collectors.distributed.DistributedCollector)(create_env_fn, policy, ...) | A distributed data collector with torch.distributed backend. |
| --- | --- |
| [`RPCCollector`](generated/torchrl.collectors.distributed.RPCCollector.html#torchrl.collectors.distributed.RPCCollector)(create_env_fn, policy, ...) | An RPC-based distributed data collector. |
| [`DistributedSyncCollector`](generated/torchrl.collectors.distributed.DistributedSyncCollector.html#torchrl.collectors.distributed.DistributedSyncCollector)(create_env_fn, ...) | A distributed synchronous data collector with torch.distributed backend. |
| [`submitit_delayed_launcher`](generated/torchrl.collectors.distributed.submitit_delayed_launcher.html#torchrl.collectors.distributed.submitit_delayed_launcher)(num_jobs[, ...]) | Delayed launcher for submitit. |
| [`RayCollector`](generated/torchrl.collectors.distributed.RayCollector.html#torchrl.collectors.distributed.RayCollector)(create_env_fn, policy, ...[, ...]) | Distributed data collector with [Ray](https://docs.ray.io/) backend. |
| [`RayEvalWorker`](generated/torchrl.collectors.distributed.RayEvalWorker.html#torchrl.collectors.distributed.RayEvalWorker)(init_fn, env_maker, ...[, ...]) | Asynchronous evaluation worker backed by a Ray actor. |

## Removed legacy names

The deprecated distributed collector aliases were removed in v0.13. Use
`DistributedCollector`, `RPCCollector`, and `DistributedSyncCollector`
directly.