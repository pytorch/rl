# MultiSyncCollector

*class*torchrl.collectors.MultiSyncCollector(**args*, *sync: bool | None = None*, ***kwargs*)[[source]](../../_modules/torchrl/collectors/_multi_sync.html#MultiSyncCollector)

Runs a given number of DataCollectors on separate processes synchronously.

[![../../_images/aafig-0afdfc2530573f394c19f5999361d016302d1e1b.svg](../../_images/aafig-0afdfc2530573f394c19f5999361d016302d1e1b.svg)](../../_images/aafig-0afdfc2530573f394c19f5999361d016302d1e1b.svg)

Envs can be identical or different.

The collection starts when the next item of the collector is queried,
and no environment step is computed in between the reception of a batch of
trajectory and the start of the next collection.
This class can be safely used with online RL sota-implementations.

Note

Python requires multiprocessed code to be instantiated within a main guard:

```
>>> from torchrl.collectors import MultiSyncCollector
>>> if __name__ == "__main__":
... # Create your collector here
... collector = MultiSyncCollector(...)
```

See [https://docs.python.org/3/library/multiprocessing.html](https://docs.python.org/3/library/multiprocessing.html) for more info.

Examples

```
>>> from torchrl.envs.libs.gym import GymEnv
>>> from tensordict.nn import TensorDictModule
>>> from torch import nn
>>> from torchrl.collectors import MultiSyncCollector
>>> if __name__ == "__main__":
... env_maker = lambda: GymEnv("Pendulum-v1", device="cpu")
... policy = TensorDictModule(nn.Linear(3, 1), in_keys=["observation"], out_keys=["action"])
... collector = MultiSyncCollector(
... create_env_fn=[env_maker, env_maker],
... policy=policy,
... total_frames=2000,
... max_frames_per_traj=50,
... frames_per_batch=200,
... init_random_frames=-1,
... reset_at_each_iter=False,
... device="cpu",
... storing_device="cpu",
... cat_results="stack",
... )
... for i, data in enumerate(collector):
... if i == 2:
... print(data)
... break
... collector.shutdown()
... del collector
TensorDict(
 fields={
 action: Tensor(shape=torch.Size([200, 1]), device=cpu, dtype=torch.float32, is_shared=False),
 collector: TensorDict(
 fields={
 traj_ids: Tensor(shape=torch.Size([200]), device=cpu, dtype=torch.int64, is_shared=False)},
 batch_size=torch.Size([200]),
 device=cpu,
 is_shared=False),
 done: Tensor(shape=torch.Size([200, 1]), device=cpu, dtype=torch.bool, is_shared=False),
 next: TensorDict(
 fields={
 done: Tensor(shape=torch.Size([200, 1]), device=cpu, dtype=torch.bool, is_shared=False),
 observation: Tensor(shape=torch.Size([200, 3]), device=cpu, dtype=torch.float32, is_shared=False),
 reward: Tensor(shape=torch.Size([200, 1]), device=cpu, dtype=torch.float32, is_shared=False),
 step_count: Tensor(shape=torch.Size([200, 1]), device=cpu, dtype=torch.int64, is_shared=False),
 truncated: Tensor(shape=torch.Size([200, 1]), device=cpu, dtype=torch.bool, is_shared=False)},
 batch_size=torch.Size([200]),
 device=cpu,
 is_shared=False),
 observation: Tensor(shape=torch.Size([200, 3]), device=cpu, dtype=torch.float32, is_shared=False),
 step_count: Tensor(shape=torch.Size([200, 1]), device=cpu, dtype=torch.int64, is_shared=False),
 truncated: Tensor(shape=torch.Size([200, 1]), device=cpu, dtype=torch.bool, is_shared=False)},
 batch_size=torch.Size([200]),
 device=cpu,
 is_shared=False)
```

Runs a given number of DataCollectors on separate processes.

Parameters:

- **create_env_fn** (*List**[**Callabled**]*) - list of Callables, each returning an
instance of [`EnvBase`](torchrl.envs.EnvBase.html#torchrl.envs.EnvBase).
- **policy** (*Callable*) -

Policy to be executed in the environment.
Must accept `tensordict.tensordict.TensorDictBase` object as input.
If `None` is provided (default), the policy used will be a
`RandomPolicy` instance with the environment
`action_spec`.
Accepted policies are usually subclasses of [`TensorDictModuleBase`](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.nn.TensorDictModuleBase.html#tensordict.nn.TensorDictModuleBase).
This is the recommended usage of the collector.
Other callables are accepted too:
If the policy is not a `TensorDictModuleBase` (e.g., a regular [`Module`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module)
instances) it will be wrapped in a nn.Module first.
Then, the collector will try to assess if these
modules require wrapping in a [`TensorDictModule`](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.nn.TensorDictModule.html#tensordict.nn.TensorDictModule) or not.

- If the policy forward signature matches any of `forward(self, tensordict)`,
`forward(self, td)` or `forward(self, <anything>: TensorDictBase)` (or
any typing with a single argument typed as a subclass of `TensorDictBase`)
then the policy won't be wrapped in a [`TensorDictModule`](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.nn.TensorDictModule.html#tensordict.nn.TensorDictModule).
- In all other cases an attempt to wrap it will be undergone as such:
`TensorDictModule(policy, in_keys=env_obs_key, out_keys=env.action_keys)`.

Note

If the policy needs to be passed as a policy factory (e.g., in case it mustn't be serialized /
pickled directly), the `policy_factory` should be used instead.

Note

When using `weight_sync_schemes`, both `policy` and `policy_factory` can be provided together.
In this case, the `policy` is used ONLY for weight extraction (via `TensorDict.from_module()`) to
set up weight synchronization, but it is NOT sent to workers and its weights are NOT depopulated.
The `policy_factory` is what actually gets passed to workers to create their local policy instances.
This is useful when the policy is hard to serialize but you have a copy on the main node for
weight synchronization purposes.

Keyword Arguments:

- **sync** (*bool**,**optional*) - if `True`, the collector will run in sync mode (`MultiSyncCollector`). If
False, the collector will run in async mode ([`MultiAsyncCollector`](torchrl.collectors.MultiAsyncCollector.html#torchrl.collectors.MultiAsyncCollector)).
- **policy_factory** (*Callable**[**[**]**,**Callable**]**,*[*list*](torchrl.services.RayService.html#torchrl.services.RayService.list)*of**Callable**[**[**]**,**Callable**]**,**optional*) -

a callable
(or list of callables) that returns a policy instance.

When not using `weight_sync_schemes`, this is mutually exclusive with the `policy` argument.

When using `weight_sync_schemes`, both `policy` and `policy_factory` can be provided:
the `policy` is used for weight extraction only, while `policy_factory` creates policies on workers.

Note

policy_factory comes in handy whenever the policy cannot be serialized.

Warning

policy_factory is currently not compatible with multiprocessed data
collectors.
- **num_workers** (*int**,**optional*) - number of workers to use. If create_env_fn is a list, this will be ignored.
Defaults to None (workers determined by the create_env_fn length).
- **frames_per_batch** (*int**,**Sequence**[**int**]*) - A keyword-only argument representing the
total number of elements in a batch. If a sequence is provided, represents the number of elements in a
batch per worker. Total number of elements in a batch is then the sum over the sequence.
- **total_frames** (*int**,**optional*) - A keyword-only argument representing the
total number of frames returned by the collector
during its lifespan. If the `total_frames` is not divisible by
`frames_per_batch`, an exception is raised.
Endless collectors can be created by passing `total_frames=-1`.
Defaults to `-1` (never ending collector).
- **device** (*int**,**str**or*[*torch.device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)*,**optional*) - The generic device of the
collector. The `device` args fills any non-specified device: if
`device` is not `None` and any of `storing_device`, `policy_device` or
`env_device` is not specified, its value will be set to `device`.
Defaults to `None` (No default device).
Supports a list of devices if one wishes to indicate a different device
for each worker. The list must be as long as the number of workers.
- **storing_device** (*int**,**str**or*[*torch.device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)*,**optional*) - The device on which
the output [`TensorDict`](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDict.html#tensordict.TensorDict) will be stored.
If `device` is passed and `storing_device` is `None`, it will
default to the value indicated by `device`.
For long trajectories, it may be necessary to store the data on a different
device than the one where the policy and env are executed.
Defaults to `None` (the output tensordict isn't on a specific device,
leaf tensors sit on the device where they were created).
Supports a list of devices if one wishes to indicate a different device
for each worker. The list must be as long as the number of workers.
- **env_device** (*int**,**str**or*[*torch.device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)*,**optional*) - The device on which
the environment should be cast (or executed if that functionality is
supported). If not specified and the env has a non-`None` device,
`env_device` will default to that value. If `device` is passed
and `env_device=None`, it will default to `device`. If the value
as such specified of `env_device` differs from `policy_device`
and one of them is not `None`, the data will be cast to `env_device`
before being passed to the env (i.e., passing different devices to
policy and env is supported). Defaults to `None`.
Supports a list of devices if one wishes to indicate a different device
for each worker. The list must be as long as the number of workers.
- **policy_device** (*int**,**str**or*[*torch.device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)*,**optional*) - The device on which
the policy should be cast.
If `device` is passed and `policy_device=None`, it will default
to `device`. If the value as such specified of `policy_device`
differs from `env_device` and one of them is not `None`,
the data will be cast to `policy_device` before being passed to
the policy (i.e., passing different devices to policy and env is
supported). Defaults to `None`.
Supports a list of devices if one wishes to indicate a different device
for each worker. The list must be as long as the number of workers.
- **create_env_kwargs** (*dict**,**optional*) - A dictionary with the
keyword arguments used to create an environment. If a list is
provided, each of its elements will be assigned to a sub-collector.
- **collector_class** (*Python class**or**constructor*) - a collector class to be remotely instantiated. Can be
[`Collector`](torchrl.collectors.Collector.html#torchrl.collectors.Collector),
`MultiSyncCollector`,
[`MultiAsyncCollector`](torchrl.collectors.MultiAsyncCollector.html#torchrl.collectors.MultiAsyncCollector)
or a derived class of these.
Defaults to [`Collector`](torchrl.collectors.Collector.html#torchrl.collectors.Collector).
- **max_frames_per_traj** (*int**,**optional*) - Maximum steps per trajectory.
Note that a trajectory can span across multiple batches (unless
`reset_at_each_iter` is set to `True`, see below).
Once a trajectory reaches `n_steps`, the environment is reset.
If the environment wraps multiple environments together, the number
of steps is tracked for each environment independently. Negative
values are allowed, in which case this argument is ignored.
Defaults to `None` (i.e. no maximum number of steps).
- **init_random_frames** (*int**,**optional*) - Number of frames for which the
policy is ignored before it is called. This feature is mainly
intended to be used in offline/model-based settings, where a
batch of random trajectories can be used to initialize training.
If provided, it will be rounded up to the closest multiple of frames_per_batch.
Defaults to `None` (i.e. no random frames).
- **reset_at_each_iter** (*bool**,**optional*) - Whether environments should be reset
at the beginning of a batch collection.
Defaults to `False`.
- **postproc** (*Callable**,**optional*) - A post-processing transform, such as
a `Transform` or a `MultiStep`
instance.
Defaults to `None`.
- **split_trajs** (*bool**,**optional*) - Boolean indicating whether the resulting
TensorDict should be split according to the trajectories.
See [`split_trajectories()`](torchrl.collectors.utils.split_trajectories.html#torchrl.collectors.utils.split_trajectories) for more
information.
Defaults to `False`.
- **exploration_type** (*ExplorationType**,**optional*) - interaction mode to be used when
collecting data. Must be one of `torchrl.envs.utils.ExplorationType.DETERMINISTIC`,
`torchrl.envs.utils.ExplorationType.RANDOM`, `torchrl.envs.utils.ExplorationType.MODE`
or `torchrl.envs.utils.ExplorationType.MEAN`.
- **reset_when_done** (*bool**,**optional*) - if `True` (default), an environment
that return a `True` value in its `"done"` or `"truncated"`
entry will be reset at the corresponding indices.
- **update_at_each_batch** (*boolm optional*) - if `True`, `update_policy_weights_()`
will be called before (sync) or after (async) each data collection.
Defaults to `False`.
- **preemptive_threshold** (`float`, optional) - a value between 0.0 and 1.0 that specifies the ratio of workers
that will be allowed to finished collecting their rollout before the rest are forced to end early.
- **num_threads** (*int**,**optional*) - number of threads for this process.
Defaults to the number of workers.
- **num_sub_threads** (*int**,**optional*) - number of threads of the subprocesses.
Should be equal to one plus the number of processes launched within
each subprocess (or one if a single process is launched).
Defaults to 1 for safety: if none is indicated, launching multiple
workers may charge the cpu load too much and harm performance.
- **cat_results** (*str**,**int**or**None*) -

(`MultiSyncCollector` exclusively).
If `"stack"`, the data collected from the workers will be stacked along the
first dimension. This is the preferred behavior as it is the most compatible
with the rest of the library.
If `0`, results will be concatenated along the first dimension
of the outputs, which can be the batched dimension if the environments are
batched or the time dimension if not.
A `cat_results` value of `-1` will always concatenate results along the
time dimension. This should be preferred over the default. Intermediate values
are also accepted.
Defaults to `"stack"`.

Note

From v0.5, this argument will default to `"stack"` for a better
interoperability with the rest of the library.
- **set_truncated** (*bool**,**optional*) - if `True`, the truncated signals (and corresponding
`"done"` but not `"terminated"`) will be set to `True` when the last frame of
a rollout is reached. If no `"truncated"` key is found, an exception is raised.
Truncated keys can be set through `env.add_truncated_keys`.
Defaults to `False`.
- **trajs_per_batch** (*int**,**optional*) -

When set together with `replay_buffer`,
trajectory assembly is delegated to each worker's inner
[`Collector`](torchrl.collectors.Collector.html#torchrl.collectors.Collector). Each worker calls
`_iter_by_trajectories()`
independently and writes **complete trajectories** (episodes whose
last step has `("next", "done") == True`) to the shared replay
buffer as flat 1-D sequences -- no padding, no accumulation.

When set *without* `replay_buffer`, each worker assembles
trajectories and the multi-collector yields zero-padded batches
of shape `(trajs_per_batch, max_traj_len)` with a
`("collector", "mask")` boolean field.

Both the iteration pattern (`for data in collector`) and the
async `start()` pattern are supported.

Defaults to `None` (fixed-frame batches).

See [`BaseCollector`](torchrl.collectors.BaseCollector.html#torchrl.collectors.BaseCollector) for the full
description of the completeness guarantee and replay-buffer
storage contract.
- **use_buffers** (*bool**,**optional*) - if `True`, a buffer will be used to stack the data.
This isn't compatible with environments with dynamic specs. Defaults to `True`
for envs without dynamic specs, `False` for others.
- **replay_buffer** ([*ReplayBuffer*](torchrl.data.ReplayBuffer.html#torchrl.data.ReplayBuffer)*,**optional*) - if provided, the collector will not yield tensordicts
but populate the buffer instead. Defaults to `None`.
- **extend_buffer** (*bool**,**optional*) - if True, the replay buffer is extended with entire rollouts and not
with single steps. Defaults to True for multiprocessed data collectors.
- **trust_policy** (*bool**,**optional*) - if `True`, a non-TensorDictModule policy will be trusted to be
assumed to be compatible with the collector. This defaults to `True` for CudaGraphModules
and `False` otherwise.
- **compile_policy** (*bool**or**Dict**[**str**,**Any**]**,**optional*) - if `True`, the policy will be compiled
using [`compile()`](https://docs.pytorch.org/docs/stable/generated/torch.compile.html#torch.compile) default behaviour. If a dictionary of kwargs is passed, it
will be used to compile the policy.
- **cudagraph_policy** (*bool**or**Dict**[**str**,**Any**]**,**optional*) - if `True`, the policy will be wrapped
in [`CudaGraphModule`](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.nn.CudaGraphModule.html#tensordict.nn.CudaGraphModule) with default kwargs.
If a dictionary of kwargs is passed, it will be used to wrap the policy.
- **no_cuda_sync** (*bool*) - if `True`, explicit CUDA synchronizations calls will be bypassed.
For environments running directly on CUDA ([IsaacLab](https://github.com/isaac-sim/IsaacLab/)
or [ManiSkills](https://github.com/haosulab/ManiSkill/)) cuda synchronization may cause unexpected
crashes.
Defaults to `False`.
- **auto_register_policy_transforms** (*bool**,**optional*) - forwarded to each
worker [`Collector`](torchrl.collectors.Collector.html#torchrl.collectors.Collector). When `True`,
workers append [`InitTracker`](torchrl.envs.transforms.InitTracker.html#torchrl.envs.transforms.InitTracker) and
recurrent-state [`TensorDictPrimer`](torchrl.envs.transforms.TensorDictPrimer.html#torchrl.envs.transforms.TensorDictPrimer)
transforms to their envs if the env specs don't already provide
them. `False` disables it; `None` (default through v0.14)
preserves pre-v0.15 behavior and emits a
`FutureWarning` when wrapping would have been needed.
Default flips to `True` in v0.15.
- **weight_updater** ([*WeightUpdaterBase*](torchrl.collectors.WeightUpdaterBase.html#torchrl.collectors.WeightUpdaterBase)*or**constructor**,**optional*) - An instance of [`WeightUpdaterBase`](torchrl.collectors.WeightUpdaterBase.html#torchrl.collectors.WeightUpdaterBase)
or its subclass, responsible for updating the policy weights on remote inference workers.
If not provided, a [`MultiProcessedWeightUpdater`](torchrl.collectors.MultiProcessedWeightUpdater.html#torchrl.collectors.MultiProcessedWeightUpdater) will be used by default,
which handles weight synchronization across multiple processes.
Consider using a constructor if the updater needs to be serialized.
- **weight_sync_schemes** (*dict**[**str**,*[*WeightSyncScheme*](torchrl.weight_update.WeightSyncScheme.html#torchrl.weight_update.WeightSyncScheme)*]**,**optional*) - Dictionary of weight sync schemes for
SENDING weights to worker sub-collectors. Keys are model identifiers (e.g., "policy")
and values are WeightSyncScheme instances configured to send weights to child processes.
If not provided, a `MultiProcessWeightSyncScheme` will be used by default.
This is for propagating weights DOWN the hierarchy (parent -> children).
- **weight_recv_schemes** (*dict**[**str**,*[*WeightSyncScheme*](torchrl.weight_update.WeightSyncScheme.html#torchrl.weight_update.WeightSyncScheme)*]**,**optional*) - Dictionary of weight sync schemes for
RECEIVING weights from parent collectors. Keys are model identifiers (e.g., "policy")
and values are WeightSyncScheme instances configured to receive weights.
This enables cascading in hierarchies like: RPCCollector -> MultiSyncCollector -> Collector.
Received weights are automatically propagated to sub-collectors if matching model_ids exist.
Defaults to `None`.
- **track_policy_version** (*bool**or*[*PolicyVersion*](torchrl.envs.llm.transforms.PolicyVersion.html#torchrl.envs.llm.transforms.PolicyVersion)*,**optional*) -

if `True`, the collector will track the version of the policy.
A [`PolicyVersion`](torchrl.envs.llm.transforms.PolicyVersion.html#torchrl.envs.llm.transforms.PolicyVersion) transform is
installed on each worker's environment, tagging every collected frame with the
current version under the `"policy_version"` key. Each worker's transform is
bumped after the new weights have actually been applied in that worker, so
per-frame tagging tracks real weight updates rather than rollout iterations.

Note that in asynchronous mode a batch that was already in flight when
`update_policy_weights_()` is called may straddle the bump (some frames
tagged with the old version, the remainder with the new). Treat the value as
the version under which each individual frame was produced, not as a batch-level
label.

For multi-process collectors, the `"policy_version"` entries in the
collected tensordict are produced by worker-local transforms and are the
source of truth for data provenance. The parent collector's
`policy_version` property exposes only the parent-side tracker state
and should not be used as a label for a returned batch.

The recommended path is `track_policy_version=True`: let the collector own
the transform. Passing a [`PolicyVersion`](torchrl.envs.llm.transforms.PolicyVersion.html#torchrl.envs.llm.transforms.PolicyVersion)
instance directly is reserved for advanced use cases that wire up a
`PolicyVersion` **without** going through a collector. With multi-process
collectors that pre-built tracker lives in the *parent* and is not propagated
into workers, so per-frame tagging will still be driven by per-worker
transforms -- favor `True`.

Defaults to `False`.
- **compact_obs** (*bool**,**optional*) - if `True`, each worker drops the
observation and state keys from the `("next", ...)` sub-tensordict
before stacking. See
[`Collector`](torchrl.collectors.Collector.html#torchrl.collectors.Collector) for the full
explanation and tradeoffs (most notably:
`MultiStepTransform` cannot be used
in compact mode), plus the pairing with
[`NextStateReconstructor`](torchrl.envs.transforms.NextStateReconstructor.html#torchrl.envs.transforms.NextStateReconstructor) at
sampling time, the boundary-preserving lossy alternative
[`NextObservationDelta`](torchrl.envs.transforms.NextObservationDelta.html#torchrl.envs.transforms.NextObservationDelta), and the
*Memory-efficient RL training* tutorial. Defaults to `False`.
- **worker_idx** (*int**,**optional*) - the index of the worker.

Examples

```
>>> from torchrl.collectors import MultiCollector
>>> from torchrl.envs import GymEnv
>>>
>>> def make_env():
... return GymEnv("CartPole-v1")
>>>
>>> # Synchronous collection (for on-policy algorithms like PPO)
>>> sync_collector = MultiCollector(
... create_env_fn=[make_env] * 4, # 4 parallel workers
... policy=my_policy,
... frames_per_batch=1000,
... total_frames=100000,
... sync=True, # All workers complete before batch is delivered
... )
>>>
>>> # Asynchronous collection (for off-policy algorithms like SAC)
>>> async_collector = MultiCollector(
... create_env_fn=[make_env] * 4,
... policy=my_policy,
... frames_per_batch=1000,
... total_frames=100000,
... sync=False, # First-come-first-serve delivery
... )
>>>
>>> # Iterate over collected data
>>> for data in sync_collector:
... # data is a TensorDict with collected transitions
... pass
>>> sync_collector.shutdown()
```

async_shutdown(*timeout: float | None = None*)

Shuts down the collector when started asynchronously with the start method.

Parameters:

- **timeout** (*float**,**optional*) - The maximum time to wait for the collector to shutdown.
- **close_env** (*bool**,**optional*) - If True, the collector will close the contained environment.
Defaults to True.

See also

`start()`

cascade_execute(*attr_path: str*, **args*, ***kwargs*) → Any

Execute a method on a nested attribute of this collector.

This method allows remote callers to invoke methods on nested attributes
of the collector without needing to know the full structure. It's particularly
useful for calling methods on weight sync schemes from the sender side.

Parameters:

- **attr_path** - Full path to the callable, e.g.,
"_receiver_schemes['model_id']._set_dist_connection_info"
- ***args** - Positional arguments to pass to the method.
- ****kwargs** - Keyword arguments to pass to the method.

Returns:

The return value of the method call.

Examples

```
>>> collector.cascade_execute(
... "_receiver_schemes['policy']._set_dist_connection_info",
... connection_info_ref,
... worker_idx=0
... )
```

disable_profile() → None

Stop any in-flight profiler and restore the prior `post_collect_hook`.

Safe to call when profiling was never enabled (becomes a no-op). When
the profiler was already self-stopped after `num_rollouts`, this just
clears the hook and restores any user-set `post_collect_hook`.

enable_profile(***, *workers: list[int] | None = None*, *num_rollouts: int = 3*, *warmup_rollouts: int = 1*, *save_path: str | Path | None = None*, *activities: list[str] | None = None*, *record_shapes: bool = True*, *profile_memory: bool = False*, *with_stack: bool = True*, *with_flops: bool = False*, *on_trace_ready: Callable | None = None*) → None

Enable profiling for collector worker rollouts.

This method configures the collector to profile rollouts using PyTorch's
profiler. For multi-process collectors, profiling happens in the worker
processes. For single-process collectors (Collector), profiling happens
in the main process.

Parameters:

- **workers** - List of worker indices to profile. Defaults to [0].
For single-process collectors, this is ignored.
- **num_rollouts** - Total number of rollouts to run the profiler for
(including warmup). Profiling stops after this many rollouts.
Defaults to 3.
- **warmup_rollouts** - Number of rollouts to skip before starting actual
profiling. Useful for JIT/compile warmup. The profiler runs
but discards data during warmup. Defaults to 1.
- **save_path** - Path to save the profiling trace. Supports {worker_idx}
placeholder for worker-specific files. If None, traces are
saved to "./collector_profile_{worker_idx}.json".
- **activities** - List of profiler activities ("cpu", "cuda").
Defaults to ["cpu", "cuda"].
- **record_shapes** - Whether to record tensor shapes. Defaults to True.
- **profile_memory** - Whether to profile memory usage. Defaults to False.
- **with_stack** - Whether to record Python stack traces. Defaults to True.
- **with_flops** - Whether to compute FLOPS. Defaults to False.
- **on_trace_ready** - Optional callback when trace is ready. If None,
traces are exported to Chrome trace format at save_path.

Raises:

- **RuntimeError** - If called after iteration has started.
- **ValueError** - If num_rollouts <= warmup_rollouts.

Example

```
>>> from torchrl.collectors import MultiSyncCollector
>>> collector = MultiSyncCollector(
... create_env_fn=[make_env] * 4,
... policy=policy,
... frames_per_batch=1000,
... total_frames=100000,
... )
>>> collector.enable_profile(
... workers=[0],
... num_rollouts=5,
... warmup_rollouts=2,
... save_path="./traces/worker_{worker_idx}.json",
... )
>>> # Worker 0 will be profiled for rollouts 2, 3, 4
>>> for data in collector:
... train(data)
>>> collector.shutdown()
```

Note

- Profiling adds overhead, so only profile specific workers
- The trace file can be viewed in Chrome's trace viewer
(chrome://tracing) or with PyTorch's TensorBoard plugin
- For multi-process collectors, this must be called BEFORE
iteration starts as it needs to configure workers

fake_tensordict() → [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)

Not implemented for multi-process collectors.

Honoring the multi-collector contract here would require either
creating an env in the main process (which defeats the purpose of
a multi-process collector -- Isaac Lab / mujoco-mjx etc. can only
run in workers) or routing a request to a worker over the pipe
(which requires workers to be alive and adds protocol surface).
Neither is implemented; call [`fake_tensordict()`](torchrl.collectors.Collector.html#torchrl.collectors.Collector.fake_tensordict)
on a single [`Collector`](torchrl.collectors.Collector.html#torchrl.collectors.Collector) instead, or
build the template directly from the env spec.

get_cached_weights(*model_id: str*)

Get cached shared memory weights if available (for weight sync schemes).

Parameters:

**model_id** - Model identifier

Returns:

Cached TensorDict weights or None if not available

get_distant_attr(*attr: str*) → list[Any]

Get a nested attribute from each worker collector.

get_model(*model_id: str*)

Get model instance by ID (for weight sync schemes).

Parameters:

**model_id** - Model identifier (e.g., "policy", "value_net")

Returns:

The model instance

Raises:

**ValueError** - If model_id is not recognized

get_policy_version() → str | int | None

Get the parent-side policy version.

This method exists to support remote calls in Ray actors, since properties
cannot be accessed directly through Ray's RPC mechanism.

Returns:

The parent-side version number (int) or UUID (str), or `None` if
version tracking is disabled. For collected data, prefer the
per-frame `"policy_version"` tensor in returned batches.

getattr_env(*attr*)

Get an attribute from the environment of the first worker.

Parameters:

**attr** (*str*) - The attribute name to retrieve from the environment.

Returns:

The attribute value from the environment of the first worker.

Raises:

**AttributeError** - If the attribute doesn't exist on the environment.

getattr_policy(*attr*)

Get an attribute from the policy of the first worker.

Parameters:

**attr** (*str*) - The attribute name to retrieve from the policy.

Returns:

The attribute value from the policy of the first worker.

Raises:

**AttributeError** - If the attribute doesn't exist on the policy.

getattr_rb(*attr*)

Get an attribute from the replay buffer.

increment_version()

Increment the policy version.

init_updater(**args*, ***kwargs*)

Initialize the weight updater with custom arguments.

This method passes the arguments to the weight updater's init method.
If no weight updater is set, this is a no-op.

Parameters:

- ***args** - Positional arguments for weight updater initialization
- ****kwargs** - Keyword arguments for weight updater initialization

load_state_dict(*state_dict: OrderedDict*) → None[[source]](../../_modules/torchrl/collectors/_multi_sync.html#MultiSyncCollector.load_state_dict)

Loads the state_dict on the workers.

Parameters:

**state_dict** (*OrderedDict*) - state_dict of the form
`{"worker0": state_dict0, "worker1": state_dict1}`.

map_fn(*method_name: str*, *list_of_args: list[tuple] | None = None*, *list_of_kwargs: list[dict] | None = None*) → list[Any]

Apply a method to each worker collector.

pause()

Context manager that pauses the collector if it is running free.

*property*policy_version*: str | int | None*

The parent-side policy version.

For multi-process collectors, worker-local
[`PolicyVersion`](torchrl.envs.llm.transforms.PolicyVersion.html#torchrl.envs.llm.transforms.PolicyVersion)
transforms write the per-frame `"policy_version"` values in returned
batches. Those tensor entries are the source of truth for collected
data; this property is only the parent-side tracker state.

*property*post_collect_hook*: Callable[[[TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)], None] | None*

Get the post-collection hook.

Returns:

A callable to be executed after each rollout, receiving the collected
TensorDict as argument, or None.

*property*postprocs

use `postproc` instead. Will be removed in v0.14.

Type:

Deprecated

*property*pre_collect_hook*: Callable[[], None] | None*

Get the pre-collection hook.

Returns:

A callable to be executed before each rollout, or None.

*property*profile_config*: ProfileConfig | None*

Get the profiling configuration.

Returns:

ProfileConfig if profiling is enabled, None otherwise.

receive_weights(*policy_or_weights: [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase) | None = None*)[[source]](../../_modules/torchrl/collectors/_multi_sync.html#MultiSyncCollector.receive_weights)

Receive and apply weights to the collector's policy.

This method applies weights to the local policy. When receiver schemes are
registered, it delegates to those schemes. Otherwise, it directly applies
the provided weights.

The method accepts weights in multiple forms for convenience:

Examples

```
>>> # Receive from registered schemes (distributed collectors)
>>> collector.receive_weights()
>>>
>>> # Apply weights from a policy module (positional)
>>> collector.receive_weights(trained_policy)
>>>
>>> # Apply weights from a TensorDict (positional)
>>> collector.receive_weights(weights_tensordict)
>>>
>>> # Use keyword arguments for clarity
>>> collector.receive_weights(weights=weights_td)
>>> collector.receive_weights(policy=trained_policy)
```

Parameters:

**policy_or_weights** -

The weights to apply. Can be:

- `nn.Module`: A policy module whose weights will be extracted and applied
- `TensorDictModuleBase`: A TensorDict module whose weights will be extracted
- `TensorDictBase`: A TensorDict containing weights
- `dict`: A regular dict containing weights
- `None`: Receive from registered schemes or mirror from original policy

Keyword Arguments:

- **weights** - Alternative to positional argument. A TensorDict or dict containing
weights to apply. Cannot be used together with `policy_or_weights` or `policy`.
- **policy** - Alternative to positional argument. An `nn.Module` or `TensorDictModuleBase`
whose weights will be extracted. Cannot be used together with `policy_or_weights`
or `weights`.

Raises:

**ValueError** - If conflicting parameters are provided or if arguments are passed
 when receiver schemes are registered.

register_scheme_receiver(*weight_recv_schemes: dict[str, [WeightSyncScheme](torchrl.weight_update.WeightSyncScheme.html#torchrl.weight_update.WeightSyncScheme)]*, ***, *synchronize_weights: bool = True*)

Set up receiver schemes for this collector to receive weights from parent collectors.

This method initializes receiver schemes and stores them in _receiver_schemes
for later use by _receive_weights_scheme() and receive_weights().

Receiver schemes enable cascading weight updates across collector hierarchies:
- Parent collector sends weights via its weight_sync_schemes (senders)
- Child collector receives weights via its weight_recv_schemes (receivers)
- If child is also a parent (intermediate node), it can propagate to its own children

Parameters:

**weight_recv_schemes** (*dict**[**str**,*[*WeightSyncScheme*](torchrl.weight_update.WeightSyncScheme.html#torchrl.weight_update.WeightSyncScheme)*]*) - Dictionary of {model_id: WeightSyncScheme} to set up as receivers.
These schemes will receive weights from parent collectors.

Keyword Arguments:

**synchronize_weights** (*bool**,**optional*) - If True, synchronize weights immediately after registering the schemes.
Defaults to True.

reset(*reset_idx: Sequence[bool] | None = None*) → None

Resets the environments to a new initial state.

Parameters:

**reset_idx** - Optional. Sequence indicating which environments have
to be reset. If None, all environments are reset.

set_post_collect_hook(*hook: Callable[[[TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)], None] | None*) → None

Method form of the `post_collect_hook` setter.

Exposed because Ray actor handles can call methods (actor.method.remote(...))
but cannot directly invoke property setters. Keeping the actual setter
for in-process use and this method for remote-actor use.

set_seed(*seed: int*, *static_seed: bool = False*) → int[[source]](../../_modules/torchrl/collectors/_multi_sync.html#MultiSyncCollector.set_seed)

Sets the seeds of the environments stored in the DataCollector.

Parameters:

- **seed** - integer representing the seed to be used for the environment.
- **static_seed** (*bool**,**optional*) - if `True`, the seed is not incremented.
Defaults to False

Returns:

Output seed. This is useful when more than one environment is
contained in the DataCollector, as the seed will be incremented for
each of these. The resulting seed is the seed of the last
environment.

Examples

```
>>> from torchrl.envs import ParallelEnv
>>> from torchrl.envs.libs.gym import GymEnv
>>> from tensordict.nn import TensorDictModule
>>> from torch import nn
>>> env_fn = lambda: GymEnv("Pendulum-v1")
>>> env_fn_parallel = lambda: ParallelEnv(6, env_fn)
>>> policy = TensorDictModule(nn.Linear(3, 1), in_keys=["observation"], out_keys=["action"])
>>> collector = Collector(env_fn_parallel, policy, frames_per_batch=100, total_frames=300)
>>> out_seed = collector.set_seed(1) # out_seed = 6
```

shutdown(*timeout: float | None = None*, *close_env: bool = True*, *raise_on_error: bool = True*) → None[[source]](../../_modules/torchrl/collectors/_multi_sync.html#MultiSyncCollector.shutdown)

Shuts down all processes. This operation is irreversible.

Parameters:

- **timeout** (*float**,**optional*) - The timeout for closing pipes between workers.
- **close_env** (*bool**,**optional*) - Whether to close the environment. Defaults to True.
- **raise_on_error** (*bool**,**optional*) - Whether to raise an error if the shutdown fails. Defaults to True.

start()

Starts the collector(s) for asynchronous data collection.

The collected data is stored in the provided replay buffer. This method initiates the background collection of
data across multiple processes, allowing for decoupling of data collection and training.

Raises:

**RuntimeError** - If no replay buffer is defined during the collector's initialization.

Example

```
>>> from torchrl.modules import RandomPolicy >>> >>> import time
>>> from functools import partial
>>>
>>> import tqdm
>>>
>>> from torchrl.collectors import MultiAsyncCollector
>>> from torchrl.data import LazyTensorStorage, ReplayBuffer
>>> from torchrl.envs import GymEnv, set_gym_backend
>>> import ale_py
>>>
>>> # Set the gym backend to gymnasium
>>> set_gym_backend("gymnasium").set()
>>>
>>> if __name__ == "__main__":
... # Create a random policy for the Pong environment
... env_fn = partial(GymEnv, "ALE/Pong-v5")
... policy = RandomPolicy(env_fn().action_spec)
...
... # Initialize a shared replay buffer
... rb = ReplayBuffer(storage=LazyTensorStorage(10000), shared=True)
...
... # Create a multi-async data collector with 16 environments
... num_envs = 16
... collector = MultiAsyncCollector(
... [env_fn] * num_envs,
... policy=policy,
... replay_buffer=rb,
... frames_per_batch=num_envs * 16,
... total_frames=-1,
... )
...
... # Progress bar to track the number of collected frames
... pbar = tqdm.tqdm(total=100_000)
...
... # Start the collector asynchronously
... collector.start()
...
... # Track the write count of the replay buffer
... prec_wc = 0
... while True:
... wc = rb.write_count
... c = wc - prec_wc
... prec_wc = wc
...
... # Update the progress bar
... pbar.update(c)
... pbar.set_description(f"Write Count: {rb.write_count}")
...
... # Check the write count every 0.5 seconds
... time.sleep(0.5)
...
... # Stop when the desired number of frames is reached
... if rb.write_count . 100_000:
... break
...
... # Shut down the collector
... collector.async_shutdown()
```

state_dict() → OrderedDict[[source]](../../_modules/torchrl/collectors/_multi_sync.html#MultiSyncCollector.state_dict)

Returns the state_dict of the data collector.

Each field represents a worker containing its own state_dict.

update_policy_weights_(*policy_or_weights: [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase) | [TensorDictModuleBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.nn.TensorDictModuleBase.html#tensordict.nn.TensorDictModuleBase) | dict | None = None*, ***, *worker_ids: int | list[int] | [device](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device) | list[[device](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)] | None = None*, ***kwargs*) → None[[source]](../../_modules/torchrl/collectors/_multi_sync.html#MultiSyncCollector.update_policy_weights_)

Update policy weights for the data collector.

This method synchronizes the policy weights used by the collector with the latest
trained weights. It supports both local and remote weight updates, depending on
the collector configuration.

The method accepts weights in multiple forms for convenience:

Examples

```
>>> # Pass policy module as positional argument
>>> collector.update_policy_weights_(policy_module)
>>>
>>> # Pass TensorDict weights as positional argument
>>> collector.update_policy_weights_(weights_tensordict)
>>>
>>> # Use keyword arguments for clarity
>>> collector.update_policy_weights_(weights=weights_td, model_id="actor")
>>> collector.update_policy_weights_(policy=actor_module, model_id="actor")
>>>
>>> # Update multiple models atomically
>>> collector.update_policy_weights_(weights_dict={
... "actor": actor_weights,
... "critic": critic_weights,
... })
>>>
>>> # Per-worker weight updates (for distinct policy factories)
>>> # Each worker can have independently updated weights
>>> collector.update_policy_weights_({
... 0: worker_0_weights,
... 1: worker_1_weights,
... 2: worker_2_weights,
... })
```

Parameters:

**policy_or_weights** -

The weights to update with. Can be:

- `nn.Module`: A policy module whose weights will be extracted
- `TensorDictModuleBase`: A TensorDict module whose weights will be extracted
- `TensorDictBase`: A TensorDict containing weights
- `dict`: A regular dict containing weights
- `dict[int, TensorDictBase]`: Per-worker weights where keys are worker indices.
This is used with distinct policy factories where each worker has independent weights.
- `None`: Will try to get weights from server using `_get_server_weights()`

Keyword Arguments:

- **weights** - Alternative to positional argument. A TensorDict or dict containing
weights to update. Cannot be used together with `policy_or_weights` or `policy`.
- **policy** - Alternative to positional argument. An `nn.Module` or `TensorDictModuleBase`
whose weights will be extracted. Cannot be used together with `policy_or_weights`
or `weights`.
- **worker_ids** - Identifiers for the workers to update. Relevant when the collector
has multiple workers. Can be int, list of ints, device, or list of devices.
- **model_id** - The model identifier to update (default: `"policy"`).
Cannot be used together with `weights_dict`.
- **weights_dict** - Dictionary mapping model_id to weights for updating
multiple models atomically. Keys should match model_ids registered in
`weight_sync_schemes`. Cannot be used together with `model_id`,
`policy_or_weights`, `weights`, or `policy`.

Raises:

- **TypeError** - If `worker_ids` is provided but no `weight_updater` is configured.
- **ValueError** - If conflicting parameters are provided.

Note

Users should extend the `WeightUpdaterBase` classes to customize
the weight update logic for specific use cases.

See also

`LocalWeightsUpdaterBase` and
`RemoteWeightsUpdaterBase()`.

*property*worker_idx*: int | None*

Get the worker index for this collector.

Returns:

The worker index (0-indexed).

Raises:

**RuntimeError** - If worker_idx has not been set.