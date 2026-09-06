# DistributedSyncCollector

*class*torchrl.collectors.distributed.DistributedSyncCollector(*create_env_fn, policy: Callable[[TensorDictBase], TensorDictBase] | None = None, *, policy_factory: Callable[[], Callable] | list[Callable[[], Callable]] | None = None, frames_per_batch: int, total_frames: int = -1, device: torch.device | list[torch.device] = None, storing_device: torch.device | list[torch.device] = None, env_device: torch.device | list[torch.device] = None, policy_device: torch.device | list[torch.device] = None, max_frames_per_traj: int = -1, init_random_frames: int = -1, reset_at_each_iter: bool = False, postproc: Callable | None = None, split_trajs: bool = False, exploration_type: ExporationType = InteractionType.RANDOM, collector_class: type | Callable[[], BaseCollector] = <class 'torchrl.collectors._single.Collector'>, collector_kwargs: dict[str, Any] | None = None, num_workers_per_collector: int = 1, slurm_kwargs: dict[str, Any] | None = None, backend: Literal['gloo', 'nccl'] = 'gloo', max_weight_update_interval: int = -1, update_interval: int = 1, launcher: str = 'submitit', tcp_port: str | None = None*)[[source]](../../_modules/torchrl/collectors/distributed/sync.html#DistributedSyncCollector)

A distributed synchronous data collector with torch.distributed backend.

Parameters:

- **create_env_fn** (*Callable**or**List**[**Callabled**]*) - list of Callables, each returning an
instance of [`EnvBase`](torchrl.envs.EnvBase.html#torchrl.envs.EnvBase).
- **policy** (*Callable*) -

Policy to be executed in the environment.
Must accept `tensordict.tensordict.TensorDictBase` object as input.
If `None` is provided, the policy used will be a
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
- In all other cases an attempt to wrap it will be undergone as such: `TensorDictModule(policy, in_keys=env_obs_key, out_keys=env.action_keys)`.

Note

If the policy needs to be passed as a policy factory (e.g., in case it mustn't be serialized /
pickled directly), the `policy_factory` should be used instead.

Keyword Arguments:

- **policy_factory** (*Callable**[**[**]**,**Callable**]**,*[*list*](torchrl.services.RayService.html#torchrl.services.RayService.list)*of**Callable**[**[**]**,**Callable**]**,**optional*) -

a callable
(or list of callables) that returns a policy instance. This is exclusive with the policy argument.

Note

policy_factory comes in handy whenever the policy cannot be serialized.
- **frames_per_batch** (*int*) - A keyword-only argument representing the total
number of elements in a batch.
- **total_frames** (*int*) - A keyword-only argument representing the total
number of frames returned by the collector
during its lifespan. If the `total_frames` is not divisible by
`frames_per_batch`, an exception is raised.
Endless collectors can be created by passing `total_frames=-1`.
Defaults to `-1` (endless collector).
- **device** (*int**,**str**or*[*torch.device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)*,**optional*) - The generic device of the
collector. The `device` args fills any non-specified device: if
`device` is not `None` and any of `storing_device`, `policy_device` or
`env_device` is not specified, its value will be set to `device`.
Defaults to `None` (No default device).
Lists of devices are supported.
- **storing_device** (*int**,**str**or*[*torch.device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)*,**optional*) - The *remote* device on which
the output [`TensorDict`](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDict.html#tensordict.TensorDict) will be stored.
If `device` is passed and `storing_device` is `None`, it will
default to the value indicated by `device`.
For long trajectories, it may be necessary to store the data on a different
device than the one where the policy and env are executed.
Defaults to `None` (the output tensordict isn't on a specific device,
leaf tensors sit on the device where they were created).
Lists of devices are supported.
- **env_device** (*int**,**str**or*[*torch.device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)*,**optional*) - The *remote* device on which
the environment should be cast (or executed if that functionality is
supported). If not specified and the env has a non-`None` device,
`env_device` will default to that value. If `device` is passed
and `env_device=None`, it will default to `device`. If the value
as such specified of `env_device` differs from `policy_device`
and one of them is not `None`, the data will be cast to `env_device`
before being passed to the env (i.e., passing different devices to
policy and env is supported). Defaults to `None`.
Lists of devices are supported.
- **policy_device** (*int**,**str**or*[*torch.device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)*,**optional*) - The *remote* device on which
the policy should be cast.
If `device` is passed and `policy_device=None`, it will default
to `device`. If the value as such specified of `policy_device`
differs from `env_device` and one of them is not `None`,
the data will be cast to `policy_device` before being passed to
the policy (i.e., passing different devices to policy and env is
supported). Defaults to `None`.
Lists of devices are supported.
- **max_frames_per_traj** (*int**,**optional*) - Maximum steps per trajectory.
Note that a trajectory can span across multiple batches (unless
`reset_at_each_iter` is set to `True`, see below).
Once a trajectory reaches `n_steps`, the environment is reset.
If the environment wraps multiple environments together, the number
of steps is tracked for each environment independently. Negative
values are allowed, in which case this argument is ignored.
Defaults to `None` (i.e., no maximum number of steps).
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
- **collector_class** (*Type**or**str**,**optional*) - a collector class for the remote node. Can be
[`Collector`](torchrl.collectors.Collector.html#torchrl.collectors.Collector),
[`MultiSyncCollector`](torchrl.collectors.MultiSyncCollector.html#torchrl.collectors.MultiSyncCollector),
[`MultiAsyncCollector`](torchrl.collectors.MultiAsyncCollector.html#torchrl.collectors.MultiAsyncCollector)
or a derived class of these. The strings "single", "sync" and
"async" correspond to respective class.
Defaults to [`Collector`](torchrl.collectors.Collector.html#torchrl.collectors.Collector).
- **collector_kwargs** (*dict**or*[*list*](torchrl.services.RayService.html#torchrl.services.RayService.list)*,**optional*) - a dictionary of parameters to be passed to the
remote data-collector. If a list is provided, each element will
correspond to an individual set of keyword arguments for the
dedicated collector.
- **num_workers_per_collector** (*int**,**optional*) - the number of copies of the
env constructor that is to be used on the remote nodes.
Defaults to 1 (a single env per collector).
On a single worker node all the sub-workers will be
executing the same environment. If different environments need to
be executed, they should be dispatched across worker nodes, not
subnodes.
- **slurm_kwargs** (*dict*) - a dictionary of parameters to be passed to the
submitit executor.
- **backend** (*str**,**optional*) - must a string "<distributed_backed>" where
<distributed_backed> is one of `"gloo"`, `"mpi"`, `"nccl"` or `"ucc"`. See
the torch.distributed documentation for more information.
Defaults to `"gloo"`.
- **max_weight_update_interval** (*int**,**optional*) - the maximum number of
batches that can be collected before the policy weights of a worker
is updated.
For sync collections, this parameter is overwritten by `update_after_each_batch`.
For async collections, it may be that one worker has not seen its
parameters being updated for a certain time even if `update_after_each_batch`
is turned on.
Defaults to -1 (no forced update).
- **update_interval** (*int**,**optional*) - the frequency at which the policy is
updated. Defaults to 1.
- **launcher** (*str**,**optional*) - how jobs should be launched.
Can be one of "submitit" or "mp" for multiprocessing. The former
can launch jobs across multiple nodes, whilst the latter will only
launch jobs on a single machine. "submitit" requires the homonymous
library to be installed.
To find more about submitit, visit
[facebookincubator/submitit](https://github.com/facebookincubator/submitit)
Defaults to "submitit".
- **tcp_port** (*int**,**optional*) - the TCP port to be used. Defaults to 10003.

async_shutdown(*timeout: float | None = None*, *close_env: bool = True*) → None

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

get_distant_attr(*attr: str*) → Any

Get a nested attribute of this collector.

This method allows remote callers to retrieve attributes from nested
structures of the collector without needing to know the full structure.

Parameters:

**attr** - Full path to the attribute, e.g.,
"_receiver_schemes['model_id'].some_attribute"

Returns:

The value of the attribute.

Examples

```
>>> collector.get_distant_attr("_receiver_schemes['policy']._sync_interval")
```

init_updater(**args*, ***kwargs*)

Initialize the weight updater with custom arguments.

This method passes the arguments to the weight updater's init method.
If no weight updater is set, this is a no-op.

Parameters:

- ***args** - Positional arguments for weight updater initialization
- ****kwargs** - Keyword arguments for weight updater initialization

map_fn(*method_name: str*, *list_of_args: list[tuple] | None = None*, *list_of_kwargs: list[dict] | None = None*) → list[Any]

Apply a method to each set of arguments.

This method executes a method on the collector with different arguments,
returning a list of results.

Parameters:

- **method_name** - Name of the method to call on the collector.
- **list_of_args** - List of positional argument tuples. Each tuple
contains the arguments for one call.
- **list_of_kwargs** - List of keyword argument dicts. Each dict
contains the kwargs for one call.

Returns:

List of return values from each method call.

Examples

```
>>> # Call a method with different arguments
>>> collector.map_fn("update_policy_weights_", list_of_args=[(weights1,), (weights2,)])
>>>
>>> # Call with kwargs
>>> collector.map_fn("update_policy_weights_", list_of_kwargs=[{"weights": w1}, {"weights": w2}])
```

pause()

Context manager that pauses the collector if it is running free.

*property*post_collect_hook*: Callable[[[TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)], None] | None*

Get the post-collection hook.

Returns:

A callable to be executed after each rollout, receiving the collected
TensorDict as argument, or None.

*property*pre_collect_hook*: Callable[[], None] | None*

Get the pre-collection hook.

Returns:

A callable to be executed before each rollout, or None.

*property*profile_config*: ProfileConfig | None*

Get the profiling configuration.

Returns:

ProfileConfig if profiling is enabled, None otherwise.

receive_weights(*policy_or_weights: [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase) | [TensorDictModuleBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.nn.TensorDictModuleBase.html#tensordict.nn.TensorDictModuleBase) | [Module](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module) | dict | None = None*, ***, *weights: [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase) | dict | None = None*, *policy: [TensorDictModuleBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.nn.TensorDictModuleBase.html#tensordict.nn.TensorDictModuleBase) | [Module](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module) | None = None*) → None

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

set_post_collect_hook(*hook: Callable[[[TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)], None] | None*) → None

Method form of the `post_collect_hook` setter.

Exposed because Ray actor handles can call methods (actor.method.remote(...))
but cannot directly invoke property setters. Keeping the actual setter
for in-process use and this method for remote-actor use.

start()

Starts the collector for asynchronous data collection.

This method initiates the background collection of data, allowing for decoupling of data collection and training.

The collected data is typically stored in a replay buffer passed during the collector's initialization.

Note

After calling this method, it's essential to shut down the collector using `async_shutdown()`
when you're done with it to free up resources.

Warning

Asynchronous data collection can significantly impact training performance due to its decoupled nature.
Ensure you understand the implications for your specific algorithm before using this mode.

Raises:

**NotImplementedError** - If not implemented by a subclass.

stats() → dict[str, int | float | bool]

Returns a cheap, serializable snapshot of the collector's progress.

The snapshot only contains scalar counters and gauges: it never
includes policy, environment or batch data, does not modify the
collector state and is safe to call while the collector is running.
Cumulative counters such as `frames` are meant to be converted into
rates by an external monitor such as
[`LoggerMonitor`](torchrl.record.loggers.monitoring.LoggerMonitor.html#torchrl.record.loggers.monitoring.LoggerMonitor).

Entries are only present when the corresponding state exists on the
collector:

- `"frames"`: total number of frames collected so far;
- `"batches"`: number of batches delivered so far;
- `"total_frames"`: requested total frames (absent for endless collectors);
- `"completed"`: whether the frame budget has been reached;
- `"requested_frames_per_batch"`: the per-batch frame budget;
- `"policy_version"`: current policy version, when the collector
tracks it with an integer version.

Multi-worker collectors extend this signature with a `workers`
argument controlling aggregate versus per-worker views.

Examples

```
>>> from torchrl.collectors import Collector
>>> from torchrl.envs import GymEnv
>>> from torchrl.envs.utils import RandomPolicy
>>> env = GymEnv("Pendulum-v1")
>>> collector = Collector(
... env,
... RandomPolicy(env.action_spec),
... frames_per_batch=10,
... total_frames=20,
... )
>>> for batch in collector:
... print(collector.stats()["frames"])
10
20
```

update_policy_weights_(*policy_or_weights: [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase) | None = None*, ***, *worker_ids=None*, *wait=True*, ***kwargs*) → None[[source]](../../_modules/torchrl/collectors/distributed/sync.html#DistributedSyncCollector.update_policy_weights_)

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