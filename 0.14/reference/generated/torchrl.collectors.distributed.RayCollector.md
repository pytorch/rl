# RayCollector

*class*torchrl.collectors.distributed.RayCollector(*create_env_fn: ~collections.abc.Callable | ~torchrl.envs.common.EnvBase | list[~collections.abc.Callable] | list[~torchrl.envs.common.EnvBase], policy: ~collections.abc.Callable[[~tensordict.base.TensorDictBase], ~tensordict.base.TensorDictBase] | None = None, *, policy_factory: ~collections.abc.Callable[[], ~collections.abc.Callable] | list[~collections.abc.Callable[[], ~collections.abc.Callable]] | None = None, trust_policy: bool | None = None, frames_per_batch: int, total_frames: int = -1, device: ~torch.device | list[~torch.device] | None = None, storing_device: ~torch.device | list[~torch.device] | None = None, env_device: ~torch.device | list[~torch.device] | None = None, policy_device: ~torch.device | list[~torch.device] | None = None, max_frames_per_traj=-1, init_random_frames=-1, reset_at_each_iter=False, postproc=None, split_trajs=False, exploration_type=InteractionType.RANDOM, collector_class: ~collections.abc.Callable[[~tensordict._td.TensorDict], ~tensordict._td.TensorDict] = <class 'torchrl.collectors._single.Collector'>, collector_kwargs: dict[str, ~typing.Any] | list[dict] | None = None, num_workers_per_collector: int = 1, sync: bool = False, ray_init_config: dict[str, ~typing.Any] | None = None, remote_configs: dict[str, ~typing.Any] | list[dict[str, ~typing.Any]] | None = None, num_collectors: int | None = None, update_after_each_batch: bool = False, max_weight_update_interval: int = -1, replay_buffer: ~torchrl.data.replay_buffers.replay_buffers.ReplayBuffer | None = None, weight_updater: ~torchrl.collectors.weight_update.WeightUpdaterBase | ~collections.abc.Callable[[], ~torchrl.collectors.weight_update.WeightUpdaterBase] | None = None, weight_sync_schemes: dict[str, ~torchrl.weight_update.weight_sync_schemes.WeightSyncScheme] | None = None, weight_recv_schemes: dict[str, ~torchrl.weight_update.weight_sync_schemes.WeightSyncScheme] | None = None, use_env_creator: bool = False, no_cuda_sync: bool | None = None, trajs_per_batch: int | None = None, pre_collect_hook: ~collections.abc.Callable[[], None] | None = None, post_collect_hook: ~collections.abc.Callable[[~tensordict.base.TensorDictBase], None] | None = None*)[[source]](../../_modules/torchrl/collectors/distributed/ray.html#RayCollector)

Distributed data collector with [Ray](https://docs.ray.io/) backend.

Note

Prefer `Collector(backend="ray", ...)` for construction in new code.
Pass Ray-specific arguments through `backend_options`. This class
remains the concrete Ray implementation API.

This Python class serves as a ray-based solution to instantiate and coordinate multiple
data collectors in a distributed cluster. Like TorchRL non-distributed collectors, this
collector is an iterable that yields TensorDicts until a target number of collected
frames is reached, but handles distributed data collection under the hood.

The class dictionary input parameter "ray_init_config" can be used to provide the kwargs to
call Ray initialization method ray.init(). If "ray_init_config" is not provided, the default
behavior is to autodetect an existing Ray cluster or start a new Ray instance locally if no
existing cluster is found. Refer to Ray documentation for advanced initialization kwargs.

Similarly, dictionary input parameter "remote_configs" can be used to specify the kwargs for
ray.remote() when called to create each remote collector actor, including collector compute
resources.The sum of all collector resources should be available in the cluster. Refer to Ray
documentation for advanced configuration of the ray.remote() method. Default kwargs are:

```
>>> kwargs = {
... "num_cpus": 1,
... "num_gpus": 0.2,
... "memory": 2 * 1024 ** 3,
... }
```

The coordination between collector instances can be specified as "synchronous" or "asynchronous".
In synchronous coordination, this class waits for all remote collectors to collect a rollout,
concatenates all rollouts into a single TensorDict instance and finally yields the concatenated
data. On the other hand, if the coordination is to be carried out asynchronously, this class
provides the rollouts as they become available from individual remote collectors.

Parameters:

- **create_env_fn** (*Callable**or**List**[**Callabled**]*) - list of Callables, each returning an
instance of [`EnvBase`](torchrl.envs.EnvBase.html#torchrl.envs.EnvBase).
- **policy** (*Callable**,**optional*) -

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

A Ray-owned
[`InferenceServer`](torchrl.modules.inference_server.InferenceServer.html#torchrl.modules.inference_server.InferenceServer) is also
accepted. In that case each collector worker receives an independent
restricted inference client and keeps no local policy copy.

Keyword Arguments:

- **policy_factory** (*Callable**[**[**]**,**Callable**]**,*[*list*](torchrl.services.RayService.html#torchrl.services.RayService.list)*of**Callable**[**[**]**,**Callable**]**,**optional*) -

a callable
(or list of callables) that returns a policy instance. This is exclusive with the policy argument.

Note

policy_factory comes in handy whenever the policy cannot be serialized.
- **trust_policy** (*bool**,**optional*) - if `True`, a non-TensorDictModule policy will be trusted to be
assumed to be compatible with the collector. This defaults to `True` for CudaGraphModules
and `False` otherwise.
- **frames_per_batch** (*int*) - A keyword-only argument representing the
total number of elements in a batch.
- **total_frames** (*int**,**Optional*) - lower bound of the total number of frames returned by the collector.
The iterator will stop once the total number of frames equates or exceeds the total number of
frames passed to the collector. Default value is -1, which mean no target total number of frames
(i.e. the collector will run indefinitely).
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
- **create_env_kwargs** (*dict**,**optional*) - Dictionary of kwargs for
`create_env_fn`.
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
- **collector_class** (*Python class**or**constructor*) - a collector class to be remotely instantiated. Can be
[`Collector`](torchrl.collectors.Collector.html#torchrl.collectors.Collector),
[`MultiSyncCollector`](torchrl.collectors.MultiSyncCollector.html#torchrl.collectors.MultiSyncCollector),
[`MultiAsyncCollector`](torchrl.collectors.MultiAsyncCollector.html#torchrl.collectors.MultiAsyncCollector)
or a derived class of these.
Defaults to [`Collector`](torchrl.collectors.Collector.html#torchrl.collectors.Collector).
- **collector_kwargs** (*dict**or*[*list*](torchrl.services.RayService.html#torchrl.services.RayService.list)*,**optional*) - a dictionary of parameters to be passed to the
remote data-collector. If a list is provided, each element will
correspond to an individual set of keyword arguments for the
dedicated collector.
- **num_workers_per_collector** (*int*) - the number of copies of the
env constructor that is to be used on the remote nodes.
Defaults to 1 (a single env per collector).
On a single worker node all the sub-workers will be
executing the same environment. If different environments need to
be executed, they should be dispatched across worker nodes, not
subnodes.
- **ray_init_config** (*dict**,**Optional*) - kwargs used to call ray.init().
- **remote_configs** ([*list*](torchrl.services.RayService.html#torchrl.services.RayService.list)*of**dicts**,**Optional*) - ray resource specs for each remote collector.
A single dict can be provided as well, and will be used in all collectors.
- **num_collectors** (*int**,**Optional*) - total number of collectors to be instantiated.
- **sync** (*bool*) - if `True`, the resulting tensordict is a stack of all the
tensordicts collected on each node. If `False` (default), each
tensordict results from a separate node in a "first-ready,
first-served" fashion.
- **update_after_each_batch** (*bool**,**optional*) - if `True`, the weights will
be updated after each collection. For `sync=True`, this means that
all workers will see their weights updated. For `sync=False`,
only the worker from which the data has been gathered will be
updated.
This is equivalent to max_weight_update_interval=0.
Defaults to `False`, i.e. updates have to be executed manually
through
[`torchrl.collectors.Collector.update_policy_weights_()`](torchrl.collectors.Collector.html#torchrl.collectors.Collector.update_policy_weights_)
- **max_weight_update_interval** (*int**,**optional*) - the maximum number of
batches that can be collected before the policy weights of a worker
is updated.
For sync collections, this parameter is overwritten by `update_after_each_batch`.
For async collections, it may be that one worker has not seen its
parameters being updated for a certain time even if `update_after_each_batch`
is turned on.
Defaults to -1 (no forced update).
- **replay_buffer** ([*ReplayBuffer*](torchrl.data.ReplayBuffer.html#torchrl.data.ReplayBuffer)*,**optional*) - if provided, the collector will
populate it instead of yielding TensorDicts. The replay buffer must
use `service_backend="ray"`; the collector creates restricted
worker clients internally. A regular in-process replay buffer is
rejected because serializing it into Ray actors would create remote
copies rather than populate the driver-owned buffer. For large,
fixed-layout TensorDict payloads, `transport="distributed"` is
the recommended data path (Gloo for CPU tensors and NCCL for CUDA
tensors). Defaults to `None`.
- **weight_updater** ([*WeightUpdaterBase*](torchrl.collectors.WeightUpdaterBase.html#torchrl.collectors.WeightUpdaterBase)*or**constructor**,**optional*) - (Deprecated) An instance of [`WeightUpdaterBase`](torchrl.collectors.WeightUpdaterBase.html#torchrl.collectors.WeightUpdaterBase)
or its subclass, responsible for updating the policy weights on remote inference workers managed by Ray.
If not provided, a [`RayWeightUpdater`](torchrl.collectors.RayWeightUpdater.html#torchrl.collectors.RayWeightUpdater) will be used by default, leveraging
Ray's distributed capabilities.
Consider using a constructor if the updater needs to be serialized.
- **weight_sync_schemes** (*dict**[**str**,*[*WeightSyncScheme*](torchrl.weight_update.WeightSyncScheme.html#torchrl.weight_update.WeightSyncScheme)*]**,**optional*) -

Dictionary of weight sync schemes for
SENDING weights to remote collector workers. Keys are model identifiers (e.g., "policy")
and values are WeightSyncScheme instances configured to send weights via Ray.
This is the recommended way to configure weight synchronization for propagating weights
from the main process to remote collectors. If not provided,
defaults to `{"policy": RayWeightSyncScheme()}`.

Note

Weight synchronization is lazily initialized. When using `policy_factory`
without a central `policy`, weight sync is deferred until the first call to
[`update_policy_weights_()`](torchrl.collectors.Collector.html#torchrl.collectors.Collector.update_policy_weights_) with actual weights.
This allows sub-collectors to each have their own independent policies created via
the factory. If you have a central policy and want to sync its weights to remote
collectors, call `update_policy_weights_(policy)` before starting iteration.
- **weight_recv_schemes** (*dict**[**str**,*[*WeightSyncScheme*](torchrl.weight_update.WeightSyncScheme.html#torchrl.weight_update.WeightSyncScheme)*]**,**optional*) - Dictionary of weight sync schemes for
RECEIVING weights from a parent process or training loop. Keys are model identifiers (e.g., "policy")
and values are WeightSyncScheme instances configured to receive weights.
This is typically used when RayCollector is itself a worker in a larger distributed setup.
Defaults to `None`.
- **use_env_creator** (*bool**,**optional*) - if `True`, the environment constructor functions will be wrapped
in [`EnvCreator`](torchrl.envs.EnvCreator.html#torchrl.envs.EnvCreator). This is useful for multiprocessed settings where shared memory
needs to be managed, but Ray has its own object storage mechanism, so this is typically not needed.
Defaults to `False`.
- **trajs_per_batch** (*int**,**optional*) -

When set, each remote collector
assembles complete trajectories (episodes ending with
`("next", "done") == True`) before writing them to the replay
buffer as flat 1-D sequences. Passed through to
`collector_kwargs` so that each worker's inner collector calls
`_iter_by_trajectories()`.

See [`BaseCollector`](torchrl.collectors.BaseCollector.html#torchrl.collectors.BaseCollector) for the full
description of the completeness guarantee and storage contract.
Defaults to `None`.

Examples

```
>>> from torch import nn
>>> from tensordict.nn import TensorDictModule
>>> from torchrl.envs.libs.gym import GymEnv
>>> from torchrl.collectors import Collector
>>> from torchrl.collectors.distributed import RayCollector
>>> env_maker = lambda: GymEnv("Pendulum-v1", device="cpu")
>>> policy = TensorDictModule(nn.Linear(3, 1), in_keys=["observation"], out_keys=["action"])
>>> distributed_collector = RayCollector(
... create_env_fn=[env_maker],
... policy=policy,
... collector_class=Collector,
... max_frames_per_traj=50,
... init_random_frames=-1,
... reset_at_each_iter=-False,
... collector_kwargs={
... "device": "cpu",
... "storing_device": "cpu",
... },
... num_collectors=1,
... total_frames=10000,
... frames_per_batch=200,
... )
>>> for i, data in enumerate(collector):
... if i == 2:
... print(data)
... break
```

add_collectors(*create_env_fn*, *num_envs*, *policy*, *collector_kwargs*, *remote_configs*)[[source]](../../_modules/torchrl/collectors/distributed/ray.html#RayCollector.add_collectors)

Creates and adds a number of remote collectors to the set.

*async*async_shutdown(*shutdown_ray: bool = False*)[[source]](../../_modules/torchrl/collectors/distributed/ray.html#RayCollector.async_shutdown)

Finishes processes started by the collector during async execution.

Parameters:

**shutdown_ray** (*bool*) - If True, also shutdown the Ray cluster. Defaults to False.
Note: Setting this to True will kill all Ray actors in the cluster, including
any replay buffers or other services. Only set to True if you're sure you want
to shut down the entire Ray cluster.

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

get_distant_attr(*attr: str*) → list[Any][[source]](../../_modules/torchrl/collectors/distributed/ray.html#RayCollector.get_distant_attr)

Get a nested attribute from each remote collector.

init_updater(**args*, ***kwargs*)

Initialize the weight updater with custom arguments.

This method passes the arguments to the weight updater's init method.
If no weight updater is set, this is a no-op.

Parameters:

- ***args** - Positional arguments for weight updater initialization
- ****kwargs** - Keyword arguments for weight updater initialization

load_state_dict(*state_dict: OrderedDict | list[OrderedDict]*) → None[[source]](../../_modules/torchrl/collectors/distributed/ray.html#RayCollector.load_state_dict)

Calls parent method for each remote collector.

local_policy()[[source]](../../_modules/torchrl/collectors/distributed/ray.html#RayCollector.local_policy)

Returns local collector.

map_fn(*method_name: str*, *list_of_args: list[tuple] | None = None*, *list_of_kwargs: list[dict] | None = None*) → list[Any][[source]](../../_modules/torchrl/collectors/distributed/ray.html#RayCollector.map_fn)

Apply a method to each remote collector.

pause(*timeout: float = 30.0*, ***, *resume: bool = True*) → Iterator[None][[source]](../../_modules/torchrl/collectors/distributed/ray.html#RayCollector.pause)

Pause background collection while the context is active.

Any in-flight actor requests are drained before entering the context.
The remote collectors remain alive and collection resumes when the
context exits unless `resume=False`. This provides a quiescent
boundary for checkpointing without changing the
[`BaseCollector`](torchrl.collectors.BaseCollector.html#torchrl.collectors.BaseCollector) context-manager contract.

Parameters:

- **timeout** (*float*) - Maximum time to wait for in-flight collection.
Defaults to `30.0` seconds.
- **resume** (*bool*) - Whether to resume collection when the context exits.
Defaults to `True`.

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

*property*remote_collectors

Returns list of remote collectors.

set_post_collect_hook(*hook: Callable[[[TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)], None] | None*) → None

Method form of the `post_collect_hook` setter.

Exposed because Ray actor handles can call methods (actor.method.remote(...))
but cannot directly invoke property setters. Keeping the actual setter
for in-process use and this method for remote-actor use.

set_seed(*seed: int*, *static_seed: bool = False*) → list[int][[source]](../../_modules/torchrl/collectors/distributed/ray.html#RayCollector.set_seed)

Calls parent method for each remote collector iteratively and returns final seed.

shutdown(*timeout: float | None = None*, *shutdown_ray: bool = False*) → None[[source]](../../_modules/torchrl/collectors/distributed/ray.html#RayCollector.shutdown)

Finishes processes started by the collector.

Parameters:

- **timeout** (*float**,**optional*) - Timeout for stopping the collection thread.
- **shutdown_ray** (*bool*) - If True, also shutdown the Ray cluster. Defaults to False.
Note: Setting this to True will kill all Ray actors in the cluster, including
any replay buffers or other services. Only set to True if you're sure you want
to shut down the entire Ray cluster.

start()[[source]](../../_modules/torchrl/collectors/distributed/ray.html#RayCollector.start)

Starts the RayCollector in a background thread.

state_dict() → list[OrderedDict][[source]](../../_modules/torchrl/collectors/distributed/ray.html#RayCollector.state_dict)

Calls parent method for each remote collector and returns a list of results.

stats(*workers: Literal['aggregate', 'per_worker', 'both'] = 'aggregate'*, ***, *timeout: float | None = 10.0*) → dict[str, int | float | bool][[source]](../../_modules/torchrl/collectors/distributed/ray.html#RayCollector.stats)

Returns a cheap, serializable snapshot of the collector's progress.

See [`stats()`](torchrl.collectors.BaseCollector.html#torchrl.collectors.BaseCollector.stats) for the general
contract. Worker snapshots are requested from all remote collectors
concurrently, one RPC per worker bounded by `timeout`; a worker
whose request fails or does not reply in time is counted as dead and
skipped. Note that, unlike multiprocessing collectors, every call
(including `workers="aggregate"`) contacts each remote collector to
derive `"workers_alive"` and `"worker_frames"`.

Parameters:

**workers** (*str**,**optional*) - controls the worker view. With
`"aggregate"` (default), the snapshot contains the
coordinator counters plus `"worker_frames"`, the sum of the
frame counters reported by the remote collectors. With
`"per_worker"`, each remote snapshot is namespaced as
`"worker_<idx>/<metric>"` instead. `"both"` returns the
union. `"workers"` and `"workers_alive"` are always
present.

Keyword Arguments:

**timeout** ([*float*](torchrl.data.llm.TopKRewardSelector.html#torchrl.data.llm.TopKRewardSelector.float)*,**optional*) - how long to wait for the worker
snapshots, in seconds, so that a hung worker cannot block the
caller (for example a monitoring thread) indefinitely.
`None` waits forever. Defaults to `10.0`.

The coordinator-side `"frames"` counter tracks frames dispatched
through the iterator. When remote collectors write directly into a
replay buffer, the buffer's `write_count` is the authoritative
production counter and `"worker_frames"` is the closest
collector-side estimate.

stop_remote_collectors()[[source]](../../_modules/torchrl/collectors/distributed/ray.html#RayCollector.stop_remote_collectors)

Stops all remote collectors.

update_policy_weights_(*policy_or_weights: [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase) | [TensorDictModuleBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.nn.TensorDictModuleBase.html#tensordict.nn.TensorDictModuleBase) | [Module](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module) | dict | None = None*, ***, *weights: [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase) | dict | None = None*, *policy: [TensorDictModuleBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.nn.TensorDictModuleBase.html#tensordict.nn.TensorDictModuleBase) | [Module](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module) | None = None*, *worker_ids: int | list[int] | [device](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device) | list[[device](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)] | None = None*, *model_id: str | None = None*, *weights_dict: dict[str, Any] | None = None*, ***kwargs*) → None

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