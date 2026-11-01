# AsyncBatchedCollector

*class*torchrl.collectors.AsyncBatchedCollector(*create_env_fn: list[Callable[[], [EnvBase](torchrl.envs.EnvBase.html#torchrl.envs.EnvBase)]]*, ***, *policy: Callable | None = None*, *policy_factory: Callable[[], Callable] | None = None*, *frames_per_batch: int*, *total_frames: int = -1*, *max_batch_size: int = 64*, *min_batch_size: int = 1*, *server_timeout: float = 0.01*, *transport: [InferenceTransport](torchrl.modules.inference_server.InferenceTransport.html#torchrl.modules.inference_server.InferenceTransport) | None = None*, *device: [device](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device) | str | None = None*, *backend: Literal['threading', 'multiprocessing', 'ray', 'monarch'] = 'threading'*, *env_backend: Literal['threading', 'multiprocessing'] | None = None*, *policy_backend: Literal['threading', 'multiprocessing', 'ray', 'monarch'] | None = None*, *reset_at_each_iter: bool = False*, *postproc: Callable[[[TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)], [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)] | None = None*, *yield_completed_trajectories: bool = False*, *weight_sync=None*, *weight_sync_model_id: str = 'policy'*, *verbose: bool = False*, *create_env_kwargs: dict | list[dict] | None = None*)[[source]](../../_modules/torchrl/collectors/_async_batched.html#AsyncBatchedCollector)

Asynchronous collector that pairs per-env threads with an [`AsyncEnvPool`](torchrl.envs.AsyncEnvPool.html#torchrl.envs.AsyncEnvPool) and an `InferenceServer`.

Unlike [`Collector`](torchrl.collectors.Collector.html#torchrl.collectors.Collector), this collector fully
decouples environment stepping from policy inference:

- An [`AsyncEnvPool`](torchrl.envs.AsyncEnvPool.html#torchrl.envs.AsyncEnvPool) runs *N* environments using
whatever backend the user chooses (`"threading"`,
`"multiprocessing"`).
- *N* lightweight coordinator threads - one per environment - each own
a slot in the pool and an inference client. A thread sends its env's
observation to the `InferenceServer`, blocks
until the batched action is returned, then sends the action back to
the pool for stepping.
- The `InferenceServer` running in a background
thread continuously drains observation submissions, batches them, runs
a single forward pass, and fans actions back out.

There is **no global synchronisation barrier**: fast environments keep
stepping while slow ones wait for inference, and the server always
processes whatever observations have accumulated.

The user simply provides env factories and a policy; the collector
handles all wiring internally.

Parameters:

**create_env_fn** (*list**[**Callable**[**[**]**,*[*EnvBase*](torchrl.envs.EnvBase.html#torchrl.envs.EnvBase)*]**]*) - a list of callables, each
returning an [`EnvBase`](torchrl.envs.EnvBase.html#torchrl.envs.EnvBase) instance. The list
length determines the number of parallel environments.

Keyword Arguments:

- **policy** (*nn.Module**or**Callable**,**optional*) - the policy module.
Mutually exclusive with `policy_factory`.
- **policy_factory** (*Callable**[**[**]**,**Callable**]**,**optional*) - a zero-argument
callable that returns the policy. Useful when the policy cannot
be pickled. Mutually exclusive with `policy`.
- **frames_per_batch** (*int*) - number of environment frames to collect per
batch. Required.
- **total_frames** (*int**,**optional*) - total number of frames the collector
should return during its lifespan. `-1` means endless.
Defaults to `-1`.
- **max_batch_size** (*int**,**optional*) - upper bound on the number of
requests the inference server processes in a single forward pass.
Defaults to `64`.
- **min_batch_size** (*int**,**optional*) - minimum number of requests the
inference server accumulates before dispatching a batch. After
the first request arrives the server keeps draining for up to
`server_timeout` seconds until this many items are collected.
`1` (default) dispatches immediately.
- **server_timeout** ([*float*](torchrl.data.llm.TopKRewardSelector.html#torchrl.data.llm.TopKRewardSelector.float)*,**optional*) - seconds the server waits for work
before dispatching a partial batch. Defaults to `0.01`.
- **transport** ([*InferenceTransport*](torchrl.modules.inference_server.InferenceTransport.html#torchrl.modules.inference_server.InferenceTransport)*,**optional*) - a pre-built transport
object. When provided, it takes precedence over
`policy_backend`. When `None` (default) a transport is
created automatically from the resolved `policy_backend`.
- **device** ([*torch.device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)*or**str**,**optional*) - device for policy inference.
Passed to the inference server. Defaults to `None`.
- **backend** (*str**,**optional*) - global default backend for both
environments and policy inference. Specific overrides
`env_backend` and `policy_backend` take precedence when set.
One of `"threading"`, `"multiprocessing"`, `"ray"`, or
`"monarch"`. Defaults to `"threading"`.
- **env_backend** (*str**,**optional*) - backend for the
[`AsyncEnvPool`](torchrl.envs.AsyncEnvPool.html#torchrl.envs.AsyncEnvPool) that runs environments. One
of `"threading"` or `"multiprocessing"`. Falls back to
`backend` when `None`. The coordinator threads are always
Python threads regardless of this setting. Defaults to `None`.
- **policy_backend** (*str**,**optional*) - backend for the inference transport
used to communicate with the
`InferenceServer`. One of
`"threading"`, `"multiprocessing"`, `"ray"`, or
`"monarch"`. Falls back to `backend` when `None`.
Defaults to `None`.
- **reset_at_each_iter** (*bool**,**optional*) - whether to reset all envs at the
start of every collection batch. Defaults to `False`.
- **postproc** (*Callable**,**optional*) - post-processing transform applied to
each collected batch before yielding. Defaults to `None`.
- **yield_completed_trajectories** (*bool**,**optional*) - if `True`, the
collector yields individual completed trajectories as they finish
rather than fixed-size batches. `frames_per_batch` acts as the
*minimum* number of frames to accumulate before yielding.
Defaults to `False`.
- **weight_sync** - an optional
[`WeightSyncScheme`](torchrl.weight_update.WeightSyncScheme.html#torchrl.weight_update.WeightSyncScheme) forwarded to the
inference server for receiving weight updates.
- **weight_sync_model_id** (*str**,**optional*) - model id for weight sync.
Defaults to `"policy"`.
- **verbose** (*bool**,**optional*) - if `True`, log progress messages.
Defaults to `False`.
- **create_env_kwargs** (*dict**or*[*list*](torchrl.services.RayService.html#torchrl.services.RayService.list)*[**dict**]**,**optional*) - keyword arguments
forwarded to each environment factory. A single dict is broadcast
to all factories.

Examples

```
>>> from torchrl.collectors import AsyncBatchedCollector
>>> from torchrl.envs import GymEnv
>>> from tensordict.nn import TensorDictModule
>>> import torch.nn as nn
>>> policy = TensorDictModule(
... nn.Linear(4, 2), in_keys=["observation"], out_keys=["action"]
... )
>>> collector = AsyncBatchedCollector(
... create_env_fn=[lambda: GymEnv("CartPole-v1")] * 4,
... policy=policy,
... frames_per_batch=200,
... total_frames=1000,
... )
>>> for batch in collector:
... print(batch.shape)
... break
>>> collector.shutdown()
```

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

*property*env*: [AsyncEnvPool](torchrl.envs.AsyncEnvPool.html#torchrl.envs.AsyncEnvPool)*

The underlying `AsyncEnvPool`.

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

iterator() → Iterator[[TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)][[source]](../../_modules/torchrl/collectors/_async_batched.html#AsyncBatchedCollector.iterator)

Iterate over collected batches.

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

*property*policy*: Callable*

The policy passed to the inference server.

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

set_seed(*seed: int*, *static_seed: bool = False*) → int[[source]](../../_modules/torchrl/collectors/_async_batched.html#AsyncBatchedCollector.set_seed)

Set the seed (no-op; envs are created inside the pool).

shutdown(*timeout: float | None = None*, *close_env: bool = True*, *raise_on_error: bool = True*) → None[[source]](../../_modules/torchrl/collectors/_async_batched.html#AsyncBatchedCollector.shutdown)

Shut down the collector, inference server, threads and env pool.

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