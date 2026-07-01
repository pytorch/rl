# BaseCollector

*class*torchrl.collectors.BaseCollector(***, *pre_collect_hook: Callable[[], None] | None = None*, *post_collect_hook: Callable[[[TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)], None] | None = None*)[[source]](../../_modules/torchrl/collectors/_base.html#BaseCollector)

Base class for data collectors.

Keyword Arguments:

- **trajs_per_batch** (*int**,**optional*) -

When set, the collector yields batches
of exactly this many complete trajectories instead of fixed-frame
batches. By default each yielded [`TensorDict`](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDict.html#tensordict.TensorDict)
has shape `(trajs_per_batch, max_traj_len)`, zero-padded along
time, and includes a `("collector", "mask")` boolean field
marking valid time steps; with `traj_format="cat"` the
trajectories are instead concatenated along time into a flat,
unpadded batch. Trajectories that span multiple internal
collection steps are reassembled automatically.
`frames_per_batch` still controls how often the environment is
polled internally, but the output batch size is determined by
`trajs_per_batch`.
([`AsyncBatchedCollector`](torchrl.collectors.AsyncBatchedCollector.html#torchrl.collectors.AsyncBatchedCollector) exposes the
same capability through its `yield_completed_trajectories`
flag.)

**Replay buffer integration**

When combined with a `replay_buffer`, each complete trajectory is
written to the buffer as a **flat 1-D sequence** of valid timesteps
(no padding, no accumulation to `trajs_per_batch`). The method
yields `None` on every write -- matching the standard replay-buffer
collection convention. This flat storage is directly compatible
with `SliceSampler` using
`end_key=("next", "done")`.

Important

When using a **multi-process** collector with a shared replay
buffer and a `SliceSampler`, setting
`trajs_per_batch` is strongly recommended. Without it,
different workers write batches independently and adjacent
frames in the buffer can come from unrelated episodes without
an intervening `done` signal, causing the sampler to draw
slices that cross trajectory boundaries.

**Completeness guarantee**: only trajectories whose last step has
`("next", "done") == True` are written to the buffer. Partial
trajectories (episodes still in flight) are held internally until
the episode terminates. This means every trajectory in the buffer
is guaranteed to be a complete episode segment.

**Batched environments**: when the environment has a batch size > 1
(e.g. [`SerialEnv`](torchrl.envs.SerialEnv.html#torchrl.envs.SerialEnv)), steps are disassembled by
`traj_id` and each trajectory is written individually as a flat
sequence. The buffer storage should use `ndim=1` -- `ndim=2`
is incompatible because variable-length trajectories cannot fill a
fixed second dimension.

**Multi-process and distributed collectors**: `trajs_per_batch`
combined with `replay_buffer` is supported for
[`MultiSyncCollector`](torchrl.collectors.MultiSyncCollector.html#torchrl.collectors.MultiSyncCollector),
[`MultiAsyncCollector`](torchrl.collectors.MultiAsyncCollector.html#torchrl.collectors.MultiAsyncCollector),
[`RayCollector`](torchrl.collectors.distributed.RayCollector.html#torchrl.collectors.distributed.RayCollector), and
[`RPCCollector`](torchrl.collectors.distributed.RPCCollector.html#torchrl.collectors.distributed.RPCCollector).
Trajectory assembly is delegated to each worker's inner collector,
which calls `_iter_by_trajectories()` independently and writes
complete trajectories to the shared replay buffer. Both the
iteration pattern (`for data in collector`) and the async
`start()` pattern are supported.

```
rb = ReplayBuffer(
 storage=LazyTensorStorage(10_000),
 sampler=SliceSampler(slice_len=16, end_key=("next", "done")),
 shared=True,
)
collector = MultiSyncCollector(
 [env_fn] * 4, policy,
 replay_buffer=rb,
 frames_per_batch=200,
 total_frames=-1,
 trajs_per_batch=32,
)
collector.start() # workers fill rb with complete trajectories
```

Defaults to `None` (fixed-frame batches).
- **trajs_per_write** (*int**,**optional*) - When `trajs_per_batch` is used with
a replay buffer, write this many completed trajectories to the
buffer per `extend` call. Larger values reduce Python overhead
for highly batched environments. For example, if 10 complete
trajectories are queued for replay-buffer insertion,
`trajs_per_write=2` makes 5 writes, while
`trajs_per_write=10` or larger makes 1 write. Defaults to
`None` (write all currently queued completed trajectories).
- **traj_format** (*str**,**optional*) - layout of the batches yielded when
`trajs_per_batch` is set. `"padded"` stacks the
trajectories into a `(trajs_per_batch, max_traj_len)` batch,
zero-padded along time, with a `("collector", "mask")` entry
marking the valid steps. `"cat"` concatenates them along time
into a flat, unpadded `[sum_i T_i]` batch -- no mask, no wasted
memory on padding; trajectories are contiguous, delimited by
`("next", "done")` (`True` at the last step of each, by the
completeness guarantee) and `("collector", "traj_ids")`.
`"cat"` matches the layout the replay-buffer write path uses
and the one `SliceSampler` expects. Has no
effect on replay-buffer writes (always flat); raises if set
without `trajs_per_batch`. Defaults to `None`, which
currently resolves to `"padded"` and emits a
`FutureWarning` when `trajs_per_batch` batches are
yielded without an explicit choice: the default will change to
`"cat"` in torchrl v0.16.

async_shutdown(*timeout: float | None = None*, *close_env: bool = True*) → None[[source]](../../_modules/torchrl/collectors/_base.html#BaseCollector.async_shutdown)

Shuts down the collector when started asynchronously with the start method.

Parameters:

- **timeout** (*float**,**optional*) - The maximum time to wait for the collector to shutdown.
- **close_env** (*bool**,**optional*) - If True, the collector will close the contained environment.
Defaults to True.

See also

`start()`

cascade_execute(*attr_path: str*, **args*, ***kwargs*) → Any[[source]](../../_modules/torchrl/collectors/_base.html#BaseCollector.cascade_execute)

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

disable_profile() → None[[source]](../../_modules/torchrl/collectors/_base.html#BaseCollector.disable_profile)

Stop any in-flight profiler and restore the prior `post_collect_hook`.

Safe to call when profiling was never enabled (becomes a no-op). When
the profiler was already self-stopped after `num_rollouts`, this just
clears the hook and restores any user-set `post_collect_hook`.

enable_profile(***, *workers: list[int] | None = None*, *num_rollouts: int = 3*, *warmup_rollouts: int = 1*, *save_path: str | Path | None = None*, *activities: list[str] | None = None*, *record_shapes: bool = True*, *profile_memory: bool = False*, *with_stack: bool = True*, *with_flops: bool = False*, *on_trace_ready: Callable | None = None*) → None[[source]](../../_modules/torchrl/collectors/_base.html#BaseCollector.enable_profile)

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

get_distant_attr(*attr: str*) → Any[[source]](../../_modules/torchrl/collectors/_base.html#BaseCollector.get_distant_attr)

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

init_updater(**args*, ***kwargs*)[[source]](../../_modules/torchrl/collectors/_base.html#BaseCollector.init_updater)

Initialize the weight updater with custom arguments.

This method passes the arguments to the weight updater's init method.
If no weight updater is set, this is a no-op.

Parameters:

- ***args** - Positional arguments for weight updater initialization
- ****kwargs** - Keyword arguments for weight updater initialization

map_fn(*method_name: str*, *list_of_args: list[tuple] | None = None*, *list_of_kwargs: list[dict] | None = None*) → list[Any][[source]](../../_modules/torchrl/collectors/_base.html#BaseCollector.map_fn)

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

pause()[[source]](../../_modules/torchrl/collectors/_base.html#BaseCollector.pause)

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

receive_weights() → None[[source]](../../_modules/torchrl/collectors/_base.html#BaseCollector.receive_weights)

receive_weights(*policy_or_weights: [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase) | [TensorDictModuleBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.nn.TensorDictModuleBase.html#tensordict.nn.TensorDictModuleBase) | [Module](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module) | dict*, */*) → None

receive_weights(***, *weights: [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase) | dict*) → None

receive_weights(***, *policy: [TensorDictModuleBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.nn.TensorDictModuleBase.html#tensordict.nn.TensorDictModuleBase) | [Module](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module)*) → None

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

register_scheme_receiver(*weight_recv_schemes: dict[str, [WeightSyncScheme](torchrl.weight_update.WeightSyncScheme.html#torchrl.weight_update.WeightSyncScheme)]*, ***, *synchronize_weights: bool = True*)[[source]](../../_modules/torchrl/collectors/_base.html#BaseCollector.register_scheme_receiver)

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

set_post_collect_hook(*hook: Callable[[[TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)], None] | None*) → None[[source]](../../_modules/torchrl/collectors/_base.html#BaseCollector.set_post_collect_hook)

Method form of the `post_collect_hook` setter.

Exposed because Ray actor handles can call methods (actor.method.remote(...))
but cannot directly invoke property setters. Keeping the actual setter
for in-process use and this method for remote-actor use.

start()[[source]](../../_modules/torchrl/collectors/_base.html#BaseCollector.start)

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

update_policy_weights_(*policy_or_weights: [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase) | [TensorDictModuleBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.nn.TensorDictModuleBase.html#tensordict.nn.TensorDictModuleBase) | [Module](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module) | dict*, */*) → None[[source]](../../_modules/torchrl/collectors/_base.html#BaseCollector.update_policy_weights_)

update_policy_weights_(*policy_or_weights: [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase) | [TensorDictModuleBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.nn.TensorDictModuleBase.html#tensordict.nn.TensorDictModuleBase) | [Module](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module) | dict*, */*, ***, *worker_ids: int | list[int] | [device](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device) | list[[device](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)] | None = None*, *model_id: str | None = None*) → None

update_policy_weights_(***, *weights: [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase) | dict*, *model_id: str | None = None*, *worker_ids: int | list[int] | [device](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device) | list[[device](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)] | None = None*) → None

update_policy_weights_(***, *policy: [TensorDictModuleBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.nn.TensorDictModuleBase.html#tensordict.nn.TensorDictModuleBase) | [Module](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module)*, *model_id: str | None = None*, *worker_ids: int | list[int] | [device](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device) | list[[device](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)] | None = None*) → None

update_policy_weights_(***, *weights_dict: dict[str, [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase) | [TensorDictModuleBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.nn.TensorDictModuleBase.html#tensordict.nn.TensorDictModuleBase) | [Module](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module) | dict]*, *worker_ids: int | list[int] | [device](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device) | list[[device](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)] | None = None*) → None

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