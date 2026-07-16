# LLMCollector

*class*torchrl.collectors.llm.LLMCollector(*env: [EnvBase](torchrl.envs.EnvBase.html#torchrl.envs.EnvBase) | Callable[[], [EnvBase](torchrl.envs.EnvBase.html#torchrl.envs.EnvBase)]*, ***, *policy: Callable[[TensorDictBase], TensorDictBase] | None = None*, *policy_factory: Callable[[], Callable[[TensorDictBase], TensorDictBase]] | None = None*, *dialog_turns_per_batch: int | None = None*, *yield_only_last_steps: bool | None = None*, *yield_completed_trajectories: bool | None = None*, *postproc: Callable[[TensorDictBase], TensorDictBase] | None = None*, *total_dialog_turns: int = -1*, *async_envs: bool | None = None*, *replay_buffer: [ReplayBuffer](torchrl.data.ReplayBuffer.html#torchrl.data.ReplayBuffer) | None = None*, *reset_at_each_iter: bool = False*, *flatten_data: bool | None = None*, *weight_updater: [WeightUpdaterBase](torchrl.collectors.WeightUpdaterBase.html#torchrl.collectors.WeightUpdaterBase) | Callable[[], [WeightUpdaterBase](torchrl.collectors.WeightUpdaterBase.html#torchrl.collectors.WeightUpdaterBase)] | None = None*, *queue: Any | None = None*, *track_policy_version: bool | [PolicyVersion](torchrl.envs.llm.transforms.PolicyVersion.html#torchrl.envs.llm.transforms.PolicyVersion) = False*, *verbose: bool = False*)[[source]](../../_modules/torchrl/collectors/llm/base.html#LLMCollector)

A simplified version of Collector for LLM inference.

Parameters:

**env** ([*EnvBase*](torchrl.envs.EnvBase.html#torchrl.envs.EnvBase)*or**EnvBase constructor*) - the environment to be used for data collection.

Keyword Arguments:

- **policy** (*Callable**[**[**TensorDictBase**]**,**TensorDictBase**]*) - the policy to be used for data collection.
- **policy_factory** (*Callable**[**[**]**,**Callable**]**,**optional*) -

a callable that returns
a policy instance. This is exclusive with the policy argument.

Note

policy_factory comes in handy whenever the policy cannot be serialized.
- **dialog_turns_per_batch** (*int**,**optional*) - A keyword-only argument representing the total
number of elements in a batch. It is always required except when yield_completed_trajectories=True.
- **total_dialog_turns** (*int*) - A keyword-only argument representing the total
number of steps returned by the collector during its lifespan. -1 is never ending (until shutdown).
Defaults to -1.
- **yield_completed_trajectories** (*bool**,**optional*) -

whether to yield batches of rollouts with a given number of steps
(yield_completed_trajectories=False, default) or single, completed trajectories
(yield_completed_trajectories=True).
Defaults to False unless yield_only_last_steps=True, where it cannot be False.

Warning

If the done state of the environment is not properly set, this may lead to a collector
that never leads any data.
- **yield_only_last_steps** (*bool**,**optional*) -

whether to yield every step of a trajectory, or only the
last (done) steps.
If True, a single trajectory is yielded (or written in the buffer) at a time.

Warning

If the done state of the environment is not properly set, this may lead to a collector
that never leads any data.
- **postproc** (*Callable**,**optional*) - A post-processing transform, such as
a `Transform` or a `MultiStep`
instance.
Defaults to `None`.
- **async_envs** (*bool**,**optional*) - if `True`, the environment will be run asynchronously. Defaults to True if the
environment is a [`AsyncEnvPool`](torchrl.envs.AsyncEnvPool.html#torchrl.envs.AsyncEnvPool) instance.
- **replay_buffer** ([*ReplayBuffer*](torchrl.data.ReplayBuffer.html#torchrl.data.ReplayBuffer)*,**optional*) - if provided, the collector will not yield tensordicts
but populate the buffer instead. Defaults to `None`.
- **reset_at_each_iter** (*bool**,**optional*) - if `True`, the environment will be reset at each iteration.
- **flatten_data** (*bool**,**optional*) - if `True`, the collector will flatten the collected data
before returning it. In practice, this means that if an environment of batch-size (B,) is used
and run for T steps, flatten_data=True will present data of shape (B*T,), whereas
flatten_data=False will not present data of shape (B, T).
Defaults to True when replay_buffer is provided, False otherwise.
- **weight_updater** ([*WeightUpdaterBase*](torchrl.collectors.WeightUpdaterBase.html#torchrl.collectors.WeightUpdaterBase)*or**constructor**,**optional*) - An instance of [`WeightUpdaterBase`](torchrl.collectors.WeightUpdaterBase.html#torchrl.collectors.WeightUpdaterBase)
or its subclass, responsible for updating the policy weights on remote inference workers.
This is typically not used in [`Collector`](torchrl.collectors.Collector.html#torchrl.collectors.Collector) as it operates in a single-process environment.
Consider using a constructor if the updater needs to be serialized.
- **track_policy_version** (*bool**or*[*PolicyVersion*](torchrl.envs.llm.transforms.PolicyVersion.html#torchrl.envs.llm.transforms.PolicyVersion)*,**optional*) - if `True`, the collector will track the version of the policy.
This will be mediated by the [`PolicyVersion`](torchrl.envs.llm.transforms.PolicyVersion.html#torchrl.envs.llm.transforms.PolicyVersion) transform, which will be added to the environment.
Alternatively, a [`PolicyVersion`](torchrl.envs.llm.transforms.PolicyVersion.html#torchrl.envs.llm.transforms.PolicyVersion) instance can be passed, which will be used to track
the policy version.
Defaults to False.
- **verbose** (*bool**,**optional*) - if `True`, the collector will print progress information.
Defaults to False.

Examples

```
>>> import vllm
>>> from torchrl.modules import vLLMWrapper
>>> from torchrl.testing.mocking_classes import DummyStrDataLoader
>>> from torchrl.envs import LLMEnv
>>> llm_model = vllm.LLM("gpt2")
>>> tokenizer = llm_model.get_tokenizer()
>>> tokenizer.pad_token = tokenizer.eos_token
>>> policy = vLLMWrapper(llm_model)
>>> dataloader = DummyStrDataLoader(1)
>>> env = LLMEnv.from_dataloader(
... dataloader=dataloader,
... tokenizer=tokenizer,
... from_text=True,
... batch_size=1,
... group_repeats=True,
... )
>>> collector = LLMCollector(
... env=env,
... policy_factory=lambda: policy,
... dialog_turns_per_batch=env.batch_size[0],
... total_dialog_turns=3,
... )
>>> for i, data in enumerate(collector):
... if i == 2:
... print(data)
... break
LazyStackedTensorDict(
fields={
 attention_mask: Tensor(shape=torch.Size([1, 1, 22]), device=cpu, dtype=torch.int64, is_shared=False),
 collector: LazyStackedTensorDict(
 fields={
 traj_ids: Tensor(shape=torch.Size([1, 1]), device=cpu, dtype=torch.int64, is_shared=False)},
 exclusive_fields={
 },
 batch_size=torch.Size([1, 1]),
 device=None,
 is_shared=False,
 stack_dim=1),
 done: Tensor(shape=torch.Size([1, 1, 1]), device=cpu, dtype=torch.bool, is_shared=False),
 terminated: Tensor(shape=torch.Size([1, 1, 1]), device=cpu, dtype=torch.bool, is_shared=False),
 text: NonTensorStack(
 [['plsgqejeyd']],
 batch_size=torch.Size([1, 1]),
 device=None),
 text_response: NonTensorStack(
 [['ec.n.n.n.tjbjz3perwhz']],
 batch_size=torch.Size([1, 1]),
 device=None),
 tokens: Tensor(shape=torch.Size([1, 1, 22]), device=cpu, dtype=torch.int64, is_shared=False),
 tokens_response: Tensor(shape=torch.Size([1, 1, 16]), device=cpu, dtype=torch.int64, is_shared=False)},
exclusive_fields={
},
batch_size=torch.Size([1, 1]),
device=None,
is_shared=False,
stack_dim=1)
>>> del collector
```

*classmethod*as_remote(*remote_config: dict[str, Any] | None = None*)

Creates an instance of a remote ray class.

Parameters:

- **cls** (*Python Class*) - class to be remotely instantiated.
- **remote_config** (*dict*) - the quantity of CPU cores to reserve for this class.

Returns:

A function that creates ray remote class instances.

async_shutdown(*timeout: float | None = None*, *close_env: bool = True*) → None

Finishes processes started by ray.init() during async execution.

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

*property*dialog_turns_per_batch*: int*

Alias to frames_per_batch.

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

Return a zero-filled tensordict shaped like one batch from this collector.

The result mirrors what `next(iter(collector))` would yield:

- batch shape `(*env.batch_size, frames_per_batch)` with the last
dim named `"time"`;
- env keys (observation / reward / done / terminated / truncated /
`is_init` when an `InitTracker` is on the
env), policy out-keys, and `("collector", "traj_ids")` when
trajectory tracking is enabled;
- `compact_obs=True` exclusions applied;
- `set_truncated=True` last-step `truncated`/`done` masking
applied;
- `postproc` / `split_trajs` / private-key exclusion applied,
mirroring `_postproc()`.

Intended for storage initialization and `torch.compile` /
cudagraph warmup without having to step the environment first.

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

get_model(*model_id: str*)

Get model instance by ID (for weight sync schemes).

Parameters:

**model_id** - Model identifier (e.g., "policy", "value_net")

Returns:

The model instance

Raises:

**ValueError** - If model_id is not recognized

get_policy_model()[[source]](../../_modules/torchrl/collectors/llm/base.html#LLMCollector.get_policy_model)

Get the policy model.

This method is used by RayLLMCollector to get the remote LLM instance
for weight updates.

Returns:

The policy model instance

get_policy_version() → str | int | None[[source]](../../_modules/torchrl/collectors/llm/base.html#LLMCollector.get_policy_version)

Get the current policy version.

This method exists to support remote calls in Ray actors, since properties
cannot be accessed directly through Ray's RPC mechanism.

Returns:

The current version number (int) or UUID (str), or None if version tracking is disabled.

getattr_env(*attr*)

Get an attribute from the environment.

getattr_policy(*attr*)

Get an attribute from the policy.

getattr_rb(*attr*)

Get an attribute from the replay buffer.

increment_version()[[source]](../../_modules/torchrl/collectors/llm/base.html#LLMCollector.increment_version)

Increment the policy version.

init_updater(**args*, ***kwargs*)

Initialize the weight updater with custom arguments.

This method passes the arguments to the weight updater's init method.
If no weight updater is set, this is a no-op.

Parameters:

- ***args** - Positional arguments for weight updater initialization
- ****kwargs** - Keyword arguments for weight updater initialization

is_initialized() → bool[[source]](../../_modules/torchrl/collectors/llm/base.html#LLMCollector.is_initialized)

Check if the collector is initialized and ready.

Returns:

True if the collector is initialized and ready to collect data.

Return type:

bool

iterator() → Iterator[[TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)]

Iterates through the DataCollector.

Yields: TensorDictBase objects containing (chunks of) trajectories

load_state_dict(*state_dict: OrderedDict*, ***kwargs*) → None

Loads a state_dict on the environment and policy.

Parameters:

**state_dict** (*OrderedDict*) - ordered dictionary containing the fields
"policy_state_dict" and `"env_state_dict"`.

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

*property*policy_version*: str | int | None*

The current policy version.

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

reset(*index=None*, ***kwargs*) → None

Resets the environments to a new initial state.

When `trajs_per_batch` is in use, also drops in-flight episodes and
completed-but-not-yet-yielded trajectories, so post-reset batches
contain only post-reset data.

*property*rollout*: Callable[[], [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)]*

Computes a rollout in the environment using the provided policy.

Each call runs `frames_per_batch` env steps and returns (or writes
to the replay buffer) the resulting batch. The per-timestep flow is:

1. **Carrier prep** -- read `self._carrier`, the persistent
tensordict that survives across timesteps (allocated once in
`_make_carrier()`). If `reset_at_each_iter=True`, reset
the env first.
2. **Policy step** -- cast the carrier to `policy_device` if it
differs from `env_device` (then `_sync_policy()`), invoke
the policy, and merge its outputs back into the carrier.
3. **Env step** -- cast the carrier to `env_device` if needed
(then `_sync_env()`), call `env.step_and_maybe_reset`, and
write the returned `"next"` sub-tensordict back into the
carrier.
4. **Persist** -- append the per-step snapshot to a list after
casting to `storing_device` and `_sync_storage()` if needed,
or write it directly with `replay_buffer.add(...)` when direct
replay-buffer writes are enabled.
5. **Advance** -- swap the carrier for the post-reset
`env_next_output` and update `("collector", "traj_ids")` for
any envs that finished.

See [Collector Internals](../collectors_internals.html#ref-collectors-internals) for the full flow diagram and
an explanation of the carrier / sync / device-cast machinery.

Returns:

TensorDictBase containing the computed rollout.

set_post_collect_hook(*hook: Callable[[[TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)], None] | None*) → None

Method form of the `post_collect_hook` setter.

Exposed because Ray actor handles can call methods (actor.method.remote(...))
but cannot directly invoke property setters. Keeping the actual setter
for in-process use and this method for remote-actor use.

set_seed(*seed: int*, *static_seed: bool = False*) → int

Sets the seeds of the environments stored in the DataCollector.

Parameters:

- **seed** (*int*) - integer representing the seed to be used for the environment.
- **static_seed** (*bool**,**optional*) - if `True`, the seed is not incremented.
Defaults to False

Returns:

Output seed. This is useful when more than one environment is contained in the DataCollector, as the
seed will be incremented for each of these. The resulting seed is the seed of the last environment.

Examples

```
>>> from torchrl.envs import ParallelEnv
>>> from torchrl.envs.libs.gym import GymEnv
>>> from tensordict.nn import TensorDictModule
>>> from torch import nn
>>> env_fn = lambda: GymEnv("Pendulum-v1")
>>> env_fn_parallel = ParallelEnv(6, env_fn)
>>> policy = TensorDictModule(nn.Linear(3, 1), in_keys=["observation"], out_keys=["action"])
>>> collector = Collector(env_fn_parallel, policy, total_frames=300, frames_per_batch=100)
>>> out_seed = collector.set_seed(1) # out_seed = 6
```

shutdown(*timeout: float | None = None*, *close_env: bool = True*, *raise_on_error: bool = True*) → None

Shuts down all workers and/or closes the local environment.

Parameters:

- **timeout** (*float**,**optional*) - The timeout for closing pipes between workers.
No effect for this class.
- **close_env** (*bool**,**optional*) - Whether to close the environment. Defaults to True.
- **raise_on_error** (*bool**,**optional*) - Whether to raise an error if the shutdown fails. Defaults to True.

start()

Starts the collector in a separate thread for asynchronous data collection.

The collected data is stored in the provided replay buffer. This method is useful when you want to decouple data
collection from training, allowing your training loop to run independently of the data collection process.

Raises:

**RuntimeError** - If no replay buffer is defined during the collector's initialization.

Example

```
>>> from torchrl.modules import RandomPolicy >>> >>> import time
>>> from functools import partial
>>>
>>> import tqdm
>>>
>>> from torchrl.collectors import Collector
>>> from torchrl.data import LazyTensorStorage, ReplayBuffer
>>> from torchrl.envs import GymEnv, set_gym_backend
>>> import ale_py
>>>
>>> # Set the gym backend to gymnasium
>>> set_gym_backend("gymnasium").set()
>>>
>>> if __name__ == "__main__":
... # Create a random policy for the Pong environment
... env = GymEnv("ALE/Pong-v5")
... policy = RandomPolicy(env.action_spec)
...
... # Initialize a shared replay buffer
... rb = ReplayBuffer(storage=LazyTensorStorage(1000), shared=True)
...
... # Create a synchronous data collector
... collector = Collector(
... env,
... policy=policy,
... replay_buffer=rb,
... frames_per_batch=256,
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

state_dict() → OrderedDict

Returns the local state_dict of the data collector (environment and policy).

Returns:

an ordered dictionary with fields `"policy_state_dict"` and
"env_state_dict".

update_policy_weights_(*policy_or_weights: [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase) | [TensorDictModuleBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.nn.TensorDictModuleBase.html#tensordict.nn.TensorDictModuleBase) | dict | None = None*, ***, *worker_ids: int | list[int] | [device](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device) | list[[device](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)] | None = None*, ***kwargs*) → None

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