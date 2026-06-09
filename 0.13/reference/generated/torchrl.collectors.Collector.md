# Collector

*class*torchrl.collectors.Collector(*create_env_fn: [EnvBase](torchrl.envs.EnvBase.html#torchrl.envs.EnvBase) | [EnvCreator](torchrl.envs.EnvCreator.html#torchrl.envs.EnvCreator) | Sequence[Callable[[], [EnvBase](torchrl.envs.EnvBase.html#torchrl.envs.EnvBase)]]*, *policy: None | [TensorDictModule](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.nn.TensorDictModule.html#tensordict.nn.TensorDictModule) | Callable[[[TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)], [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)] = None*, ***, *policy_factory: Callable[[], Callable] | None = None*, *frames_per_batch: int*, *total_frames: int = -1*, *device: [device](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device) | str | int | None = None*, *storing_device: [device](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device) | str | int | None = None*, *policy_device: [device](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device) | str | int | None = None*, *env_device: [device](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device) | str | int | None = None*, *create_env_kwargs: dict[str, Any] | None = None*, *max_frames_per_traj: int | None = None*, *init_random_frames: int | None = None*, *reset_at_each_iter: bool = False*, *postproc: Callable[[[TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)], [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)] | None = None*, *split_trajs: bool | None = None*, *track_traj_ids: bool = True*, *exploration_type: [InteractionType](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.nn.InteractionType.html#tensordict.nn.InteractionType) = InteractionType.RANDOM*, *return_same_td: bool = False*, *reset_when_done: bool = True*, *interruptor=None*, *set_truncated: bool = False*, *use_buffers: bool | None = None*, *replay_buffer: [ReplayBuffer](torchrl.data.ReplayBuffer.html#torchrl.data.ReplayBuffer) | None = None*, *extend_buffer: bool = True*, *trust_policy: bool | None = None*, *compile_policy: bool | dict[str, Any] | None = None*, *cudagraph_policy: bool | dict[str, Any] | None = None*, *no_cuda_sync: bool = False*, *weight_updater: [WeightUpdaterBase](torchrl.collectors.WeightUpdaterBase.html#torchrl.collectors.WeightUpdaterBase) | Callable[[], [WeightUpdaterBase](torchrl.collectors.WeightUpdaterBase.html#torchrl.collectors.WeightUpdaterBase)] | None = None*, *weight_sync_schemes: dict[str, [WeightSyncScheme](torchrl.weight_update.WeightSyncScheme.html#torchrl.weight_update.WeightSyncScheme)] | None = None*, *weight_recv_schemes: dict[str, [WeightSyncScheme](torchrl.weight_update.WeightSyncScheme.html#torchrl.weight_update.WeightSyncScheme)] | None = None*, *track_policy_version: bool = False*, *worker_idx: int | None = None*, *trajs_per_batch: int | None = None*, *trajs_per_write: int | None = None*, *auto_register_policy_transforms: bool | None = None*, *pre_collect_hook: Callable[[], None] | None = None*, *post_collect_hook: Callable[[[TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)], None] | None = None*, *compact_obs: bool = False*, ***kwargs*)[[source]](../../_modules/torchrl/collectors/_single.html#Collector)

Generic data collector for RL problems. Requires an environment constructor and a policy.

Parameters:

- **create_env_fn** (*Callable**or*[*EnvBase*](torchrl.envs.EnvBase.html#torchrl.envs.EnvBase)) - a callable that returns an instance of
[`EnvBase`](torchrl.envs.EnvBase.html#torchrl.envs.EnvBase) class, or the env itself.
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

- **policy_factory** (*Callable**[**[**]**,**Callable**]**,**optional*) -

a callable that returns
a policy instance. This is exclusive with the policy argument.

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
- **storing_device** (*int**,**str**or*[*torch.device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)*,**optional*) - The device on which
the output [`TensorDict`](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDict.html#tensordict.TensorDict) will be stored.
If `device` is passed and `storing_device` is `None`, it will
default to the value indicated by `device`.
For long trajectories, it may be necessary to store the data on a different
device than the one where the policy and env are executed.
Defaults to `None` (the output tensordict isn't on a specific device,
leaf tensors sit on the device where they were created).
- **env_device** (*int**,**str**or*[*torch.device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)*,**optional*) - The device on which
the environment should be cast (or executed if that functionality is
supported). If not specified and the env has a non-`None` device,
`env_device` will default to that value. If `device` is passed
and `env_device=None`, it will default to `device`. If the value
as such specified of `env_device` differs from `policy_device`
and one of them is not `None`, the data will be cast to `env_device`
before being passed to the env (i.e., passing different devices to
policy and env is supported). Defaults to `None`.
- **policy_device** (*int**,**str**or*[*torch.device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)*,**optional*) - The device on which
the policy should be cast.
If `device` is passed and `policy_device=None`, it will default
to `device`. If the value as such specified of `policy_device`
differs from `env_device` and one of them is not `None`,
the data will be cast to `policy_device` before being passed to
the policy (i.e., passing different devices to policy and env is
supported). Defaults to `None`.
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
- **postproc** (*Callable**,**optional*) -

A post-processing transform, such as
a `Transform` or a `MultiStep`
instance.

Warning

Postproc is not applied when a replay buffer is used and items are added to the buffer
as they are produced (extend_buffer=False). The recommended usage is to use extend_buffer=True.

Defaults to `None`.
- **split_trajs** (*bool**,**optional*) - Boolean indicating whether the resulting
TensorDict should be split according to the trajectories.
See [`split_trajectories()`](torchrl.collectors.utils.split_trajectories.html#torchrl.collectors.utils.split_trajectories) for more
information.
Defaults to `False`.
- **track_traj_ids** (*bool**,**optional*) - if `False`, the collector will not
write `("collector", "traj_ids")` in the rollout nor update
trajectory identifiers at every environment step. This is useful
when trajectory splitting or trajectory-aware replay sampling is not
needed. Defaults to `True`.
- **exploration_type** (*ExplorationType**,**optional*) - interaction mode to be used when
collecting data. Must be one of `torchrl.envs.utils.ExplorationType.DETERMINISTIC`,
`torchrl.envs.utils.ExplorationType.RANDOM`, `torchrl.envs.utils.ExplorationType.MODE`
or `torchrl.envs.utils.ExplorationType.MEAN`.
- **return_same_td** (*bool**,**optional*) - if `True`, the same TensorDict
will be returned at each iteration, with its values
updated. This feature should be used cautiously: if the same
tensordict is added to a replay buffer for instance,
the whole content of the buffer will be identical.
Default is `False`.
- **interruptor** (*_Interruptor**,**optional*) - An _Interruptor object that can be used from outside the class to control rollout collection.
The _Interruptor class has methods ´start_collection´ and ´stop_collection´, which allow to implement
strategies such as preeptively stopping rollout collection.
Default is `False`.
- **set_truncated** (*bool**,**optional*) - if `True`, the truncated signals (and corresponding
`"done"` but not `"terminated"`) will be set to `True` when the last frame of
a rollout is reached. If no `"truncated"` key is found, an exception is raised.
Truncated keys can be set through `env.add_truncated_keys`.
Defaults to `False`.
- **use_buffers** (*bool**,**optional*) - if `True`, a buffer will be used to stack the data.
This isn't compatible with environments with dynamic specs. Defaults to `True`
for envs without dynamic specs, `False` for others.
- **replay_buffer** ([*ReplayBuffer*](torchrl.data.ReplayBuffer.html#torchrl.data.ReplayBuffer)*,**optional*) -

if provided, the collector will not yield tensordicts
but populate the buffer instead.
Defaults to `None`.

See also

By default (`extend_buffer=True`), the buffer is extended with entire rollouts.
If the buffer needs to be populated with individual frames as they are collected,
set `extend_buffer=False` (deprecated).

Warning

Using a replay buffer with a postproc or split_trajs=True requires
extend_buffer=True, as the whole batch needs to be observed to apply these transforms.
- **extend_buffer** (*bool**,**optional*) -

if True, the replay buffer is extended with entire rollouts and not
with single steps. Defaults to True.

Note

Setting this to False is deprecated and will be removed in a future version.
Extending the buffer with entire rollouts is the recommended approach for better
compatibility with postprocessing and trajectory splitting.
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
- **auto_register_policy_transforms** (*bool**,**optional*) -

if `True`, the
collector inspects the policy for recurrent submodules
([`LSTMModule`](torchrl.modules.LSTMModule.html#torchrl.modules.LSTMModule),
[`GRUModule`](torchrl.modules.GRUModule.html#torchrl.modules.GRUModule), anything implementing
`make_tensordict_primer()`) and appends the matching
[`InitTracker`](torchrl.envs.transforms.InitTracker.html#torchrl.envs.transforms.InitTracker) and
[`TensorDictPrimer`](torchrl.envs.transforms.TensorDictPrimer.html#torchrl.envs.transforms.TensorDictPrimer) transforms to
the env if the env's specs don't already provide them. The check
is spec-based and idempotent, so passing an env that was already
wrapped via [`EnvBase`](torchrl.envs.EnvBase.html#torchrl.envs.EnvBase)'s `policy=`
constructor argument is safe. If `False`, the collector never
modifies the env. Defaults to `None` through v0.14, which
preserves the pre-v0.15 behavior (no auto-registration) but emits
a `FutureWarning` if the env was missing transforms the
policy needed. The default flips to `True` in v0.15.

See also

[Auto-wrapping recurrent transforms via the
policy= argument](../envs_api.html#environment-policy-arg).
- **weight_updater** ([*WeightUpdaterBase*](torchrl.collectors.WeightUpdaterBase.html#torchrl.collectors.WeightUpdaterBase)*or**constructor**,**optional*) - An instance of [`WeightUpdaterBase`](torchrl.collectors.WeightUpdaterBase.html#torchrl.collectors.WeightUpdaterBase)
or its subclass, responsible for updating the policy weights on remote inference workers.
This is typically not used in `Collector` as it operates in a single-process environment.
Consider using a constructor if the updater needs to be serialized.
- **weight_sync_schemes** (*dict**[**str**,*[*WeightSyncScheme*](torchrl.weight_update.WeightSyncScheme.html#torchrl.weight_update.WeightSyncScheme)*]**,**optional*) - **Not supported for Collector**.
Collector is a leaf collector and cannot send weights to sub-collectors.
Providing this parameter will raise a ValueError.
Use `weight_recv_schemes` if you need to receive weights from a parent collector.
- **weight_recv_schemes** (*dict**[**str**,*[*WeightSyncScheme*](torchrl.weight_update.WeightSyncScheme.html#torchrl.weight_update.WeightSyncScheme)*]**,**optional*) - Dictionary of weight sync schemes for
RECEIVING weights from parent collectors. Keys are model identifiers (e.g., "policy")
and values are WeightSyncScheme instances configured to receive weights.
This enables cascading weight updates in hierarchies like:
RPCCollector -> MultiSyncCollector -> Collector.
Defaults to `None`.
- **track_policy_version** (*bool**or*[*PolicyVersion*](torchrl.envs.llm.transforms.PolicyVersion.html#torchrl.envs.llm.transforms.PolicyVersion)*,**optional*) -

if `True`, the collector will track the version of the policy.
A [`PolicyVersion`](torchrl.envs.llm.transforms.PolicyVersion.html#torchrl.envs.llm.transforms.PolicyVersion) transform is
installed on the environment, tagging every collected frame with the current version
under the `"policy_version"` key. The transform's version is bumped exactly once
per `update_policy_weights_()` call -- for multi-process collectors this happens
in each worker after the new weights have actually been applied, so per-frame
tagging tracks real weight updates rather than rollout iterations.

The recommended path is `track_policy_version=True`: let the collector own the
transform. Passing a [`PolicyVersion`](torchrl.envs.llm.transforms.PolicyVersion.html#torchrl.envs.llm.transforms.PolicyVersion)
instance directly is reserved for advanced use cases that wire up a `PolicyVersion`
**without** going through a collector (e.g. a hand-rolled rollout loop). Pre-creating
a transform and passing it to a collector is supported but discouraged because it
invites a divergence between the transform on the env and the one the collector
increments.

Defaults to `False`.
- **compact_obs** (*bool**,**optional*) -

if `True`, the collector drops the
observation and state keys from the `("next", ...)` sub-tensordict
before stacking per-step data. Those keys are bit-for-bit identical
to the root keys of the next step (modulo the last frame of each
trajectory), so storing both copies roughly doubles the observation
footprint for nothing. `("next", "reward")`, `("next", "done")`
and `("next", "truncated")` are preserved because they cannot be
reconstructed from the root keys. The dropped keys can be
re-hydrated at sampling time with
[`NextStateReconstructor`](torchrl.envs.transforms.NextStateReconstructor.html#torchrl.envs.transforms.NextStateReconstructor); trajectory
ends will carry `NaN` for the missing `("next", obs)` and the
value-estimator forward pass substitutes a finite placeholder so
GAE / TD targets stay numerically defined (see
`_sanitize_next_obs_nan()`).

`compact_obs=True` composes cleanly with
[`GAE`](torchrl.objectives.value.GAE.html#torchrl.objectives.value.GAE) configured with
`shifted=True`: the budgeted shifted path can run the on-policy
advantage pass without rehydrating every per-step
`("next", "observation")` mirror. For vectorized environments
with large observations this is typically a sizeable GPU-memory
win at near-zero CPU cost.

Default is `False` because the canonical `("next", obs)` is
still required by some downstream losses -- most notably
`MultiStepTransform`, which uses
the n-step `("next", obs)` (and its in-trajectory fallback at
the last `n - 1` frames) and cannot reconstruct that from root
obs alone. For a lossy-precision alternative that *does* preserve
boundary transitions (at the cost of a smaller memory saving), see
[`NextObservationDelta`](torchrl.envs.transforms.NextObservationDelta.html#torchrl.envs.transforms.NextObservationDelta). See also
the *Memory-efficient RL training* tutorial for an end-to-end
pipeline. Defaults to `False`.

Examples

```
>>> from torchrl.envs.libs.gym import GymEnv
>>> from tensordict.nn import TensorDictModule
>>> from torch import nn
>>> env_maker = lambda: GymEnv("Pendulum-v1", device="cpu")
>>> policy = TensorDictModule(nn.Linear(3, 1), in_keys=["observation"], out_keys=["action"])
>>> collector = Collector(
... create_env_fn=env_maker,
... policy=policy,
... total_frames=2000,
... max_frames_per_traj=50,
... frames_per_batch=200,
... init_random_frames=-1,
... reset_at_each_iter=False,
... device="cpu",
... storing_device="cpu",
... )
>>> for i, data in enumerate(collector):
... if i == 2:
... print(data)
... break
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
>>> del collector
```

The collector delivers batches of data that are marked with a `"time"`
dimension.

Examples

```
>>> assert data.names[-1] == "time"
```

async_shutdown(*timeout: float | None = None*, *close_env: bool = True*) → None[[source]](../../_modules/torchrl/collectors/_single.html#Collector.async_shutdown)

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

fake_tensordict() → [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)[[source]](../../_modules/torchrl/collectors/_single.html#Collector.fake_tensordict)

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

get_model(*model_id: str*)[[source]](../../_modules/torchrl/collectors/_single.html#Collector.get_model)

Get model instance by ID (for weight sync schemes).

Parameters:

**model_id** - Model identifier (e.g., "policy", "value_net")

Returns:

The model instance

Raises:

**ValueError** - If model_id is not recognized

get_policy_version() → str | int | None[[source]](../../_modules/torchrl/collectors/_single.html#Collector.get_policy_version)

Get the current policy version.

This method exists to support remote calls in Ray actors, since properties
cannot be accessed directly through Ray's RPC mechanism.

Returns:

The current version number (int) or UUID (str), or None if version tracking is disabled.

getattr_env(*attr*)[[source]](../../_modules/torchrl/collectors/_single.html#Collector.getattr_env)

Get an attribute from the environment.

getattr_policy(*attr*)[[source]](../../_modules/torchrl/collectors/_single.html#Collector.getattr_policy)

Get an attribute from the policy.

getattr_rb(*attr*)[[source]](../../_modules/torchrl/collectors/_single.html#Collector.getattr_rb)

Get an attribute from the replay buffer.

increment_version()[[source]](../../_modules/torchrl/collectors/_single.html#Collector.increment_version)

Increment the policy version.

init_updater(**args*, ***kwargs*)

Initialize the weight updater with custom arguments.

This method passes the arguments to the weight updater's init method.
If no weight updater is set, this is a no-op.

Parameters:

- ***args** - Positional arguments for weight updater initialization
- ****kwargs** - Keyword arguments for weight updater initialization

iterator() → Iterator[[TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)][[source]](../../_modules/torchrl/collectors/_single.html#Collector.iterator)

Iterates through the DataCollector.

Yields: TensorDictBase objects containing (chunks of) trajectories

load_state_dict(*state_dict: OrderedDict*, ***kwargs*) → None[[source]](../../_modules/torchrl/collectors/_single.html#Collector.load_state_dict)

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

reset(*index=None*, ***kwargs*) → None[[source]](../../_modules/torchrl/collectors/_single.html#Collector.reset)

Resets the environments to a new initial state.

rollout() → [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)[[source]](../../_modules/torchrl/collectors/_single.html#Collector.rollout)

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

set_seed(*seed: int*, *static_seed: bool = False*) → int[[source]](../../_modules/torchrl/collectors/_single.html#Collector.set_seed)

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

shutdown(*timeout: float | None = None*, *close_env: bool = True*, *raise_on_error: bool = True*) → None[[source]](../../_modules/torchrl/collectors/_single.html#Collector.shutdown)

Shuts down all workers and/or closes the local environment.

Parameters:

- **timeout** (*float**,**optional*) - The timeout for closing pipes between workers.
No effect for this class.
- **close_env** (*bool**,**optional*) - Whether to close the environment. Defaults to True.
- **raise_on_error** (*bool**,**optional*) - Whether to raise an error if the shutdown fails. Defaults to True.

start()[[source]](../../_modules/torchrl/collectors/_single.html#Collector.start)

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

state_dict() → OrderedDict[[source]](../../_modules/torchrl/collectors/_single.html#Collector.state_dict)

Returns the local state_dict of the data collector (environment and policy).

Returns:

an ordered dictionary with fields `"policy_state_dict"` and
"env_state_dict".

update_policy_weights_(*policy_or_weights: [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase) | [TensorDictModuleBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.nn.TensorDictModuleBase.html#tensordict.nn.TensorDictModuleBase) | dict | None = None*, ***, *worker_ids: int | list[int] | [device](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device) | list[[device](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)] | None = None*, ***kwargs*) → None[[source]](../../_modules/torchrl/collectors/_single.html#Collector.update_policy_weights_)

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