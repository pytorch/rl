# Evaluator

*class*torchrl.collectors.Evaluator(*env: [EnvBase](torchrl.envs.EnvBase.html#torchrl.envs.EnvBase) | Callable[[], [EnvBase](torchrl.envs.EnvBase.html#torchrl.envs.EnvBase)]*, *policy: [TensorDictModuleBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.nn.TensorDictModuleBase.html#tensordict.nn.TensorDictModuleBase) | Callable | None = None*, ***, *policy_factory: Callable[[...], Callable] | None = None*, *num_trajectories: int = 10*, *max_steps: int | None = None*, *frames_per_batch: int | None = None*, *collector_cls: type | str | None = None*, *collector_kwargs: dict | None = None*, *weight_sync_schemes: dict[str, Any] | None = None*, *log_prefix: str = 'eval'*, *reward_keys: str | tuple[str, ...] = ('next', 'reward')*, *done_keys: str | tuple[str, ...] = ('next', 'done')*, *device: [device](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device) | str | None = None*, *exploration_type: [InteractionType](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.nn.InteractionType.html#tensordict.nn.InteractionType) = InteractionType.DETERMINISTIC*, *metrics_fn: Callable[[[TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)], dict[str, float]] | None = None*, *dump_video: bool = True*, *on_result: Callable[[[TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)], None] | None = None*, *busy_policy: str = 'skip'*, *backend: str = 'thread'*, *init_fn: Callable[[], None] | None = None*, *num_gpus: int = 1*, *ray_kwargs: dict | None = None*)[[source]](../../_modules/torchrl/collectors/_evaluator.html#Evaluator)

Unified sync / async evaluator with pluggable backend.

The evaluator wraps an environment and a policy and provides two modes of
operation:

- **Synchronous** - call `evaluate()` to run a blocking evaluation
and get metrics back immediately.
- **Asynchronous** - call `trigger_eval()` to kick off an evaluation
in the background, then `poll()` (non-blocking) or `wait()`
(blocking) to retrieve the result. Use the `pending` property
to check whether an evaluation is currently in progress. Results can
also be consumed via an `on_result` callback.

Internally, a [`Collector`](torchrl.collectors.Collector.html#torchrl.collectors.Collector) is used with
`trajs_per_batch=num_trajectories` to collect complete episodes. The
collector pre-allocates buffers and writes in-place -- O(1) GPU
allocations vs O(n) per step -- yielding significant speedups for
batched eval environments.

Three backends are available:

- `"thread"` (default) - runs in a daemon thread. Low overhead,
well suited for GPU-bound evaluation where the GIL is released by
CUDA ops. When *env* is a callable **and** *policy_factory* is
provided, both are created lazily inside the worker thread, which is
useful for dedicated eval devices.
- `"process"` - runs in a child process (`spawn` context). The
env and policy are always created inside the child process, giving
full CUDA context isolation and avoiding the GIL entirely. Requires
*env* to be a callable and *policy_factory* to be provided.
- `"ray"` - runs in a Ray actor, suitable for distributed setups.
Requires *env* to be a callable and *policy_factory* to be provided.

**Backpressure / overlap policy**: calling `trigger_eval()` while a
previous evaluation is still running skips the new request by default
(`busy_policy="skip"`), raises immediately
(`busy_policy="error"`), or queues the new request
(`busy_policy="queue"`). `trigger_eval()` returns `True` when a
request was accepted and `False` when it was skipped.

**Callback thread-safety**: when `on_result` is provided, it is
invoked from the evaluator's async coordination thread after the
rollout completes. If the callback writes to a logger, the callback is
responsible for any locking it needs.

**Dedicated eval device** (multi-GPU example):

```
evaluator = Evaluator(
 lambda: make_env(device="cuda:7"),
 policy_factory=lambda env: make_policy(env).to("cuda:7"),
 max_steps=1000,
 backend="process", # or "thread"
)
```

**Batched eval environments**: for best results, add a
[`RewardSum`](torchrl.envs.transforms.RewardSum.html#torchrl.envs.transforms.RewardSum) transform to the eval
env so that per-episode returns are tracked. Without it, the
evaluator falls back to summing raw rewards over each trajectory.

Parameters:

- **env** - An [`EnvBase`](torchrl.envs.EnvBase.html#torchrl.envs.EnvBase) instance **or** a callable
that returns one. For the `"process"` and `"ray"` backends
the callable form is required. For the `"thread"` backend,
when combined with *policy_factory*, passing a callable defers
construction to the worker thread.
- **policy** - The evaluation policy. Mutually exclusive with
*policy_factory*.

Keyword Arguments:

- **policy_factory** - A callable `(env) -> policy` used to build the
policy. Required for the `"process"` and `"ray"` backends.
For `"thread"`, if both *env* (callable) and *policy_factory*
are provided, construction is deferred to the worker thread.
- **num_trajectories** (*int*) - Number of complete episodes to collect per
evaluation round. A [`Collector`](torchrl.collectors.Collector.html#torchrl.collectors.Collector) is
used internally with `trajs_per_batch=num_trajectories`.
Default: `10`.
- **max_steps** (*int**or**None*) - Maximum environment steps per episode,
passed as `max_frames_per_traj` to the internal collector.
When `None`, episodes run until done with no step limit.
Default: `None`.
- **frames_per_batch** (*int**or**None*) - Internal collection batch size
(env steps per collector iteration). If `None`, defaults to
`max_steps`. This is purely internal -- output granularity
is controlled by *num_trajectories*.
- **collector_cls** - Which collector class to use. Accepts a class or a
string name resolved from `torchrl.collectors` (e.g.
`"Collector"`).
Default: `None` (uses [`Collector`](torchrl.collectors.Collector.html#torchrl.collectors.Collector)).
- **collector_kwargs** (*dict**or**None*) - Extra keyword arguments forwarded
to the collector constructor.
- **log_prefix** (*str*) - Prefix prepended to all logged metric names.
Default: `"eval"`.
- **reward_keys** - Nested key(s) for reading the reward from the
tensordict. Default: `("next", "reward")`.
- **done_keys** - Nested key(s) for reading the done flag.
Default: `("next", "done")`.
- **device** - Device for the evaluation policy. If `None`, inferred
from the policy parameters.
- **exploration_type** - Exploration mode during evaluation.
Default: `ExplorationType.DETERMINISTIC`.
- **metrics_fn** - Optional `(TensorDictBase) -> dict[str, float]`
called on every trajectory batch to extract custom metrics.
- **dump_video** (*bool*) - Call `dump()` on `VideoRecorder`
transforms after each evaluation. Process-backed collectors invoke
the transform in their worker and can use a service-backed logger.
Default: `True`.
- **on_result** - Optional `(TensorDictBase) -> None` invoked after each
completed evaluation. The callback receives a flat tensordict
with the same prefixed metric names returned by
`evaluate()`, `poll()`, and `wait()`.
- **busy_policy** (*str*) -

Behaviour when `trigger_eval()` is called
while another async evaluation is still pending. `"skip"`
returns `False` without scheduling a new request (default).
`"error"` raises immediately. `"queue"` enqueues the new
request and runs it when the current evaluation finishes.

Warning

With `busy_policy="queue"`, each queued request stores a
copy of the weights dict. For large models this can consume
significant memory. Prefer checking `pending` and
skipping triggers instead.
- **weight_sync_schemes** (*dict**or**None*) -

A dict mapping model IDs to
[`WeightSyncScheme`](torchrl.weight_update.WeightSyncScheme.html#torchrl.weight_update.WeightSyncScheme) instances.
When provided, a [`MultiSyncCollector`](torchrl.collectors.MultiSyncCollector.html#torchrl.collectors.MultiSyncCollector)
with a single worker is used for process-level CUDA isolation
and scheme-based weight transfer. Model IDs follow the
collector convention: `"policy"` for the main policy,
`"env.transform[0]"` for env transforms, etc.
Example:

```
from torchrl.weight_update import MultiProcessedWeightSyncScheme
evaluator = Evaluator(
 env=make_eval_env,
 policy_factory=make_eval_policy,
 weight_sync_schemes={
 "policy": MultiProcessedWeightSyncScheme(),
 "env.transform[0]": MultiProcessedWeightSyncScheme(),
 },
 max_steps=1000,
)
```
- **backend** (*str*) - `"thread"` (default), `"process"`, or `"ray"`.
The `"process"` backend is implemented as a thread backend
with a [`MultiSyncCollector`](torchrl.collectors.MultiSyncCollector.html#torchrl.collectors.MultiSyncCollector) (1
worker) running in a child process. This provides full CUDA
context isolation without custom queue management.
- **init_fn** - Callable invoked at the start of the worker / actor
process, before any evaluation work (and, ideally, before any
`torch` import inside that process). Used by both the
`"process"` and `"ray"` backends. Typical use case: start
Isaac Lab's `AppLauncher` in headless mode. Ignored by the
`"thread"` backend because no new process is spawned.
- **num_gpus** (*int*) - (*Ray only*) GPUs requested for the actor.
Default: `1`.
- **ray_kwargs** (*dict*) - (*Ray only*) Extra keyword arguments forwarded
to `ray.remote()`.

evaluate(*weights: [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase) | [Module](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module) | None = None*, *step: int | None = None*, ***, *weights_dict: dict[str, [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase) | [Module](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module)] | None = None*) → dict[str, Any][[source]](../../_modules/torchrl/collectors/_evaluator.html#Evaluator.evaluate)

Run a blocking evaluation rollout.

Parameters:

- **weights** - Policy weights to load before the rollout. Accepts a
`TensorDictBase` (e.g. from
`TensorDict.from_module(policy).data`) or an
`nn.Module` (weights are extracted automatically).
If `None` the current policy weights are used.
- **step** - Logging step. If `None` an internal counter is used.

Keyword Arguments:

**weights_dict** - A dict mapping `model_id` strings to weight
sources (`nn.Module` or `TensorDictBase`). Use this
to sync multiple models (e.g. policy + env transforms).
When provided, *weights* is treated as
`weights_dict["policy"]` if `"policy"` is not already
in the dict.

Returns:

dict with at least `"<prefix>/reward"` and
`"<prefix>/episode_length"` keys.

*static*extract_weights(*policy: [Module](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module)*) → [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)[[source]](../../_modules/torchrl/collectors/_evaluator.html#Evaluator.extract_weights)

Extract detached, cloned, CPU weights from a policy.

This is a convenience helper; the returned TensorDict is safe to
pass across threads.

*property*pending*: bool*

Return `True` if an async evaluation is currently in progress.

This can be used to avoid triggering overlapping evaluations:

```
if not evaluator.pending:
 evaluator.trigger_eval(weights, step=step)
```

poll(*timeout: float = 0*) → dict[str, Any] | None[[source]](../../_modules/torchrl/collectors/_evaluator.html#Evaluator.poll)

Return the latest evaluation result if ready, else `None`.

Parameters:

**timeout** - Seconds to wait. `0` means non-blocking.

shutdown(*timeout: float = 5.0*) → None[[source]](../../_modules/torchrl/collectors/_evaluator.html#Evaluator.shutdown)

Cancel any running evaluation, clean up resources.

trigger_eval(*weights: [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase) | [Module](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module) | None = None*, *step: int | None = None*, ***, *weights_dict: dict[str, [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase) | [Module](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module)] | None = None*) → bool[[source]](../../_modules/torchrl/collectors/_evaluator.html#Evaluator.trigger_eval)

Start an async evaluation.

Parameters:

- **weights** - Policy weights to load. See `evaluate()`.
- **step** - Logging step. See `evaluate()`.
- **weights_dict** - Multi-model weights dict. See `evaluate()`.

Returns:

`True` if an evaluation request was scheduled, `False` if it
was skipped because another request was pending and
`busy_policy="skip"`.

wait(*timeout: float | None = None*) → dict[str, Any] | None[[source]](../../_modules/torchrl/collectors/_evaluator.html#Evaluator.wait)

Block until the current evaluation finishes.

Parameters:

**timeout** - Max seconds to wait. `None` waits forever.