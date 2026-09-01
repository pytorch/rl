# RayEvalWorker

*class*torchrl.collectors.distributed.RayEvalWorker(*init_fn: Callable[[], None] | None*, *env_maker: Callable[[], Any]*, *policy_maker: Callable[[Any], Any]*, ***, *num_gpus: int = 1*, *reward_keys: tuple[str, ...] = ('next', 'reward')*, *name: str | None = None*, ***remote_kwargs: Any*)[[source]](../../_modules/torchrl/collectors/distributed/ray_eval_worker.html#RayEvalWorker)

Asynchronous evaluation worker backed by a Ray actor.

The worker creates a **new Python process** (via Ray) and inside it:

1. Calls *init_fn* - use this for any process-level setup that must happen
before other imports (e.g. Isaac Lab `AppLauncher`).
2. Creates the environment via *env_maker*.
3. Creates the policy via *policy_maker(env)*.

Thereafter, `submit()` sends new policy weights and triggers an
evaluation rollout. `poll()` returns the result (reward and optional
video frames) when the rollout finishes, or `None` if it is still
running.

If a *name* is provided the actor is registered with Ray under that name,
allowing other processes (or a later session) to reconnect to the same
running actor via `from_name()`.

Parameters:

- **init_fn** - Optional callable invoked at the very start of the actor
process, before *env_maker* or *policy_maker*. All imports should
be **local** inside this callable so that the actor's fresh Python
process can control import order. Set to `None` to skip.
- **env_maker** - Callable that returns a TorchRL environment. Called once
inside the actor after *init_fn*. If the underlying environment
supports `render_mode="rgb_array"`, the actor will call
`render()` on each evaluation step and return the frames.
- **policy_maker** - Callable `(env) -> policy` that builds the policy
module given the environment. Called once inside the actor after
the environment has been created.
- **num_gpus** - Number of GPUs to request from Ray for this actor.
Defaults to 1.
- **reward_keys** - Nested key(s) used to read the reward from the rollout
tensordict. Defaults to `("next", "reward")`.
- **name** - Optional name for the Ray actor. When set, the actor is
registered under this name and can be retrieved later with
`from_name()`.
- ****remote_kwargs** - Extra keyword arguments forwarded to
`ray.remote()` when creating the actor class (e.g.
`num_cpus`, `runtime_env`).

*classmethod*from_name(*name: str*, ***, *reward_keys: tuple[str, ...] = ('next', 'reward')*) → RayEvalWorker[[source]](../../_modules/torchrl/collectors/distributed/ray_eval_worker.html#RayEvalWorker.from_name)

Connect to an existing named `RayEvalWorker` actor.

This is useful when one process creates the worker (with a *name*)
and another process wants to submit evaluations or poll results on
the same actor.

Parameters:

- **name** - The actor name that was passed to the constructor.
- **reward_keys** - Nested key(s) used to read the reward from the
rollout tensordict. Defaults to `("next", "reward")`.

poll(*timeout: float = 0*) → dict | None[[source]](../../_modules/torchrl/collectors/distributed/ray_eval_worker.html#RayEvalWorker.poll)

Return the evaluation result if ready, otherwise `None`.

The returned dict contains:

- `"reward"` - scalar mean episode reward.
- `"frames"` - `(T, H, W, 3)` uint8 CPU tensor of rendered
frames, or `None` if the environment does not render.

Parameters:

**timeout** - Seconds to wait for the result. `0` means
non-blocking (return immediately if not ready).

shutdown() → None[[source]](../../_modules/torchrl/collectors/distributed/ray_eval_worker.html#RayEvalWorker.shutdown)

Close the environment and kill the actor.

Safe to call multiple times or after `ray.shutdown()` has already
torn down the actor (e.g. via a test fixture).

submit(*weights: Any*, *max_steps: int*, ***, *deterministic: bool = True*, *break_when_any_done: bool = True*) → None[[source]](../../_modules/torchrl/collectors/distributed/ray_eval_worker.html#RayEvalWorker.submit)

Start an asynchronous evaluation rollout.

If a previous rollout is still running its result is silently
discarded (fire-and-forget semantics).

Parameters:

- **weights** - Policy weights, typically obtained via
`TensorDict.from_module(policy).data.detach().cpu()`.
- **max_steps** - Maximum number of environment steps per rollout.
- **deterministic** - If `True`, use deterministic exploration.
- **break_when_any_done** - If `True`, stop the rollout as soon as
any sub-environment reports `done`.