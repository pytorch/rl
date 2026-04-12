# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Unified evaluator with pluggable thread / process / Ray backends.

This module provides :class:`Evaluator`, a single entry-point for running
evaluation rollouts either synchronously (blocking) or asynchronously
(fire-and-forget) during RL training.

Internally, the evaluator uses a :class:`~torchrl.collectors.Collector` with
``trajs_per_batch`` to collect complete trajectories.  The collector
pre-allocates buffers and writes in-place — O(1) GPU allocations vs O(n)
per step with ``env.rollout()`` — yielding significant speedups for batched
eval environments.

Typical usage -- **thread backend** (default)::

    from torchrl.collectors import Evaluator

    evaluator = Evaluator(
        make_eval_env,
        eval_policy,
        max_steps=1000,
        logger=logger,
    )

    # Non-blocking: trigger and poll later
    evaluator.trigger_eval(train_policy, step=collected_frames)
    result = evaluator.poll()  # None while still running

    # Or blocking:
    metrics = evaluator.evaluate(train_policy, step=collected_frames)

    evaluator.shutdown()

Typical usage -- **process backend** (dedicated eval device)::

    evaluator = Evaluator(
        lambda: make_eval_env(device="cuda:7"),
        policy_factory=lambda env: make_eval_policy(env).to("cuda:7"),
        max_steps=1000,
        logger=logger,
        backend="process",
    )

    evaluator.trigger_eval(train_policy, step=collected_frames)
    # ... training continues ...
    if not evaluator.pending:
        result = evaluator.poll()

    evaluator.shutdown()

Typical usage -- **Ray backend**::

    evaluator = Evaluator(
        make_eval_env,
        policy_factory=make_eval_policy,
        max_steps=1000,
        logger=logger,
        backend="ray",
        init_fn=my_init_fn,
        num_gpus=1,
    )

    evaluator.trigger_eval(weights, step=step)
    result = evaluator.poll()
    evaluator.shutdown()
"""
from __future__ import annotations

import abc
import importlib
import logging
import multiprocessing as mp
import threading
from collections.abc import Callable
from typing import Any

import torch
import torch.nn as nn

from tensordict import TensorDict, TensorDictBase
from tensordict.nn import TensorDictModuleBase

from torchrl.envs import EnvBase
from torchrl.envs.utils import ExplorationType, set_exploration_type

_has_ray = importlib.util.find_spec("ray") is not None

logger = logging.getLogger(__name__)

NestedKey = str | tuple[str, ...]


class Evaluator:
    """Unified sync / async evaluator with pluggable backend.

    The evaluator wraps an environment and a policy and provides two modes of
    operation:

    * **Synchronous** -- call :meth:`evaluate` to run a blocking evaluation
      and get metrics back immediately.
    * **Asynchronous** -- call :meth:`trigger_eval` to kick off an evaluation
      in the background, then :meth:`poll` (non-blocking) or :meth:`wait`
      (blocking) to retrieve the result.  Use the :attr:`pending` property
      to check whether an evaluation is currently in progress.
      Results are also auto-logged if a *logger* is provided.

    Internally, a :class:`~torchrl.collectors.Collector` is used with
    ``trajs_per_batch=num_trajectories`` to collect complete episodes.  The
    collector pre-allocates buffers and writes in-place — O(1) GPU
    allocations vs O(n) per step — yielding significant speedups for
    batched eval environments.

    Three backends are available:

    * ``"thread"`` (default) -- runs in a daemon thread.  Low overhead,
      well suited for GPU-bound evaluation where the GIL is released by
      CUDA ops.  When *env* is a callable **and** *policy_factory* is
      provided, both are created lazily inside the worker thread, which is
      useful for dedicated eval devices.
    * ``"process"`` -- runs in a child process (``spawn`` context).  The
      env and policy are always created inside the child process, giving
      full CUDA context isolation and avoiding the GIL entirely.  Requires
      *env* to be a callable and *policy_factory* to be provided.
    * ``"ray"`` -- runs in a Ray actor, suitable for distributed setups.
      Requires *env* to be a callable and *policy_factory* to be provided.

    **Backpressure / overlap policy**: calling :meth:`trigger_eval` while a
    previous evaluation is still running **drops** the in-progress result
    (fire-and-forget).  The new evaluation starts as soon as the background
    worker finishes the current collection — there is no queue, no
    coalescing, and no error.  Only the most-recently-triggered evaluation
    will produce a result.  Use :attr:`pending` to conditionally skip
    trigger calls::

        if not evaluator.pending:
            evaluator.trigger_eval(weights, step=step)

    **Logging thread-safety**: all logger writes (scalar metrics, video
    encoding) happen on the **caller thread** inside :meth:`poll`,
    :meth:`wait`, or :meth:`evaluate`.  The background worker only computes
    plain metrics and returns them; it never touches the logger.

    **Dedicated eval device** (multi-GPU example)::

        evaluator = Evaluator(
            lambda: make_env(device="cuda:7"),
            policy_factory=lambda env: make_policy(env).to("cuda:7"),
            max_steps=1000,
            backend="process",  # or "thread"
        )

    **Batched eval environments**: for best results, add a
    :class:`~torchrl.envs.transforms.RewardSum` transform to the eval
    env so that per-episode returns are tracked.  Without it, the
    evaluator falls back to summing raw rewards over each trajectory.

    Args:
        env: An :class:`~torchrl.envs.EnvBase` instance **or** a callable
            that returns one.  For the ``"process"`` and ``"ray"`` backends
            the callable form is required.  For the ``"thread"`` backend,
            when combined with *policy_factory*, passing a callable defers
            construction to the worker thread.
        policy: The evaluation policy.  Mutually exclusive with
            *policy_factory*.

    Keyword Args:
        policy_factory: A callable ``(env) -> policy`` used to build the
            policy.  Required for the ``"process"`` and ``"ray"`` backends.
            For ``"thread"``, if both *env* (callable) and *policy_factory*
            are provided, construction is deferred to the worker thread.
        num_trajectories (int): Number of complete episodes to collect per
            evaluation round.  A :class:`~torchrl.collectors.Collector` is
            used internally with ``trajs_per_batch=num_trajectories``.
            Default: ``10``.
        max_steps (int): Maximum environment steps per episode, passed as
            ``max_frames_per_traj`` to the internal collector.
        frames_per_batch (int or None): Internal collection batch size
            (env steps per collector iteration).  If ``None``, defaults to
            ``max_steps``.  This is purely internal — output granularity
            is controlled by *num_trajectories*.
        collector_cls: Which collector class to use.  Accepts a class or a
            string name resolved from :mod:`torchrl.collectors` (e.g.
            ``"Collector"``).
            Default: ``None`` (uses :class:`~torchrl.collectors.Collector`).
        collector_kwargs (dict or None): Extra keyword arguments forwarded
            to the collector constructor.
        logger: Optional :class:`~torchrl.record.loggers.Logger` for
            automatic metric / video logging.
        log_prefix (str): Prefix prepended to all logged metric names.
            Default: ``"eval"``.
        reward_keys: Nested key(s) for reading the reward from the
            tensordict.  Default: ``("next", "reward")``.
        done_keys: Nested key(s) for reading the done flag.
            Default: ``("next", "done")``.
        device: Device for the evaluation policy.  If ``None``, inferred
            from the policy parameters.
        exploration_type: Exploration mode during evaluation.
            Default: :attr:`ExplorationType.DETERMINISTIC`.
        metrics_fn: Optional ``(TensorDictBase) -> dict[str, float]``
            called on every trajectory batch to extract custom metrics.
        dump_video (bool): Call ``dump()`` on :class:`VideoRecorder`
            transforms after each evaluation (thread backend only).
            Default: ``True``.
        callback: Optional ``(dict, int) -> None`` invoked with
            ``(metrics, step)`` after each completed evaluation.
        logger_lock: A :class:`threading.Lock` shared with the training
            loop to serialise logger access.  If ``None`` a private lock
            is created.
        backend (str): ``"thread"`` (default), ``"process"``, or ``"ray"``.
        init_fn: (*Ray only*) Callable invoked at the start of the actor
            process, before any ``torch`` import.
        num_gpus (int): (*Ray only*) GPUs requested for the actor.
            Default: ``1``.
        ray_kwargs (dict): (*Ray only*) Extra keyword arguments forwarded
            to ``ray.remote()``.
    """

    def __init__(
        self,
        env: EnvBase | Callable[[], EnvBase],
        policy: TensorDictModuleBase | Callable | None = None,
        *,
        policy_factory: Callable[..., Callable] | None = None,
        num_trajectories: int = 10,
        max_steps: int,
        frames_per_batch: int | None = None,
        collector_cls: type | str | None = None,
        collector_kwargs: dict | None = None,
        logger=None,
        log_prefix: str = "eval",
        reward_keys: NestedKey = ("next", "reward"),
        done_keys: NestedKey = ("next", "done"),
        device: torch.device | str | None = None,
        exploration_type: ExplorationType = ExplorationType.DETERMINISTIC,
        metrics_fn: Callable[[TensorDictBase], dict[str, float]] | None = None,
        dump_video: bool = True,
        callback: Callable[[dict, int], None] | None = None,
        logger_lock: threading.Lock | None = None,
        # Backend selection
        backend: str = "thread",
        # Ray-specific
        init_fn: Callable[[], None] | None = None,
        num_gpus: int = 1,
        ray_kwargs: dict | None = None,
    ) -> None:
        self._logger = logger
        self._log_prefix = log_prefix
        self._reward_keys = reward_keys
        self._done_keys = done_keys
        self._exploration_type = exploration_type
        self._metrics_fn = metrics_fn
        self._callback = callback
        self._logger_lock = logger_lock or threading.Lock()
        self._step_counter = 0
        self._dump_video = dump_video

        if backend == "thread":
            self._backend: _EvalBackend = _ThreadEvalBackend(
                env=env,
                policy=policy,
                policy_factory=policy_factory,
                num_trajectories=num_trajectories,
                max_steps=max_steps,
                frames_per_batch=frames_per_batch,
                collector_cls=collector_cls,
                collector_kwargs=collector_kwargs,
                device=device,
                exploration_type=exploration_type,
                reward_keys=reward_keys,
                done_keys=done_keys,
                metrics_fn=metrics_fn,
            )
        elif backend == "process":
            env_is_callable = callable(env) and not isinstance(env, EnvBase)
            if not env_is_callable:
                raise ValueError(
                    "The 'process' backend requires `env` to be a callable "
                    "(factory function) because the env is created inside "
                    "the child process."
                )
            if policy_factory is None:
                raise ValueError(
                    "The 'process' backend requires `policy_factory` (a "
                    "callable `(env) -> policy`) because the policy is "
                    "created inside the child process."
                )
            self._backend = _ProcessEvalBackend(
                env_factory=env,
                policy_factory=policy_factory,
                num_trajectories=num_trajectories,
                max_steps=max_steps,
                frames_per_batch=frames_per_batch,
                collector_cls=collector_cls,
                collector_kwargs=collector_kwargs,
                exploration_type=exploration_type,
                reward_keys=reward_keys,
                done_keys=done_keys,
                metrics_fn=metrics_fn,
            )
        elif backend == "ray":
            self._backend = _RayEvalBackend(
                env_maker=env,
                policy_factory=policy_factory,
                max_steps=max_steps,
                reward_keys=reward_keys,
                init_fn=init_fn,
                num_gpus=num_gpus,
                ray_kwargs=ray_kwargs or {},
            )
        else:
            raise ValueError(
                f"Unknown backend {backend!r}. " "Choose 'thread', 'process', or 'ray'."
            )

    # ------------------------------------------------------------------
    # Synchronous API
    # ------------------------------------------------------------------

    def evaluate(
        self,
        weights: TensorDictBase | nn.Module | None = None,
        step: int | None = None,
    ) -> dict[str, Any]:
        """Run a blocking evaluation rollout.

        Args:
            weights: Policy weights to load before the rollout.  Accepts a
                :class:`TensorDictBase` (e.g. from
                ``TensorDict.from_module(policy).data``) or an
                :class:`nn.Module` (weights are extracted automatically).
                If ``None`` the current policy weights are used.
            step: Logging step.  If ``None`` an internal counter is used.

        Returns:
            dict with at least ``"<prefix>/reward"`` and
            ``"<prefix>/episode_length"`` keys.
        """
        weights = self._prepare_weights(weights)
        step = self._next_step(step)
        raw = self._backend.run_sync(weights, step)
        return self._finalize(raw)

    # ------------------------------------------------------------------
    # Asynchronous API
    # ------------------------------------------------------------------

    def trigger_eval(
        self,
        weights: TensorDictBase | nn.Module | None = None,
        step: int | None = None,
    ) -> None:
        """Start an async evaluation (fire-and-forget).

        If a previous evaluation is still running its result will be
        discarded.

        Args:
            weights: See :meth:`evaluate`.
            step: See :meth:`evaluate`.
        """
        weights = self._prepare_weights(weights)
        step = self._next_step(step)
        self._backend.submit(weights, step)

    def poll(self, timeout: float = 0) -> dict[str, Any] | None:
        """Return the latest evaluation result if ready, else ``None``.

        Args:
            timeout: Seconds to wait.  ``0`` means non-blocking.
        """
        raw = self._backend.poll(timeout)
        if raw is None:
            return None
        return self._finalize(raw)

    def wait(self, timeout: float | None = None) -> dict[str, Any] | None:
        """Block until the current evaluation finishes.

        Args:
            timeout: Max seconds to wait.  ``None`` waits forever.
        """
        raw = self._backend.wait(timeout)
        if raw is None:
            return None
        return self._finalize(raw)

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    @property
    def pending(self) -> bool:
        """Return ``True`` if an async evaluation is currently in progress.

        This can be used to avoid triggering overlapping evaluations::

            if not evaluator.pending:
                evaluator.trigger_eval(weights, step=step)
        """
        return self._backend.pending

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def shutdown(self, timeout: float = 5.0) -> None:
        """Cancel any running evaluation, clean up resources."""
        self._backend.shutdown(timeout)

    def __del__(self):
        try:
            self.shutdown(timeout=2.0)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def extract_weights(policy: nn.Module) -> TensorDictBase:
        """Extract detached, cloned, CPU weights from a policy.

        This is a convenience helper; the returned TensorDict is safe to
        pass across threads.
        """
        return TensorDict.from_module(policy).data.detach().clone().cpu()

    def _prepare_weights(
        self, weights: TensorDictBase | nn.Module | None
    ) -> TensorDictBase | None:
        if weights is None:
            return None
        if isinstance(weights, nn.Module):
            return self.extract_weights(weights)
        return weights.detach().clone().cpu()

    def _next_step(self, step: int | None) -> int:
        if step is not None:
            return step
        s = self._step_counter
        self._step_counter += 1
        return s

    def _finalize(self, raw: dict[str, Any]) -> dict[str, Any]:
        """Format metrics, dump video, log, and invoke callback.

        All heavy I/O (video dump, logger writes) happens here on the
        **caller thread**, never inside the background eval thread.
        """
        step = raw.get("_step")
        metrics = self._format_metrics(raw)

        # Video dump -- on the caller thread so logger writes are safe
        if self._dump_video:
            self._backend.dump_video(step=step)

        self._auto_log(metrics, step)
        if self._callback is not None:
            self._callback(metrics, step)
        return metrics

    def _format_metrics(self, raw: dict[str, Any]) -> dict[str, Any]:
        prefix = self._log_prefix
        out: dict[str, Any] = {}
        for key in ("reward", "reward_std", "num_episodes", "episode_length"):
            if key in raw:
                out[f"{prefix}/{key}"] = raw[key]
        if "frames" in raw and raw["frames"] is not None:
            out[f"{prefix}/video"] = raw["frames"]
        if "_step" in raw:
            out[f"{prefix}/step"] = raw["_step"]
        # Custom metrics (already prefixed by backend or metrics_fn)
        for k, v in raw.items():
            if k.startswith("custom/"):
                out[f"{prefix}/{k[7:]}"] = v
        return out

    def _auto_log(self, metrics: dict[str, Any], step: int | None) -> None:
        if self._logger is None:
            return
        with self._logger_lock:
            # Separate non-scalar entries from scalar metrics for logging
            video = metrics.get(f"{self._log_prefix}/video")
            step_key = f"{self._log_prefix}/step"
            scalars = {
                k: v
                for k, v in metrics.items()
                if k != f"{self._log_prefix}/video" and k != step_key
            }
            if scalars:
                self._logger.log_metrics(scalars, step=step)
            if video is not None:
                self._logger.log_video(f"{self._log_prefix}/video", video, step=step)


# ======================================================================
# Backend interface
# ======================================================================


class _EvalBackend(abc.ABC):
    """Internal contract that each backend implements."""

    @abc.abstractmethod
    def run_sync(self, weights: TensorDictBase | None, step: int) -> dict[str, Any]:
        """Run a blocking evaluation and return raw results."""
        ...

    @abc.abstractmethod
    def submit(self, weights: TensorDictBase | None, step: int) -> None:
        """Start an async evaluation (fire-and-forget)."""
        ...

    @abc.abstractmethod
    def poll(self, timeout: float) -> dict[str, Any] | None:
        """Non-blocking check for results."""
        ...

    @abc.abstractmethod
    def wait(self, timeout: float | None) -> dict[str, Any] | None:
        """Blocking wait for results."""
        ...

    @property
    @abc.abstractmethod
    def pending(self) -> bool:
        """Return ``True`` if an async evaluation is currently in progress."""
        ...

    @abc.abstractmethod
    def shutdown(self, timeout: float) -> None:
        """Clean up resources."""
        ...

    def dump_video(self, step: int | None = None) -> None:
        """Dump accumulated video frames (called on caller thread).

        Default is a no-op; overridden by backends that support video.
        """
        return None


# ======================================================================
# Shared metric helpers
# ======================================================================

_EPISODE_REWARD_KEY = ("next", "episode_reward")


def _extract_metrics_from_trajectories(
    traj_batch: TensorDictBase,
    reward_keys: NestedKey,
    done_keys: NestedKey,
    metrics_fn: Callable[[TensorDictBase], dict[str, float]] | None,
) -> dict[str, Any]:
    """Extract evaluation metrics from a trajectory batch produced by a collector.

    *traj_batch* has shape ``(num_trajectories, max_traj_len)`` with a
    ``("collector", "mask")`` boolean field marking valid timesteps.
    """
    mask = traj_batch.get(("collector", "mask"))  # [N, T]
    num_trajectories = traj_batch.shape[0]

    episode_rewards = []
    episode_lengths = []

    ep_reward_td = traj_batch.get(_EPISODE_REWARD_KEY, None)
    step_count_td = traj_batch.get(("next", "step_count"), None)
    reward_td = traj_batch.get(reward_keys, None)

    for i in range(num_trajectories):
        traj_mask = mask[i]  # [T]
        if traj_mask.ndim > 1:
            traj_mask = traj_mask.squeeze(-1)
        valid_len = traj_mask.sum().item()
        if valid_len == 0:
            continue

        # Last valid index
        last_idx = int(valid_len) - 1

        if ep_reward_td is not None:
            # Prefer episode_reward from RewardSum (cumulative return)
            r = ep_reward_td[i, last_idx]
            if r.ndim > 0:
                r = r.squeeze(-1)
            episode_rewards.append(r.item())
        elif reward_td is not None:
            # Fallback: sum raw rewards over valid trajectory steps
            valid_rewards = reward_td[i, : int(valid_len)]
            if valid_rewards.ndim > 1:
                valid_rewards = valid_rewards.squeeze(-1)
            episode_rewards.append(valid_rewards.sum().item())

        if step_count_td is not None:
            ep_len = step_count_td[i, last_idx]
            if ep_len.ndim > 0:
                ep_len = ep_len.squeeze(-1)
            episode_lengths.append(ep_len.item())
        else:
            episode_lengths.append(float(valid_len))

    num_episodes = len(episode_rewards)

    if num_episodes > 0:
        rewards_t = torch.tensor(episode_rewards)
        mean_reward = rewards_t.mean().item()
        std_reward = rewards_t.std().item() if num_episodes > 1 else 0.0
        mean_length = sum(episode_lengths) / len(episode_lengths)
    else:
        mean_reward = float("nan")
        std_reward = float("nan")
        mean_length = float("nan")

    metrics: dict[str, Any] = {
        "reward": mean_reward,
        "reward_std": std_reward,
        "num_episodes": num_episodes,
        "episode_length": mean_length,
    }

    if metrics_fn is not None:
        custom = metrics_fn(traj_batch)
        for k, v in custom.items():
            metrics[f"custom/{k}"] = v

    return metrics


def _env_has_step_count(env: EnvBase) -> bool:
    """Check if the environment already has a StepCounter transform."""
    for key in env.output_spec.keys(True, True):
        if isinstance(key, str):
            key = (key,)
        if "step_count" in key:
            return True
    return False


def _resolve_collector_cls(cls_or_name: type | str | None):
    """Resolve a collector class from a string name or return as-is."""
    if cls_or_name is None:
        from torchrl.collectors import Collector

        return Collector
    if isinstance(cls_or_name, str):
        import torchrl.collectors as mod

        return getattr(mod, cls_or_name)
    return cls_or_name


# ======================================================================
# Thread backend
# ======================================================================


class _ThreadEvalBackend(_EvalBackend):
    """Runs evaluation in a daemon thread using an internal collector.

    When *policy_factory* is provided and *env* is a callable, the
    environment and policy are created **lazily** on the first evaluation
    call.  For async evaluation this means construction happens inside the
    worker thread, which is critical for multi-device setups where the
    eval environment lives on a dedicated GPU.
    """

    def __init__(
        self,
        env: EnvBase | Callable[[], EnvBase],
        policy: TensorDictModuleBase | Callable | None,
        policy_factory: Callable[..., Callable] | None,
        num_trajectories: int,
        max_steps: int,
        frames_per_batch: int | None,
        collector_cls: type | str | None,
        collector_kwargs: dict | None,
        device: torch.device | str | None,
        exploration_type: ExplorationType,
        reward_keys: NestedKey,
        done_keys: NestedKey,
        metrics_fn: Callable[[TensorDictBase], dict[str, float]] | None,
    ) -> None:
        if policy is not None and policy_factory is not None:
            raise ValueError("Provide either `policy` or `policy_factory`, not both.")

        self._env_factory: Callable[[], EnvBase] | None = None
        self._policy_factory: Callable[..., Callable] | None = None

        env_is_callable = callable(env) and not isinstance(env, EnvBase)

        # Lazy path: defer both env and policy creation to the worker thread
        if policy_factory is not None and env_is_callable:
            self._env_factory = env
            self._policy_factory = policy_factory
            self._env: EnvBase | None = None
            self._policy = None
        else:
            # Eager path (existing behaviour)
            if env_is_callable:
                env = env()
            self._env = env

            if policy_factory is not None:
                policy = policy_factory(self._env)
            if policy is None:
                raise ValueError(
                    "Either `policy` or `policy_factory` must be provided."
                )
            self._policy = policy

        # Device -- may be set lazily after env/policy creation
        if device is not None:
            self._device = torch.device(device)
        elif self._policy is not None:
            try:
                self._device = next(self._policy.parameters()).device
            except (StopIteration, AttributeError):
                self._device = torch.device("cpu")
        else:
            self._device: torch.device | None = None

        self._num_trajectories = num_trajectories
        self._max_steps = max_steps
        self._frames_per_batch = frames_per_batch
        self._collector_cls = collector_cls
        self._collector_kwargs = collector_kwargs
        self._exploration_type = exploration_type
        self._reward_keys = reward_keys
        self._done_keys = done_keys
        self._metrics_fn = metrics_fn

        # Collector (created lazily)
        self._collector = None

        # Threading state
        self._lock = threading.Lock()
        self._cancel = threading.Event()
        self._eval_ready = threading.Event()
        self._result_ready = threading.Event()
        self._pending = threading.Event()  # set while an eval is in-flight
        self._pending_request: tuple[TensorDictBase | None, int] | None = None
        self._result: dict[str, Any] | None = None
        self._shutdown_flag = False
        self._thread: threading.Thread | None = None

    # ---- sync ----

    def run_sync(self, weights: TensorDictBase | None, step: int) -> dict[str, Any]:
        self._ensure_collector()
        metrics = self._run_eval(weights)
        metrics["_step"] = step
        return metrics

    # ---- async ----

    def submit(self, weights: TensorDictBase | None, step: int) -> None:
        with self._lock:
            self._cancel.set()  # discard any in-progress result
            self._pending_request = (weights, step)
            self._result = None
            self._result_ready.clear()
            self._pending.set()
        self._eval_ready.set()
        self._ensure_thread()

    def poll(self, timeout: float) -> dict[str, Any] | None:
        if timeout > 0:
            self._result_ready.wait(timeout=timeout)
        with self._lock:
            result = self._result
            if result is not None:
                self._result = None
                self._pending.clear()
            return result

    def wait(self, timeout: float | None) -> dict[str, Any] | None:
        self._result_ready.wait(timeout=timeout)
        with self._lock:
            result = self._result
            if result is not None:
                self._result = None
                self._pending.clear()
            return result

    @property
    def pending(self) -> bool:
        return self._pending.is_set()

    def shutdown(self, timeout: float) -> None:
        self._shutdown_flag = True
        self._cancel.set()
        self._eval_ready.set()  # wake thread so it can exit
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=timeout)
        self._pending.clear()
        if self._collector is not None:
            self._collector.shutdown()

    # ---- internals ----

    def _ensure_env_and_policy(self) -> None:
        """Create env and policy from factories (lazy initialisation)."""
        if self._env is not None:
            return
        if self._env_factory is None:
            raise RuntimeError(
                "Evaluator backend has no env and no env factory -- "
                "this should not happen."
            )
        self._env = self._env_factory()
        self._policy = self._policy_factory(self._env)
        # Free the factories
        self._env_factory = None
        self._policy_factory = None
        # Infer device if not set explicitly
        if self._device is None:
            try:
                self._device = next(self._policy.parameters()).device
            except (StopIteration, AttributeError):
                self._device = torch.device("cpu")

    def _ensure_thread(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._shutdown_flag = False
        self._thread = threading.Thread(target=self._eval_loop, daemon=True)
        self._thread.start()

    def _eval_loop(self) -> None:
        self._ensure_collector()
        while not self._shutdown_flag:
            self._eval_ready.wait(timeout=1.0)
            if self._shutdown_flag:
                break
            self._eval_ready.clear()

            with self._lock:
                request = self._pending_request
                self._pending_request = None
                self._cancel.clear()

            if request is None:
                continue

            weights, step = request
            metrics = self._run_eval(weights)

            if not self._cancel.is_set():
                metrics["_step"] = step
                with self._lock:
                    self._result = metrics
                self._result_ready.set()
            else:
                # Cancelled -- clear pending since the result was discarded
                # (submit() will re-set it for the new request)
                pass

    def _ensure_collector(self) -> None:
        """Create the collector lazily (inside the worker thread)."""
        if self._collector is not None:
            return
        self._ensure_env_and_policy()
        cls = _resolve_collector_cls(self._collector_cls)
        fpb = self._frames_per_batch or self._max_steps or 1000
        # If the env already has a StepCounter (step_count in output),
        # set max_frames_per_traj=0 to avoid conflict with the collector
        # trying to add a second StepCounter.
        max_frames = self._max_steps
        if _env_has_step_count(self._env):
            max_frames = 0
        self._collector = cls(
            create_env_fn=self._env,
            policy=self._policy,
            frames_per_batch=fpb,
            total_frames=-1,
            max_frames_per_traj=max_frames,
            trajs_per_batch=self._num_trajectories,
            exploration_type=self._exploration_type,
            **(self._collector_kwargs or {}),
        )

    def _run_eval(self, weights: TensorDictBase | None) -> dict[str, Any]:
        """Run evaluation using the internal collector."""
        self._ensure_collector()

        if weights is not None:
            weights.to(self._device).to_module(self._policy)

        if isinstance(self._policy, nn.Module):
            self._policy.eval()

        # Reset collector for clean episode boundaries
        self._collector.reset()

        with set_exploration_type(self._exploration_type), torch.no_grad():
            # Each yield gives exactly num_trajectories complete,
            # zero-padded episodes with ("collector", "mask").
            traj_batch = next(iter(self._collector))

        if isinstance(self._policy, nn.Module):
            self._policy.train()

        return _extract_metrics_from_trajectories(
            traj_batch,
            self._reward_keys,
            self._done_keys,
            self._metrics_fn,
        )

    def dump_video(self, step: int | None = None) -> None:
        """Dump accumulated video frames from VideoRecorder transforms.

        Called on the caller thread so that logger writes are thread-safe.
        """
        if self._env is None or not hasattr(self._env, "transform"):
            return
        transform = self._env.transform
        try:
            transforms = iter(transform)
        except TypeError:
            # Single transform, not Compose — wrap in a list
            transforms = [transform]
        for t in transforms:
            if hasattr(t, "dump"):
                t.dump(step=step)


# ======================================================================
# Ray backend
# ======================================================================


class _RayEvalBackend(_EvalBackend):
    """Wraps :class:`RayEvalWorker` to match the backend contract."""

    def __init__(
        self,
        env_maker: Callable[[], Any],
        policy_factory: Callable[..., Any] | None,
        max_steps: int,
        reward_keys: NestedKey,
        init_fn: Callable[[], None] | None,
        num_gpus: int,
        ray_kwargs: dict,
    ) -> None:
        if not _has_ray:
            raise RuntimeError(
                "Ray is required for backend='ray' but could not be found. "
                "Install it with: pip install ray"
            )
        if policy_factory is None:
            raise ValueError(
                "The 'ray' backend requires `policy_factory` (a callable "
                "`(env) -> policy`) because the policy is created inside "
                "the Ray actor process."
            )
        from torchrl.collectors.distributed import RayEvalWorker

        self._worker = RayEvalWorker(
            init_fn=init_fn,
            env_maker=env_maker,
            policy_maker=policy_factory,
            num_gpus=num_gpus,
            reward_keys=reward_keys
            if isinstance(reward_keys, tuple)
            else (reward_keys,),
            **ray_kwargs,
        )
        self._max_steps = max_steps
        self._last_step: int | None = None
        self._pending_flag = False

    def run_sync(self, weights: TensorDictBase | None, step: int) -> dict[str, Any]:
        self._worker.submit(
            weights,
            self._max_steps,
        )
        # Block until done
        result = self._worker.poll(timeout=None)
        result["_step"] = step
        result.setdefault("episode_length", self._max_steps)
        return result

    def submit(self, weights: TensorDictBase | None, step: int) -> None:
        self._last_step = step
        self._pending_flag = True
        self._worker.submit(
            weights,
            self._max_steps,
        )

    @property
    def pending(self) -> bool:
        return self._pending_flag

    def poll(self, timeout: float) -> dict[str, Any] | None:
        result = self._worker.poll(timeout=timeout)
        if result is None:
            return None
        self._pending_flag = False
        result["_step"] = self._last_step
        result.setdefault("episode_length", self._max_steps)
        return result

    def wait(self, timeout: float | None) -> dict[str, Any] | None:
        # RayEvalWorker.poll supports timeout=None for blocking wait
        return self.poll(timeout=timeout if timeout is not None else 1e9)

    def shutdown(self, timeout: float) -> None:
        self._pending_flag = False
        try:
            self._worker.shutdown()
        except Exception:
            logger.warning("RayEvalBackend: error during shutdown", exc_info=True)


# ======================================================================
# Process backend
# ======================================================================


def _process_eval_worker(
    env_factory: Callable[[], EnvBase],
    policy_factory: Callable,
    request_queue: mp.Queue,
    result_queue: mp.Queue,
    num_trajectories: int,
    max_steps: int,
    frames_per_batch: int | None,
    collector_cls_name: str | None,
    collector_kwargs: dict | None,
    exploration_type: ExplorationType,
    reward_keys: NestedKey,
    done_keys: NestedKey,
    metrics_fn: Callable[[TensorDictBase], dict[str, float]] | None,
) -> None:
    """Entry point for the evaluator child process.

    Creates env, policy, and collector inside the process, then loops
    waiting for ``(weights, step)`` requests on *request_queue* and puts
    result dicts on *result_queue*.  A ``None`` sentinel terminates the loop.
    """
    env = env_factory()
    policy = policy_factory(env)

    try:
        device = next(policy.parameters()).device
    except (StopIteration, AttributeError):
        device = torch.device("cpu")

    cls = _resolve_collector_cls(collector_cls_name)
    fpb = frames_per_batch or max_steps or 1000
    max_frames = 0 if _env_has_step_count(env) else max_steps
    collector = cls(
        create_env_fn=env,
        policy=policy,
        frames_per_batch=fpb,
        total_frames=-1,
        max_frames_per_traj=max_frames,
        trajs_per_batch=num_trajectories,
        exploration_type=exploration_type,
        **(collector_kwargs or {}),
    )

    while True:
        request = request_queue.get()
        if request is None:
            break

        weights, step = request

        if weights is not None:
            weights.to(device).to_module(policy)

        if isinstance(policy, nn.Module):
            policy.eval()

        collector.reset()
        with set_exploration_type(exploration_type), torch.no_grad():
            traj_batch = next(iter(collector))
        metrics = _extract_metrics_from_trajectories(
            traj_batch, reward_keys, done_keys, metrics_fn
        )

        if isinstance(policy, nn.Module):
            policy.train()

        metrics["_step"] = step
        result_queue.put(metrics)

    collector.shutdown()


class _ProcessEvalBackend(_EvalBackend):
    """Runs evaluation in a child process.

    The environment and policy are created **inside** the child process
    from the provided factories, which means they live in an entirely
    separate address space.  This avoids GIL contention for CPU-bound
    work and gives clean CUDA context isolation for multi-GPU setups.

    Like the thread backend, only the most-recently-triggered evaluation
    produces a result (fire-and-forget).
    """

    def __init__(
        self,
        env_factory: Callable[[], EnvBase],
        policy_factory: Callable[..., Callable],
        num_trajectories: int,
        max_steps: int,
        frames_per_batch: int | None,
        collector_cls: type | str | None,
        collector_kwargs: dict | None,
        exploration_type: ExplorationType,
        reward_keys: NestedKey,
        done_keys: NestedKey,
        metrics_fn: Callable[[TensorDictBase], dict[str, float]] | None,
    ) -> None:
        # Serialise collector_cls as string for pickling
        if collector_cls is not None and not isinstance(collector_cls, str):
            collector_cls_name = (
                collector_cls.__name__
                if hasattr(collector_cls, "__name__")
                else str(collector_cls)
            )
        else:
            collector_cls_name = collector_cls

        ctx = mp.get_context("spawn")
        self._request_queue: mp.Queue = ctx.Queue(maxsize=1)
        self._result_queue: mp.Queue = ctx.Queue(maxsize=1)
        self._pending_flag = False
        self._last_step: int | None = None

        self._process = ctx.Process(
            target=_process_eval_worker,
            kwargs={
                "env_factory": env_factory,
                "policy_factory": policy_factory,
                "request_queue": self._request_queue,
                "result_queue": self._result_queue,
                "num_trajectories": num_trajectories,
                "max_steps": max_steps,
                "frames_per_batch": frames_per_batch,
                "collector_cls_name": collector_cls_name,
                "collector_kwargs": collector_kwargs,
                "exploration_type": exploration_type,
                "reward_keys": reward_keys,
                "done_keys": done_keys,
                "metrics_fn": metrics_fn,
            },
            daemon=False,
        )
        self._process.start()

    # ---- sync ----

    def run_sync(self, weights: TensorDictBase | None, step: int) -> dict[str, Any]:
        self._request_queue.put((weights, step))
        return self._result_queue.get()

    # ---- async ----

    def submit(self, weights: TensorDictBase | None, step: int) -> None:
        # Drain any stale result from a previous (possibly cancelled) eval
        self._drain_result_queue()
        self._last_step = step
        self._pending_flag = True
        self._request_queue.put((weights, step))

    def poll(self, timeout: float) -> dict[str, Any] | None:
        try:
            result = self._result_queue.get(timeout=timeout if timeout > 0 else 0.001)
            self._pending_flag = False
            return result
        except Exception:
            # queue.Empty
            return None

    def wait(self, timeout: float | None) -> dict[str, Any] | None:
        try:
            result = self._result_queue.get(timeout=timeout)
            self._pending_flag = False
            return result
        except Exception:
            return None

    @property
    def pending(self) -> bool:
        return self._pending_flag

    def shutdown(self, timeout: float) -> None:
        self._pending_flag = False
        try:
            self._request_queue.put_nowait(None)  # sentinel
        except Exception:
            pass
        if self._process.is_alive():
            self._process.join(timeout=timeout)
            if self._process.is_alive():
                self._process.terminate()

    def _drain_result_queue(self) -> None:
        """Remove any stale result left in the queue."""
        while not self._result_queue.empty():
            try:
                self._result_queue.get_nowait()
            except Exception:
                break
