# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Unified evaluator with pluggable thread / process / Ray backends.

This module provides :class:`Evaluator`, a single entry-point for running
evaluation rollouts either synchronously (blocking) or asynchronously
(in the background) during RL training.

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
        on_result=lambda result: logger.log_metrics(
            {k: v.item() for k, v in result.items() if k != "eval/step"},
            step=result["eval/step"].item(),
        ),
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
import threading
import time
from collections import deque
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
      to check whether an evaluation is currently in progress. Results can
      also be consumed via an ``on_result`` callback.

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
    previous evaluation is still running either raises immediately
    (``busy_policy="error"``; default) or queues the new request
    (``busy_policy="queue"``). Use :attr:`pending` to conditionally skip
    trigger calls::

        if not evaluator.pending:
            evaluator.trigger_eval(weights, step=step)

    **Callback thread-safety**: when ``on_result`` is provided, it is
    invoked from the evaluator's async coordination thread after the
    rollout completes. If the callback writes to a logger, the callback is
    responsible for any locking it needs.

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
        on_result: Optional ``(TensorDictBase) -> None`` invoked after each
            completed evaluation. The callback receives a flat tensordict
            with the same prefixed metric names returned by
            :meth:`evaluate`, :meth:`poll`, and :meth:`wait`.
        busy_policy (str): Behaviour when :meth:`trigger_eval` is called
            while another async evaluation is still pending. ``"error"``
            raises immediately (default; recommended). ``"queue"`` enqueues
            the new request and runs it when the current evaluation
            finishes.

            .. warning::
                With ``busy_policy="queue"``, each queued request stores a
                copy of the weights dict. For large models this can consume
                significant memory. Prefer checking :attr:`pending` and
                skipping triggers instead.
        weight_sync_schemes (dict or None): A dict mapping model IDs to
            :class:`~torchrl.weight_update.WeightSyncScheme` instances.
            When provided, a :class:`~torchrl.collectors.MultiSyncCollector`
            with a single worker is used for process-level CUDA isolation
            and scheme-based weight transfer.  Model IDs follow the
            collector convention: ``"policy"`` for the main policy,
            ``"env.transform[0]"`` for env transforms, etc.
            Example::

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
        backend (str): ``"thread"`` (default), ``"process"``, or ``"ray"``.
            The ``"process"`` backend is implemented as a thread backend
            with a :class:`~torchrl.collectors.MultiSyncCollector` (1
            worker) running in a child process.  This provides full CUDA
            context isolation without custom queue management.
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
        weight_sync_schemes: dict[str, Any] | None = None,
        log_prefix: str = "eval",
        reward_keys: NestedKey = ("next", "reward"),
        done_keys: NestedKey = ("next", "done"),
        device: torch.device | str | None = None,
        exploration_type: ExplorationType = ExplorationType.DETERMINISTIC,
        metrics_fn: Callable[[TensorDictBase], dict[str, float]] | None = None,
        dump_video: bool = True,
        on_result: Callable[[TensorDictBase], None] | None = None,
        busy_policy: str = "error",
        # Backend selection
        backend: str = "thread",
        # Ray-specific
        init_fn: Callable[[], None] | None = None,
        num_gpus: int = 1,
        ray_kwargs: dict | None = None,
    ) -> None:
        self._log_prefix = log_prefix
        self._reward_keys = reward_keys
        self._done_keys = done_keys
        self._exploration_type = exploration_type
        self._metrics_fn = metrics_fn
        self._on_result = on_result
        self._step_counter = 0
        self._dump_video = dump_video
        if busy_policy not in {"error", "queue"}:
            raise ValueError(
                f"Unknown busy_policy {busy_policy!r}. Choose 'error' or 'queue'."
            )
        self._busy_policy = busy_policy
        self._async_lock = threading.Lock()
        self._async_trigger = threading.Event()
        self._ready_result = threading.Event()
        self._async_requests: deque[
            tuple[dict[str, TensorDictBase] | None, int]
        ] = deque()
        self._ready_results: deque[dict[str, Any]] = deque()
        self._async_shutdown = False
        self._async_thread: threading.Thread | None = None

        if backend in ("thread", "process"):
            # The process backend is implemented as a thread backend
            # with a MultiSyncCollector (1 worker) running in a child
            # process.  This eliminates custom process management and
            # uses the weight_sync_schemes infrastructure for weight
            # transfer.
            use_multi_collector = (
                backend == "process" or weight_sync_schemes is not None
            )
            if use_multi_collector:
                env_is_callable = callable(env) and not isinstance(env, EnvBase)
                if not env_is_callable:
                    raise ValueError(
                        f"The {backend!r} backend with weight_sync_schemes "
                        "(or backend='process') requires `env` to be a callable "
                        "(factory function) because the env is created inside "
                        "a child process."
                    )
                if policy_factory is None:
                    raise ValueError(
                        f"The {backend!r} backend with weight_sync_schemes "
                        "(or backend='process') requires `policy_factory` "
                        "(a callable `(env) -> policy`) because the policy is "
                        "created inside a child process."
                    )
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
                weight_sync_schemes=weight_sync_schemes,
                use_multi_collector=use_multi_collector,
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
        *,
        weights_dict: dict[str, TensorDictBase | nn.Module] | None = None,
    ) -> dict[str, Any]:
        """Run a blocking evaluation rollout.

        Args:
            weights: Policy weights to load before the rollout.  Accepts a
                :class:`TensorDictBase` (e.g. from
                ``TensorDict.from_module(policy).data``) or an
                :class:`nn.Module` (weights are extracted automatically).
                If ``None`` the current policy weights are used.
            step: Logging step.  If ``None`` an internal counter is used.

        Keyword Args:
            weights_dict: A dict mapping ``model_id`` strings to weight
                sources (``nn.Module`` or ``TensorDictBase``).  Use this
                to sync multiple models (e.g. policy + env transforms).
                When provided, *weights* is treated as
                ``weights_dict["policy"]`` if ``"policy"`` is not already
                in the dict.

        Returns:
            dict with at least ``"<prefix>/reward"`` and
            ``"<prefix>/episode_length"`` keys.
        """
        prepared = self._prepare_weights_dict(weights, weights_dict)
        step = self._next_step(step)
        raw = self._backend.run_sync(prepared, step)
        result = self._finalize(raw)
        self._invoke_on_result(result)
        return result

    # ------------------------------------------------------------------
    # Asynchronous API
    # ------------------------------------------------------------------

    def trigger_eval(
        self,
        weights: TensorDictBase | nn.Module | None = None,
        step: int | None = None,
        *,
        weights_dict: dict[str, TensorDictBase | nn.Module] | None = None,
    ) -> None:
        """Start an async evaluation.

        Args:
            weights: Policy weights to load.  See :meth:`evaluate`.
            step: Logging step.  See :meth:`evaluate`.
            weights_dict: Multi-model weights dict.  See :meth:`evaluate`.
        """
        prepared = self._prepare_weights_dict(weights, weights_dict)
        step = self._next_step(step)
        with self._async_lock:
            if self._busy_policy == "error" and (
                self._backend.pending or self._async_requests
            ):
                raise RuntimeError(
                    "Evaluation already pending. Wait for completion or set "
                    "busy_policy='queue'."
                )
            self._async_requests.append((prepared, step))
            self._async_trigger.set()
        self._ensure_async_thread()

    def poll(self, timeout: float = 0) -> dict[str, Any] | None:
        """Return the latest evaluation result if ready, else ``None``.

        Args:
            timeout: Seconds to wait.  ``0`` means non-blocking.
        """
        if timeout > 0:
            self._ready_result.wait(timeout=timeout)
        return self._pop_ready_result()

    def wait(self, timeout: float | None = None) -> dict[str, Any] | None:
        """Block until the current evaluation finishes.

        Args:
            timeout: Max seconds to wait.  ``None`` waits forever.
        """
        self._ready_result.wait(timeout=timeout)
        return self._pop_ready_result()

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
        with self._async_lock:
            return self._backend.pending or bool(self._async_requests)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def shutdown(self, timeout: float = 5.0) -> None:
        """Cancel any running evaluation, clean up resources."""
        self._async_shutdown = True
        self._async_trigger.set()
        if self._async_thread is not None and self._async_thread.is_alive():
            self._async_thread.join(timeout=timeout)
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

    @staticmethod
    def _prepare_single(weights: TensorDictBase | nn.Module) -> TensorDictBase:
        """Prepare a single weight source for cross-thread transfer."""
        if isinstance(weights, nn.Module):
            return Evaluator.extract_weights(weights)
        return weights.detach().clone().cpu()

    def _prepare_weights_dict(
        self,
        weights: TensorDictBase | nn.Module | None,
        weights_dict: dict[str, TensorDictBase | nn.Module] | None,
    ) -> dict[str, TensorDictBase] | None:
        """Build a ``{model_id: TensorDictBase}`` dict from user inputs.

        When *weights_dict* is ``None`` and *weights* is provided, the
        result is ``{"policy": prepared_weights}`` for backward
        compatibility.  When both are ``None``, returns ``None``.
        """
        result: dict[str, TensorDictBase] = {}
        if weights_dict:
            for k, v in weights_dict.items():
                result[k] = self._prepare_single(v)
        if weights is not None and "policy" not in result:
            result["policy"] = self._prepare_single(weights)
        return result or None

    def _next_step(self, step: int | None) -> int:
        if step is not None:
            return step
        s = self._step_counter
        self._step_counter += 1
        return s

    def _ensure_async_thread(self) -> None:
        if self._async_thread is not None and self._async_thread.is_alive():
            return
        self._async_shutdown = False
        self._async_thread = threading.Thread(
            target=self._async_loop,
            name="evaluator-async",
            daemon=True,
        )
        self._async_thread.start()

    def _async_loop(self) -> None:
        while not self._async_shutdown:
            request = None
            with self._async_lock:
                if not self._backend.pending and self._async_requests:
                    request = self._async_requests.popleft()
                else:
                    self._async_trigger.clear()
            if request is not None:
                self._backend.submit(*request)
                continue

            raw = self._backend.poll(timeout=0.1)
            if raw is not None:
                result = self._finalize(raw)
                with self._async_lock:
                    self._ready_results.append(result)
                    self._ready_result.set()
                    has_more_requests = bool(self._async_requests)
                self._invoke_on_result(result)
                if has_more_requests:
                    self._async_trigger.set()
                continue

            self._async_trigger.wait(timeout=0.1)

    def _pop_ready_result(self) -> dict[str, Any] | None:
        with self._async_lock:
            if not self._ready_results:
                self._ready_result.clear()
                return None
            result = self._ready_results.popleft()
            if not self._ready_results:
                self._ready_result.clear()
            return result

    def _invoke_on_result(self, result: dict[str, Any]) -> None:
        if self._on_result is None:
            return
        try:
            self._on_result(self._to_callback_tensordict(result))
        except Exception:
            logger.warning("Evaluator on_result callback failed", exc_info=True)

    def _finalize(self, raw: dict[str, Any]) -> dict[str, Any]:
        """Format metrics and dump video on the caller or coordinator thread."""
        step = raw.get("_step")
        metrics = self._format_metrics(raw)
        if self._dump_video:
            self._backend.dump_video(step=step)
        return metrics

    def _format_metrics(self, raw: dict[str, Any]) -> dict[str, Any]:
        prefix = self._log_prefix
        out: dict[str, Any] = {}
        for key in ("reward", "reward_std", "num_episodes", "episode_length", "fps"):
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

    @staticmethod
    def _to_result_tensor(value: Any) -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            return value.detach().cpu()
        return torch.as_tensor(value)

    def _to_callback_tensordict(self, metrics: dict[str, Any]) -> TensorDictBase:
        data = {
            key: self._to_result_tensor(value)
            for key, value in metrics.items()
            if value is not None
        }
        return TensorDict(data, batch_size=[])


# ======================================================================
# Backend interface
# ======================================================================


class _EvalBackend(abc.ABC):
    """Internal contract that each backend implements.

    All weight arguments are now ``dict[str, TensorDictBase] | None``
    mapping model IDs to prepared weight tensordicts.  For backward
    compatibility, the ``"policy"`` key holds the main policy weights.
    """

    @abc.abstractmethod
    def run_sync(
        self, weights_dict: dict[str, TensorDictBase] | None, step: int
    ) -> dict[str, Any]:
        """Run a blocking evaluation and return raw results."""
        ...

    @abc.abstractmethod
    def submit(self, weights_dict: dict[str, TensorDictBase] | None, step: int) -> None:
        """Start an async evaluation."""
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
    eval_time: float | None = None,
) -> dict[str, Any]:
    """Extract evaluation metrics from a trajectory batch produced by a collector.

    *traj_batch* has shape ``(num_trajectories, max_traj_len)`` with a
    ``("collector", "mask")`` boolean field marking valid timesteps.
    """
    mask = traj_batch.get(("collector", "mask"))  # [N, T]
    num_trajectories = traj_batch.shape[0]

    episode_rewards = []
    episode_lengths = []
    total_frames = 0

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
        total_frames += int(valid_len)

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
    if eval_time is not None and eval_time > 0:
        metrics["fps"] = total_frames / eval_time
    metrics["frame_count"] = total_frames

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


def _freeze_vecnorm(env: EnvBase) -> EnvBase:
    """Freeze all VecNorm / VecNormV2 transforms in the env.

    Evaluation environments should not update running statistics —
    they receive stats from the training process via weight sync and
    use them as-is.
    """
    from torchrl.envs.transforms import Compose, TransformedEnv
    from torchrl.envs.transforms.vecnorm import VecNormV2

    # Also handle the legacy VecNorm
    try:
        from torchrl.envs.transforms.transforms import VecNorm
    except ImportError:
        VecNorm = None  # noqa: N806

    def _freeze_transforms(transform):
        if isinstance(transform, VecNormV2):
            transform.freeze()
        elif VecNorm is not None and isinstance(transform, VecNorm):
            transform.freeze()
        elif isinstance(transform, Compose):
            for t in transform:
                _freeze_transforms(t)

    if isinstance(env, TransformedEnv):
        _freeze_transforms(env.transform)
    return env


def _wrap_env_factory_frozen(
    env_factory: Callable[[], EnvBase],
) -> Callable[[], EnvBase]:
    """Wrap an env factory to freeze VecNorm transforms after creation.

    If *env_factory* is an :class:`~torchrl.envs.EnvCreator`, the returned
    object is also an ``EnvCreator`` (preserving pre-computed ``meta_data``
    and shared-memory state dicts) whose ``__call__`` freezes VecNorm
    transforms on the newly-created environment.
    """
    from torchrl.envs.env_creator import EnvCreator

    if isinstance(env_factory, EnvCreator):

        class _FrozenEnvCreator(EnvCreator):
            """Thin ``EnvCreator`` wrapper that freezes VecNorm after creation."""

            def __init__(self, original: EnvCreator):
                # Skip parent __init__ (avoids recreating shadow env).
                # Copy all state from the original instead.
                self.__dict__.update(original.__dict__)
                self._original = original

            def __call__(self, **kwargs) -> EnvBase:
                env = self._original(**kwargs)
                return _freeze_vecnorm(env)

        return _FrozenEnvCreator(env_factory)

    def wrapper():
        env = env_factory()
        return _freeze_vecnorm(env)

    return wrapper


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

    When *weight_sync_schemes* is provided (or *use_multi_collector* is
    ``True``), a :class:`~torchrl.collectors.MultiSyncCollector` with a
    single worker is used instead of a plain
    :class:`~torchrl.collectors.Collector`.  This provides process-level
    CUDA isolation and uses the weight-sync-scheme infrastructure for
    cross-process weight transfer — replacing the old
    ``_ProcessEvalBackend``.
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
        weight_sync_schemes: dict[str, Any] | None = None,
        use_multi_collector: bool = False,
    ) -> None:
        if policy is not None and policy_factory is not None:
            raise ValueError("Provide either `policy` or `policy_factory`, not both.")

        self._env_factory: Callable[[], EnvBase] | None = None
        self._policy_factory: Callable[..., Callable] | None = None
        self._use_multi_collector = use_multi_collector or (
            weight_sync_schemes is not None
        )
        self._weight_sync_schemes = weight_sync_schemes

        env_is_callable = callable(env) and not isinstance(env, EnvBase)

        if self._use_multi_collector:
            # MultiSyncCollector path: always defer construction.
            # Wrap the factory to freeze VecNorm transforms in the worker.
            self._env_factory = _wrap_env_factory_frozen(env)
            self._policy_factory = policy_factory
            self._env: EnvBase | None = None
            self._policy = None
        elif policy_factory is not None and env_is_callable:
            # Lazy path: defer both env and policy creation to the worker thread
            self._env_factory = env
            self._policy_factory = policy_factory
            self._env = None
            self._policy = None
        else:
            # Eager path (existing behaviour)
            if env_is_callable:
                env = env()
            # Freeze VecNorm transforms so eval doesn't update running stats
            _freeze_vecnorm(env)
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
        self._collector_iter = None  # persistent iterator for multi-collector

        # Threading state
        self._lock = threading.Lock()
        self._eval_ready = threading.Event()
        self._result_ready = threading.Event()
        self._pending = threading.Event()
        self._pending_request: tuple[
            dict[str, TensorDictBase] | None, int
        ] | None = None
        self._result: dict[str, Any] | None = None
        self._shutdown_flag = False
        self._thread: threading.Thread | None = None

    # ---- sync ----

    def run_sync(
        self, weights_dict: dict[str, TensorDictBase] | None, step: int
    ) -> dict[str, Any]:
        self._ensure_collector()
        metrics = self._run_eval(weights_dict)
        metrics["_step"] = step
        return metrics

    # ---- async ----

    def submit(self, weights_dict: dict[str, TensorDictBase] | None, step: int) -> None:
        with self._lock:
            if self._pending.is_set():
                raise RuntimeError("Evaluation already pending.")
            self._pending_request = (weights_dict, step)
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
        # Freeze VecNorm transforms so eval doesn't update running stats
        _freeze_vecnorm(self._env)
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

            if request is None:
                continue

            weights_dict, step = request
            metrics = self._run_eval(weights_dict)
            metrics["_step"] = step
            with self._lock:
                self._result = metrics
                self._pending.clear()
            self._result_ready.set()

    def _ensure_collector(self) -> None:
        """Create the collector lazily (inside the worker thread)."""
        if self._collector is not None:
            return

        fpb = self._frames_per_batch or self._max_steps or 1000
        kwargs = dict(self._collector_kwargs or {})

        if self._use_multi_collector:
            # Process isolation via MultiSyncCollector (1 worker).
            # The env and policy are created inside the child process
            # by the collector, so we pass factories — not instances.
            from torchrl.collectors import MultiSyncCollector

            self._collector = MultiSyncCollector(
                create_env_fn=[self._env_factory],
                policy_factory=self._policy_factory,
                frames_per_batch=fpb,
                total_frames=-1,
                max_frames_per_traj=self._max_steps,
                trajs_per_batch=self._num_trajectories,
                exploration_type=self._exploration_type,
                weight_sync_schemes=self._weight_sync_schemes,
                **kwargs,
            )
        else:
            self._ensure_env_and_policy()
            cls = _resolve_collector_cls(self._collector_cls)
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
                **kwargs,
            )

    def _run_eval(
        self, weights_dict: dict[str, TensorDictBase] | None
    ) -> dict[str, Any]:
        """Run evaluation using the internal collector."""
        self._ensure_collector()

        if weights_dict:
            if self._use_multi_collector:
                # Multi-process: use scheme-based sync via the collector
                self._collector.update_policy_weights_(weights_dict=weights_dict)
            else:
                # Same process: apply weights directly
                for model_id, w in weights_dict.items():
                    if model_id == "policy":
                        w.to(self._device).to_module(self._policy)
                    else:
                        from torchrl.weight_update.utils import _resolve_model

                        target = _resolve_model(self._collector, model_id)
                        w.to(self._device).to_module(target)

        if not self._use_multi_collector and isinstance(self._policy, nn.Module):
            self._policy.eval()

        eval_start = time.perf_counter()
        with set_exploration_type(self._exploration_type), torch.no_grad():
            if self._use_multi_collector:
                # MultiSyncCollector: use a persistent iterator because
                # re-creating the iterator (next(iter(...))) after the first
                # batch causes data loss from the queue/pipe based workers.
                if self._collector_iter is None:
                    self._collector_iter = iter(self._collector)
                traj_batch = next(self._collector_iter)
            else:
                # Single-process Collector: reset and use fresh iterator
                self._collector.reset()
                traj_batch = next(iter(self._collector))

        if not self._use_multi_collector and isinstance(self._policy, nn.Module):
            self._policy.train()

        return _extract_metrics_from_trajectories(
            traj_batch,
            self._reward_keys,
            self._done_keys,
            self._metrics_fn,
            eval_time=time.perf_counter() - eval_start,
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

    def run_sync(
        self, weights_dict: dict[str, TensorDictBase] | None, step: int
    ) -> dict[str, Any]:
        weights = weights_dict.get("policy") if weights_dict else None
        self._worker.submit(
            weights,
            self._max_steps,
        )
        # Block until done
        result = self._worker.poll(timeout=None)
        result["_step"] = step
        result.setdefault("episode_length", self._max_steps)
        return result

    def submit(self, weights_dict: dict[str, TensorDictBase] | None, step: int) -> None:
        weights = weights_dict.get("policy") if weights_dict else None
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
