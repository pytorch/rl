# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Unified evaluator with pluggable thread / Ray backends.

This module provides :class:`Evaluator`, a single entry-point for running
evaluation rollouts either synchronously (blocking) or asynchronously
(fire-and-forget) during RL training.

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

    * **Synchronous** -- call :meth:`evaluate` to run a blocking rollout and
      get metrics back immediately.
    * **Asynchronous** -- call :meth:`trigger_eval` to kick off a rollout in
      the background, then :meth:`poll` (non-blocking) or :meth:`wait`
      (blocking) to retrieve the result.  Results are also auto-logged if a
      *logger* is provided.

    Fire-and-forget semantics: calling :meth:`trigger_eval` while a previous
    evaluation is still running discards the in-progress result.

    Args:
        env: An :class:`~torchrl.envs.EnvBase` instance **or** a callable
            that returns one.  For the ``"ray"`` backend the callable form is
            required (the env is created inside the Ray actor process).
        policy: The evaluation policy.  Mutually exclusive with
            *policy_factory*.

    Keyword Args:
        policy_factory: A callable ``(env) -> policy`` used to build the
            policy.  Required for the ``"ray"`` backend; optional for
            ``"thread"`` (if provided, called once at construction time).
        max_steps (int): Maximum environment steps per rollout.
        logger: Optional :class:`~torchrl.record.loggers.Logger` for
            automatic metric / video logging.
        log_prefix (str): Prefix prepended to all logged metric names.
            Default: ``"eval"``.
        reward_keys: Nested key(s) for reading the reward from the rollout
            tensordict.  Default: ``("next", "reward")``.
        done_keys: Nested key(s) for reading the done flag.
            Default: ``("next", "done")``.
        device: Device for the evaluation policy.  If ``None``, inferred
            from the policy parameters.
        exploration_type: Exploration mode during evaluation.
            Default: :attr:`ExplorationType.DETERMINISTIC`.
        metrics_fn: Optional ``(TensorDictBase) -> dict[str, float]``
            called on every rollout result to extract custom metrics.
        break_when_any_done (bool): Stop the rollout as soon as any
            sub-environment reports done.  Default: ``True``.
        auto_cast_to_device (bool): Auto-cast tensordicts to policy device.
            Default: ``True``.
        dump_video (bool): Call ``dump()`` on :class:`VideoRecorder`
            transforms after each rollout (thread backend only).
            Default: ``True``.
        callback: Optional ``(dict, int) -> None`` invoked with
            ``(metrics, step)`` after each completed evaluation.
        logger_lock: A :class:`threading.Lock` shared with the training
            loop to serialise logger access.  If ``None`` a private lock
            is created.
        backend (str): ``"thread"`` (default) or ``"ray"``.
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
        max_steps: int,
        logger=None,
        log_prefix: str = "eval",
        reward_keys: NestedKey = ("next", "reward"),
        done_keys: NestedKey = ("next", "done"),
        device: torch.device | str | None = None,
        exploration_type: ExplorationType = ExplorationType.DETERMINISTIC,
        metrics_fn: Callable[[TensorDictBase], dict[str, float]] | None = None,
        break_when_any_done: bool = True,
        auto_cast_to_device: bool = True,
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

        if backend == "thread":
            self._backend: _EvalBackend = _ThreadEvalBackend(
                env=env,
                policy=policy,
                policy_factory=policy_factory,
                max_steps=max_steps,
                device=device,
                exploration_type=exploration_type,
                reward_keys=reward_keys,
                break_when_any_done=break_when_any_done,
                auto_cast_to_device=auto_cast_to_device,
                dump_video=dump_video,
                metrics_fn=metrics_fn,
            )
        elif backend == "ray":
            self._backend = _RayEvalBackend(
                env_maker=env,
                policy_factory=policy_factory,
                max_steps=max_steps,
                reward_keys=reward_keys,
                break_when_any_done=break_when_any_done,
                init_fn=init_fn,
                num_gpus=num_gpus,
                ray_kwargs=ray_kwargs or {},
            )
        else:
            raise ValueError(f"Unknown backend {backend!r}. Choose 'thread' or 'ray'.")

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
        metrics = self._format_metrics(raw)
        self._auto_log(metrics, step)
        if self._callback is not None:
            self._callback(metrics, step)
        return metrics

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
        metrics = self._format_metrics(raw)
        step = raw.get("_step")
        self._auto_log(metrics, step)
        if self._callback is not None:
            self._callback(metrics, step)
        return metrics

    def wait(self, timeout: float | None = None) -> dict[str, Any] | None:
        """Block until the current evaluation finishes.

        Args:
            timeout: Max seconds to wait.  ``None`` waits forever.
        """
        raw = self._backend.wait(timeout)
        if raw is None:
            return None
        metrics = self._format_metrics(raw)
        step = raw.get("_step")
        self._auto_log(metrics, step)
        if self._callback is not None:
            self._callback(metrics, step)
        return metrics

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

    def _format_metrics(self, raw: dict[str, Any]) -> dict[str, Any]:
        prefix = self._log_prefix
        out: dict[str, Any] = {}
        if "reward" in raw:
            out[f"{prefix}/reward"] = raw["reward"]
        if "episode_length" in raw:
            out[f"{prefix}/episode_length"] = raw["episode_length"]
        if "frames" in raw and raw["frames"] is not None:
            out[f"{prefix}/video"] = raw["frames"]
        # Custom metrics (already prefixed by backend or metrics_fn)
        for k, v in raw.items():
            if k.startswith("custom/"):
                out[f"{prefix}/{k[7:]}"] = v
        return out

    def _auto_log(self, metrics: dict[str, Any], step: int | None) -> None:
        if self._logger is None:
            return
        with self._logger_lock:
            # Separate video from scalar metrics
            video = metrics.pop(f"{self._log_prefix}/video", None)
            if metrics:
                self._logger.log_metrics(metrics, step=step)
            if video is not None:
                self._logger.log_video(f"{self._log_prefix}/video", video, step=step)
                # Put it back so the caller still sees it
                metrics[f"{self._log_prefix}/video"] = video


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

    @abc.abstractmethod
    def shutdown(self, timeout: float) -> None:
        """Clean up resources."""
        ...


# ======================================================================
# Thread backend
# ======================================================================


class _ThreadEvalBackend(_EvalBackend):
    """Runs evaluation in a daemon thread using ``env.rollout()``."""

    def __init__(
        self,
        env: EnvBase | Callable[[], EnvBase],
        policy: TensorDictModuleBase | Callable | None,
        policy_factory: Callable[..., Callable] | None,
        max_steps: int,
        device: torch.device | str | None,
        exploration_type: ExplorationType,
        reward_keys: NestedKey,
        break_when_any_done: bool,
        auto_cast_to_device: bool,
        dump_video: bool,
        metrics_fn: Callable[[TensorDictBase], dict[str, float]] | None,
    ) -> None:
        # Build env
        if callable(env) and not isinstance(env, EnvBase):
            env = env()
        self._env: EnvBase = env

        # Build policy
        if policy is not None and policy_factory is not None:
            raise ValueError("Provide either `policy` or `policy_factory`, not both.")
        if policy_factory is not None:
            policy = policy_factory(self._env)
        if policy is None:
            raise ValueError("Either `policy` or `policy_factory` must be provided.")
        self._policy = policy

        # Device
        if device is None:
            try:
                device = next(self._policy.parameters()).device
            except StopIteration:
                device = torch.device("cpu")
        self._device = torch.device(device)

        self._max_steps = max_steps
        self._exploration_type = exploration_type
        self._reward_keys = reward_keys
        self._break_when_any_done = break_when_any_done
        self._auto_cast_to_device = auto_cast_to_device
        self._dump_video = dump_video
        self._metrics_fn = metrics_fn

        # Threading state
        self._lock = threading.Lock()
        self._cancel = threading.Event()
        self._eval_ready = threading.Event()
        self._result_ready = threading.Event()
        self._pending_request: tuple[TensorDictBase | None, int] | None = None
        self._result: dict[str, Any] | None = None
        self._shutdown_flag = False
        self._thread: threading.Thread | None = None

    # ---- sync ----

    def run_sync(self, weights: TensorDictBase | None, step: int) -> dict[str, Any]:
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
        self._eval_ready.set()
        self._ensure_thread()

    def poll(self, timeout: float) -> dict[str, Any] | None:
        if timeout > 0:
            self._result_ready.wait(timeout=timeout)
        with self._lock:
            result = self._result
            if result is not None:
                self._result = None
            return result

    def wait(self, timeout: float | None) -> dict[str, Any] | None:
        self._result_ready.wait(timeout=timeout)
        with self._lock:
            result = self._result
            if result is not None:
                self._result = None
            return result

    def shutdown(self, timeout: float) -> None:
        self._shutdown_flag = True
        self._cancel.set()
        self._eval_ready.set()  # wake thread so it can exit
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=timeout)
        if not self._env.is_closed:
            self._env.close()

    # ---- internals ----

    def _ensure_thread(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._shutdown_flag = False
        self._thread = threading.Thread(target=self._eval_loop, daemon=True)
        self._thread.start()

    def _eval_loop(self) -> None:
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

    def _run_eval(self, weights: TensorDictBase | None) -> dict[str, Any]:
        # Apply weights
        if weights is not None:
            weights.to(self._device).to_module(self._policy)

        if isinstance(self._policy, nn.Module):
            self._policy.eval()

        with set_exploration_type(self._exploration_type), torch.no_grad():
            rollout_td = self._env.rollout(
                self._max_steps,
                self._policy,
                auto_cast_to_device=self._auto_cast_to_device,
                break_when_any_done=self._break_when_any_done,
            )

        if isinstance(self._policy, nn.Module):
            self._policy.train()

        # Compute metrics
        reward = rollout_td.get(self._reward_keys, None)
        if reward is not None:
            # sum over time, mean over batch
            episode_reward = reward.sum(-2).mean().item()
        else:
            episode_reward = float("nan")

        episode_length = rollout_td.shape[-1]

        metrics: dict[str, Any] = {
            "reward": episode_reward,
            "episode_length": episode_length,
        }

        # Custom metrics
        if self._metrics_fn is not None:
            custom = self._metrics_fn(rollout_td)
            for k, v in custom.items():
                metrics[f"custom/{k}"] = v

        # Video dump
        if self._dump_video and hasattr(self._env, "transform"):
            for transform in self._env.transform:
                if hasattr(transform, "dump"):
                    transform.dump()

        return metrics


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
        break_when_any_done: bool,
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
        self._break_when_any_done = break_when_any_done
        self._last_step: int | None = None

    def run_sync(self, weights: TensorDictBase | None, step: int) -> dict[str, Any]:
        self._worker.submit(
            weights,
            self._max_steps,
            break_when_any_done=self._break_when_any_done,
        )
        # Block until done
        result = self._worker.poll(timeout=None)
        result["_step"] = step
        result.setdefault("episode_length", self._max_steps)
        return result

    def submit(self, weights: TensorDictBase | None, step: int) -> None:
        self._last_step = step
        self._worker.submit(
            weights,
            self._max_steps,
            break_when_any_done=self._break_when_any_done,
        )

    def poll(self, timeout: float) -> dict[str, Any] | None:
        result = self._worker.poll(timeout=timeout)
        if result is None:
            return None
        result["_step"] = self._last_step
        result.setdefault("episode_length", self._max_steps)
        return result

    def wait(self, timeout: float | None) -> dict[str, Any] | None:
        # RayEvalWorker.poll supports timeout=None for blocking wait
        return self.poll(timeout=timeout if timeout is not None else 1e9)

    def shutdown(self, timeout: float) -> None:
        try:
            self._worker.shutdown()
        except Exception:
            logger.warning("RayEvalBackend: error during shutdown", exc_info=True)
