# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Ray-based asynchronous evaluation worker.

This module provides :class:`RayEvalWorker`, a generic helper that runs an
environment and policy inside a dedicated Ray actor process.  This is useful
when the evaluation environment requires special process-level initialisation
(e.g. Isaac Lab's ``AppLauncher`` must run before ``import torch``) or when
evaluation should happen concurrently with training on a separate GPU.

Typical usage::

    from torchrl.collectors.distributed import RayEvalWorker

    worker = RayEvalWorker(
        init_fn=my_init,          # called first in the actor process
        env_maker=make_eval_env,  # returns a TorchRL env
        policy_maker=make_policy, # returns a TorchRL policy module
        num_gpus=1,
        name="my_eval_worker",    # optional: allows others to connect
    )

    # Non-blocking: submit weights and start a rollout
    weights = TensorDict.from_module(policy).data.detach().cpu()
    worker.submit(weights, max_steps=500)

    # Later -- check if the rollout finished
    result = worker.poll()       # None while still running
    if result is not None:
        print(result["reward"])  # scalar mean episode reward
        print(result["frames"]) # (T, H, W, 3) uint8 tensor or None

    # From another process, connect to the same actor by name:
    worker2 = RayEvalWorker.from_name("my_eval_worker")
"""
from __future__ import annotations

import importlib
import logging
from collections.abc import Callable
from typing import Any


_has_ray = importlib.util.find_spec("ray") is not None
_ray = None

logger = logging.getLogger(__name__)


def _get_ray():
    """Lazily import the optional Ray dependency."""
    if not _has_ray:
        raise RuntimeError(
            "Ray is required for RayEvalWorker but could not be found. "
            "Install it with: pip install ray"
        )
    global _ray
    if _ray is None:
        _ray = importlib.import_module("ray")
    return _ray


class RayEvalWorker:
    """Asynchronous evaluation worker backed by a Ray actor.

    The worker creates a **new Python process** (via Ray) and inside it:

    1. Calls *init_fn* -- use this for any process-level setup that must happen
       before other imports (e.g. Isaac Lab ``AppLauncher``).
    2. Creates the environment via *env_maker*.
    3. Creates the policy via *policy_maker(env)*.

    Thereafter, :meth:`submit` sends new policy weights and triggers an
    evaluation rollout.  :meth:`poll` returns the result (reward and optional
    video frames) when the rollout finishes, or ``None`` if it is still
    running.

    If a *name* is provided the actor is registered with Ray under that name,
    allowing other processes (or a later session) to reconnect to the same
    running actor via :meth:`from_name`.

    Args:
        init_fn: Optional callable invoked at the very start of the actor
            process, before *env_maker* or *policy_maker*.  All imports should
            be **local** inside this callable so that the actor's fresh Python
            process can control import order.  Set to ``None`` to skip.
        env_maker: Callable that returns a TorchRL environment.  Called once
            inside the actor after *init_fn*.  If the underlying environment
            supports ``render_mode="rgb_array"``, the actor will call
            ``render()`` on each evaluation step and return the frames.
        policy_maker: Callable ``(env) -> policy`` that builds the policy
            module given the environment.  Called once inside the actor after
            the environment has been created.
        num_gpus: Number of GPUs to request from Ray for this actor.
            Defaults to 1.
        reward_keys: Nested key(s) used to read the reward from the rollout
            tensordict.  Defaults to ``("next", "reward")``.
        name: Optional name for the Ray actor.  When set, the actor is
            registered under this name and can be retrieved later with
            :meth:`from_name`.
        **remote_kwargs: Extra keyword arguments forwarded to
            ``ray.remote()`` when creating the actor class (e.g.
            ``num_cpus``, ``runtime_env``).
    """

    def __init__(
        self,
        init_fn: Callable[[], None] | None,
        env_maker: Callable[[], Any],
        policy_maker: Callable[[Any], Any],
        *,
        num_gpus: int = 1,
        reward_keys: tuple[str, ...] = ("next", "reward"),
        name: str | None = None,
        **remote_kwargs: Any,
    ) -> None:
        ray = _get_ray()

        self._reward_keys = reward_keys

        # Build the remote actor class dynamically so that the caller does not
        # need to depend on Ray at import time.
        actor_cls = ray.remote(num_gpus=num_gpus, **remote_kwargs)(_EvalActor)

        actor_kwargs = {}
        if name is not None:
            actor_kwargs["name"] = name
            actor_kwargs["lifetime"] = "detached"
        self._actor = actor_cls.options(**actor_kwargs).remote(
            init_fn, env_maker, policy_maker
        )
        self._pending_ref: ray.ObjectRef | None = None

    # ------------------------------------------------------------------
    # Alternative constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_name(
        cls,
        name: str,
        *,
        reward_keys: tuple[str, ...] = ("next", "reward"),
    ) -> RayEvalWorker:
        """Connect to an existing named :class:`RayEvalWorker` actor.

        This is useful when one process creates the worker (with a *name*)
        and another process wants to submit evaluations or poll results on
        the same actor.

        Args:
            name: The actor name that was passed to the constructor.
            reward_keys: Nested key(s) used to read the reward from the
                rollout tensordict.  Defaults to ``("next", "reward")``.
        """
        ray = _get_ray()

        worker = object.__new__(cls)
        worker._reward_keys = reward_keys
        worker._actor = ray.get_actor(name)
        worker._pending_ref = None
        return worker

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def submit(
        self,
        weights: Any,
        max_steps: int,
        *,
        deterministic: bool = True,
        break_when_any_done: bool = True,
    ) -> None:
        """Start an asynchronous evaluation rollout.

        If a previous rollout is still running its result is silently
        discarded (fire-and-forget semantics).

        Args:
            weights: Policy weights, typically obtained via
                ``TensorDict.from_module(policy).data.detach().cpu()``.
            max_steps: Maximum number of environment steps per rollout.
            deterministic: If ``True``, use deterministic exploration.
            break_when_any_done: If ``True``, stop the rollout as soon as
                any sub-environment reports ``done``.
        """
        # Discard any previous un-polled result
        self._pending_ref = self._actor.eval.remote(
            weights,
            max_steps,
            self._reward_keys,
            deterministic,
            break_when_any_done,
        )

    def poll(self, timeout: float = 0) -> dict | None:
        """Return the evaluation result if ready, otherwise ``None``.

        The returned dict contains:

        - ``"reward"`` -- scalar mean episode reward.
        - ``"frames"`` -- ``(T, H, W, 3)`` uint8 CPU tensor of rendered
          frames, or ``None`` if the environment does not render.

        Args:
            timeout: Seconds to wait for the result.  ``0`` means
                non-blocking (return immediately if not ready).
        """
        if self._pending_ref is None:
            return None

        ray = _get_ray()

        ready, _ = ray.wait([self._pending_ref], timeout=timeout)
        if not ready:
            return None

        result = ray.get(self._pending_ref)
        self._pending_ref = None
        return result

    def shutdown(self) -> None:
        """Close the environment and kill the actor.

        Safe to call multiple times or after ``ray.shutdown()`` has already
        torn down the actor (e.g. via a test fixture).
        """
        ray = _get_ray()

        if self._actor is None:
            return
        try:
            ray.get(self._actor.shutdown.remote())
        except Exception:
            logger.warning("RayEvalWorker: error during shutdown", exc_info=True)
        try:
            ray.kill(self._actor)
        except Exception:
            # The actor may already be dead (e.g. ray.shutdown() ran first).
            logger.debug("RayEvalWorker: actor already terminated", exc_info=True)
        self._actor = None
        self._pending_ref = None


# ======================================================================
# Inner actor -- runs inside the Ray worker process
# ======================================================================


class _EvalActor:
    """Plain class turned into a Ray actor by :class:`RayEvalWorker`.

    Environments like Isaac Lab require their ``AppLauncher`` to be initialised
    before ``torch`` is imported. The torch-dependent runtime is therefore kept
    in a private module and imported only after *init_fn* has run in the actor
    process.
    """

    def __init__(
        self,
        init_fn: Callable[[], None] | None,
        env_maker: Callable[[], Any],
        policy_maker: Callable[[Any], Any],
    ) -> None:
        if init_fn is not None:
            init_fn()

        runtime_mod = importlib.import_module(
            "torchrl.collectors.distributed._ray_eval_runtime"
        )
        env = env_maker()
        self._runtime = runtime_mod.RayEvalRuntime(env, policy_maker(env))

    def eval(
        self,
        weights: Any,
        max_steps: int,
        reward_keys: tuple[str, ...],
        deterministic: bool,
        break_when_any_done: bool,
    ) -> dict:
        """Run an evaluation rollout with the given weights."""
        return self._runtime.eval(
            weights=weights,
            max_steps=max_steps,
            reward_keys=reward_keys,
            deterministic=deterministic,
            break_when_any_done=break_when_any_done,
        )

    def shutdown(self) -> None:
        """Shut down the environment."""
        self._runtime.shutdown()
