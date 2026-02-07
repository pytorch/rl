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

    from torchrl.eval import RayEvalWorker

    worker = RayEvalWorker(
        init_fn=my_init,          # called first in the actor process
        env_maker=make_eval_env,  # returns a TorchRL env
        policy_maker=make_policy, # returns a TorchRL policy module
        num_gpus=1,
    )

    # Non-blocking: submit weights and start a rollout
    weights = TensorDict.from_module(policy).data.detach().cpu()
    worker.submit(weights, max_steps=500)

    # Later – check if the rollout finished
    result = worker.poll()       # None while still running
    if result is not None:
        print(result["reward"])  # scalar mean episode reward
        print(result["frames"]) # (T, H, W, 3) uint8 tensor or None
"""
from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)


class RayEvalWorker:
    """Asynchronous evaluation worker backed by a Ray actor.

    The worker creates a **new Python process** (via Ray) and inside it:

    1. Calls *init_fn* – use this for any process-level setup that must happen
       before other imports (e.g. Isaac Lab ``AppLauncher``).
    2. Creates the environment via *env_maker*.
    3. Creates the policy via *policy_maker(env)*.

    Thereafter, :meth:`submit` sends new policy weights and triggers an
    evaluation rollout.  :meth:`poll` returns the result (reward and optional
    video frames) when the rollout finishes, or ``None`` if it is still
    running.

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
        **remote_kwargs: Any,
    ) -> None:
        import ray

        self._reward_keys = reward_keys

        # Build the remote actor class dynamically so that the caller does not
        # need to depend on Ray at import time.
        actor_cls = ray.remote(num_gpus=num_gpus, **remote_kwargs)(_EvalActor)

        self._actor = actor_cls.remote(init_fn, env_maker, policy_maker)
        self._pending_ref: ray.ObjectRef | None = None

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

        - ``"reward"`` – scalar mean episode reward.
        - ``"frames"`` – ``(T, H, W, 3)`` uint8 CPU tensor of rendered
          frames, or ``None`` if the environment does not render.

        Args:
            timeout: Seconds to wait for the result.  ``0`` means
                non-blocking (return immediately if not ready).
        """
        if self._pending_ref is None:
            return None

        import ray

        ready, _ = ray.wait([self._pending_ref], timeout=timeout)
        if not ready:
            return None

        result = ray.get(self._pending_ref)
        self._pending_ref = None
        return result

    def shutdown(self) -> None:
        """Close the environment and kill the actor."""
        import ray

        try:
            ray.get(self._actor.shutdown.remote())
        except Exception:
            logger.warning("RayEvalWorker: error during shutdown", exc_info=True)
        ray.kill(self._actor)
        self._actor = None
        self._pending_ref = None


# ======================================================================
# Inner actor – runs inside the Ray worker process
# ======================================================================


class _EvalActor:
    """Plain class turned into a Ray actor by :class:`RayEvalWorker`.

    All heavy imports happen inside methods so that the module-level import
    of this file does **not** pull in torch, torchrl, or any simulator SDK.
    """

    def __init__(
        self,
        init_fn: Callable[[], None] | None,
        env_maker: Callable[[], Any],
        policy_maker: Callable[[Any], Any],
    ) -> None:
        # --- process-level initialisation (e.g. AppLauncher) ---
        if init_fn is not None:
            init_fn()

        # --- now safe to import torch / torchrl ---
        import torch  # noqa: F401

        self.env = env_maker()
        self.policy = policy_maker(self.env)
        # Cache device before any to_module call can replace nn.Parameter
        # with plain tensors (which makes .parameters() empty).
        self._device = next(self.policy.parameters()).device

    def eval(
        self,
        weights,
        max_steps: int,
        reward_keys: tuple[str, ...],
        deterministic: bool,
        break_when_any_done: bool,
    ) -> dict:
        import torch
        from torchrl.envs.utils import ExplorationType, set_exploration_type, step_mdp

        # Load weights into the eval policy (move to policy device first)
        weights.to(self._device).to_module(self.policy)

        frames = []
        total_reward = 0.0
        num_steps = 0

        exploration = (
            ExplorationType.DETERMINISTIC if deterministic else ExplorationType.RANDOM
        )
        with set_exploration_type(exploration), torch.no_grad():
            td = self.env.reset()
            for _i in range(max_steps):
                td = self.policy(td)
                td = self.env.step(td)

                total_reward += td[reward_keys].mean().item()
                num_steps += 1

                frame = self._try_render()
                if frame is not None:
                    frames.append(frame)

                done = td.get(("next", "done"), None)
                if break_when_any_done and done is not None and done.any():
                    break

                td = step_mdp(td)

        mean_reward = total_reward / max(1, num_steps)

        # Format video: (1, T, C, H, W) uint8 CPU tensor
        video = None
        if frames:
            video = torch.stack(frames, dim=0).unsqueeze(0).cpu()

        return {"reward": mean_reward, "frames": video}

    def _try_render(self):
        """Render one frame from the underlying environment.

        Walks the wrapper chain to find a callable ``render()`` method
        and returns the result as a ``(C, H, W)`` uint8 tensor, or
        ``None`` if rendering is unavailable.
        """
        import numpy as np
        import torch

        # Walk through TransformedEnv / wrapper chain to the base env.
        env = self.env
        while hasattr(env, "base_env"):
            env = env.base_env
        render_fn = getattr(env, "render", None)
        # If the base env delegates to a gymnasium env, prefer that.
        if hasattr(env, "_env") and hasattr(env._env, "render"):
            render_fn = env._env.render
        if render_fn is None:
            return None

        raw = render_fn()
        if raw is None:
            return None

        if isinstance(raw, np.ndarray):
            raw = torch.from_numpy(raw.copy())

        # (H, W, C) -> (C, H, W)
        if raw.ndim == 3 and raw.shape[-1] in (3, 4):
            raw = raw[..., :3]
            raw = raw.permute(2, 0, 1)

        return raw.to(torch.uint8)

    def shutdown(self) -> None:
        if hasattr(self, "env") and not self.env.is_closed:
            self.env.close()
