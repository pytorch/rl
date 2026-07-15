# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

from typing import Any

import numpy as np
import torch

from torchrl.envs.utils import ExplorationType, set_exploration_type, step_mdp
from torchrl.weight_update.weight_sync_schemes import WeightStrategy


class RayEvalRuntime:
    """Torch-dependent runtime for :class:`~torchrl.collectors.RayEvalWorker`."""

    def __init__(self, env: Any, policy: Any) -> None:
        self.env = env
        self.policy = policy
        # Cache device before any weight application can affect parameter
        # registration.
        self._device = next(self.policy.parameters()).device
        self._weight_strategy = WeightStrategy(extract_as="tensordict")

    def eval(
        self,
        weights: Any,
        max_steps: int,
        reward_keys: tuple[str, ...],
        deterministic: bool,
        break_when_any_done: bool,
    ) -> dict:
        """Run an evaluation rollout with the given weights."""
        # Load weights into the eval policy (move to policy device first).
        # ``weights`` can legitimately be None when the caller asks for an
        # evaluation of the current policy (no fresh weights to inject).
        if weights is not None:
            self._weight_strategy.apply_weights(self.policy, weights.to(self._device))

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

        # Format video: (1, T, C, H, W) uint8 CPU tensor.
        video = None
        if frames:
            video = torch.stack(frames, dim=0).unsqueeze(0).cpu()

        return {"reward": mean_reward, "frames": video}

    def _try_render(self) -> torch.Tensor | None:
        """Render one frame from the underlying environment, if available."""
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

        # (H, W, C) -> (C, H, W).
        if raw.ndim == 3 and raw.shape[-1] in (3, 4):
            raw = raw[..., :3]
            raw = raw.permute(2, 0, 1)

        return raw.to(torch.uint8)

    def shutdown(self) -> None:
        """Shut down the environment."""
        is_closed = getattr(self.env, "is_closed", False)
        if not is_closed:
            self.env.close()
