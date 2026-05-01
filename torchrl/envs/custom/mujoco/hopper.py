# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Hopper-v4 single-leg locomotion env."""
from __future__ import annotations

import torch
from tensordict import TensorDictBase

from torchrl.envs.custom.mujoco.base import MujocoEnv


class HopperEnv(MujocoEnv):
    """Single-legged hopping task (6-DoF, 3 actuators).

    Args: see :class:`~torchrl.envs.custom.mujoco.MujocoEnv`.

    Example:
        >>> from torchrl.envs import HopperEnv  # doctest: +SKIP
        >>> env = HopperEnv(num_envs=4)         # doctest: +SKIP
        >>> td = env.rollout(10)                # doctest: +SKIP
    """

    XML_PATH = "hopper.xml"
    FRAME_SKIP = 5
    SKIP_QPOS = 1
    HEALTHY_Z_MIN = 0.7
    HEALTHY_ANGLE_MAX = 0.2
    HEALTHY_REWARD = 1.0
    CTRL_COST_WEIGHT = 1e-3

    def _make_obs(self, state: TensorDictBase) -> torch.Tensor:
        qpos = state["qpos"].to(self.dtype)
        qvel = state["qvel"].to(self.dtype).clamp(-10.0, 10.0)
        return torch.cat([qpos[..., self.SKIP_QPOS :], qvel], dim=-1)

    def _is_healthy(self, qpos: torch.Tensor) -> torch.Tensor:
        z = qpos[..., 1]
        angle = qpos[..., 2]
        return (z >= self.HEALTHY_Z_MIN) & (angle.abs() <= self.HEALTHY_ANGLE_MAX)

    def _compute_reward(
        self,
        state: TensorDictBase,
        action: torch.Tensor,
        next_state: TensorDictBase,
    ) -> torch.Tensor:
        dt = self._backend.timestep * self.frame_skip
        forward_vel = (next_state["qpos"][..., 0] - state["qpos"][..., 0]) / dt
        ctrl_cost = self.CTRL_COST_WEIGHT * (action.to(self.dtype) ** 2).sum(dim=-1)
        healthy = self._is_healthy(next_state["qpos"]).to(self.dtype)
        reward = forward_vel.to(self.dtype) + self.HEALTHY_REWARD * healthy - ctrl_cost
        return reward.unsqueeze(-1)

    def _compute_done(
        self,
        state: TensorDictBase,
        next_state: TensorDictBase,
    ) -> torch.Tensor:
        return (~self._is_healthy(next_state["qpos"])).unsqueeze(-1)
