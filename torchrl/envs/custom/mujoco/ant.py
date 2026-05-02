# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Ant-v4 quadruped locomotion env."""

from __future__ import annotations

import re

import torch
from tensordict import TensorDictBase
from torchrl.envs.custom.mujoco.base import MujocoEnv


class AntEnv(MujocoEnv):
    """Quadruped locomotion (15-DoF, 8 actuators).

    The bundled ``ant.xml`` is a fixed-base ant; we patch it at load
    time to insert a free joint on the torso and set ``timestep=0.01``,
    matching Gymnasium ``Ant-v4`` semantics.

    Args: see :class:`~torchrl.envs.custom.mujoco.MujocoEnv`.

    Example:
        >>> from torchrl.envs import AntEnv  # doctest: +SKIP
        >>> env = AntEnv(num_envs=8)         # doctest: +SKIP
        >>> td = env.rollout(50)             # doctest: +SKIP
    """

    XML_PATH = "ant.xml"
    FRAME_SKIP = 5
    RESET_NOISE_SCALE = 0.1
    SKIP_QPOS = 2
    HEALTHY_Z_LOW = 0.2
    HEALTHY_Z_HIGH = 1.0
    HEALTHY_REWARD = 1.0
    CTRL_COST_WEIGHT = 0.5

    @classmethod
    def _patch_xml(cls, xml: str) -> str:
        xml = super()._patch_xml(xml)
        xml = re.sub(
            r'(<body\s+name="torso"[^>]*>)',
            r"\1\n      <freejoint name='root'/>",
            xml,
            count=1,
        )
        xml = re.sub(
            r"(<compiler\b[^/]*/>\s*)",
            r'\1<option timestep="0.01"/>\n  ',
            xml,
            count=1,
        )
        return xml

    def _is_healthy(self, qpos: torch.Tensor) -> torch.Tensor:
        z = qpos[..., 2]
        return (z >= self.HEALTHY_Z_LOW) & (z <= self.HEALTHY_Z_HIGH)

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
