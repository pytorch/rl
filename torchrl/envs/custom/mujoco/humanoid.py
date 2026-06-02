# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Humanoid-v4 locomotion env, matching the Gymnasium task spec."""

from __future__ import annotations

import torch
from tensordict import TensorDictBase
from torchrl.envs.custom.mujoco.base import MujocoEnv
from torchrl.envs.transforms._base import Transform
from torchrl.envs.transforms._primitive import MacroPrimitiveTransform


class HumanoidEnv(MujocoEnv):
    """Bipedal humanoid locomotion task (28 DoF, 21 actuators).

    Reward is forward x-velocity plus a healthy bonus minus a control
    cost. Termination occurs when the torso z-height leaves
    ``[HEALTHY_Z_LOW, HEALTHY_Z_HIGH]``.

    Args: see :class:`~torchrl.envs.custom.mujoco.MujocoEnv`.

    Example:
        >>> from torchrl.envs import HumanoidEnv  # doctest: +SKIP
        >>> env = HumanoidEnv(num_envs=4)         # doctest: +SKIP
        >>> td = env.rollout(10)                  # doctest: +SKIP

    Reference:
        Tassa et al., "Synthesis and Stabilization of Complex Behaviors
        through Online Trajectory Optimization", IROS 2012.
    """

    XML_PATH = "humanoid.xml"
    FRAME_SKIP = 5
    SKIP_QPOS = 2
    HEALTHY_Z_LOW = 1.0
    HEALTHY_Z_HIGH = 2.0
    HEALTHY_REWARD = 5.0
    CTRL_COST_WEIGHT = 0.1

    def make_control_transform(
        self,
        *,
        execute: bool = False,
        multi_action_dim: int = 1,
        stack_rewards: bool = True,
        stack_observations: bool = False,
        macro_steps: int = 16,
        settle_steps: int = 0,
    ) -> Transform:
        """Build the transform that expands actuator targets into sequences.

        Examples:
            >>> from torchrl.envs import HumanoidEnv  # doctest: +SKIP
            >>> env = HumanoidEnv()  # doctest: +SKIP
            >>> transform = env.make_control_transform(execute=True)  # doctest: +SKIP
        """
        return MacroPrimitiveTransform(
            action_dim=self.action_spec.shape[-1],
            execute=execute,
            multi_action_dim=multi_action_dim,
            stack_rewards=stack_rewards,
            stack_observations=stack_observations,
            macro_steps=macro_steps,
            settle_steps=settle_steps,
        )

    def _make_obs(self, state: TensorDictBase) -> torch.Tensor:
        qpos = state["qpos"].to(self.dtype)
        qvel = state["qvel"].to(self.dtype).clamp(-10.0, 10.0)
        return torch.cat([qpos[..., self.SKIP_QPOS :], qvel], dim=-1)

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
