# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Satellite attitude-control task with Control Moment Gyros (CMGs).

The agent commands gimbal rates of either 4 (pyramid) or 6 (orthogonal)
CMGs to slew the satellite to a target attitude sampled uniformly on
``SO(3)`` at reset, while *avoiding internal singularities* of the CMG
cluster. The singularity penalty uses the manipulability metric of the
gimbal Jacobian: ``m(J) = sqrt(det(J J^T) + eps)``.

Reward
    ``r = - ||q_err|| - lambda_s / m(J) - lambda_u * ||a||^2``

Termination
    Never (the satellite cannot crash). Use a
    :class:`~torchrl.envs.transforms.TimeLimit` transform for truncation.
"""
from __future__ import annotations

import re
from typing import Literal

import torch
from tensordict import TensorDictBase

from torchrl.data.tensor_specs import Bounded, Composite, Unbounded
from torchrl.envs.custom.mujoco._math import (
    cmg_jacobian,
    manipulability,
    orthogonal_6cmg_geometry,
    pyramid_4cmg_geometry,
    quat_conj,
    quat_log,
    quat_mul,
    random_unit_quat,
)
from torchrl.envs.custom.mujoco.base import MujocoEnv


class SatelliteEnv(MujocoEnv):
    """Attitude-control task with 4 or 6 CMGs.

    Args:
        num_cmgs: ``4`` (pyramid) or ``6`` (orthogonal cluster).
        attitude_weight: weight on ``-||q_err||``. Default ``1.0``.
        singularity_weight: ``lambda_s`` -- weight on ``1 / m(J)``.
            Default ``0.05``. Increase to push the policy harder away
            from internal singularities; values too large dominate the
            attitude term and stall slewing.
        ctrl_cost_weight: weight on ``||a||^2``. Default ``0.01``.
        rotor_h: scalar rotor angular momentum used in the Jacobian.
            Defaults to :attr:`ROTOR_SPEED` (proxy unit, since the
            absolute scale only changes the singularity threshold).
        \\*\\*kwargs: forwarded to :class:`MujocoEnv`.

    Example:
        >>> from torchrl.envs import SatelliteEnv  # doctest: +SKIP
        >>> env = SatelliteEnv(num_cmgs=4, num_envs=4)  # doctest: +SKIP
        >>> td = env.rollout(50)  # doctest: +SKIP

    Reference:
        Wie, Bong (2008). "Space Vehicle Dynamics and Control" --
        chapter on CMG steering laws and singularity escape.
    """

    FRAME_SKIP = 10
    RESET_NOISE_SCALE = 0.001
    ROTOR_SPEED_4 = 100.0
    ROTOR_SPEED_6 = 200.0
    RENDER_BACKGROUND = (0.0, 0.0, 0.05)

    def __init__(
        self,
        *,
        num_cmgs: Literal[4, 6] = 4,
        attitude_weight: float = 1.0,
        singularity_weight: float = 0.05,
        ctrl_cost_weight: float = 0.01,
        rotor_h: float | None = None,
        **kwargs,
    ) -> None:
        if num_cmgs == 4:
            self.N_GIMBALS = 4
            self.ROTOR_SPEED = self.ROTOR_SPEED_4
            self.XML_PATH = "satellite_large.xml"
        elif num_cmgs == 6:
            self.N_GIMBALS = 6
            self.ROTOR_SPEED = self.ROTOR_SPEED_6
            self.XML_PATH = "satellite_small.xml"
        else:
            raise ValueError(f"num_cmgs must be 4 or 6, got {num_cmgs}")
        self.num_cmgs = num_cmgs
        self.attitude_weight = float(attitude_weight)
        self.singularity_weight = float(singularity_weight)
        self.ctrl_cost_weight = float(ctrl_cost_weight)
        self._rotor_h = float(rotor_h) if rotor_h is not None else self.ROTOR_SPEED

        super().__init__(**kwargs)

        # CMG geometry, cached on the env's device / dtype.
        if num_cmgs == 4:
            g, r0 = pyramid_4cmg_geometry(device=self.device, dtype=self.dtype)
        else:
            g, r0 = orthogonal_6cmg_geometry(device=self.device, dtype=self.dtype)
        self._gimbal_axes = g
        self._rotor_axes_ref = r0

        # Per-env target attitude, populated on reset.
        self._target_quat = torch.zeros(
            self.num_envs, 4, device=self.device, dtype=self.dtype
        )
        self._target_quat[:, 0] = 1.0

    # ------------------------------------------------------------------
    # XML patching: no ground floor in space.
    # ------------------------------------------------------------------

    @classmethod
    def _patch_xml(cls, xml: str) -> str:
        xml = re.sub(r"<camera\b[^/]*/>\s*", "", xml)
        xml = re.sub(r"<light\b[^/]*/>\s*", "", xml)
        camera = (
            '<camera name="side" pos="3 -3 2" '
            'xyaxes="0.707 0.707 0 -0.302 0.302 0.905" fovy="60"/>'
        )
        light = (
            '<light name="top" pos="0 0 4" dir="0 0 -1" '
            'diffuse="0.8 0.8 0.8" ambient="0.3 0.3 0.3" directional="true"/>'
        )
        return xml.replace("<worldbody>", f"<worldbody>\n  {camera}\n  {light}")

    # ------------------------------------------------------------------
    # Specs
    # ------------------------------------------------------------------

    def _make_obs_spec(self) -> Composite:
        # obs = quat_err(3) + bus_omega(3) + gimbal_angles(N) + gimbal_rates(N)
        obs_dim = 6 + 2 * self.N_GIMBALS
        return Composite(
            observation=Unbounded(
                shape=(self.num_envs, obs_dim),
                dtype=self.dtype,
                device=self.device,
            ),
            shape=(self.num_envs,),
            device=self.device,
        )

    def _make_specs(self) -> None:
        super()._make_specs()
        # Override the action spec: agent controls only gimbals, not rotors.
        self.action_spec = Bounded(
            low=-1.0,
            high=1.0,
            shape=(self.num_envs, self.N_GIMBALS),
            dtype=self.dtype,
            device=self.device,
        )

    # ------------------------------------------------------------------
    # State plumbing
    # ------------------------------------------------------------------

    def _sample_initial_state(self, n: int) -> tuple[torch.Tensor, torch.Tensor]:
        qpos, qvel = super()._sample_initial_state(n)
        # Pin rotor velocities (qvel layout: [bus_lin(3), bus_ang(3),
        # then per-CMG (gimbal_rate, rotor_rate) pairs]).
        rotor_vel_idx = [6 + 2 * i + 1 for i in range(self.N_GIMBALS)]
        qvel[..., rotor_vel_idx] = self.ROTOR_SPEED
        return qpos, qvel

    def _prepare_ctrl(self, action: torch.Tensor) -> torch.Tensor:
        rotor_ctrl = torch.full(
            (action.shape[0], self.N_GIMBALS),
            self.ROTOR_SPEED,
            dtype=action.dtype,
            device=action.device,
        )
        return torch.cat([action, rotor_ctrl], dim=-1)

    def _on_reset_all(self) -> None:
        self._target_quat = random_unit_quat(
            (self.num_envs,),
            generator=self.rng,
            device=self.device,
            dtype=self.dtype,
        )

    def _on_reset_mask(self, mask: torch.Tensor) -> None:
        n = int(mask.sum().item())
        if n == 0:
            return
        new_targets = random_unit_quat(
            (n,), generator=self.rng, device=self.device, dtype=self.dtype
        )
        self._target_quat[mask] = new_targets

    # ------------------------------------------------------------------
    # Observation, reward, done
    # ------------------------------------------------------------------

    def _attitude_error(self, qpos: torch.Tensor) -> torch.Tensor:
        bus_quat = qpos[..., 3:7].to(self.dtype)
        # q_err such that target = current * q_err  =>  q_err = current^-1 * target
        q_err = quat_mul(quat_conj(bus_quat), self._target_quat)
        return quat_log(q_err)  # (B, 3)

    def _make_obs(self, state: TensorDictBase) -> torch.Tensor:
        qpos = state["qpos"].to(self.dtype)
        qvel = state["qvel"].to(self.dtype)
        n = self.N_GIMBALS
        gimbal_idx = list(range(7, 7 + n))
        gimbal_rate_idx = [6 + 2 * i for i in range(n)]
        return torch.cat(
            [
                self._attitude_error(qpos),
                qvel[..., 3:6],
                qpos[..., gimbal_idx],
                qvel[..., gimbal_rate_idx],
            ],
            dim=-1,
        )

    def _compute_reward(
        self,
        state: TensorDictBase,
        action: torch.Tensor,
        next_state: TensorDictBase,
    ) -> torch.Tensor:
        qpos = next_state["qpos"].to(self.dtype)
        gimbal_idx = list(range(7, 7 + self.N_GIMBALS))
        gimbal_angles = qpos[..., gimbal_idx]

        att_err = self._attitude_error(qpos).norm(dim=-1)
        jac = cmg_jacobian(
            gimbal_angles, self._gimbal_axes, self._rotor_axes_ref, self._rotor_h
        )
        manip = manipulability(jac)
        ctrl_cost = (action.to(self.dtype) ** 2).sum(dim=-1)
        reward = (
            -self.attitude_weight * att_err
            - self.singularity_weight / manip
            - self.ctrl_cost_weight * ctrl_cost
        )
        return reward.unsqueeze(-1)

    def _compute_done(
        self,
        state: TensorDictBase,
        next_state: TensorDictBase,
    ) -> torch.Tensor:
        return torch.zeros(
            self.num_envs, 1, dtype=torch.bool, device=self.device
        )
