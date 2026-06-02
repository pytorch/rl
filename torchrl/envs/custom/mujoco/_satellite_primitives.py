# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Satellite attitude macro actions for MuJoCo CMG examples."""

from __future__ import annotations

from typing import Literal

import torch
from tensordict import TensorDictBase
from tensordict.tensorclass import TensorClass
from tensordict.utils import NestedKey
from torchrl.envs.custom.mujoco._math import (
    cmg_jacobian,
    orthogonal_6cmg_geometry,
    pyramid_4cmg_geometry,
    quat_conj,
    quat_log,
    quat_mul,
)
from torchrl.envs.transforms._primitive import MacroPrimitive, MacroPrimitiveTransform

__all__ = ["SatelliteAction", "SatelliteAttitudeTransform"]


def _as_batch(value: torch.Tensor, last_dim: int) -> torch.Tensor:
    if value.ndim == 0 or value.shape[-1] != last_dim:
        raise ValueError(
            f"Expected a tensor with trailing dimension {last_dim}, got {value.shape}."
        )
    if value.ndim == 1:
        return value.unsqueeze(0)
    return value


def _normalize_quat_or_identity(quat: torch.Tensor) -> torch.Tensor:
    norm = quat.norm(dim=-1, keepdim=True)
    normalized = quat / norm.clamp_min(1e-12)
    identity = torch.zeros_like(quat)
    identity[..., 0] = 1.0
    return torch.where(norm > 1e-12, normalized, identity)


class SatelliteAction(TensorClass["nocast"]):
    r"""Structured action containing a desired satellite attitude.

    The action stores the target attitude quaternion. A
    :class:`SatelliteAttitudeTransform` reads the current satellite attitude
    observations, computes a local CMG steering command, and expands that
    command into a low-level gimbal-rate action sequence.

    Examples:
        >>> import torch
        >>> from torchrl.envs import SatelliteAction
        >>> action = SatelliteAction.slew_attitude(torch.tensor([1.0, 0.0, 0.0, 0.0]))
        >>> action.target_quat.shape
        torch.Size([1, 4])
    """

    target_quat: torch.Tensor
    steps: torch.Tensor
    settle_steps: torch.Tensor

    @classmethod
    def slew_attitude(
        cls,
        target_quat: torch.Tensor,
        *,
        steps: int = 36,
        settle_steps: int = 8,
    ) -> SatelliteAction:
        """Ask the transform to steer the satellite toward ``target_quat``."""
        if steps <= 0:
            raise ValueError("steps must be strictly positive.")
        if settle_steps < 0:
            raise ValueError("settle_steps must be non-negative.")
        target_quat = _normalize_quat_or_identity(_as_batch(target_quat, 4))
        batch_size = target_quat.shape[:-1]
        return cls(
            target_quat=target_quat,
            steps=torch.full(
                batch_size + (1,),
                steps,
                dtype=torch.long,
                device=target_quat.device,
            ),
            settle_steps=torch.full(
                batch_size + (1,),
                settle_steps,
                dtype=torch.long,
                device=target_quat.device,
            ),
            batch_size=batch_size,
        )


class SatelliteAttitudeTransform(MacroPrimitiveTransform):
    r"""Expand desired satellite attitudes into CMG gimbal-rate sequences.

    This transform is a satellite-specific preset. The policy-facing action is
    a :class:`SatelliteAction` containing a target quaternion. The transform
    computes the current attitude error, applies a small proportional-derivative
    steering law in body-rate coordinates, maps it through the instantaneous
    CMG Jacobian, and delegates fixed-length interpolation/execution to
    :class:`~torchrl.envs.transforms.MacroPrimitiveTransform`.

    Args:
        num_cmgs: ``4`` for the pyramid CMG cluster or ``6`` for the orthogonal
            cluster.
        action_scale: scale used by :class:`~torchrl.envs.SatelliteEnv` to map
            normalized actions to physical gimbal rates. If ``None``, the
            transform tries to read ``action_scale`` from its parent env and
            falls back to ``3.0``.
        attitude_gain: proportional gain applied to the quaternion log error.
        angular_rate_gain: damping gain applied to ``bus_omega``.
        jacobian_rotor_h: rotor-momentum scale used in the steering Jacobian.
            The examples use ``1.0`` because the gains are tuned in normalized
            command units.

    Examples:
        >>> import torch
        >>> from tensordict import TensorDict
        >>> from torchrl.envs import SatelliteAction, SatelliteAttitudeTransform
        >>> td = TensorDict({
        ...     "action": SatelliteAction.slew_attitude(
        ...         torch.tensor([[1.0, 0.0, 0.0, 0.0]]), steps=2
        ...     ),
        ...     "bus_quat": torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        ...     "bus_omega": torch.zeros(1, 3),
        ...     "gimbal_angles": torch.cat([torch.zeros(1, 4), torch.ones(1, 4)], -1),
        ... }, batch_size=[1])
        >>> SatelliteAttitudeTransform(num_cmgs=4).inv(td)["action"].shape
        torch.Size([1, 10, 4])
    """

    def __init__(
        self,
        *,
        num_cmgs: Literal[4, 6] = 4,
        action_scale: float | None = None,
        attitude_gain: float = 5.0,
        angular_rate_gain: float = 8.0,
        jacobian_rotor_h: float = 1.0,
        execute: bool = False,
        multi_action_dim: int = 1,
        stack_rewards: bool = True,
        stack_observations: bool = False,
        action_key: NestedKey = "action",
        primitive_id_key: NestedKey = "primitive_id",
        target_qpos_key: NestedKey = "target_qpos",
        target_pose_key: NestedKey = "target_pose",
        macro_steps: int = 36,
        settle_steps: int = 8,
    ) -> None:
        if num_cmgs == 4:
            gimbal_axes, rotor_axes_ref = pyramid_4cmg_geometry()
        elif num_cmgs == 6:
            gimbal_axes, rotor_axes_ref = orthogonal_6cmg_geometry()
        else:
            raise ValueError(f"num_cmgs must be 4 or 6, got {num_cmgs}.")
        self.num_cmgs = int(num_cmgs)
        self.action_scale = None if action_scale is None else float(action_scale)
        self.attitude_gain = float(attitude_gain)
        self.angular_rate_gain = float(angular_rate_gain)
        self.jacobian_rotor_h = float(jacobian_rotor_h)
        self.gimbal_axes = gimbal_axes
        self.rotor_axes_ref = rotor_axes_ref
        super().__init__(
            primitive_library=None,
            adapter=None,
            solver=None,
            execute=execute,
            multi_action_dim=multi_action_dim,
            stack_rewards=stack_rewards,
            stack_observations=stack_observations,
            action_key=action_key,
            primitive_id_key=primitive_id_key,
            target_qpos_key=target_qpos_key,
            target_pose_key=target_pose_key,
            macro_steps=macro_steps,
            settle_steps=settle_steps,
            action_dim=self.num_cmgs,
            cartesian_solver=None,
        )

    def attitude_action_target(
        self,
        tensordict: TensorDictBase,
        target_quat: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the normalized gimbal-rate target for ``target_quat``."""
        required = ("bus_quat", "bus_omega", "gimbal_angles")
        missing = [key for key in required if key not in tensordict.keys(True, True)]
        if missing:
            raise RuntimeError(
                "SatelliteAttitudeTransform requires SatelliteEnv observations "
                f"{required}; missing {tuple(missing)}."
            )
        bus_quat = _normalize_quat_or_identity(tensordict["bus_quat"])
        target_quat = _normalize_quat_or_identity(
            target_quat.to(dtype=bus_quat.dtype, device=bus_quat.device)
        )
        quat_err = quat_log(quat_mul(quat_conj(bus_quat), target_quat))
        bus_omega = tensordict["bus_omega"].to(
            dtype=quat_err.dtype,
            device=quat_err.device,
        )
        gimbal_obs = tensordict["gimbal_angles"].to(
            dtype=quat_err.dtype,
            device=quat_err.device,
        )
        n_gimbals = self.num_cmgs
        gimbal_angles = torch.atan2(
            gimbal_obs[..., :n_gimbals],
            gimbal_obs[..., n_gimbals:],
        )
        jacobian = cmg_jacobian(
            gimbal_angles,
            self.gimbal_axes.to(device=quat_err.device, dtype=quat_err.dtype),
            self.rotor_axes_ref.to(device=quat_err.device, dtype=quat_err.dtype),
            self.jacobian_rotor_h,
        )
        desired_bus_accel = (
            self.attitude_gain * quat_err - self.angular_rate_gain * bus_omega
        )
        gimbal_rate = -torch.linalg.pinv(jacobian).matmul(
            desired_bus_accel.unsqueeze(-1)
        )
        return (gimbal_rate.squeeze(-1) / self._action_scale()).clamp(-1.0, 1.0)

    def _has_structured_action(self, tensordict: TensorDictBase) -> bool:
        if self.action_key not in tensordict.keys(True, True):
            return False
        action = tensordict.get(self.action_key)
        if isinstance(action, torch.Tensor):
            return False
        if not hasattr(action, "keys") or not hasattr(action, "get"):
            return False
        keys = action.keys(True, True)
        return "target_quat" in keys or super()._has_structured_action(tensordict)

    def _unpack_structured_action(
        self, tensordict: TensorDictBase
    ) -> tuple[TensorDictBase, int, int]:
        action = tensordict.get(self.action_key)
        if "target_quat" not in action.keys(True, True):
            return super()._unpack_structured_action(tensordict)
        target = self.attitude_action_target(tensordict, action.get("target_quat"))
        out = tensordict.clone()
        batch_shape = target.shape[:-1]
        out.set(
            self.primitive_id_key,
            torch.full(
                batch_shape + (1,),
                int(MacroPrimitive.MOVEJ),
                dtype=torch.long,
                device=target.device,
            ),
        )
        out.set(self.target_qpos_key, target)
        return (
            out,
            self._structured_action_int(action, "steps", self.macro_steps),
            self._structured_action_int(action, "settle_steps", self.settle_steps),
        )

    def _action_scale(self) -> float:
        if self.action_scale is not None:
            return self.action_scale
        env = self._find_parent_env_with("action_scale")
        if env is not None:
            return float(env.action_scale)
        return 3.0
