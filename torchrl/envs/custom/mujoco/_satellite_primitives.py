# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Satellite attitude macro actions for MuJoCo CMG examples."""

from __future__ import annotations

from typing import Any, Literal

import torch
from tensordict import TensorDictBase
from tensordict.utils import NestedKey
from torchrl.data.tensor_specs import Categorical, Composite, Unbounded
from torchrl.envs.custom.mujoco._math import (
    cmg_jacobian,
    orthogonal_6cmg_geometry,
    pyramid_4cmg_geometry,
    quat_conj,
    quat_log,
    quat_mul,
)
from torchrl.envs.transforms._primitive import (
    MacroAction,
    MacroPrimitive,
    MacroPrimitiveTransform,
    TargetMacroAction,
)

__all__ = ["SatelliteMacroAction", "SatelliteAttitudeTransform"]


def _normalize_quat_or_identity(quat: torch.Tensor) -> torch.Tensor:
    norm = quat.norm(dim=-1, keepdim=True)
    normalized = quat / norm.clamp_min(1e-12)
    identity = torch.zeros_like(quat)
    identity[..., 0] = 1.0
    return torch.where(norm > 1e-12, normalized, identity)


class SatelliteMacroAction(TargetMacroAction):
    r"""Structured action containing a desired satellite attitude.

    ``SatelliteMacroAction`` is a :class:`~torchrl.envs.transforms.TargetMacroAction`
    whose ``target`` field holds a desired attitude quaternion (``w, x, y, z``).
    A :class:`SatelliteAttitudeTransform` reads the current satellite attitude
    observations, computes a local CMG steering command, and expands that
    command into a low-level gimbal-rate action sequence.

    Examples:
        >>> import torch
        >>> from torchrl.envs import SatelliteMacroAction
        >>> action = SatelliteMacroAction.slew_attitude(torch.tensor([1.0, 0.0, 0.0, 0.0]))
        >>> action.target.shape
        torch.Size([1, 4])
    """

    @classmethod
    def slew_attitude(
        cls,
        target_quat: torch.Tensor,
        *,
        steps: int = 36,
        settle_steps: int = 8,
    ) -> SatelliteMacroAction:
        """Ask the transform to steer the satellite toward ``target_quat``."""
        if target_quat.ndim == 0 or target_quat.shape[-1] != 4:
            raise ValueError(
                f"target_quat must have trailing dimension 4, got {target_quat.shape}."
            )
        target_quat = _normalize_quat_or_identity(target_quat)
        return cls.move(target_quat, steps=steps, settle_steps=settle_steps)


class SatelliteAttitudeTransform(MacroPrimitiveTransform):
    r"""Expand desired satellite attitudes into CMG gimbal-rate sequences.

    This transform is a satellite-specific preset. The policy-facing action is a
    desired attitude quaternion, provided either as a raw tensor under
    ``action_key``, under ``(action_key, "target")`` / ``(action_key,
    "attitude")``, or through a :class:`SatelliteMacroAction` (which also carries
    per-action durations). The transform computes the current attitude error,
    applies a small proportional-derivative steering law in body-rate
    coordinates, maps it through the instantaneous CMG Jacobian, and delegates
    fixed-length interpolation / execution to
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

    Examples:
        >>> import torch
        >>> from tensordict import TensorDict
        >>> from torchrl.envs import SatelliteAttitudeTransform
        >>> td = TensorDict({
        ...     "action": torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        ...     "bus_quat": torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        ...     "bus_omega": torch.zeros(1, 3),
        ...     "gimbal_angles": torch.cat([torch.zeros(1, 4), torch.ones(1, 4)], -1),
        ... }, batch_size=[1])
        >>> SatelliteAttitudeTransform(num_cmgs=4, macro_steps=2, settle_steps=0).inv(td)["action"].shape
        torch.Size([1, 2, 4])
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
            action_key=action_key,
            macro_steps=macro_steps,
            settle_steps=settle_steps,
            action_dim=self.num_cmgs,
            execute=execute,
            multi_action_dim=multi_action_dim,
            stack_rewards=stack_rewards,
            stack_observations=stack_observations,
        )

    # ------------------------------------------------------------------ #
    # Resolve
    # ------------------------------------------------------------------ #
    def _resolve(
        self, tensordict: TensorDictBase, action: Any
    ) -> tuple[torch.Tensor, torch.Tensor, int, int]:
        target_quat, steps, settle_steps, mode = self._read_target_quat(
            tensordict, action
        )
        target = self.attitude_action_target(tensordict, target_quat)
        batch_shape = target.shape[:-1]
        start = self.current_action(
            tensordict, batch_shape, target.device, target.dtype, self.action_dim
        )
        if mode is not None:
            # ``WAIT`` holds the bus still (zero gimbal-rate command).
            mode = mode.to(device=target.device).reshape(batch_shape + (1,))
            target = torch.where(mode == int(MacroPrimitive.WAIT), start, target)
        return start, target, steps, settle_steps

    def current_action(
        self,
        tensordict: TensorDictBase,
        batch_shape: torch.Size,
        device: torch.device,
        dtype: torch.dtype,
        action_dim: int,
    ) -> torch.Tensor:
        # The gimbal-rate command is computed afresh each macro step, so the
        # interpolation starts from the zero command.
        return torch.zeros(batch_shape + (self.action_dim,), dtype=dtype, device=device)

    def _read_target_quat(
        self, tensordict: TensorDictBase, action: Any
    ) -> tuple[torch.Tensor, int, int, torch.Tensor | None]:
        steps, settle_steps = self.macro_steps, self.settle_steps
        mode: torch.Tensor | None = None
        if action is None:
            raise RuntimeError(
                f"{type(self).__name__} found no attitude action under "
                f"{self.action_key!r}."
            )
        if isinstance(action, torch.Tensor):
            target_quat = action
        elif isinstance(action, (TensorDictBase, MacroAction)):
            keys = action.keys(True, True)
            for candidate in ("target", "attitude", "target_quat"):
                if candidate in keys:
                    target_quat = action.get(candidate)
                    break
            else:
                raise RuntimeError(
                    f"{type(self).__name__} expected a 'target' (or 'attitude') "
                    f"attitude quaternion under {self.action_key!r}; got keys "
                    f"{tuple(keys)}."
                )
            steps = self._field_int(action, "steps", self.macro_steps)
            settle_steps = self._field_int(action, "settle_steps", self.settle_steps)
            if "mode" in keys:
                mode = action.get("mode").to(torch.long)
        else:
            raise TypeError(
                f"{type(self).__name__} expected a SatelliteMacroAction, a "
                f"TensorDict or an attitude quaternion tensor; got "
                f"{type(action).__name__}."
            )
        if target_quat.shape[-1:] != torch.Size([4]):
            raise ValueError(
                f"{type(self).__name__} expected an attitude quaternion with "
                f"trailing dimension 4, got {target_quat.shape}."
            )
        if target_quat.ndim != len(tensordict.batch_size) + 1:
            raise ValueError(
                f"{type(self).__name__} expected an attitude quaternion with shape "
                f"{tuple(tensordict.batch_size)} + (4,), got {target_quat.shape}."
            )
        return target_quat, steps, settle_steps, mode

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
        bus_quat = tensordict["bus_quat"]
        dtype = bus_quat.dtype
        device = bus_quat.device
        bus_quat = _normalize_quat_or_identity(bus_quat)
        if target_quat.dtype != dtype or target_quat.device != device:
            target_quat = target_quat.to(dtype=dtype, device=device)
        target_quat = _normalize_quat_or_identity(target_quat)
        quat_err = quat_log(quat_mul(quat_conj(bus_quat), target_quat))
        bus_omega = tensordict["bus_omega"]
        if bus_omega.dtype != dtype or bus_omega.device != device:
            bus_omega = bus_omega.to(dtype=dtype, device=device)
        gimbal_obs = tensordict["gimbal_angles"]
        if gimbal_obs.dtype != dtype or gimbal_obs.device != device:
            gimbal_obs = gimbal_obs.to(dtype=dtype, device=device)
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

    # ------------------------------------------------------------------ #
    # Specs
    # ------------------------------------------------------------------ #
    def transform_input_spec(self, input_spec: Composite) -> Composite:
        input_spec = input_spec.clone()
        batch_size = input_spec.shape
        device = input_spec.device
        dtype = self._attitude_spec_dtype(input_spec)
        full_action_spec = Composite(shape=batch_size, device=device)
        full_action_spec[self.action_key] = Composite(
            mode=Categorical(
                n=len(MacroPrimitive),
                shape=(*batch_size, 1),
                dtype=torch.long,
                device=device,
            ),
            target=Unbounded(shape=(*batch_size, 4), dtype=dtype, device=device),
            shape=batch_size,
            device=device,
        )
        input_spec["full_action_spec"] = full_action_spec
        return input_spec

    def _attitude_spec_dtype(self, input_spec: Composite) -> torch.dtype:
        for spec_name, key in (
            ("full_observation_spec", "bus_quat"),
            ("full_state_spec", "target_quat"),
            ("full_action_spec", self.action_key),
        ):
            spec = input_spec.get(spec_name, None)
            if not isinstance(spec, Composite) or key not in spec.keys(True, True):
                continue
            leaf = spec[key]
            dtype = getattr(leaf, "dtype", None)
            if dtype is not None:
                return dtype
        return torch.get_default_dtype()

    def _action_scale(self) -> float:
        if self.action_scale is not None:
            return self.action_scale
        env = self._find_parent_env_with("action_scale")
        if env is not None:
            return float(env.action_scale)
        return 3.0
