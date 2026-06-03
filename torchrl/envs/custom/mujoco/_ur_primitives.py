# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""UR-style macro primitives used by MuJoCo manipulation examples.

This module is the manipulation specialization of the generic macro machinery in
:mod:`torchrl.envs.transforms._primitive`. It is intentionally scoped to a
six-joint UR-style arm with a scalar gripper command and is a worked example of
how to specialize :class:`~torchrl.envs.transforms.MacroPrimitiveTransform`:

* :class:`RobotMacroAction` extends :class:`~torchrl.envs.transforms.MacroAction`
  with pose / joint / gripper fields and readable factory methods;
* :class:`URScriptPrimitiveTransform` overrides the three transform hooks
  (:meth:`~URScriptPrimitiveTransform._resolve`,
  :meth:`~URScriptPrimitiveTransform.current_action`,
  :meth:`~URScriptPrimitiveTransform.transform_input_spec`) and keeps the
  Cartesian inverse-kinematics step local to the robot.
"""

from __future__ import annotations

from collections.abc import Callable
from enum import IntEnum
from typing import Any, ClassVar, Literal

import torch
from tensordict import TensorDictBase
from tensordict.utils import NestedKey
from torchrl.data.tensor_specs import Categorical, Composite, Unbounded
from torchrl.envs.transforms._primitive import (
    MacroAction,
    MacroPrimitive,
    MacroPrimitiveTransform,
)

__all__ = [
    "RobotMacroAction",
    "RobotMacroActionMode",
    "URScriptPrimitive",
    "URScriptPrimitiveTransform",
    "CartesianSolver",
]

GripperCommand = Literal["keep", "open", "closed"]
CartesianSolver = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


class URScriptPrimitive(IntEnum):
    r"""Integer ids for URScript-style robot primitives.

    The ids are specific to UR-style arm control with a binary gripper command;
    they extend the generic :class:`~torchrl.envs.transforms.MacroPrimitive`
    vocabulary (``WAIT``/``MOVE``) with joint, Cartesian and gripper moves.

    Examples:
        >>> from torchrl.envs import URScriptPrimitive
        >>> str(URScriptPrimitive.OPEN_GRIPPER)
        'open_gripper'
    """

    WAIT = int(MacroPrimitive.WAIT)
    MOVEJ = int(MacroPrimitive.MOVE)
    MOVEL = 2
    OPEN_GRIPPER = 3
    CLOSE_GRIPPER = 4

    def __str__(self) -> str:
        return self.name.lower()


class RobotMacroActionMode(IntEnum):
    r"""Readable modes for :class:`RobotMacroAction`.

    ``RobotMacroActionMode`` mirrors the URScript primitive set and adds
    ``RESET``. The reset mode requires a parent environment exposing
    ``robot_home_qpos``.

    Examples:
        >>> from torchrl.envs import RobotMacroActionMode
        >>> RobotMacroActionMode.REACH_POSE.name
        'REACH_POSE'
    """

    WAIT = int(URScriptPrimitive.WAIT)
    REACH_JOINTS = int(URScriptPrimitive.MOVEJ)
    REACH_POSE = int(URScriptPrimitive.MOVEL)
    OPEN_GRIPPER = int(URScriptPrimitive.OPEN_GRIPPER)
    CLOSE_GRIPPER = int(URScriptPrimitive.CLOSE_GRIPPER)
    RESET = len(URScriptPrimitive)


class _RobotMacroActionReset:
    def __repr__(self) -> str:
        return "RobotMacroAction.RESET"


def _unwrap_robot_macro_action(action: Any) -> Any:
    data = getattr(action, "data", None)
    if isinstance(data, _RobotMacroActionReset):
        return data
    return action


def _as_batch(value: torch.Tensor, last_dim: int) -> torch.Tensor:
    if value.shape[-1] != last_dim:
        raise ValueError(
            f"Expected a tensor with trailing dimension {last_dim}, got {value.shape}."
        )
    if value.ndim == 1:
        return value.unsqueeze(0)
    return value


def _identity_quaternion_like(position: torch.Tensor) -> torch.Tensor:
    quaternion = torch.zeros(
        position.shape[:-1] + (4,), dtype=position.dtype, device=position.device
    )
    quaternion[..., 0] = 1.0
    return quaternion


def _gripper_code(gripper: GripperCommand) -> int:
    if gripper == "keep":
        return RobotMacroAction.GRIPPER_KEEP
    if gripper == "open":
        return RobotMacroAction.GRIPPER_OPEN
    if gripper == "closed":
        return RobotMacroAction.GRIPPER_CLOSED
    raise ValueError(
        f"gripper must be one of 'keep', 'open' or 'closed', got {gripper!r}."
    )


def _batch_size(batch_size: torch.Size | tuple[int, ...] | None) -> torch.Size:
    if batch_size is None:
        return torch.Size([1])
    return torch.Size(batch_size)


def _optional_gripper_command(
    value: float | torch.Tensor | None,
    batch_size: torch.Size,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    if value is None:
        return torch.full(batch_size + (1,), float("nan"), dtype=dtype, device=device)
    value = torch.as_tensor(value, dtype=dtype, device=device)
    if value.ndim == 0:
        return value.reshape(1).expand(batch_size + (1,)).clone()
    if value.shape == batch_size:
        return value.unsqueeze(-1)
    return value.reshape(batch_size + (1,))


class RobotMacroAction(MacroAction):
    r"""Human-writable UR-style macro action.

    ``RobotMacroAction`` extends :class:`~torchrl.envs.transforms.MacroAction`
    with the fields needed for UR/gripper control. It is used by the
    cube-to-bowl tutorial to store a readable command under ``td["action"]``;
    :class:`URScriptPrimitiveTransform` expands it to low-level joint-position
    and gripper commands.

    Examples:
        >>> import torch
        >>> from torchrl.envs import RobotMacroAction
        >>> action = RobotMacroAction.reach_joints(joints=torch.zeros(1, 6))
        >>> action.joints.shape
        torch.Size([1, 6])
    """

    position: torch.Tensor
    quaternion: torch.Tensor
    joints: torch.Tensor
    gripper: torch.Tensor
    gripper_command: torch.Tensor

    GRIPPER_KEEP: ClassVar[int] = -1
    GRIPPER_OPEN: ClassVar[int] = 0
    GRIPPER_CLOSED: ClassVar[int] = 1
    RESET: ClassVar[_RobotMacroActionReset]

    @classmethod
    def reach_pose(
        cls,
        *,
        position: torch.Tensor,
        quaternion: torch.Tensor | None = None,
        gripper: GripperCommand = "keep",
        gripper_command: float | torch.Tensor | None = None,
        steps: int = 16,
        settle_steps: int = 0,
    ) -> RobotMacroAction:
        """Ask the end effector to reach a Cartesian pose."""
        position = _as_batch(position, 3)
        if quaternion is None:
            quaternion = _identity_quaternion_like(position)
        else:
            quaternion = _as_batch(quaternion, 4).to(
                dtype=position.dtype, device=position.device
            )
        return cls._make(
            RobotMacroActionMode.REACH_POSE,
            position=position,
            quaternion=quaternion,
            gripper=gripper,
            gripper_command=gripper_command,
            steps=steps,
            settle_steps=settle_steps,
        )

    @classmethod
    def reach_joints(
        cls,
        *,
        joints: torch.Tensor,
        gripper: GripperCommand = "keep",
        gripper_command: float | torch.Tensor | None = None,
        steps: int = 16,
        settle_steps: int = 0,
    ) -> RobotMacroAction:
        """Ask the arm to reach a six-joint UR configuration."""
        return cls._make(
            RobotMacroActionMode.REACH_JOINTS,
            joints=_as_batch(joints, 6),
            gripper=gripper,
            gripper_command=gripper_command,
            steps=steps,
            settle_steps=settle_steps,
        )

    @classmethod
    def home(
        cls,
        *,
        joints: torch.Tensor,
        gripper: GripperCommand = "open",
        gripper_command: float | torch.Tensor | None = None,
        steps: int = 16,
        settle_steps: int = 0,
    ) -> RobotMacroAction:
        """Ask the arm to return to an explicit home joint configuration."""
        return cls.reach_joints(
            joints=joints,
            gripper=gripper,
            gripper_command=gripper_command,
            steps=steps,
            settle_steps=settle_steps,
        )

    @classmethod
    def reset(
        cls,
        *,
        gripper: GripperCommand = "open",
        gripper_command: float | torch.Tensor | None = None,
        steps: int = 16,
        settle_steps: int = 0,
        batch_size: torch.Size | tuple[int, ...] | None = None,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ) -> RobotMacroAction:
        """Ask the transform to resolve the environment's reset/home posture."""
        dtype = torch.get_default_dtype() if dtype is None else dtype
        device = torch.device("cpu") if device is None else device
        batch_size = _batch_size(batch_size)
        position = torch.zeros(batch_size + (3,), dtype=dtype, device=device)
        return cls._make(
            RobotMacroActionMode.RESET,
            position=position,
            gripper=gripper,
            gripper_command=gripper_command,
            steps=steps,
            settle_steps=settle_steps,
        )

    @classmethod
    def open_gripper(
        cls,
        *,
        steps: int = 16,
        settle_steps: int = 0,
        batch_size: torch.Size | tuple[int, ...] | None = None,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ) -> RobotMacroAction:
        """Open the gripper while keeping the current arm state."""
        return cls._empty(
            RobotMacroActionMode.OPEN_GRIPPER,
            gripper="open",
            steps=steps,
            settle_steps=settle_steps,
            batch_size=batch_size,
            dtype=dtype,
            device=device,
        )

    @classmethod
    def close_gripper(
        cls,
        *,
        command: float | torch.Tensor | None = None,
        steps: int = 16,
        settle_steps: int = 0,
        batch_size: torch.Size | tuple[int, ...] | None = None,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ) -> RobotMacroAction:
        """Close the gripper while keeping the current arm state."""
        return cls._empty(
            RobotMacroActionMode.CLOSE_GRIPPER,
            gripper="closed",
            gripper_command=command,
            steps=steps,
            settle_steps=settle_steps,
            batch_size=batch_size,
            dtype=dtype,
            device=device,
        )

    @classmethod
    def wait(
        cls,
        *,
        gripper: GripperCommand = "keep",
        gripper_command: float | torch.Tensor | None = None,
        steps: int = 1,
        settle_steps: int = 0,
        batch_size: torch.Size | tuple[int, ...] | None = None,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ) -> RobotMacroAction:
        """Hold the current arm target for a number of low-level steps."""
        return cls._empty(
            RobotMacroActionMode.WAIT,
            gripper=gripper,
            gripper_command=gripper_command,
            steps=steps,
            settle_steps=settle_steps,
            batch_size=batch_size,
            dtype=dtype,
            device=device,
        )

    @classmethod
    def _empty(
        cls,
        mode: RobotMacroActionMode,
        *,
        gripper: GripperCommand,
        steps: int,
        settle_steps: int,
        batch_size: torch.Size | tuple[int, ...] | None,
        dtype: torch.dtype | None,
        device: torch.device | None,
        gripper_command: float | torch.Tensor | None = None,
    ) -> RobotMacroAction:
        dtype = torch.get_default_dtype() if dtype is None else dtype
        device = torch.device("cpu") if device is None else device
        batch_size = _batch_size(batch_size)
        return cls._make(
            mode,
            position=torch.zeros(batch_size + (3,), dtype=dtype, device=device),
            gripper=gripper,
            gripper_command=gripper_command,
            steps=steps,
            settle_steps=settle_steps,
        )

    @classmethod
    def _make(
        cls,
        mode: RobotMacroActionMode,
        *,
        position: torch.Tensor | None = None,
        quaternion: torch.Tensor | None = None,
        joints: torch.Tensor | None = None,
        gripper: GripperCommand = "keep",
        gripper_command: float | torch.Tensor | None = None,
        steps: int = 16,
        settle_steps: int = 0,
    ) -> RobotMacroAction:
        if position is not None:
            position = _as_batch(position, 3)
            batch_size = position.shape[:-1]
            dtype = position.dtype
            device = position.device
        elif joints is not None:
            joints = _as_batch(joints, 6)
            batch_size = joints.shape[:-1]
            dtype = joints.dtype
            device = joints.device
        else:
            batch_size = torch.Size([1])
            dtype = torch.get_default_dtype()
            device = torch.device("cpu")
            position = torch.zeros(batch_size + (3,), dtype=dtype, device=device)

        if position is None:
            position = torch.zeros(batch_size + (3,), dtype=dtype, device=device)
        if quaternion is None:
            quaternion = _identity_quaternion_like(position)
        else:
            quaternion = _as_batch(quaternion, 4).to(dtype=dtype, device=device)
        if joints is None:
            joints = torch.zeros(batch_size + (6,), dtype=dtype, device=device)

        return cls(
            position=position,
            quaternion=quaternion,
            joints=joints,
            gripper=torch.full(
                batch_size + (1,),
                _gripper_code(gripper),
                dtype=torch.long,
                device=device,
            ),
            gripper_command=_optional_gripper_command(
                gripper_command, batch_size, dtype, device
            ),
            batch_size=batch_size,
            **cls._duration_fields(
                mode=mode,
                steps=steps,
                settle_steps=settle_steps,
                batch_size=batch_size,
                device=device,
            ),
        )


RobotMacroAction.RESET = _RobotMacroActionReset()


class URScriptPrimitiveTransform(MacroPrimitiveTransform):
    r"""URScript-style preset of :class:`MacroPrimitiveTransform`.

    This specialization is scoped to six-joint UR-style arms with a scalar
    gripper command. The policy-facing action is a :class:`RobotMacroAction`
    placed under ``action_key``. The transform reads the arm and gripper joint
    observations to build the interpolation start, maps each primitive to a
    seven-dimensional joint-position + gripper destination (running Cartesian
    inverse kinematics for ``reach_pose``), and delegates fixed-length
    interpolation / execution to the generic base.

    Args:
        execute: if ``True``, return ``Compose(MultiAction(...), transform)``.
        action_key: key carrying the macro action and the expanded low-level
            sequence.
        robot_qpos_key: observation key for the six arm joints.
        gripper_qpos_key: observation key for the gripper joints.
        macro_steps: interpolated low-level actions per primitive.
        settle_steps: repeated final actions appended per primitive.
        action_dim: low-level action dimension (six joints + one gripper).
        cartesian_solver: optional callable mapping ``(target_pose,
            start_action)`` to a low-level action. When omitted, the transform
            uses a parent env's ``_cartesian_pose_to_joint_target`` hook.
        open_gripper_ctrl: low-level gripper command for an open gripper.
        close_gripper_ctrl: low-level gripper command for a closed gripper.

    Examples:
        >>> import torch
        >>> from tensordict import TensorDict
        >>> from torchrl.envs import RobotMacroAction, URScriptPrimitiveTransform
        >>> td = TensorDict({
        ...     "action": RobotMacroAction.reach_joints(joints=torch.ones(1, 6), steps=2),
        ...     "robot_qpos": torch.zeros(1, 6),
        ...     "gripper_qpos": torch.zeros(1, 2),
        ... }, batch_size=[1])
        >>> URScriptPrimitiveTransform().inv(td)["action"].shape
        torch.Size([1, 2, 7])
    """

    primitive_enum = URScriptPrimitive

    WAIT = URScriptPrimitive.WAIT
    MOVEJ = URScriptPrimitive.MOVEJ
    MOVEL = URScriptPrimitive.MOVEL
    OPEN_GRIPPER = URScriptPrimitive.OPEN_GRIPPER
    CLOSE_GRIPPER = URScriptPrimitive.CLOSE_GRIPPER

    def __init__(
        self,
        *,
        execute: bool = False,
        multi_action_dim: int = 1,
        stack_rewards: bool = True,
        stack_observations: bool = False,
        action_key: NestedKey = "action",
        robot_qpos_key: NestedKey = "robot_qpos",
        gripper_qpos_key: NestedKey = "gripper_qpos",
        macro_steps: int = 16,
        settle_steps: int = 0,
        action_dim: int = 7,
        cartesian_solver: CartesianSolver | None = None,
        open_gripper_ctrl: float = 0.0,
        close_gripper_ctrl: float = 255.0,
    ) -> None:
        super().__init__(
            action_key=action_key,
            macro_steps=macro_steps,
            settle_steps=settle_steps,
            action_dim=action_dim,
            execute=execute,
            multi_action_dim=multi_action_dim,
            stack_rewards=stack_rewards,
            stack_observations=stack_observations,
        )
        self.robot_qpos_key = robot_qpos_key
        self.gripper_qpos_key = gripper_qpos_key
        self.cartesian_solver = cartesian_solver
        self.open_gripper_ctrl = float(open_gripper_ctrl)
        self.close_gripper_ctrl = float(close_gripper_ctrl)

    # ------------------------------------------------------------------ #
    # Low-level action helpers
    # ------------------------------------------------------------------ #
    def low_level_action(
        self,
        robot_qpos: torch.Tensor,
        gripper: float | torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Build a low-level joint-position + gripper action."""
        action = torch.zeros(
            robot_qpos.shape[:-1] + (self.action_dim,),
            dtype=robot_qpos.dtype,
            device=robot_qpos.device,
        )
        n = min(robot_qpos.shape[-1], self.action_dim - 1)
        action[..., :n] = robot_qpos[..., :n]
        if gripper is None:
            action[..., -1] = self.open_gripper_ctrl
        elif isinstance(gripper, torch.Tensor):
            gripper = gripper.to(dtype=robot_qpos.dtype, device=robot_qpos.device)
            if gripper.numel() == 1:
                action[..., -1] = gripper.reshape(())
            else:
                action[..., -1:] = gripper.reshape(robot_qpos.shape[:-1] + (1,))
        else:
            action[..., -1] = float(gripper)
        return action

    @staticmethod
    def _set_gripper(action: torch.Tensor, gripper: torch.Tensor) -> torch.Tensor:
        out = action.clone()
        out[..., -1:] = gripper
        return out

    # ------------------------------------------------------------------ #
    # Convenience constructors (inspection / scripting helpers)
    # ------------------------------------------------------------------ #
    def make_primitive(
        self,
        tensordict: TensorDictBase,
        primitive_id: int | IntEnum,
        *,
        target_pose: torch.Tensor | None = None,
        target_qpos: torch.Tensor | None = None,
        gripper: float | torch.Tensor | None = None,
        steps: int | None = None,
        settle_steps: int | None = None,
    ) -> TensorDictBase:
        """Return a copy of ``tensordict`` carrying one :class:`RobotMacroAction`.

        Maps a URScript primitive id (and its pose / joint / gripper arguments)
        onto a :class:`RobotMacroAction` placed under ``action_key``.
        """
        pid = int(primitive_id)
        steps = self.macro_steps if steps is None else steps
        settle_steps = self.settle_steps if settle_steps is None else settle_steps
        batch_size = tensordict.batch_size
        device = tensordict.device or torch.device("cpu")
        dtype = self._ur_dtype(tensordict)
        common = {"steps": steps, "settle_steps": settle_steps}
        empty = {"batch_size": batch_size, "dtype": dtype, "device": device}
        if pid == int(URScriptPrimitive.WAIT):
            action = RobotMacroAction.wait(
                gripper="keep", gripper_command=gripper, **common, **empty
            )
        elif pid == int(URScriptPrimitive.MOVEJ):
            if target_qpos is None:
                raise ValueError("MOVEJ requires target_qpos.")
            joints = target_qpos[..., : self.action_dim - 1]
            action = RobotMacroAction.reach_joints(
                joints=joints, gripper="keep", gripper_command=gripper, **common
            )
        elif pid == int(URScriptPrimitive.MOVEL):
            if target_pose is None:
                raise ValueError("MOVEL requires target_pose.")
            quaternion = target_pose[..., 3:7] if target_pose.shape[-1] >= 7 else None
            action = RobotMacroAction.reach_pose(
                position=target_pose[..., :3],
                quaternion=quaternion,
                gripper="keep",
                gripper_command=gripper,
                **common,
            )
        elif pid == int(URScriptPrimitive.OPEN_GRIPPER):
            action = RobotMacroAction.open_gripper(**common, **empty)
        elif pid == int(URScriptPrimitive.CLOSE_GRIPPER):
            action = RobotMacroAction.close_gripper(command=gripper, **common, **empty)
        else:
            raise ValueError(f"Unknown URScript primitive id: {primitive_id!r}.")
        out = tensordict.copy()
        out.set(self.action_key, action)
        return out

    def action_sequence(
        self,
        tensordict: TensorDictBase,
        primitive_id: int | IntEnum | None = None,
        *,
        target_pose: torch.Tensor | None = None,
        target_qpos: torch.Tensor | None = None,
        gripper: float | torch.Tensor | None = None,
        steps: int | None = None,
        settle_steps: int | None = None,
    ) -> torch.Tensor:
        """Expand a UR primitive into its low-level sequence without executing."""
        if primitive_id is not None:
            tensordict = self.make_primitive(
                tensordict,
                primitive_id,
                target_pose=target_pose,
                target_qpos=target_qpos,
                gripper=gripper,
                steps=steps,
                settle_steps=settle_steps,
            )
        return self.inv(tensordict).get(self.action_key)

    def current_action(
        self,
        tensordict: TensorDictBase,
        batch_shape: torch.Size,
        device: torch.device,
        dtype: torch.dtype,
        action_dim: int,
    ) -> torch.Tensor:
        start = torch.zeros(
            batch_shape + (self.action_dim,), dtype=dtype, device=device
        )
        keys = tensordict.keys(True, True)
        if self.robot_qpos_key in keys:
            robot_qpos = tensordict.get(self.robot_qpos_key).to(
                dtype=dtype, device=device
            )
            n = min(robot_qpos.shape[-1], self.action_dim - 1)
            start[..., :n] = robot_qpos[..., :n]
        if self.gripper_qpos_key in keys:
            gripper_qpos = tensordict.get(self.gripper_qpos_key).to(
                dtype=dtype, device=device
            )
            start[..., -1] = gripper_qpos[..., 0]
        return start

    # ------------------------------------------------------------------ #
    # Resolve
    # ------------------------------------------------------------------ #
    def _resolve(
        self, tensordict: TensorDictBase, action: Any
    ) -> tuple[torch.Tensor, torch.Tensor, int, int]:
        action = _unwrap_robot_macro_action(action)
        batch_shape = tensordict.batch_size
        device = self._device(tensordict, action)
        dtype = self._ur_dtype(tensordict)
        start = self.current_action(
            tensordict, batch_shape, device, dtype, self.action_dim
        )

        if isinstance(action, _RobotMacroActionReset):
            action = RobotMacroAction.reset(
                batch_size=batch_shape,
                dtype=dtype,
                device=device,
                steps=self.macro_steps,
                settle_steps=self.settle_steps,
            )
        if isinstance(action, torch.Tensor):
            target = action.to(dtype=dtype, device=device).reshape(
                batch_shape + (self.action_dim,)
            )
            return start, target, self.macro_steps, self.settle_steps
        if not isinstance(action, (TensorDictBase, MacroAction)):
            raise TypeError(
                f"{type(self).__name__} expected a RobotMacroAction, a TensorDict "
                f"or a low-level action tensor; got {type(action).__name__}."
            )

        keys = action.keys(True, True)
        if "mode" not in keys:
            raise RuntimeError(
                f"{type(self).__name__} expected a RobotMacroAction with a 'mode' "
                f"field under {self.action_key!r}; got keys {tuple(keys)}."
            )
        mode = action.get("mode").to(torch.long).reshape(batch_shape + (1,))
        steps = self._field_int(action, "steps", self.macro_steps)
        settle_steps = self._field_int(action, "settle_steps", self.settle_steps)

        lib = self.primitive_enum
        joint_dim = self.action_dim - 1
        reset_mask = mode == int(RobotMacroActionMode.RESET)
        primitive_id = torch.where(
            reset_mask, torch.full_like(mode, int(lib.MOVEJ)), mode
        )

        position = self._field(action, "position", batch_shape, dtype, device, 3)
        quaternion_default = torch.zeros(batch_shape + (4,), dtype=dtype, device=device)
        quaternion_default[..., 0] = 1.0
        quaternion = self._field(
            action, "quaternion", batch_shape, dtype, device, 4, quaternion_default
        )
        joints = self._field(
            action,
            "joints",
            batch_shape,
            dtype,
            device,
            joint_dim,
            start[..., :joint_dim],
        )
        if reset_mask.any():
            joints = torch.where(
                reset_mask, self._env_home_qpos(batch_shape, dtype, device), joints
            )
        joints = torch.where(
            primitive_id == int(lib.MOVEJ), joints, start[..., :joint_dim]
        )
        gripper = self._structured_gripper(action, start, batch_shape, dtype, device)

        pose = torch.cat([position, quaternion], dim=-1)
        movej_target = self.low_level_action(joints)
        movel_target = self._solve_cartesian(pose, start)

        target = start.clone()
        target = torch.where(primitive_id == int(lib.MOVEJ), movej_target, target)
        target = torch.where(primitive_id == int(lib.MOVEL), movel_target, target)
        if gripper is not None:
            target = self._set_gripper(target, gripper)
            target = torch.where(
                primitive_id == int(lib.OPEN_GRIPPER),
                self._set_gripper(start, gripper),
                target,
            )
            target = torch.where(
                primitive_id == int(lib.CLOSE_GRIPPER),
                self._set_gripper(start, gripper),
                target,
            )
            hold_gripper = (
                (primitive_id == int(lib.WAIT))
                | (primitive_id == int(lib.MOVEJ))
                | (primitive_id == int(lib.MOVEL))
            )
            start = torch.where(hold_gripper, self._set_gripper(start, gripper), start)
        else:
            open_action = start.clone()
            open_action[..., -1] = self.open_gripper_ctrl
            close_action = start.clone()
            close_action[..., -1] = self.close_gripper_ctrl
            target = torch.where(
                primitive_id == int(lib.OPEN_GRIPPER), open_action, target
            )
            target = torch.where(
                primitive_id == int(lib.CLOSE_GRIPPER), close_action, target
            )
        return start, target, steps, settle_steps

    def _solve_cartesian(self, pose: torch.Tensor, start: torch.Tensor) -> torch.Tensor:
        if self.cartesian_solver is not None:
            return self.cartesian_solver(pose, start)
        env = self._find_parent_env_with("_cartesian_pose_to_joint_target")
        if env is None:
            return start
        return env._cartesian_pose_to_joint_target(pose, start)

    # ------------------------------------------------------------------ #
    # Specs
    # ------------------------------------------------------------------ #
    def transform_input_spec(self, input_spec: Composite) -> Composite:
        input_spec = input_spec.clone()
        batch_size = input_spec.shape
        device = input_spec.device
        dtype = self._spec_dtype(input_spec)
        joint_dim = self.action_dim - 1
        full_action_spec = Composite(shape=batch_size, device=device)
        full_action_spec[self.action_key] = Composite(
            mode=Categorical(
                n=len(RobotMacroActionMode),
                shape=(*batch_size, 1),
                dtype=torch.long,
                device=device,
            ),
            position=Unbounded(shape=(*batch_size, 3), dtype=dtype, device=device),
            quaternion=Unbounded(shape=(*batch_size, 4), dtype=dtype, device=device),
            joints=Unbounded(
                shape=(*batch_size, joint_dim), dtype=dtype, device=device
            ),
            # Three gripper choices (open / closed / keep). The transform reads
            # codes {0: open, 1: closed, other: keep}, so a Categorical over 3
            # values samples all three behaviours.
            gripper=Categorical(
                n=3,
                shape=(*batch_size, 1),
                dtype=torch.long,
                device=device,
            ),
            shape=batch_size,
            device=device,
        )
        input_spec["full_action_spec"] = full_action_spec
        return input_spec

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _ur_dtype(self, tensordict: TensorDictBase) -> torch.dtype:
        keys = tensordict.keys(True, True)
        for key in (self.robot_qpos_key, self.gripper_qpos_key):
            if key in keys:
                value = tensordict.get(key)
                if isinstance(value, torch.Tensor) and value.is_floating_point():
                    return value.dtype
        return torch.get_default_dtype()

    @staticmethod
    def _field(
        action: TensorDictBase,
        key: str,
        batch_shape: torch.Size,
        dtype: torch.dtype,
        device: torch.device,
        last_dim: int,
        default: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if key not in action.keys(True, True):
            if default is not None:
                return default
            return torch.zeros(batch_shape + (last_dim,), dtype=dtype, device=device)
        value = action.get(key).to(dtype=dtype, device=device)
        return value.reshape(batch_shape + (last_dim,))

    def _structured_gripper(
        self,
        action: TensorDictBase,
        start: torch.Tensor,
        batch_shape: torch.Size,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor | None:
        keys = action.keys(True, True)
        if "gripper" not in keys:
            return None
        gripper = (
            action.get("gripper")
            .to(dtype=torch.long, device=device)
            .reshape(batch_shape + (1,))
        )
        open_value = torch.full(
            batch_shape + (1,), self.open_gripper_ctrl, dtype=dtype, device=device
        )
        close_value = torch.full(
            batch_shape + (1,), self.close_gripper_ctrl, dtype=dtype, device=device
        )
        value = torch.where(
            gripper == RobotMacroAction.GRIPPER_OPEN,
            open_value,
            torch.where(
                gripper == RobotMacroAction.GRIPPER_CLOSED,
                close_value,
                start[..., -1:],
            ),
        )
        if "gripper_command" not in keys:
            return value
        gripper_command = (
            action.get("gripper_command")
            .to(dtype=dtype, device=device)
            .reshape(batch_shape + (1,))
        )
        return torch.where(torch.isfinite(gripper_command), gripper_command, value)

    def _env_home_qpos(
        self,
        batch_shape: torch.Size,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        env = self._find_parent_env_with("robot_home_qpos")
        if env is None:
            raise RuntimeError(
                "RobotMacroAction.RESET requires the parent environment to expose "
                "`robot_home_qpos`. Use RobotMacroAction.home(joints=...) when the "
                "home joint target is not environment-defined."
            )
        home_qpos = env.robot_home_qpos
        if callable(home_qpos):
            home_qpos = home_qpos()
        if home_qpos is None:
            raise RuntimeError(
                "RobotMacroAction.RESET could not resolve an environment home joint "
                "target. Use RobotMacroAction.home(joints=...) instead."
            )
        home_qpos = torch.as_tensor(home_qpos, dtype=dtype, device=device)
        joint_dim = self.action_dim - 1
        if home_qpos.shape[-1:] != torch.Size([joint_dim]):
            raise RuntimeError(
                "`robot_home_qpos` must have trailing dimension "
                f"{joint_dim}, got {home_qpos.shape}."
            )
        if home_qpos.ndim == 1:
            return home_qpos.expand(batch_shape + (joint_dim,)).clone()
        return home_qpos.reshape(batch_shape + (joint_dim,))
