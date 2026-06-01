# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""UR-style macro primitives used by MuJoCo manipulation examples."""

from __future__ import annotations

from enum import IntEnum
from typing import Any, ClassVar, Literal

import torch
from tensordict import TensorDictBase
from tensordict.tensorclass import TensorClass
from tensordict.utils import NestedKey, unravel_key
from torchrl.data.tensor_specs import Bounded, Composite, Unbounded
from torchrl.envs.transforms._primitive import (
    CartesianSolver,
    MacroPrimitiveTransform,
    MacroSolverName,
)

__all__ = [
    "RobotAction",
    "RobotActionMode",
    "URScriptPrimitive",
    "URScriptPrimitiveTransform",
]

GripperCommand = Literal["keep", "open", "closed"]


class URScriptPrimitive(IntEnum):
    r"""Integer ids for URScript-style robot primitives.

    The ids are specific to UR-style arm control with a binary gripper command;
    they are not the generic macro primitive vocabulary.

    Examples:
        >>> from torchrl.envs.custom.mujoco._ur_primitives import URScriptPrimitive
        >>> str(URScriptPrimitive.OPEN_GRIPPER)
        'open_gripper'
    """

    WAIT = 0
    MOVEJ = 1
    MOVEL = 2
    OPEN_GRIPPER = 3
    CLOSE_GRIPPER = 4

    def __str__(self) -> str:
        return self.name.lower()


class RobotActionMode(IntEnum):
    r"""Readable modes for :class:`RobotAction`.

    ``RobotActionMode`` mirrors the URScript primitive set and adds ``RESET``.
    The reset mode requires a parent environment exposing ``robot_home_qpos``.

    Examples:
        >>> from torchrl.envs.custom.mujoco._ur_primitives import RobotActionMode
        >>> RobotActionMode.REACH_POSE.name
        'REACH_POSE'
    """

    WAIT = int(URScriptPrimitive.WAIT)
    REACH_JOINTS = int(URScriptPrimitive.MOVEJ)
    REACH_POSE = int(URScriptPrimitive.MOVEL)
    OPEN_GRIPPER = int(URScriptPrimitive.OPEN_GRIPPER)
    CLOSE_GRIPPER = int(URScriptPrimitive.CLOSE_GRIPPER)
    RESET = len(URScriptPrimitive)


class _RobotActionReset:
    def __repr__(self) -> str:
        return "RobotAction.RESET"


def _unwrap_robot_action(action: Any) -> Any:
    data = getattr(action, "data", None)
    if isinstance(data, _RobotActionReset):
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
        return RobotAction.GRIPPER_KEEP
    if gripper == "open":
        return RobotAction.GRIPPER_OPEN
    if gripper == "closed":
        return RobotAction.GRIPPER_CLOSED
    raise ValueError(
        "gripper must be one of 'keep', 'open' or 'closed', " f"got {gripper!r}."
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


class RobotAction(TensorClass["nocast"]):
    r"""Human-writable UR-style macro action.

    ``RobotAction`` is intentionally UR/gripper-specific. It is used by the
    cube-to-bowl tutorial to store a readable command under ``td["action"]``;
    :class:`URScriptPrimitiveTransform` expands it to low-level joint-position
    and gripper commands.

    Examples:
        >>> import torch
        >>> from torchrl.envs.custom.mujoco._ur_primitives import RobotAction
        >>> action = RobotAction.reach_joints(joints=torch.zeros(1, 6))
        >>> action.joints.shape
        torch.Size([1, 6])
    """

    mode: torch.Tensor
    position: torch.Tensor
    quaternion: torch.Tensor
    joints: torch.Tensor
    gripper: torch.Tensor
    gripper_command: torch.Tensor
    steps: torch.Tensor
    settle_steps: torch.Tensor

    GRIPPER_KEEP: ClassVar[int] = -1
    GRIPPER_OPEN: ClassVar[int] = 0
    GRIPPER_CLOSED: ClassVar[int] = 1
    RESET: ClassVar[_RobotActionReset]

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
    ) -> RobotAction:
        """Ask the end effector to reach a Cartesian pose."""
        position = _as_batch(position, 3)
        if quaternion is None:
            quaternion = _identity_quaternion_like(position)
        else:
            quaternion = _as_batch(quaternion, 4).to(
                dtype=position.dtype, device=position.device
            )
        return cls._make(
            RobotActionMode.REACH_POSE,
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
    ) -> RobotAction:
        """Ask the arm to reach a six-joint UR configuration."""
        return cls._make(
            RobotActionMode.REACH_JOINTS,
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
    ) -> RobotAction:
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
    ) -> RobotAction:
        """Ask the transform to resolve the environment's reset/home posture."""
        dtype = torch.get_default_dtype() if dtype is None else dtype
        device = torch.device("cpu") if device is None else device
        batch_size = _batch_size(batch_size)
        position = torch.zeros(batch_size + (3,), dtype=dtype, device=device)
        return cls._make(
            RobotActionMode.RESET,
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
    ) -> RobotAction:
        """Open the gripper while keeping the current arm state."""
        return cls._empty(
            RobotActionMode.OPEN_GRIPPER,
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
    ) -> RobotAction:
        """Close the gripper while keeping the current arm state."""
        return cls._empty(
            RobotActionMode.CLOSE_GRIPPER,
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
    ) -> RobotAction:
        """Hold the current arm target for a number of low-level steps."""
        return cls._empty(
            RobotActionMode.WAIT,
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
        mode: RobotActionMode,
        *,
        gripper: GripperCommand,
        steps: int,
        settle_steps: int,
        batch_size: torch.Size | tuple[int, ...] | None,
        dtype: torch.dtype | None,
        device: torch.device | None,
        gripper_command: float | torch.Tensor | None = None,
    ) -> RobotAction:
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
        mode: RobotActionMode,
        *,
        position: torch.Tensor | None = None,
        quaternion: torch.Tensor | None = None,
        joints: torch.Tensor | None = None,
        gripper: GripperCommand = "keep",
        gripper_command: float | torch.Tensor | None = None,
        steps: int = 16,
        settle_steps: int = 0,
    ) -> RobotAction:
        if steps <= 0:
            raise ValueError("steps must be strictly positive.")
        if settle_steps < 0:
            raise ValueError("settle_steps must be non-negative.")
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
            mode=torch.full(
                batch_size + (1,), int(mode), dtype=torch.long, device=device
            ),
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
            steps=torch.full(batch_size + (1,), steps, dtype=torch.long, device=device),
            settle_steps=torch.full(
                batch_size + (1,), settle_steps, dtype=torch.long, device=device
            ),
            batch_size=batch_size,
        )


RobotAction.RESET = _RobotActionReset()


class _URScriptPrimitiveLibrary:
    WAIT = URScriptPrimitive.WAIT
    MOVEJ = URScriptPrimitive.MOVEJ
    MOVEL = URScriptPrimitive.MOVEL
    OPEN_GRIPPER = URScriptPrimitive.OPEN_GRIPPER
    CLOSE_GRIPPER = URScriptPrimitive.CLOSE_GRIPPER
    NUM_PRIMITIVES = len(URScriptPrimitive)


class _JointPositionGripperAdapter:
    """UR-style adapter for six arm joints plus one gripper command."""

    def __init__(
        self,
        *,
        action_key: NestedKey = "action",
        primitive_id_key: NestedKey = "primitive_id",
        target_qpos_key: NestedKey = "target_qpos",
        target_pose_key: NestedKey = "target_pose",
        gripper_key: NestedKey = "gripper",
        robot_qpos_key: NestedKey = "robot_qpos",
        gripper_qpos_key: NestedKey = "gripper_qpos",
        action_dim: int = 7,
        open_gripper_ctrl: float = 0.0,
        close_gripper_ctrl: float = 255.0,
    ) -> None:
        if action_dim <= 0:
            raise ValueError("action_dim must be strictly positive.")
        self.action_key = action_key
        self.primitive_id_key = primitive_id_key
        self.target_qpos_key = target_qpos_key
        self.target_pose_key = target_pose_key
        self.gripper_key = gripper_key
        self.robot_qpos_key = robot_qpos_key
        self.gripper_qpos_key = gripper_qpos_key
        self.action_dim = int(action_dim)
        self.open_gripper_ctrl = float(open_gripper_ctrl)
        self.close_gripper_ctrl = float(close_gripper_ctrl)

    def primitive_id(self, tensordict: TensorDictBase) -> torch.Tensor:
        return tensordict.get(self.primitive_id_key).to(torch.long).squeeze(-1)

    def current_action(
        self,
        tensordict: TensorDictBase,
        batch_shape: torch.Size,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if self.action_key in tensordict.keys(True, True):
            action = tensordict.get(self.action_key)
            if isinstance(action, torch.Tensor):
                action = action.to(dtype=dtype, device=device)
                return action.reshape(batch_shape + (self.action_dim,))
        start = torch.zeros(
            batch_shape + (self.action_dim,), dtype=dtype, device=device
        )
        if self.robot_qpos_key in tensordict.keys(True, True):
            robot_qpos = tensordict.get(self.robot_qpos_key).to(
                dtype=dtype, device=device
            )
            n = min(robot_qpos.shape[-1], self.action_dim - 1)
            start[..., :n] = robot_qpos[..., :n]
        if self.gripper_qpos_key in tensordict.keys(True, True):
            gripper_qpos = tensordict.get(self.gripper_qpos_key).to(
                dtype=dtype, device=device
            )
            start[..., -1] = gripper_qpos[..., 0]
        return start

    def low_level_action(
        self,
        robot_qpos: torch.Tensor,
        gripper: float | torch.Tensor | None = None,
    ) -> torch.Tensor:
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

    def target_qpos(
        self,
        tensordict: TensorDictBase,
        start: torch.Tensor,
        batch_shape: torch.Size,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        return self._get_or_default(
            tensordict,
            self.target_qpos_key,
            start,
            batch_shape,
            device,
            dtype,
            self.action_dim,
        )

    def target_pose(
        self,
        tensordict: TensorDictBase,
        batch_shape: torch.Size,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        return self._get_or_default(
            tensordict,
            self.target_pose_key,
            torch.zeros(batch_shape + (7,), dtype=dtype, device=device),
            batch_shape,
            device,
            dtype,
            7,
        )

    def gripper(
        self,
        tensordict: TensorDictBase,
        batch_shape: torch.Size,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor | None:
        if self.gripper_key not in tensordict.keys(True, True):
            return None
        gripper = tensordict.get(self.gripper_key).to(dtype=dtype, device=device)
        return gripper.reshape(batch_shape + (1,))

    def set_gripper(self, action: torch.Tensor, gripper: torch.Tensor) -> torch.Tensor:
        out = action.clone()
        out[..., -1:] = gripper
        return out

    def open_action(
        self,
        start: torch.Tensor,
        gripper: torch.Tensor | None,
    ) -> torch.Tensor:
        if gripper is not None:
            return self.set_gripper(start, gripper)
        out = start.clone()
        out[..., -1] = self.open_gripper_ctrl
        return out

    def close_action(
        self,
        start: torch.Tensor,
        gripper: torch.Tensor | None,
    ) -> torch.Tensor:
        if gripper is not None:
            return self.set_gripper(start, gripper)
        out = start.clone()
        out[..., -1] = self.close_gripper_ctrl
        return out

    def primitive_target(
        self,
        primitive_id: torch.Tensor,
        start: torch.Tensor,
        movej_target: torch.Tensor,
        movel_target: torch.Tensor,
        tensordict: TensorDictBase,
        library: _URScriptPrimitiveLibrary,
    ) -> torch.Tensor:
        batch_shape = primitive_id.shape
        device = primitive_id.device
        dtype = start.dtype
        gripper = self.gripper(tensordict, batch_shape, device, dtype)
        target = start.clone()
        target = torch.where(
            (primitive_id == int(library.MOVEJ)).unsqueeze(-1), movej_target, target
        )
        target = torch.where(
            (primitive_id == int(library.MOVEL)).unsqueeze(-1), movel_target, target
        )
        if gripper is not None:
            target = self.set_gripper(target, gripper)
        target = torch.where(
            (primitive_id == int(library.OPEN_GRIPPER)).unsqueeze(-1),
            self.open_action(start, gripper),
            target,
        )
        target = torch.where(
            (primitive_id == int(library.CLOSE_GRIPPER)).unsqueeze(-1),
            self.close_action(start, gripper),
            target,
        )
        return target

    def sequence_start(
        self,
        primitive_id: torch.Tensor,
        start: torch.Tensor,
        tensordict: TensorDictBase,
        library: _URScriptPrimitiveLibrary,
    ) -> torch.Tensor:
        batch_shape = primitive_id.shape
        device = primitive_id.device
        dtype = start.dtype
        gripper = self.gripper(tensordict, batch_shape, device, dtype)
        if gripper is None:
            return start
        hold_gripper = (
            (primitive_id == int(library.WAIT))
            | (primitive_id == int(library.MOVEJ))
            | (primitive_id == int(library.MOVEL))
        ).unsqueeze(-1)
        return torch.where(hold_gripper, self.set_gripper(start, gripper), start)

    def action_dtype(self, tensordict: TensorDictBase) -> torch.dtype:
        for key in (self.target_qpos_key, self.robot_qpos_key, self.gripper_qpos_key):
            if key in tensordict.keys(True, True):
                return tensordict.get(key).dtype
        return torch.get_default_dtype()

    def transform_input_spec(
        self, input_spec: Composite, primitive_library: _URScriptPrimitiveLibrary
    ) -> Composite:
        input_spec = input_spec.clone()
        batch_size = input_spec.shape
        device = input_spec.device
        dtype = self._spec_dtype(input_spec)
        action_dim = self._spec_action_dim(input_spec)
        full_action_spec = Composite(shape=batch_size, device=device)
        full_action_spec.set(
            self.primitive_id_key,
            Bounded(
                low=0,
                high=int(primitive_library.NUM_PRIMITIVES) - 1,
                shape=(*batch_size, 1),
                dtype=torch.long,
                device=device,
            ),
        )
        full_action_spec.set(
            self.target_qpos_key,
            Unbounded(shape=(*batch_size, action_dim), dtype=dtype, device=device),
        )
        full_action_spec.set(
            self.target_pose_key,
            Unbounded(shape=(*batch_size, 7), dtype=dtype, device=device),
        )
        full_action_spec.set(
            self.gripper_key,
            Bounded(
                low=min(self.open_gripper_ctrl, self.close_gripper_ctrl),
                high=max(self.open_gripper_ctrl, self.close_gripper_ctrl),
                shape=(*batch_size, 1),
                dtype=dtype,
                device=device,
            ),
        )
        input_spec["full_action_spec"] = full_action_spec
        return input_spec

    def _spec_dtype(self, input_spec: Composite) -> torch.dtype:
        action_spec = input_spec.get("full_action_spec", None)
        key = unravel_key(self.action_key)
        if isinstance(action_spec, Composite) and key in action_spec.keys(True, True):
            return action_spec[key].dtype
        return torch.get_default_dtype()

    def _spec_action_dim(self, input_spec: Composite) -> int:
        action_spec = input_spec.get("full_action_spec", None)
        key = unravel_key(self.action_key)
        if isinstance(action_spec, Composite) and key in action_spec.keys(True, True):
            shape = action_spec[key].shape
            if shape:
                return int(shape[-1])
        return self.action_dim

    @staticmethod
    def _get_or_default(
        tensordict: TensorDictBase,
        key: NestedKey,
        default: torch.Tensor,
        batch_shape: torch.Size,
        device: torch.device,
        dtype: torch.dtype,
        last_dim: int,
    ) -> torch.Tensor:
        if key not in tensordict.keys(True, True):
            return default
        value = tensordict.get(key).to(dtype=dtype, device=device)
        return value.reshape(batch_shape + (last_dim,))


class URScriptPrimitiveTransform(MacroPrimitiveTransform):
    r"""URScript-style preset of :class:`MacroPrimitiveTransform`.

    This specialization is intentionally scoped to six-joint UR-style arms with
    a scalar gripper command. The generic macro transform remains free of those
    assumptions.

    Examples:
        >>> import torch
        >>> from tensordict import TensorDict
        >>> from torchrl.envs.custom.mujoco._ur_primitives import RobotAction
        >>> from torchrl.envs.custom.mujoco._ur_primitives import URScriptPrimitiveTransform
        >>> td = TensorDict({
        ...     "action": RobotAction.reach_joints(joints=torch.ones(1, 6)),
        ...     "robot_qpos": torch.zeros(1, 6),
        ...     "gripper_qpos": torch.zeros(1, 2),
        ... }, batch_size=[1])
        >>> URScriptPrimitiveTransform(macro_steps=2).inv(td)["action"].shape
        torch.Size([1, 2, 7])
    """

    WAIT = _URScriptPrimitiveLibrary.WAIT
    MOVEJ = _URScriptPrimitiveLibrary.MOVEJ
    MOVEL = _URScriptPrimitiveLibrary.MOVEL
    OPEN_GRIPPER = _URScriptPrimitiveLibrary.OPEN_GRIPPER
    CLOSE_GRIPPER = _URScriptPrimitiveLibrary.CLOSE_GRIPPER
    NUM_PRIMITIVES = _URScriptPrimitiveLibrary.NUM_PRIMITIVES

    def __init__(
        self,
        *,
        primitive_library: Literal["urscript"]
        | _URScriptPrimitiveLibrary
        | None = "urscript",
        adapter: Literal["joint_position_gripper"]
        | _JointPositionGripperAdapter
        | None = "joint_position_gripper",
        solver: MacroSolverName | CartesianSolver | Any | None = "mujoco_dls_ik",
        execute: bool = False,
        multi_action_dim: int = 1,
        stack_rewards: bool = True,
        stack_observations: bool = False,
        action_key: NestedKey = "action",
        primitive_id_key: NestedKey = "primitive_id",
        target_qpos_key: NestedKey = "target_qpos",
        target_pose_key: NestedKey = "target_pose",
        gripper_key: NestedKey = "gripper",
        robot_qpos_key: NestedKey = "robot_qpos",
        gripper_qpos_key: NestedKey = "gripper_qpos",
        macro_steps: int = 16,
        settle_steps: int = 0,
        action_dim: int = 7,
        cartesian_solver: CartesianSolver | None = None,
        open_gripper_ctrl: float = 0.0,
        close_gripper_ctrl: float = 255.0,
    ) -> None:
        if primitive_library is None or primitive_library == "urscript":
            primitive_library = _URScriptPrimitiveLibrary()
        if isinstance(primitive_library, str):
            raise ValueError(f"Unknown primitive_library: {primitive_library}")
        if adapter is None or adapter == "joint_position_gripper":
            adapter = _JointPositionGripperAdapter(
                action_key=action_key,
                primitive_id_key=primitive_id_key,
                target_qpos_key=target_qpos_key,
                target_pose_key=target_pose_key,
                gripper_key=gripper_key,
                robot_qpos_key=robot_qpos_key,
                gripper_qpos_key=gripper_qpos_key,
                action_dim=action_dim,
                open_gripper_ctrl=open_gripper_ctrl,
                close_gripper_ctrl=close_gripper_ctrl,
            )
        if isinstance(adapter, str):
            raise ValueError(f"Unknown adapter: {adapter}")
        super().__init__(
            primitive_library=primitive_library,
            adapter=adapter,
            solver=solver,
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
            action_dim=action_dim,
            cartesian_solver=cartesian_solver,
        )
        self.gripper_key = self.adapter.gripper_key
        self.robot_qpos_key = self.adapter.robot_qpos_key
        self.gripper_qpos_key = self.adapter.gripper_qpos_key
        self.open_gripper_ctrl = float(self.adapter.open_gripper_ctrl)
        self.close_gripper_ctrl = float(self.adapter.close_gripper_ctrl)

    def low_level_action(
        self,
        robot_qpos: torch.Tensor,
        gripper: float | torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Build a low-level joint-position + gripper action."""
        return self.adapter.low_level_action(robot_qpos, gripper)

    def make_primitive(
        self,
        tensordict: TensorDictBase,
        primitive_id: int | IntEnum | torch.Tensor,
        *,
        target_pose: torch.Tensor | None = None,
        target_qpos: torch.Tensor | None = None,
        gripper: float | torch.Tensor | None = None,
    ) -> TensorDictBase:
        out = super().make_primitive(
            tensordict,
            primitive_id,
            target_pose=target_pose,
            target_qpos=target_qpos,
        )
        if gripper is None:
            if self.gripper_key in out.keys(True, True):
                out.del_(self.gripper_key)
        else:
            out.set(
                self.gripper_key,
                self._expand_value(
                    gripper,
                    batch_shape=tensordict.batch_size,
                    last_dim=1,
                    dtype=self.adapter.action_dtype(tensordict),
                    device=self._primitive_device(tensordict, gripper),
                ),
            )
        return out

    def action_sequence(
        self,
        tensordict: TensorDictBase,
        primitive_id: int | IntEnum | torch.Tensor | None = None,
        *,
        target_pose: torch.Tensor | None = None,
        target_qpos: torch.Tensor | None = None,
        gripper: float | torch.Tensor | None = None,
    ) -> torch.Tensor:
        if primitive_id is not None:
            tensordict = self.make_primitive(
                tensordict,
                primitive_id,
                target_pose=target_pose,
                target_qpos=target_qpos,
                gripper=gripper,
            )
        elif target_pose is not None or target_qpos is not None or gripper is not None:
            raise ValueError(
                "target_pose, target_qpos and gripper can only be passed when "
                "primitive_id is provided."
            )
        return self.inv(tensordict).get(self.action_key)

    def _primitive_target(
        self,
        primitive_id: torch.Tensor,
        start: torch.Tensor,
        movej_target: torch.Tensor,
        movel_target: torch.Tensor,
        tensordict: TensorDictBase,
    ) -> torch.Tensor:
        return self.adapter.primitive_target(
            primitive_id,
            start,
            movej_target,
            movel_target,
            tensordict,
            self.primitive_library,
        )

    def _sequence_start(
        self,
        primitive_id: torch.Tensor,
        start: torch.Tensor,
        target: torch.Tensor,
        tensordict: TensorDictBase,
    ) -> torch.Tensor:
        del target
        return self.adapter.sequence_start(
            primitive_id, start, tensordict, self.primitive_library
        )

    def _has_structured_action(self, tensordict: TensorDictBase) -> bool:
        if self.action_key not in tensordict.keys(True, True):
            return False
        action = _unwrap_robot_action(tensordict.get(self.action_key))
        if isinstance(action, _RobotActionReset):
            return True
        if isinstance(action, torch.Tensor):
            return False
        if not hasattr(action, "keys") or not hasattr(action, "get"):
            return False
        return "mode" in action.keys(True, True)

    def _unpack_structured_action(
        self, tensordict: TensorDictBase
    ) -> tuple[TensorDictBase, int, int]:
        action = _unwrap_robot_action(tensordict.get(self.action_key))
        dtype = self.adapter.action_dtype(tensordict)
        if isinstance(action, _RobotActionReset):
            action = RobotAction.reset(
                batch_size=tensordict.batch_size,
                dtype=dtype,
                device=self._primitive_device(tensordict),
                steps=self.macro_steps,
                settle_steps=self.settle_steps,
            )
        mode = action.get("mode").to(torch.long)
        if mode.shape[-1:] != torch.Size([1]):
            mode = mode.unsqueeze(-1)
        batch_shape = mode.shape[:-1]
        device = mode.device
        start = self.adapter.current_action(tensordict, batch_shape, device, dtype)
        reset_mask = mode == int(RobotActionMode.RESET)
        primitive_id = torch.where(
            reset_mask,
            torch.full_like(mode, int(self.primitive_library.MOVEJ)),
            mode,
        )
        position = self._structured_action_field(
            action,
            "position",
            torch.zeros(batch_shape + (3,), dtype=dtype, device=device),
            batch_shape,
            dtype,
            device,
            3,
        )
        quaternion_default = torch.zeros(batch_shape + (4,), dtype=dtype, device=device)
        quaternion_default[..., 0] = 1.0
        quaternion = self._structured_action_field(
            action,
            "quaternion",
            quaternion_default,
            batch_shape,
            dtype,
            device,
            4,
        )
        joints = self._structured_action_field(
            action,
            "joints",
            start[..., : self.action_dim - 1],
            batch_shape,
            dtype,
            device,
            self.action_dim - 1,
        )
        if reset_mask.any():
            joints = torch.where(
                reset_mask,
                self._env_home_qpos(batch_shape, dtype, device),
                joints,
            )
        joints = torch.where(
            primitive_id == int(self.primitive_library.MOVEJ),
            joints,
            start[..., : self.action_dim - 1],
        )
        gripper = self._structured_gripper(action, start, batch_shape, dtype, device)
        out = tensordict.clone()
        out.set(self.primitive_id_key, primitive_id)
        out.set(self.target_pose_key, torch.cat([position, quaternion], dim=-1))
        out.set(self.target_qpos_key, self.adapter.low_level_action(joints, gripper))
        out.set(self.gripper_key, gripper)
        return (
            out,
            self._structured_action_int(action, "steps", self.macro_steps),
            self._structured_action_int(action, "settle_steps", self.settle_steps),
        )

    @staticmethod
    def _structured_action_field(
        action: Any,
        key: str,
        default: torch.Tensor,
        batch_shape: torch.Size,
        dtype: torch.dtype,
        device: torch.device,
        last_dim: int,
    ) -> torch.Tensor:
        if key not in action.keys(True, True):
            return default
        value = action.get(key).to(dtype=dtype, device=device)
        return value.reshape(batch_shape + (last_dim,))

    def _env_home_qpos(
        self,
        batch_shape: torch.Size,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        env = self._find_parent_env_with("robot_home_qpos")
        if env is None:
            raise RuntimeError(
                "RobotAction.RESET requires the parent environment to expose "
                "`robot_home_qpos`. Use RobotAction.home(joints=...) when the "
                "home joint target is not environment-defined."
            )
        home_qpos = env.robot_home_qpos
        if callable(home_qpos):
            home_qpos = home_qpos()
        if home_qpos is None:
            raise RuntimeError(
                "RobotAction.RESET could not resolve an environment home joint "
                "target. Use RobotAction.home(joints=...) instead."
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

    def _structured_gripper(
        self,
        action: Any,
        start: torch.Tensor,
        batch_shape: torch.Size,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        if "gripper" not in action.keys(True, True):
            return start[..., -1:]
        gripper = (
            action.get("gripper")
            .to(dtype=torch.long, device=device)
            .reshape(batch_shape + (1,))
        )
        open_value = torch.full(
            batch_shape + (1,),
            self.open_gripper_ctrl,
            dtype=dtype,
            device=device,
        )
        close_value = torch.full(
            batch_shape + (1,),
            self.close_gripper_ctrl,
            dtype=dtype,
            device=device,
        )
        value = torch.where(
            gripper == 0,
            open_value,
            torch.where(gripper == 1, close_value, start[..., -1:]),
        )
        if "gripper_command" not in action.keys(True, True):
            return value
        gripper_command = action.get("gripper_command").to(dtype=dtype, device=device)
        gripper_command = gripper_command.reshape(batch_shape + (1,))
        return torch.where(torch.isfinite(gripper_command), gripper_command, value)

    def _primitive_device(
        self,
        tensordict: TensorDictBase,
        *values: torch.Tensor | float | None,
    ) -> torch.device:
        for value in values:
            if isinstance(value, torch.Tensor):
                return value.device
        for key in (
            self.action_key,
            self.target_qpos_key,
            self.target_pose_key,
            self.robot_qpos_key,
            self.gripper_qpos_key,
        ):
            if key in tensordict.keys(True, True):
                device = getattr(tensordict.get(key), "device", None)
                if device is not None:
                    return device
        if tensordict.device is not None:
            return tensordict.device
        return torch.device("cpu")
