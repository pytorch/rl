# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Action transforms for scripted robot-control primitives."""

from __future__ import annotations

from enum import IntEnum
from typing import Any, Callable, ClassVar, Literal

import torch
from tensordict import TensorDictBase, tensorclass
from tensordict.utils import NestedKey, unravel_key
from torchrl.data.tensor_specs import Bounded, Composite, Unbounded
from torchrl.envs.transforms._action import MultiAction
from torchrl.envs.transforms._base import Compose
from torchrl.envs.transforms._base import Transform

__all__ = [
    "MacroPrimitiveTransform",
    "RobotAction",
    "RobotActionMode",
    "URScriptPrimitive",
    "URScriptPrimitiveTransform",
]

PrimitiveLibraryName = Literal["urscript"]
MacroAdapterName = Literal["joint_position_gripper"]
MacroSolverName = Literal["mujoco_dls_ik", "joint_interpolation"]
CartesianSolver = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


class URScriptPrimitive(IntEnum):
    r"""Integer ids for URScript-style robot primitives.

    ``URScriptPrimitive`` gives readable names to the integer ids consumed by
    :class:`~torchrl.envs.transforms.URScriptPrimitiveTransform`. It derives
    from :class:`enum.IntEnum`, so enum values can be written directly into
    integer tensors with ``int(primitive)`` while keeping descriptive names in
    policies and tutorials.

    Examples:
        >>> import torch
        >>> from torchrl.envs.transforms import URScriptPrimitive
        >>> primitive = URScriptPrimitive.MOVEL
        >>> primitive.name
        'MOVEL'
        >>> torch.tensor([[int(primitive)]], dtype=torch.long)
        tensor([[2]])
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

    ``RobotActionMode`` mirrors the URScript primitive set and adds ``RESET``,
    which asks the transform to resolve the environment's configured home
    posture.

    Examples:
        >>> from torchrl.envs.transforms import RobotActionMode
        >>> RobotActionMode.REACH_POSE.name
        'REACH_POSE'
    """

    WAIT = int(URScriptPrimitive.WAIT)
    REACH_JOINTS = int(URScriptPrimitive.MOVEJ)
    REACH_POSE = int(URScriptPrimitive.MOVEL)
    OPEN_GRIPPER = int(URScriptPrimitive.OPEN_GRIPPER)
    CLOSE_GRIPPER = int(URScriptPrimitive.CLOSE_GRIPPER)
    RESET = len(URScriptPrimitive)


GripperCommand = Literal["keep", "open", "closed"]


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
        "gripper must be one of 'keep', 'open' or 'closed', "
        f"got {gripper!r}."
    )


def _batch_size(batch_size: torch.Size | tuple[int, ...] | None) -> torch.Size:
    if batch_size is None:
        return torch.Size([1])
    return torch.Size(batch_size)


@tensorclass
class RobotAction:
    r"""A human-writable robot macro action.

    ``RobotAction`` instances are structured TensorClass values meant to be
    written directly under ``td["action"]`` before calling ``env.step(td)`` with
    :class:`URScriptPrimitiveTransform`.

    Examples:
        >>> import torch
        >>> from tensordict import TensorDict
        >>> from torchrl.envs.transforms import RobotAction
        >>> td = TensorDict({"robot_qpos": torch.zeros(1, 6)}, batch_size=[1])
        >>> td["action"] = RobotAction.reach_pose(
        ...     position=torch.tensor([[0.4, 0.0, 0.2]])
        ... )
        >>> td["action"].position.shape
        torch.Size([1, 3])
    """

    mode: torch.Tensor
    position: torch.Tensor
    quaternion: torch.Tensor
    joints: torch.Tensor
    gripper: torch.Tensor
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
            steps=steps,
            settle_steps=settle_steps,
        )

    @classmethod
    def reach_joints(
        cls,
        *,
        joints: torch.Tensor,
        gripper: GripperCommand = "keep",
        steps: int = 16,
        settle_steps: int = 0,
    ) -> RobotAction:
        """Ask the arm to reach a joint configuration."""
        return cls._make(
            RobotActionMode.REACH_JOINTS,
            joints=_as_batch(joints, 6),
            gripper=gripper,
            steps=steps,
            settle_steps=settle_steps,
        )

    @classmethod
    def home(
        cls,
        *,
        joints: torch.Tensor,
        gripper: GripperCommand = "open",
        steps: int = 16,
        settle_steps: int = 0,
    ) -> RobotAction:
        """Ask the arm to return to an explicit home joint configuration."""
        return cls.reach_joints(
            joints=joints,
            gripper=gripper,
            steps=steps,
            settle_steps=settle_steps,
        )

    @classmethod
    def reset(
        cls,
        *,
        gripper: GripperCommand = "open",
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
    ) -> RobotAction:
        dtype = torch.get_default_dtype() if dtype is None else dtype
        device = torch.device("cpu") if device is None else device
        batch_size = _batch_size(batch_size)
        return cls._make(
            mode,
            position=torch.zeros(batch_size + (3,), dtype=dtype, device=device),
            gripper=gripper,
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
        steps: int = 16,
        settle_steps: int = 0,
    ) -> RobotAction:
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
            steps=torch.full(
                batch_size + (1,), steps, dtype=torch.long, device=device
            ),
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

    def set_gripper(
        self,
        action: torch.Tensor,
        gripper: torch.Tensor,
    ) -> torch.Tensor:
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

    def action_dtype(self, tensordict: TensorDictBase) -> torch.dtype:
        for key in (self.target_qpos_key, self.robot_qpos_key, self.gripper_qpos_key):
            if key in tensordict.keys(True, True):
                return tensordict.get(key).dtype
        return torch.get_default_dtype()

    def transform_input_spec(
        self, input_spec: Composite, primitive_library: Any
    ) -> Composite:
        input_spec = input_spec.clone()
        batch_size = input_spec.shape
        device = input_spec.device
        dtype = self._spec_dtype(input_spec)
        action_dim = self._spec_action_dim(input_spec)
        num_primitives = _num_primitives(primitive_library)
        full_action_spec = Composite(shape=batch_size, device=device)
        full_action_spec.set(
            self.primitive_id_key,
            Bounded(
                low=0,
                high=num_primitives - 1,
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
                low=self.open_gripper_ctrl,
                high=self.close_gripper_ctrl,
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


class _JointInterpolationSolver:
    def movej(
        self,
        target_qpos: torch.Tensor,
        start: torch.Tensor,
        *,
        transform: MacroPrimitiveTransform,
        tensordict: TensorDictBase,
    ) -> torch.Tensor:
        del start, transform, tensordict
        return target_qpos

    def movel(
        self,
        target_pose: torch.Tensor,
        start: torch.Tensor,
        fallback: torch.Tensor,
        *,
        transform: MacroPrimitiveTransform,
        tensordict: TensorDictBase,
    ) -> torch.Tensor:
        del target_pose, start, transform, tensordict
        return fallback


class _MujocoDampedLeastSquaresIK(_JointInterpolationSolver):
    def __init__(self, cartesian_solver: CartesianSolver | None = None) -> None:
        self.cartesian_solver = cartesian_solver

    def movel(
        self,
        target_pose: torch.Tensor,
        start: torch.Tensor,
        fallback: torch.Tensor,
        *,
        transform: MacroPrimitiveTransform,
        tensordict: TensorDictBase,
    ) -> torch.Tensor:
        del tensordict
        if self.cartesian_solver is not None:
            return self.cartesian_solver(target_pose, start)
        env = transform._find_parent_env_with("_cartesian_pose_to_joint_target")
        if env is None:
            return fallback
        return env._cartesian_pose_to_joint_target(target_pose, start)


class _CallableMoveLSolver(_JointInterpolationSolver):
    def __init__(self, solver: CartesianSolver) -> None:
        self.solver = solver

    def movel(
        self,
        target_pose: torch.Tensor,
        start: torch.Tensor,
        fallback: torch.Tensor,
        *,
        transform: MacroPrimitiveTransform,
        tensordict: TensorDictBase,
    ) -> torch.Tensor:
        del fallback, transform, tensordict
        return self.solver(target_pose, start)


def _num_primitives(primitive_library: Any) -> int:
    if hasattr(primitive_library, "NUM_PRIMITIVES"):
        return int(primitive_library.NUM_PRIMITIVES)
    if hasattr(primitive_library, "num_primitives"):
        return int(primitive_library.num_primitives)
    return 5


def _resolve_primitive_library(
    primitive_library: PrimitiveLibraryName | Any | None,
) -> Any:
    if primitive_library is None or primitive_library == "urscript":
        return _URScriptPrimitiveLibrary()
    if isinstance(primitive_library, str):
        raise ValueError(f"Unknown primitive_library: {primitive_library}")
    return primitive_library


def _resolve_adapter(
    adapter: MacroAdapterName | Any | None,
    *,
    action_key: NestedKey,
    primitive_id_key: NestedKey,
    target_qpos_key: NestedKey,
    target_pose_key: NestedKey,
    gripper_key: NestedKey,
    robot_qpos_key: NestedKey,
    gripper_qpos_key: NestedKey,
    action_dim: int,
    open_gripper_ctrl: float,
    close_gripper_ctrl: float,
) -> Any:
    if adapter is None or adapter == "joint_position_gripper":
        return _JointPositionGripperAdapter(
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
    return adapter


def _resolve_solver(
    solver: MacroSolverName | CartesianSolver | Any | None,
    *,
    cartesian_solver: CartesianSolver | None,
) -> Any:
    if solver is None or solver == "mujoco_dls_ik":
        return _MujocoDampedLeastSquaresIK(cartesian_solver=cartesian_solver)
    if solver == "joint_interpolation":
        if cartesian_solver is not None:
            raise ValueError(
                "cartesian_solver can only be used with solver='mujoco_dls_ik'."
            )
        return _JointInterpolationSolver()
    if isinstance(solver, str):
        raise ValueError(f"Unknown solver: {solver}")
    if cartesian_solver is not None:
        raise ValueError(
            "cartesian_solver cannot be passed together with a custom solver."
        )
    if callable(solver) and not hasattr(solver, "movel"):
        return _CallableMoveLSolver(solver)
    return solver


class MacroPrimitiveTransform(Transform):
    r"""Expand high-level robot primitives into low-level action sequences.

    ``MacroPrimitiveTransform`` is a generic action-transform shell for scripted
    robot macros. It owns the TensorDict plumbing and delegates robot-specific
    details to three extension points: a primitive library, an action adapter,
    and a solver backend. By default these extension points implement a
    URScript-like 7D joint-position + gripper setup, so the transform can be used
    directly for simple joint-position robot scenes while still accepting
    custom objects for other robots and MuJoCo environments.

    Args:
        primitive_library: primitive id library. ``None`` and ``"urscript"`` use
            the default wait/movej/movel/open/close library. A custom object may
            expose ``WAIT``, ``MOVEJ``, ``MOVEL``, ``OPEN_GRIPPER``,
            ``CLOSE_GRIPPER`` and optionally ``NUM_PRIMITIVES``. The default
            ids are exposed as
            :class:`~torchrl.envs.transforms.URScriptPrimitive`.
        adapter: env-specific action adapter. ``None`` and
            ``"joint_position_gripper"`` use the default adapter where the first
            six action entries are robot joint-position targets and the last
            entry is gripper control. Custom adapters may implement the methods
            used by this class, such as ``current_action``, ``target_qpos``,
            ``target_pose``, ``gripper``, ``set_gripper``, ``open_action`` and
            ``close_action``.
        solver: macro solver backend. ``None`` and ``"mujoco_dls_ik"`` use the
            default backend that handles ``movej`` by joint interpolation and
            handles ``movel`` with either ``cartesian_solver`` or a parent env's
            ``_cartesian_pose_to_joint_target`` method. ``"joint_interpolation"``
            falls back to ``target_qpos`` for ``movel``. A callable is treated as
            a ``movel(target_pose, start_action)`` solver; custom solver objects
            may implement ``movej`` and ``movel`` methods.
        action_key: low-level action key read by the inner env. If this key is
            already present during inverse expansion, it is used as the current
            low-level command to start the interpolation.
        primitive_id_key: policy-facing primitive id key.
        target_qpos_key: policy-facing low-level target key for ``movej`` and
            fallback ``movel``.
        target_pose_key: policy-facing Cartesian target key for ``movel``.
        gripper_key: optional desired gripper command key. When present, it is
            used as the gripper command for every emitted primitive action; if
            absent, the open/close constants are used for gripper primitives.
        robot_qpos_key: observation key used as the current robot joint start by
            the default adapter.
        gripper_qpos_key: observation key used as the current gripper start by
            the default adapter.
        macro_steps: fixed number of low-level actions emitted per primitive.
        settle_steps: optional number of extra low-level actions that repeat the
            final target after the interpolation. This is useful for position
            actuators that need a few simulator steps to settle at the macro
            target.
        action_dim: low-level action dimension for the default adapter.
        cartesian_solver: optional callable passed to the default
            ``"mujoco_dls_ik"`` solver. It maps ``(target_pose, start_action)``
            to a low-level joint target for ``movel``.
        open_gripper_ctrl: default adapter open-gripper command.
        close_gripper_ctrl: default adapter close-gripper command.

    Examples:
        >>> import torch
        >>> from tensordict import TensorDict
        >>> from torchrl.envs.transforms import MacroPrimitiveTransform
        >>> td = TensorDict({
        ...     "primitive_id": torch.tensor([[1]]),
        ...     "target_qpos": torch.ones(1, 7),
        ...     "robot_qpos": torch.zeros(1, 6),
        ...     "gripper_qpos": torch.zeros(1, 8),
        ... }, batch_size=[1])
        >>> transform = MacroPrimitiveTransform(macro_steps=2)
        >>> out = transform.inv(td)
        >>> out["action"].shape
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
        primitive_library: PrimitiveLibraryName | Any | None = None,
        adapter: MacroAdapterName | Any | None = None,
        solver: MacroSolverName | CartesianSolver | Any | None = None,
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
        super().__init__(in_keys_inv=[], out_keys_inv=[])
        if macro_steps <= 0:
            raise ValueError("macro_steps must be strictly positive.")
        if settle_steps < 0:
            raise ValueError("settle_steps must be non-negative.")
        self.primitive_library = _resolve_primitive_library(primitive_library)
        self.adapter = _resolve_adapter(
            adapter,
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
        self.solver = _resolve_solver(solver, cartesian_solver=cartesian_solver)
        self.action_key = self.adapter.action_key
        self.primitive_id_key = self.adapter.primitive_id_key
        self.target_qpos_key = self.adapter.target_qpos_key
        self.target_pose_key = self.adapter.target_pose_key
        self.gripper_key = self.adapter.gripper_key
        self.robot_qpos_key = self.adapter.robot_qpos_key
        self.gripper_qpos_key = self.adapter.gripper_qpos_key
        self.macro_steps = int(macro_steps)
        self.settle_steps = int(settle_steps)
        self.action_dim = int(self.adapter.action_dim)
        self.open_gripper_ctrl = float(self.adapter.open_gripper_ctrl)
        self.close_gripper_ctrl = float(self.adapter.close_gripper_ctrl)

    def low_level_action(
        self,
        robot_qpos: torch.Tensor,
        gripper: float | torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Build a low-level joint-position action for the default adapter.

        Args:
            robot_qpos: robot joint targets. With the default adapter, the first
                ``action_dim - 1`` entries are copied into the low-level action.
            gripper: optional gripper command. If omitted, the transform's
                ``open_gripper_ctrl`` value is used.

        Returns:
            A tensor with trailing dimension ``action_dim``.

        Examples:
            >>> import torch
            >>> from torchrl.envs.transforms import URScriptPrimitiveTransform
            >>> transform = URScriptPrimitiveTransform(open_gripper_ctrl=0.0)
            >>> transform.low_level_action(torch.zeros(1, 6)).shape
            torch.Size([1, 7])
        """
        if not hasattr(self.adapter, "low_level_action"):
            raise NotImplementedError(
                f"{type(self.adapter).__name__} does not implement low_level_action."
            )
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
        """Return a TensorDict action for one macro primitive.

        This helper centralizes the TensorDict keys used by the transform. It is
        equivalent to cloning an observation TensorDict and filling
        ``primitive_id``, ``target_pose``, ``target_qpos`` and optionally
        ``gripper`` under the keys configured on the transform.

        Args:
            tensordict: observation/action context used for batch size, dtype,
                device and current joint state.
            primitive_id: integer or enum primitive id.
            target_pose: optional Cartesian target for ``movel``.
            target_qpos: optional low-level target for ``movej``. If omitted,
                the current action inferred from ``tensordict`` is used.
            gripper: optional gripper command override.

        Returns:
            A cloned TensorDict containing the primitive action.

        Examples:
            >>> import torch
            >>> from tensordict import TensorDict
            >>> from torchrl.envs.transforms import (
            ...     URScriptPrimitive,
            ...     URScriptPrimitiveTransform,
            ... )
            >>> transform = URScriptPrimitiveTransform(macro_steps=2)
            >>> obs = TensorDict({
            ...     "robot_qpos": torch.zeros(1, 6),
            ...     "gripper_qpos": torch.zeros(1, 2),
            ... }, batch_size=[1])
            >>> primitive = transform.make_primitive(
            ...     obs, URScriptPrimitive.OPEN_GRIPPER
            ... )
            >>> int(primitive["primitive_id"].item())
            3
        """
        batch_shape = tensordict.batch_size
        device = self._primitive_device(tensordict, target_pose, target_qpos, gripper)
        dtype = self.adapter.action_dtype(tensordict)
        start = self.adapter.current_action(tensordict, batch_shape, device, dtype)
        out = tensordict.clone()
        out.set(
            self.primitive_id_key,
            self._expand_value(
                primitive_id,
                batch_shape=batch_shape,
                last_dim=1,
                dtype=torch.long,
                device=device,
            ),
        )
        if target_qpos is None:
            target_qpos = start
        else:
            target_qpos = target_qpos.to(dtype=dtype, device=device)
            target_qpos = target_qpos.reshape(batch_shape + (self.action_dim,))
        out.set(self.target_qpos_key, target_qpos)
        if target_pose is None:
            target_pose = torch.zeros(batch_shape + (7,), dtype=dtype, device=device)
        else:
            target_pose = target_pose.to(dtype=dtype, device=device)
            target_pose = target_pose.reshape(batch_shape + (7,))
        out.set(self.target_pose_key, target_pose)
        if gripper is None:
            if self.gripper_key in out.keys(True, True):
                out.del_(self.gripper_key)
        else:
            out.set(
                self.gripper_key,
                self._expand_value(
                    gripper,
                    batch_shape=batch_shape,
                    last_dim=1,
                    dtype=dtype,
                    device=device,
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
        """Expand a primitive action and return its low-level action sequence.

        Args:
            tensordict: observation/action context.
            primitive_id: optional primitive id. If provided, the primitive
                TensorDict is first built with :meth:`make_primitive`; if
                omitted, ``tensordict`` is assumed to already contain the
                primitive keys.
            target_pose: optional Cartesian target passed to
                :meth:`make_primitive`.
            target_qpos: optional joint target passed to :meth:`make_primitive`.
            gripper: optional gripper command passed to :meth:`make_primitive`.

        Returns:
            The tensor stored at ``action_key`` after inverse expansion.

        Examples:
            >>> import torch
            >>> from tensordict import TensorDict
            >>> from torchrl.envs.transforms import (
            ...     URScriptPrimitive,
            ...     URScriptPrimitiveTransform,
            ... )
            >>> transform = URScriptPrimitiveTransform(macro_steps=2)
            >>> obs = TensorDict({
            ...     "robot_qpos": torch.zeros(1, 6),
            ...     "gripper_qpos": torch.zeros(1, 2),
            ... }, batch_size=[1])
            >>> sequence = transform.action_sequence(
            ...     obs, URScriptPrimitive.OPEN_GRIPPER
            ... )
            >>> sequence.shape
            torch.Size([1, 2, 7])
        """
        if primitive_id is not None:
            tensordict = self.make_primitive(
                tensordict,
                primitive_id,
                target_pose=target_pose,
                target_qpos=target_qpos,
                gripper=gripper,
            )
        elif (
            target_pose is not None
            or target_qpos is not None
            or gripper is not None
        ):
            raise ValueError(
                "target_pose, target_qpos and gripper can only be passed when "
                "primitive_id is provided."
            )
        return self.inv(tensordict).get(self.action_key)

    def _inv_call(self, tensordict: TensorDictBase) -> TensorDictBase:
        macro_steps = self.macro_steps
        settle_steps = self.settle_steps
        if self._has_structured_action(tensordict):
            tensordict, macro_steps, settle_steps = self._unpack_structured_action(
                tensordict
            )
        primitive_id = self.adapter.primitive_id(tensordict)
        batch_shape = primitive_id.shape
        device = primitive_id.device
        dtype = self.adapter.action_dtype(tensordict)
        start = self.adapter.current_action(tensordict, batch_shape, device, dtype)
        target_qpos = self.adapter.target_qpos(
            tensordict, start, batch_shape, device, dtype
        )
        target_pose = self.adapter.target_pose(tensordict, batch_shape, device, dtype)
        gripper = self.adapter.gripper(tensordict, batch_shape, device, dtype)
        movej_target = self._movej_target(target_qpos, start, tensordict)
        movel_target = self._movel_target(target_pose, start, target_qpos, tensordict)

        library = self.primitive_library
        target = start.clone()
        target = torch.where(
            (primitive_id == library.MOVEJ).unsqueeze(-1), movej_target, target
        )
        target = torch.where(
            (primitive_id == library.MOVEL).unsqueeze(-1), movel_target, target
        )
        if gripper is not None:
            target = self.adapter.set_gripper(target, gripper)

        open_action = self.adapter.open_action(start, gripper)
        close_action = self.adapter.close_action(start, gripper)
        target = torch.where(
            (primitive_id == library.OPEN_GRIPPER).unsqueeze(-1), open_action, target
        )
        target = torch.where(
            (primitive_id == library.CLOSE_GRIPPER).unsqueeze(-1), close_action, target
        )

        alpha = torch.linspace(
            1.0 / macro_steps,
            1.0,
            macro_steps,
            dtype=dtype,
            device=device,
        ).reshape((1,) * len(batch_shape) + (macro_steps, 1))
        sequence_start = start
        if gripper is not None:
            hold_gripper = (
                (primitive_id == library.WAIT)
                | (primitive_id == library.MOVEJ)
                | (primitive_id == library.MOVEL)
            ).unsqueeze(-1)
            sequence_start = torch.where(
                hold_gripper,
                self.adapter.set_gripper(start, gripper),
                start,
            )
        sequence = sequence_start.unsqueeze(-2) + alpha * (
            target - sequence_start
        ).unsqueeze(-2)
        if settle_steps:
            settle = target.unsqueeze(-2).expand(
                batch_shape + (settle_steps, self.action_dim)
            )
            sequence = torch.cat([sequence, settle], dim=-2)
        return tensordict.set(self.action_key, sequence)

    def transform_input_spec(self, input_spec: Composite) -> Composite:
        return self.adapter.transform_input_spec(input_spec, self.primitive_library)

    def _movej_target(
        self,
        target_qpos: torch.Tensor,
        start: torch.Tensor,
        tensordict: TensorDictBase,
    ) -> torch.Tensor:
        if hasattr(self.solver, "movej"):
            return self.solver.movej(
                target_qpos, start, transform=self, tensordict=tensordict
            )
        return target_qpos

    def _movel_target(
        self,
        target_pose: torch.Tensor,
        start: torch.Tensor,
        fallback: torch.Tensor,
        tensordict: TensorDictBase,
    ) -> torch.Tensor:
        if not hasattr(self.solver, "movel"):
            return fallback
        return self.solver.movel(
            target_pose, start, fallback, transform=self, tensordict=tensordict
        )

    def _find_parent_env_with(self, attr: str) -> Any | None:
        try:
            env = self.parent
        except AttributeError:
            return None
        seen: set[int] = set()
        while env is not None and id(env) not in seen:
            seen.add(id(env))
            if hasattr(env, attr):
                return env
            env = getattr(env, "base_env", None)
        return None

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
            primitive_id == self.primitive_library.MOVEJ,
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
        home_qpos = getattr(env, "robot_home_qpos")
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
        gripper = action.get("gripper").to(dtype=torch.long, device=device).reshape(
            batch_shape + (1,)
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
        return torch.where(
            gripper == 0,
            open_value,
            torch.where(gripper == 1, close_value, start[..., -1:]),
        )

    @staticmethod
    def _structured_action_int(action: Any, key: str, default: int) -> int:
        if key not in action.keys(True, True):
            return int(default)
        value = int(action.get(key).reshape(-1)[0].item())
        if value < 0:
            raise ValueError(f"{key} must be non-negative.")
        if key == "steps" and value <= 0:
            raise ValueError("steps must be strictly positive.")
        return value

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

    @staticmethod
    def _expand_value(
        value: int | IntEnum | float | torch.Tensor,
        *,
        batch_shape: torch.Size,
        last_dim: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        value = torch.as_tensor(value, dtype=dtype, device=device)
        shape = batch_shape + (last_dim,)
        if value.numel() == 1:
            return value.reshape(()).expand(shape).clone()
        return value.reshape(shape)

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(macro_steps={self.macro_steps}, "
            f"settle_steps={self.settle_steps}, "
            f"action_dim={self.action_dim}, action_key={self.action_key!r}, "
            f"solver={type(self.solver).__name__})"
        )


class URScriptPrimitiveTransform(MacroPrimitiveTransform):
    r"""URScript-style preset of :class:`MacroPrimitiveTransform`.

    This class forwards to :class:`MacroPrimitiveTransform` with
    ``primitive_library="urscript"``,
    ``adapter="joint_position_gripper"`` and ``solver="mujoco_dls_ik"`` by
    default.

    Args:
        primitive_library: primitive id library. Defaults to ``"urscript"``.
        adapter: action adapter. Defaults to ``"joint_position_gripper"``.
        solver: macro solver backend. Defaults to ``"mujoco_dls_ik"``.
        execute: if ``True``, return a transform composition that first expands
            the primitive action and then executes the emitted low-level action
            sequence with :class:`~torchrl.envs.transforms.MultiAction`.
        action_key: low-level action key read by the inner environment.
        primitive_id_key: key containing the primitive id.
        target_qpos_key: key containing the joint target for ``movej``.
        target_pose_key: key containing the Cartesian target for ``movel``.
        gripper_key: optional key containing the gripper command.
        robot_qpos_key: observation key for the current robot joint state.
        gripper_qpos_key: observation key for the current gripper state.
        macro_steps: fixed number of low-level actions emitted per primitive.
        settle_steps: optional number of extra repeated final actions emitted
            after the interpolation.
        action_dim: low-level action dimension.
        cartesian_solver: optional Cartesian solver callable.
        open_gripper_ctrl: command used by ``open_gripper``.
        close_gripper_ctrl: command used by ``close_gripper``.

    Examples:
        >>> import torch
        >>> from tensordict import TensorDict
        >>> from torchrl.envs.transforms import RobotAction, URScriptPrimitiveTransform
        >>> robot_action = RobotAction.reach_joints(
        ...     joints=torch.ones(1, 6), steps=2
        ... )
        >>> robot_action.mode.shape
        torch.Size([1, 1])
        >>> td = TensorDict({
        ...     "action": robot_action,
        ...     "robot_qpos": torch.zeros(1, 6),
        ...     "gripper_qpos": torch.zeros(1, 8),
        ... }, batch_size=[1])
        >>> transform = URScriptPrimitiveTransform(macro_steps=2)
        >>> out = transform.inv(td)
        >>> out["action"].shape
        torch.Size([1, 2, 7])
    """

    def __new__(
        cls,
        *args,
        execute: bool = False,
        multi_action_dim: int = 1,
        stack_rewards: bool = True,
        stack_observations: bool = False,
        **kwargs,
    ):
        if execute:
            primitive = cls(*args, execute=False, **kwargs)
            return Compose(
                MultiAction(
                    dim=multi_action_dim,
                    stack_rewards=stack_rewards,
                    stack_observations=stack_observations,
                ),
                primitive,
            )
        return super().__new__(cls)

    def __init__(
        self,
        *,
        primitive_library: PrimitiveLibraryName | Any | None = "urscript",
        adapter: MacroAdapterName | Any | None = "joint_position_gripper",
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
        del execute, multi_action_dim, stack_rewards, stack_observations
        super().__init__(
            primitive_library=primitive_library,
            adapter=adapter,
            solver=solver,
            action_key=action_key,
            primitive_id_key=primitive_id_key,
            target_qpos_key=target_qpos_key,
            target_pose_key=target_pose_key,
            gripper_key=gripper_key,
            robot_qpos_key=robot_qpos_key,
            gripper_qpos_key=gripper_qpos_key,
            macro_steps=macro_steps,
            settle_steps=settle_steps,
            action_dim=action_dim,
            cartesian_solver=cartesian_solver,
            open_gripper_ctrl=open_gripper_ctrl,
            close_gripper_ctrl=close_gripper_ctrl,
        )
