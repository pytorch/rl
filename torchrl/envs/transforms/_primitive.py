# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Action transforms for scripted robot-control primitives."""

from __future__ import annotations

from enum import IntEnum
from typing import Any, Callable, Literal

import torch
from tensordict import TensorDictBase
from tensordict.utils import NestedKey, unravel_key
from torchrl.data.tensor_specs import Bounded, Composite, Unbounded
from torchrl.envs.transforms._base import Transform

__all__ = [
    "MacroPrimitiveTransform",
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
            action = tensordict.get(self.action_key).to(dtype=dtype, device=device)
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

    def _inv_call(self, tensordict: TensorDictBase) -> TensorDictBase:
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
            1.0 / self.macro_steps,
            1.0,
            self.macro_steps,
            dtype=dtype,
            device=device,
        ).reshape((1,) * len(batch_shape) + (self.macro_steps, 1))
        sequence = start.unsqueeze(-2) + alpha * (target - start).unsqueeze(-2)
        if self.settle_steps:
            settle = target.unsqueeze(-2).expand(
                batch_shape + (self.settle_steps, self.action_dim)
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
        >>> from torchrl.envs.transforms import URScriptPrimitiveTransform
        >>> td = TensorDict({
        ...     "primitive_id": torch.tensor([[1]]),
        ...     "target_qpos": torch.ones(1, 7),
        ...     "robot_qpos": torch.zeros(1, 6),
        ...     "gripper_qpos": torch.zeros(1, 8),
        ... }, batch_size=[1])
        >>> transform = URScriptPrimitiveTransform(macro_steps=2)
        >>> out = transform.inv(td)
        >>> out["action"].shape
        torch.Size([1, 2, 7])
    """

    def __init__(
        self,
        *,
        primitive_library: PrimitiveLibraryName | Any | None = "urscript",
        adapter: MacroAdapterName | Any | None = "joint_position_gripper",
        solver: MacroSolverName | CartesianSolver | Any | None = "mujoco_dls_ik",
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
