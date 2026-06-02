# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Generic action transforms for macro-control primitives."""

from __future__ import annotations

from collections.abc import Callable

from enum import IntEnum
from typing import Any, Literal, Protocol, runtime_checkable

import torch
from tensordict import TensorDictBase
from tensordict.utils import NestedKey, unravel_key
from torchrl.data.tensor_specs import Bounded, Composite, Unbounded
from torchrl.envs.transforms._action import MultiAction
from torchrl.envs.transforms._base import Compose, Transform

__all__ = [
    "CartesianSolver",
    "MacroPrimitive",
    "MacroPrimitiveTransform",
]

PrimitiveLibraryName = Literal["basic"]
MacroAdapterName = Literal["tensordict"]
MacroSolverName = Literal["joint_interpolation", "mujoco_dls_ik"]
CartesianSolver = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


class MacroPrimitive(IntEnum):
    r"""Generic primitive ids understood by :class:`MacroPrimitiveTransform`.

    The base primitive set is intentionally small and robot-agnostic: wait,
    interpolate toward a low-level action target, or ask a solver to map a pose
    target to a low-level action target. Domain-specific libraries can extend
    this enum in their own modules.

    Examples:
        >>> from torchrl.envs.transforms import MacroPrimitive
        >>> int(MacroPrimitive.MOVEJ)
        1
    """

    WAIT = 0
    MOVEJ = 1
    MOVEL = 2

    def __str__(self) -> str:
        return self.name.lower()


@runtime_checkable
class PrimitiveLibrary(Protocol):
    """Protocol for primitive id containers used by the generic transform."""

    WAIT: int | IntEnum
    MOVEJ: int | IntEnum
    MOVEL: int | IntEnum


@runtime_checkable
class MacroPrimitiveAdapter(Protocol):
    """Protocol for env-specific low-level action adapters."""

    action_key: NestedKey
    primitive_id_key: NestedKey
    target_qpos_key: NestedKey
    target_pose_key: NestedKey
    action_dim: int | None

    def primitive_id(self, tensordict: TensorDictBase) -> torch.Tensor:
        """Read primitive ids from a TensorDict."""

    def current_action(
        self,
        tensordict: TensorDictBase,
        batch_shape: torch.Size,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Return the low-level action used as interpolation start."""

    def target_qpos(
        self,
        tensordict: TensorDictBase,
        start: torch.Tensor,
        batch_shape: torch.Size,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Return the low-level action target for ``MOVEJ``."""

    def target_pose(
        self,
        tensordict: TensorDictBase,
        batch_shape: torch.Size,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Return the pose target for ``MOVEL``."""

    def action_dtype(self, tensordict: TensorDictBase) -> torch.dtype:
        """Infer the dtype to use for low-level actions."""

    def transform_input_spec(
        self, input_spec: Composite, primitive_library: PrimitiveLibrary
    ) -> Composite:
        """Expose policy-facing macro-action specs."""


class _BasicPrimitiveLibrary:
    WAIT = MacroPrimitive.WAIT
    MOVEJ = MacroPrimitive.MOVEJ
    MOVEL = MacroPrimitive.MOVEL
    NUM_PRIMITIVES = len(MacroPrimitive)


class _TensorDictActionAdapter:
    """Default adapter for action-space macro interpolation.

    The adapter has no robot or gripper assumptions. It reads a primitive id and
    a low-level target action from configurable TensorDict keys, inferring the
    current action from ``action_key`` when present or from zeros otherwise.
    """

    def __init__(
        self,
        *,
        action_key: NestedKey = "action",
        primitive_id_key: NestedKey = "primitive_id",
        target_qpos_key: NestedKey = "target_qpos",
        target_pose_key: NestedKey = "target_pose",
        action_dim: int | None = None,
    ) -> None:
        if action_dim is not None and action_dim <= 0:
            raise ValueError("action_dim must be strictly positive.")
        self.action_key = action_key
        self.primitive_id_key = primitive_id_key
        self.target_qpos_key = target_qpos_key
        self.target_pose_key = target_pose_key
        self.action_dim = int(action_dim) if action_dim is not None else None

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
                return action.reshape(batch_shape + (action.shape[-1],))
        if self.target_qpos_key in tensordict.keys(True, True):
            target = tensordict.get(self.target_qpos_key)
            action_dim = target.shape[-1]
        elif self.action_dim is not None:
            action_dim = self.action_dim
        else:
            raise RuntimeError(
                "MacroPrimitiveTransform could not infer action_dim. Pass "
                "action_dim=... or provide an action/target tensor."
            )
        return torch.zeros(batch_shape + (action_dim,), dtype=dtype, device=device)

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
            start.shape[-1],
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

    def action_dtype(self, tensordict: TensorDictBase) -> torch.dtype:
        for key in (self.target_qpos_key, self.action_key):
            if key in tensordict.keys(True, True):
                value = tensordict.get(key)
                if isinstance(value, torch.Tensor):
                    return value.dtype
        return torch.get_default_dtype()

    def transform_input_spec(
        self, input_spec: Composite, primitive_library: PrimitiveLibrary
    ) -> Composite:
        input_spec = input_spec.clone()
        batch_size = input_spec.shape
        device = input_spec.device
        dtype = self._spec_dtype(input_spec)
        action_dim = self._spec_action_dim(input_spec)
        if action_dim is None:
            raise RuntimeError(
                "MacroPrimitiveTransform needs action_dim to transform input specs."
            )
        full_action_spec = Composite(shape=batch_size, device=device)
        full_action_spec.set(
            self.primitive_id_key,
            Bounded(
                low=0,
                high=_num_primitives(primitive_library) - 1,
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
        input_spec["full_action_spec"] = full_action_spec
        return input_spec

    def _spec_dtype(self, input_spec: Composite) -> torch.dtype:
        action_spec = input_spec.get("full_action_spec", None)
        key = unravel_key(self.action_key)
        if isinstance(action_spec, Composite) and key in action_spec.keys(True, True):
            return action_spec[key].dtype
        return torch.get_default_dtype()

    def _spec_action_dim(self, input_spec: Composite) -> int | None:
        if self.action_dim is not None:
            return self.action_dim
        action_spec = input_spec.get("full_action_spec", None)
        key = unravel_key(self.action_key)
        if isinstance(action_spec, Composite) and key in action_spec.keys(True, True):
            shape = action_spec[key].shape
            if shape:
                return int(shape[-1])
        return None

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
    """Solver that treats action targets as already solved low-level actions."""

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
    """Optional MuJoCo ``movel`` bridge backed by an env-provided IK method."""

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
    """Adapter for callables with signature ``(target_pose, start_action)``."""

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


def _num_primitives(primitive_library: PrimitiveLibrary) -> int:
    if hasattr(primitive_library, "NUM_PRIMITIVES"):
        return int(primitive_library.NUM_PRIMITIVES)
    if hasattr(primitive_library, "num_primitives"):
        return int(primitive_library.num_primitives)
    return len(MacroPrimitive)


def _resolve_primitive_library(
    primitive_library: PrimitiveLibraryName | PrimitiveLibrary | None,
) -> PrimitiveLibrary:
    if primitive_library is None or primitive_library == "basic":
        return _BasicPrimitiveLibrary()
    if isinstance(primitive_library, str):
        raise ValueError(f"Unknown primitive_library: {primitive_library}")
    return primitive_library


def _resolve_adapter(
    adapter: MacroAdapterName | MacroPrimitiveAdapter | None,
    *,
    action_key: NestedKey,
    primitive_id_key: NestedKey,
    target_qpos_key: NestedKey,
    target_pose_key: NestedKey,
    action_dim: int | None,
) -> MacroPrimitiveAdapter:
    if adapter is None or adapter == "tensordict":
        return _TensorDictActionAdapter(
            action_key=action_key,
            primitive_id_key=primitive_id_key,
            target_qpos_key=target_qpos_key,
            target_pose_key=target_pose_key,
            action_dim=action_dim,
        )
    if isinstance(adapter, str):
        raise ValueError(f"Unknown adapter: {adapter}")
    return adapter


def _resolve_solver(
    solver: MacroSolverName | CartesianSolver | object | None,
    *,
    cartesian_solver: CartesianSolver | None,
) -> object:
    if solver is None or solver == "joint_interpolation":
        if cartesian_solver is not None:
            raise ValueError(
                "cartesian_solver can only be used with solver='mujoco_dls_ik'."
            )
        return _JointInterpolationSolver()
    if solver == "mujoco_dls_ik":
        return _MujocoDampedLeastSquaresIK(cartesian_solver=cartesian_solver)
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
    r"""Expand high-level macro primitives into low-level action sequences.

    The base transform is deliberately agnostic to robots, grippers, and MuJoCo
    models. It handles TensorDict plumbing, fixed-length interpolation, optional
    execution through :class:`~torchrl.envs.transforms.MultiAction`, and delegates
    domain details to an action adapter, primitive library, and solver.

    Args:
        primitive_library: primitive id library. ``None`` and ``"basic"`` use
            :class:`MacroPrimitive`; custom objects may expose ``WAIT``,
            ``MOVEJ``, ``MOVEL`` and optionally ``NUM_PRIMITIVES``.
        adapter: env-specific adapter. ``None`` and ``"tensordict"`` use a
            robot-agnostic adapter that reads low-level action targets directly
            from TensorDict keys.
        solver: macro solver backend. ``None`` and ``"joint_interpolation"``
            interpolate to the provided target action. ``"mujoco_dls_ik"`` uses
            an explicit ``cartesian_solver`` or a parent env's
            ``_cartesian_pose_to_joint_target`` hook for ``MOVEL``.
        execute: if ``True``, return ``Compose(MultiAction(...), transform)`` so
            emitted action sequences are executed by the parent environment.
        action_key: low-level action key consumed by the inner environment.
        primitive_id_key: key containing primitive ids.
        target_qpos_key: key containing low-level action targets.
        target_pose_key: key containing pose targets for solvers.
        macro_steps: number of interpolated low-level actions per primitive.
        settle_steps: number of repeated final actions appended after each
            primitive.
        action_dim: low-level action dimension. Required when it cannot be
            inferred from specs or TensorDict values.
        cartesian_solver: optional callable mapping ``(target_pose,
            start_action)`` to a low-level action target.

    Examples:
        >>> import torch
        >>> from tensordict import TensorDict
        >>> from torchrl.envs.transforms import MacroPrimitiveTransform
        >>> td = TensorDict({
        ...     "primitive_id": torch.tensor([[1]]),
        ...     "target_qpos": torch.ones(1, 3),
        ... }, batch_size=[1])
        >>> transform = MacroPrimitiveTransform(macro_steps=2, action_dim=3)
        >>> transform.inv(td)["action"].shape
        torch.Size([1, 2, 3])
    """

    WAIT = _BasicPrimitiveLibrary.WAIT
    MOVEJ = _BasicPrimitiveLibrary.MOVEJ
    MOVEL = _BasicPrimitiveLibrary.MOVEL
    NUM_PRIMITIVES = _BasicPrimitiveLibrary.NUM_PRIMITIVES

    def __new__(
        cls,
        *args: Any,
        execute: bool = False,
        multi_action_dim: int = 1,
        stack_rewards: bool = True,
        stack_observations: bool = False,
        **kwargs: Any,
    ) -> MacroPrimitiveTransform | Compose:
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
        primitive_library: PrimitiveLibraryName | PrimitiveLibrary | None = None,
        adapter: MacroAdapterName | MacroPrimitiveAdapter | None = None,
        solver: MacroSolverName | CartesianSolver | object | None = None,
        execute: bool = False,
        multi_action_dim: int = 1,
        stack_rewards: bool = True,
        stack_observations: bool = False,
        action_key: NestedKey = "action",
        primitive_id_key: NestedKey = "primitive_id",
        target_qpos_key: NestedKey = "target_qpos",
        target_pose_key: NestedKey = "target_pose",
        macro_steps: int = 16,
        settle_steps: int = 0,
        action_dim: int | None = None,
        cartesian_solver: CartesianSolver | None = None,
    ) -> None:
        del execute, multi_action_dim, stack_rewards, stack_observations
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
            action_dim=action_dim,
        )
        self.solver = _resolve_solver(solver, cartesian_solver=cartesian_solver)
        self.action_key = self.adapter.action_key
        self.primitive_id_key = self.adapter.primitive_id_key
        self.target_qpos_key = self.adapter.target_qpos_key
        self.target_pose_key = self.adapter.target_pose_key
        self.macro_steps = int(macro_steps)
        self.settle_steps = int(settle_steps)
        self.action_dim = self.adapter.action_dim

    def make_primitive(
        self,
        tensordict: TensorDictBase,
        primitive_id: int | IntEnum | torch.Tensor,
        *,
        target_pose: torch.Tensor | None = None,
        target_qpos: torch.Tensor | None = None,
    ) -> TensorDictBase:
        """Return a cloned TensorDict containing one macro primitive action."""
        batch_shape = tensordict.batch_size
        device = self._primitive_device(tensordict, target_pose, target_qpos)
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
            target_qpos = target_qpos.reshape(batch_shape + (start.shape[-1],))
        out.set(self.target_qpos_key, target_qpos)
        if target_pose is None:
            target_pose = torch.zeros(batch_shape + (7,), dtype=dtype, device=device)
        else:
            target_pose = target_pose.to(dtype=dtype, device=device)
            target_pose = target_pose.reshape(batch_shape + (7,))
        out.set(self.target_pose_key, target_pose)
        return out

    def action_sequence(
        self,
        tensordict: TensorDictBase,
        primitive_id: int | IntEnum | torch.Tensor | None = None,
        *,
        target_pose: torch.Tensor | None = None,
        target_qpos: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Expand a primitive action and return its low-level sequence."""
        if primitive_id is not None:
            tensordict = self.make_primitive(
                tensordict,
                primitive_id,
                target_pose=target_pose,
                target_qpos=target_qpos,
            )
        elif target_pose is not None or target_qpos is not None:
            raise ValueError(
                "target_pose and target_qpos can only be passed when "
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
        movej_target = self._movej_target(target_qpos, start, tensordict)
        movel_target = self._movel_target(target_pose, start, target_qpos, tensordict)
        target = self._primitive_target(
            primitive_id, start, movej_target, movel_target, tensordict
        )
        sequence_start = self._sequence_start(primitive_id, start, target, tensordict)
        sequence = self._interpolate_sequence(
            sequence_start, target, macro_steps, settle_steps
        )
        return tensordict.set(self.action_key, sequence)

    def _primitive_target(
        self,
        primitive_id: torch.Tensor,
        start: torch.Tensor,
        movej_target: torch.Tensor,
        movel_target: torch.Tensor,
        tensordict: TensorDictBase,
    ) -> torch.Tensor:
        del tensordict
        library = self.primitive_library
        target = start.clone()
        target = torch.where(
            (primitive_id == int(library.MOVEJ)).unsqueeze(-1), movej_target, target
        )
        target = torch.where(
            (primitive_id == int(library.MOVEL)).unsqueeze(-1), movel_target, target
        )
        return target

    def _sequence_start(
        self,
        primitive_id: torch.Tensor,
        start: torch.Tensor,
        target: torch.Tensor,
        tensordict: TensorDictBase,
    ) -> torch.Tensor:
        del primitive_id, target, tensordict
        return start

    @staticmethod
    def _interpolate_sequence(
        start: torch.Tensor,
        target: torch.Tensor,
        macro_steps: int,
        settle_steps: int,
    ) -> torch.Tensor:
        batch_shape = start.shape[:-1]
        dtype = start.dtype
        device = start.device
        alpha = torch.linspace(
            1.0 / macro_steps,
            1.0,
            macro_steps,
            dtype=dtype,
            device=device,
        ).reshape((1,) * len(batch_shape) + (macro_steps, 1))
        sequence = start.unsqueeze(-2) + alpha * (target - start).unsqueeze(-2)
        if settle_steps:
            settle = target.unsqueeze(-2).expand(
                batch_shape + (settle_steps, target.shape[-1])
            )
            sequence = torch.cat([sequence, settle], dim=-2)
        return sequence

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
        action = tensordict.get(self.action_key)
        if isinstance(action, torch.Tensor):
            return False
        if not hasattr(action, "keys") or not hasattr(action, "get"):
            return False
        keys = action.keys(True, True)
        return (
            "primitive_id" in keys
            or ("mode" in keys and "target" in keys)
            or "target_qpos" in keys
            or "target_pose" in keys
        )

    def _unpack_structured_action(
        self, tensordict: TensorDictBase
    ) -> tuple[TensorDictBase, int, int]:
        action = tensordict.get(self.action_key)
        out = tensordict.clone()

        keys = action.keys(True, True)
        if "primitive_id" in keys:
            primitive_id = action.get("primitive_id").to(torch.long)
        else:
            primitive_id = action.get("mode").to(torch.long)
        if primitive_id.shape[-1:] != torch.Size([1]):
            primitive_id = primitive_id.unsqueeze(-1)
        out.set(self.primitive_id_key, primitive_id)

        if "target_qpos" in keys:
            out.set(self.target_qpos_key, action.get("target_qpos"))
        if "target_pose" in keys:
            out.set(self.target_pose_key, action.get("target_pose"))
        if "target" in keys:
            target = action.get("target")
            if (primitive_id == int(MacroPrimitive.MOVEL)).all():
                out.set(self.target_pose_key, target)
            elif (primitive_id == int(MacroPrimitive.MOVEJ)).all() or (
                primitive_id == int(MacroPrimitive.WAIT)
            ).all():
                out.set(self.target_qpos_key, target)
            else:
                raise RuntimeError(
                    "Batched structured macro actions must all use the same "
                    "target kind. Use TensorDict primitive keys directly for "
                    "mixed macro batches."
                )
        return (
            out,
            self._structured_action_int(action, "steps", self.macro_steps),
            self._structured_action_int(action, "settle_steps", self.settle_steps),
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
        for key in (self.action_key, self.target_qpos_key, self.target_pose_key):
            if key in tensordict.keys(True, True):
                value = tensordict.get(key)
                device = getattr(value, "device", None)
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
            f"settle_steps={self.settle_steps}, action_key={self.action_key!r}, "
            f"adapter={type(self.adapter).__name__}, "
            f"solver={type(self.solver).__name__})"
        )
