# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Generic action transforms for macro-control primitives.

This module owns the robot-agnostic machinery shared by every macro transform:

* :class:`MacroPrimitive` -- the minimal primitive vocabulary (``WAIT`` and
  ``MOVE``);
* :class:`MacroAction` / :class:`TargetMacroAction` -- structured, human-writable
  action objects placed under the environment action key;
* :class:`MacroPrimitiveTransform` -- the inverse-action plumbing that turns one
  macro action into a fixed-length low-level action sequence and (optionally)
  executes it through :class:`~torchrl.envs.transforms.MultiAction`.

Domain specializations (a robot arm, a satellite, ...) subclass
:class:`MacroPrimitiveTransform` and override three small hooks --
:meth:`~MacroPrimitiveTransform._resolve`,
:meth:`~MacroPrimitiveTransform.current_action` and
:meth:`~MacroPrimitiveTransform.transform_input_spec` -- instead of plugging in
adapter/solver/library objects. See
:doc:`../reference/macro_primitives` for the design guide.
"""

from __future__ import annotations

from enum import IntEnum
from typing import Any

import torch
from tensordict import TensorDictBase
from tensordict.tensorclass import TensorClass
from tensordict.utils import NestedKey, unravel_key
from torchrl.data.tensor_specs import Categorical, Composite, Unbounded
from torchrl.envs.transforms._action import MultiAction
from torchrl.envs.transforms._base import Compose, Transform

__all__ = [
    "MacroPrimitive",
    "MacroAction",
    "TargetMacroAction",
    "MacroPrimitiveTransform",
]


class MacroPrimitive(IntEnum):
    r"""Generic primitive ids understood by :class:`MacroPrimitiveTransform`.

    The base vocabulary is intentionally tiny and robot-agnostic: either hold
    the current low-level action (``WAIT``) or interpolate toward a low-level
    action target (``MOVE``). Domain-specific transforms can extend this enum in
    their own modules (e.g. adding gripper or inverse-kinematics primitives).

    Examples:
        >>> from torchrl.envs.transforms import MacroPrimitive
        >>> int(MacroPrimitive.MOVE)
        1
    """

    WAIT = 0
    MOVE = 1

    def __str__(self) -> str:
        return self.name.lower()


def _ensure_batched(value: torch.Tensor) -> torch.Tensor:
    """Return ``value`` with at least one leading batch dimension."""
    if value.ndim == 0:
        raise ValueError("target must have a non-empty trailing feature dimension.")
    if value.ndim == 1:
        return value.unsqueeze(0)
    return value


class MacroAction(TensorClass["nocast"]):
    r"""Base structured macro action: a primitive id plus a duration.

    Every macro action stores a primitive ``mode`` and the number of low-level
    ``steps`` (and trailing ``settle_steps``) used to expand it. Domain actions
    subclass this base and add their own target fields; see
    :class:`TargetMacroAction` for the common single-target case and
    :class:`~torchrl.envs.RobotMacroAction` for a richer example.

    Examples:
        >>> import torch
        >>> from torchrl.envs.transforms import MacroAction
        >>> action = MacroAction(
        ...     mode=torch.zeros(1, 1, dtype=torch.long),
        ...     steps=torch.full((1, 1), 4, dtype=torch.long),
        ...     settle_steps=torch.zeros(1, 1, dtype=torch.long),
        ...     batch_size=[1],
        ... )
        >>> int(action.steps.reshape(-1)[0])
        4
    """

    mode: torch.Tensor
    steps: torch.Tensor
    settle_steps: torch.Tensor

    @classmethod
    def _duration_fields(
        cls,
        *,
        mode: int | IntEnum,
        steps: int,
        settle_steps: int,
        batch_size: torch.Size,
        device: torch.device,
    ) -> dict[str, torch.Tensor]:
        """Build the shared ``mode``/``steps``/``settle_steps`` field tensors."""
        if steps <= 0:
            raise ValueError("steps must be strictly positive.")
        if settle_steps < 0:
            raise ValueError("settle_steps must be non-negative.")
        return {
            "mode": torch.full(
                batch_size + (1,), int(mode), dtype=torch.long, device=device
            ),
            "steps": torch.full(
                batch_size + (1,), int(steps), dtype=torch.long, device=device
            ),
            "settle_steps": torch.full(
                batch_size + (1,), int(settle_steps), dtype=torch.long, device=device
            ),
        }


class TargetMacroAction(MacroAction):
    r"""Macro action carrying a single ``target`` interpreted by the transform.

    The ``target`` is whatever a :class:`MacroPrimitiveTransform` subclass knows
    how to map to a low-level action destination. For the generic transform the
    target lives directly in low-level action coordinates; for a domain preset it
    can be a semantic quantity such as a target attitude quaternion.

    Examples:
        >>> import torch
        >>> from torchrl.envs.transforms import TargetMacroAction
        >>> action = TargetMacroAction.move(torch.ones(1, 4), steps=2)
        >>> action.target.shape
        torch.Size([1, 4])
    """

    target: torch.Tensor

    @classmethod
    def move(
        cls,
        target: torch.Tensor,
        *,
        steps: int = 16,
        settle_steps: int = 0,
        mode: int | IntEnum = MacroPrimitive.MOVE,
    ) -> TargetMacroAction:
        """Interpolate toward ``target`` over ``steps`` low-level actions."""
        target = _ensure_batched(target)
        batch_size = target.shape[:-1]
        return cls(
            target=target,
            batch_size=batch_size,
            **cls._duration_fields(
                mode=mode,
                steps=steps,
                settle_steps=settle_steps,
                batch_size=batch_size,
                device=target.device,
            ),
        )

    @classmethod
    def wait(
        cls,
        *,
        action_dim: int,
        steps: int = 1,
        settle_steps: int = 0,
        batch_size: torch.Size | tuple[int, ...] | None = None,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ) -> TargetMacroAction:
        """Hold the current low-level action for ``steps`` simulator steps."""
        if action_dim <= 0:
            raise ValueError("action_dim must be strictly positive.")
        dtype = torch.get_default_dtype() if dtype is None else dtype
        device = torch.device("cpu") if device is None else device
        batch_size = torch.Size([1]) if batch_size is None else torch.Size(batch_size)
        target = torch.zeros(batch_size + (action_dim,), dtype=dtype, device=device)
        return cls.move(
            target,
            steps=steps,
            settle_steps=settle_steps,
            mode=MacroPrimitive.WAIT,
        )


class MacroPrimitiveTransform(Transform):
    r"""Expand a high-level macro action into a low-level action sequence.

    The base transform is deliberately agnostic to robots, grippers and MuJoCo
    models. Its inverse-action path reads one macro action from ``action_key``,
    resolves a ``(start, target)`` pair of low-level actions, linearly
    interpolates between them over ``macro_steps`` (plus ``settle_steps`` held
    repeats), and writes the resulting ``(..., T, action_dim)`` sequence back
    under ``action_key``. When ``execute=True`` the constructor returns
    ``Compose(MultiAction(...), self)`` so the sequence is executed by the parent
    environment in a single high-level step.

    The policy-facing action accepted under ``action_key`` may be:

    * a :class:`MacroAction` / :class:`TargetMacroAction` (or a plain
      :class:`~tensordict.TensorDict` with the same ``mode`` / ``target`` /
      ``steps`` / ``settle_steps`` schema); or
    * a raw tensor, treated as a direct low-level action target (``MOVE``).

    Domain specializations override three hooks rather than configuring adapter,
    solver and library objects:

    * :meth:`_resolve` -- map a macro action to ``(start, target, steps,
      settle_steps)`` low-level tensors;
    * :meth:`current_action` -- read the low-level action used as the
      interpolation start (defaults to zeros or a tensor already at
      ``action_key``);
    * :meth:`transform_input_spec` -- advertise the policy-facing action spec.

    Args:
        action_key: low-level action key consumed by the inner environment and
            also the key carrying the macro action on the way in.
        macro_steps: number of interpolated low-level actions per primitive.
        settle_steps: number of repeated final actions appended after each
            primitive.
        action_dim: low-level action dimension. Required when it cannot be
            inferred from specs or from the macro action target.
        execute: if ``True``, return ``Compose(MultiAction(...), transform)`` so
            emitted action sequences are executed by the parent environment.
        multi_action_dim: stack dimension consumed by ``MultiAction`` when
            ``execute=True``.
        stack_rewards: whether ``MultiAction`` returns each low-level reward.
        stack_observations: whether ``MultiAction`` returns each low-level
            observation.

    Examples:
        >>> import torch
        >>> from tensordict import TensorDict
        >>> from torchrl.envs.transforms import MacroPrimitiveTransform
        >>> td = TensorDict({"action": torch.ones(1, 3)}, batch_size=[1])
        >>> transform = MacroPrimitiveTransform(macro_steps=2, action_dim=3)
        >>> transform.inv(td)["action"].shape
        torch.Size([1, 2, 3])
    """

    primitive_enum = MacroPrimitive

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
        action_key: NestedKey = "action",
        macro_steps: int = 16,
        settle_steps: int = 0,
        action_dim: int | None = None,
        execute: bool = False,
        multi_action_dim: int = 1,
        stack_rewards: bool = True,
        stack_observations: bool = False,
    ) -> None:
        del execute, multi_action_dim, stack_rewards, stack_observations
        super().__init__(in_keys_inv=[], out_keys_inv=[])
        if macro_steps <= 0:
            raise ValueError("macro_steps must be strictly positive.")
        if settle_steps < 0:
            raise ValueError("settle_steps must be non-negative.")
        if action_dim is not None and action_dim <= 0:
            raise ValueError("action_dim must be strictly positive.")
        self.action_key = unravel_key(action_key)
        self.macro_steps = int(macro_steps)
        self.settle_steps = int(settle_steps)
        self.action_dim = int(action_dim) if action_dim is not None else None

    # ------------------------------------------------------------------ #
    # Inverse-action path
    # ------------------------------------------------------------------ #
    def _inv_call(self, tensordict: TensorDictBase) -> TensorDictBase:
        action = tensordict.get(self.action_key, default=None)
        start, target, steps, settle_steps = self._resolve(tensordict, action)
        sequence = self._interpolate_sequence(start, target, steps, settle_steps)
        return tensordict.set(self.action_key, sequence)

    def _resolve(
        self, tensordict: TensorDictBase, action: Any
    ) -> tuple[torch.Tensor, torch.Tensor, int, int]:
        """Map ``action`` to ``(start, target, steps, settle_steps)``.

        The default implementation handles a :class:`MacroAction` (or a plain
        TensorDict with the same schema) and a raw low-level action tensor.
        Subclasses override this hook to interpret domain-specific targets.
        """
        if action is None:
            raise RuntimeError(
                f"{type(self).__name__} found no action under "
                f"{self.action_key!r} to expand."
            )
        batch_shape = tensordict.batch_size
        device = self._device(tensordict, action)
        dtype = self._action_dtype(tensordict, action)

        if isinstance(action, torch.Tensor):
            action_dim = action.shape[-1]
            target = action.to(dtype=dtype, device=device).reshape(
                batch_shape + (action_dim,)
            )
            start = torch.zeros(batch_shape + (action_dim,), dtype=dtype, device=device)
            return start, target, self.macro_steps, self.settle_steps

        if isinstance(action, (TensorDictBase, MacroAction)):
            keys = action.keys(True, True)
            if "target" not in keys:
                raise RuntimeError(
                    f"{type(self).__name__} expected a macro action with a "
                    f"'target' field under {self.action_key!r}; got keys "
                    f"{tuple(keys)}."
                )
            target = action.get("target").to(dtype=dtype, device=device)
            action_dim = target.shape[-1]
            target = target.reshape(batch_shape + (action_dim,))
            start = self.current_action(
                tensordict, batch_shape, device, dtype, action_dim
            )
            if "mode" in keys:
                mode = action.get("mode").to(torch.long).reshape(batch_shape + (1,))
                target = torch.where(mode == int(MacroPrimitive.WAIT), start, target)
            steps = self._field_int(action, "steps", self.macro_steps)
            settle_steps = self._field_int(action, "settle_steps", self.settle_steps)
            return start, target, steps, settle_steps

        raise TypeError(
            f"{type(self).__name__} cannot expand an action of type "
            f"{type(action).__name__}; pass a MacroAction, a TensorDict or a "
            "low-level action tensor."
        )

    def current_action(
        self,
        tensordict: TensorDictBase,
        batch_shape: torch.Size,
        device: torch.device,
        dtype: torch.dtype,
        action_dim: int,
    ) -> torch.Tensor:
        """Return the low-level action used as the interpolation start.

        The base implementation starts every macro from the zero action: in the
        inverse path ``action_key`` carries the incoming macro action (the
        *target*), so it must not be read back here as the start. Subclasses that
        can read the controlled state from observations (e.g. joint positions)
        override this hook.
        """
        dim = self.action_dim if self.action_dim is not None else action_dim
        return torch.zeros(batch_shape + (dim,), dtype=dtype, device=device)

    # ------------------------------------------------------------------ #
    # Convenience constructors (inspection / scripting helpers)
    # ------------------------------------------------------------------ #
    def make_primitive(
        self,
        tensordict: TensorDictBase,
        mode: int | IntEnum = MacroPrimitive.MOVE,
        *,
        target: torch.Tensor | None = None,
        target_qpos: torch.Tensor | None = None,
        steps: int | None = None,
        settle_steps: int | None = None,
    ) -> TensorDictBase:
        """Return a copy of ``tensordict`` carrying one macro action.

        This is a small scripting helper: it builds a
        :class:`TargetMacroAction` and stores it under ``action_key`` so the
        result can be passed to :meth:`action_sequence` or executed.
        """
        target = target if target is not None else target_qpos
        if target is None:
            raise ValueError("make_primitive requires a target tensor.")
        steps = self.macro_steps if steps is None else steps
        settle_steps = self.settle_steps if settle_steps is None else settle_steps
        action = TargetMacroAction.move(
            target, steps=steps, settle_steps=settle_steps, mode=mode
        )
        out = tensordict.copy()
        out.set(self.action_key, action)
        return out

    def action_sequence(
        self,
        tensordict: TensorDictBase,
        mode: int | IntEnum | None = None,
        *,
        target: torch.Tensor | None = None,
        target_qpos: torch.Tensor | None = None,
        steps: int | None = None,
        settle_steps: int | None = None,
    ) -> torch.Tensor:
        """Expand a macro action into its low-level sequence without executing.

        When ``mode``/``target`` are given, a primitive is built first;
        otherwise ``tensordict`` is expected to already carry a macro action
        under ``action_key``.
        """
        if mode is not None or target is not None or target_qpos is not None:
            tensordict = self.make_primitive(
                tensordict,
                MacroPrimitive.MOVE if mode is None else mode,
                target=target,
                target_qpos=target_qpos,
                steps=steps,
                settle_steps=settle_steps,
            )
        return self.inv(tensordict).get(self.action_key)

    @staticmethod
    def _interpolate_sequence(
        start: torch.Tensor,
        target: torch.Tensor,
        macro_steps: int,
        settle_steps: int,
    ) -> torch.Tensor:
        """Linearly interpolate ``start`` -> ``target`` then hold for settle."""
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

    # ------------------------------------------------------------------ #
    # Specs
    # ------------------------------------------------------------------ #
    def transform_input_spec(self, input_spec: Composite) -> Composite:
        input_spec = input_spec.clone()
        batch_size = input_spec.shape
        device = input_spec.device
        dtype = self._spec_dtype(input_spec)
        action_dim = self._spec_action_dim(input_spec)
        if action_dim is None:
            raise RuntimeError(
                f"{type(self).__name__} needs action_dim to transform input "
                "specs. Pass action_dim=... at construction time."
            )
        full_action_spec = Composite(shape=batch_size, device=device)
        full_action_spec[self.action_key] = self._macro_action_spec(
            batch_size, device, dtype, action_dim
        )
        input_spec["full_action_spec"] = full_action_spec
        return input_spec

    def _macro_action_spec(
        self,
        batch_size: torch.Size,
        device: torch.device,
        dtype: torch.dtype,
        action_dim: int,
    ) -> Composite:
        """Build the policy-facing macro action spec (``mode`` + ``target``)."""
        return Composite(
            mode=Categorical(
                n=len(self.primitive_enum),
                shape=(*batch_size, 1),
                dtype=torch.long,
                device=device,
            ),
            target=Unbounded(
                shape=(*batch_size, action_dim), dtype=dtype, device=device
            ),
            shape=batch_size,
            device=device,
        )

    def _spec_dtype(self, input_spec: Composite) -> torch.dtype:
        action_spec = input_spec.get("full_action_spec", None)
        if isinstance(action_spec, Composite) and self.action_key in action_spec.keys(
            True, True
        ):
            leaf = action_spec[self.action_key]
            dtype = getattr(leaf, "dtype", None)
            if dtype is not None:
                return dtype
        return torch.get_default_dtype()

    def _spec_action_dim(self, input_spec: Composite) -> int | None:
        if self.action_dim is not None:
            return self.action_dim
        action_spec = input_spec.get("full_action_spec", None)
        if isinstance(action_spec, Composite) and self.action_key in action_spec.keys(
            True, True
        ):
            shape = action_spec[self.action_key].shape
            if shape:
                return int(shape[-1])
        return None

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _find_parent_env_with(self, attr: str) -> Any | None:
        """Walk the parent chain looking for an env exposing ``attr``."""
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

    def _device(self, tensordict: TensorDictBase, action: Any) -> torch.device:
        device = getattr(action, "device", None)
        if device is not None:
            return device
        if tensordict.device is not None:
            return tensordict.device
        return torch.device("cpu")

    def _action_dtype(self, tensordict: TensorDictBase, action: Any) -> torch.dtype:
        target = None
        if isinstance(action, torch.Tensor):
            target = action
        elif isinstance(action, (TensorDictBase, MacroAction)) and "target" in (
            action.keys(True, True)
        ):
            target = action.get("target")
        if isinstance(target, torch.Tensor) and target.is_floating_point():
            return target.dtype
        return torch.get_default_dtype()

    @staticmethod
    def _field_int(action: TensorDictBase, key: str, default: int) -> int:
        if key not in action.keys(True, True):
            return int(default)
        value = int(action.get(key).reshape(-1)[0].item())
        if value < 0:
            raise ValueError(f"{key} must be non-negative.")
        if key == "steps" and value <= 0:
            raise ValueError("steps must be strictly positive.")
        return value

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(macro_steps={self.macro_steps}, "
            f"settle_steps={self.settle_steps}, action_key={self.action_key!r})"
        )
