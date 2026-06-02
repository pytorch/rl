# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Humanoid macro actions used by MuJoCo examples."""

from __future__ import annotations

import torch
from tensordict.tensorclass import TensorClass
from torchrl.envs.transforms._primitive import MacroPrimitive

__all__ = ["HumanoidMacroAction"]


def _as_batch(value: torch.Tensor) -> torch.Tensor:
    if value.ndim == 0:
        raise ValueError("target must have a non-empty trailing action dimension.")
    if value.ndim == 1:
        return value.unsqueeze(0)
    return value


def _batch_size(batch_size: torch.Size | tuple[int, ...] | None) -> torch.Size:
    if batch_size is None:
        return torch.Size([1])
    return torch.Size(batch_size)


class HumanoidMacroAction(TensorClass["nocast"]):
    r"""Structured action for humanoid actuator-control macros.

    ``HumanoidMacroAction`` is intentionally small: the humanoid demo does not solve
    a Cartesian inverse-kinematics problem, it asks the base env to move toward
    a low-level actuator-control destination. A
    :class:`~torchrl.envs.transforms.MacroPrimitiveTransform` expands this
    target into a fixed-length action sequence.

    Examples:
        >>> import torch
        >>> from torchrl.envs import HumanoidMacroAction
        >>> action = HumanoidMacroAction.reach_control(torch.zeros(1, 4), steps=2)
        >>> action.target_qpos.shape
        torch.Size([1, 4])
    """

    primitive_id: torch.Tensor
    target_qpos: torch.Tensor
    steps: torch.Tensor
    settle_steps: torch.Tensor

    @classmethod
    def reach_control(
        cls,
        target: torch.Tensor,
        *,
        steps: int = 16,
        settle_steps: int = 0,
    ) -> HumanoidMacroAction:
        """Ask the transform to interpolate to a low-level control target."""
        return cls._make(
            primitive_id=MacroPrimitive.MOVEJ,
            target_qpos=_as_batch(target),
            steps=steps,
            settle_steps=settle_steps,
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
    ) -> HumanoidMacroAction:
        """Hold the current low-level action for a number of simulator steps."""
        if action_dim <= 0:
            raise ValueError("action_dim must be strictly positive.")
        dtype = torch.get_default_dtype() if dtype is None else dtype
        device = torch.device("cpu") if device is None else device
        batch_size = _batch_size(batch_size)
        target = torch.zeros(batch_size + (action_dim,), dtype=dtype, device=device)
        return cls._make(
            primitive_id=MacroPrimitive.WAIT,
            target_qpos=target,
            steps=steps,
            settle_steps=settle_steps,
        )

    @classmethod
    def _make(
        cls,
        *,
        primitive_id: MacroPrimitive,
        target_qpos: torch.Tensor,
        steps: int,
        settle_steps: int,
    ) -> HumanoidMacroAction:
        if target_qpos.ndim == 0 or target_qpos.shape[-1] <= 0:
            raise ValueError("target must have a non-empty trailing action dimension.")
        if steps <= 0:
            raise ValueError("steps must be strictly positive.")
        if settle_steps < 0:
            raise ValueError("settle_steps must be non-negative.")
        batch_size = target_qpos.shape[:-1]
        return cls(
            primitive_id=torch.full(
                batch_size + (1,),
                int(primitive_id),
                dtype=torch.long,
                device=target_qpos.device,
            ),
            target_qpos=target_qpos,
            steps=torch.full(
                batch_size + (1,),
                steps,
                dtype=torch.long,
                device=target_qpos.device,
            ),
            settle_steps=torch.full(
                batch_size + (1,),
                settle_steps,
                dtype=torch.long,
                device=target_qpos.device,
            ),
            batch_size=batch_size,
        )
