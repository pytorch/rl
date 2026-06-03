# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Humanoid macro actions used by MuJoCo examples."""

from __future__ import annotations

import torch
from torchrl.envs.transforms._primitive import TargetMacroAction

__all__ = ["HumanoidMacroAction"]


class HumanoidMacroAction(TargetMacroAction):
    r"""Structured action for humanoid actuator-control macros.

    ``HumanoidMacroAction`` is the thinnest possible specialization of
    :class:`~torchrl.envs.transforms.TargetMacroAction`: the humanoid demo does
    not solve a Cartesian inverse-kinematics problem, it asks the base env to
    interpolate toward a low-level actuator-control destination. A plain
    :class:`~torchrl.envs.transforms.MacroPrimitiveTransform` expands the target
    into a fixed-length action sequence; the named factory below only documents
    the intent and validates the target shape.

    Examples:
        >>> import torch
        >>> from torchrl.envs import HumanoidMacroAction
        >>> action = HumanoidMacroAction.reach_control(torch.zeros(1, 4), steps=2)
        >>> action.target.shape
        torch.Size([1, 4])
    """

    @classmethod
    def reach_control(
        cls,
        target: torch.Tensor,
        *,
        steps: int = 16,
        settle_steps: int = 0,
    ) -> HumanoidMacroAction:
        """Interpolate toward a low-level actuator-control target."""
        return cls.move(target, steps=steps, settle_steps=settle_steps)
