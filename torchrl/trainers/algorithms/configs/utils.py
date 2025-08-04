# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass

from torchrl.trainers.algorithms.configs.common import ConfigBase


@dataclass
class AdamConfig(ConfigBase):
    """Configuration for Adam optimizer."""

    lr: float = 1e-3
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-4
    weight_decay: float = 0.0
    amsgrad: bool = False
    _target_: str = "torch.optim.Adam"
    _partial_: bool = True

    def __post_init__(self) -> None:
        """Post-initialization hook for Adam optimizer configurations."""
