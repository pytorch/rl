# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass

from typing import Any

from torchrl.trainers.algorithms.configs.common import ConfigBase


@dataclass
class AdamConfig(ConfigBase):
    """A class to configure an Adam optimizer."""

    params: Any = None
    lr: float = 3e-4
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-4
    weight_decay: float = 0.0
    amsgrad: bool = False
    _target_: str = "torch.optim.Adam"
    _partial_: bool = True
