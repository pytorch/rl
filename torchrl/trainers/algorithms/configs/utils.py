# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Any

from torchrl.trainers.algorithms.configs.common import ConfigBase
from dataclasses import dataclass

@dataclass
class AdamConfig(ConfigBase):
    """A class to configure an Adam optimizer.

    Args:
        lr: The learning rate.
        weight_decay: The weight decay.
    """

    params: Any = None
    lr: float = 1e-3
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-4
    weight_decay: float = 0.0
    amsgrad: bool = False

    _target_: str = "torch.optim.Adam"
    _partial_: bool = True

    @classmethod
    def default_config(cls, **kwargs) -> "AdamConfig":
        """Creates a default Adam optimizer configuration.
        
        Args:
            **kwargs: Override default values
            
        Returns:
            AdamConfig with default values, overridden by kwargs
        """
        defaults = {
            "params": None,  # Will be set when instantiating
            "lr": 3e-4,
            "betas": (0.9, 0.999),
            "eps": 1e-4,
            "weight_decay": 0.0,
            "amsgrad": False,
            "_partial_": True,
        }
        defaults.update(kwargs)
        return cls(**defaults)
