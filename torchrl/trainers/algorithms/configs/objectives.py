# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from torchrl.objectives.ppo import ClipPPOLoss, KLPENPPOLoss, PPOLoss
from torchrl.trainers.algorithms.configs.common import ConfigBase


@dataclass
class LossConfig(ConfigBase):
    """A class to configure a loss.

    Args:
        loss_type: The type of loss to use.
    """

    _partial_: bool = False

    def __post_init__(self) -> None:
        """Post-initialization hook for loss configurations."""
        pass


@dataclass
class PPOLossConfig(LossConfig):
    """Configuration for PPO loss."""

    actor_network: Any = None
    critic_network: Any = None
    loss_type: str = "clip"
    entropy_bonus: bool = True
    samples_mc_entropy: int = 1
    entropy_coeff: float | None = None
    log_explained_variance: bool = True
    critic_coeff: float = 0.25
    loss_critic_type: str = "smooth_l1"
    normalize_advantage: bool = True
    normalize_advantage_exclude_dims: tuple = ()
    gamma: float | None = None
    separate_losses: bool = False
    advantage_key: str | None = None
    value_target_key: str | None = None
    value_key: str | None = None
    functional: bool = True
    actor: Any = None
    critic: Any = None
    reduction: str | None = None
    clip_value: float | None = None
    device: Any = None
    _target_: str = "torchrl.trainers.algorithms.configs.objectives._make_ppo_loss"

    def __post_init__(self) -> None:
        """Post-initialization hook for PPO loss configurations."""
        super().__post_init__()


def _make_ppo_loss(*args, **kwargs) -> PPOLoss:
    loss_type = kwargs.pop("loss_type", "clip")
    if loss_type == "clip":
        return ClipPPOLoss(*args, **kwargs)
    elif loss_type == "kl":
        return KLPENPPOLoss(*args, **kwargs)
    elif loss_type == "ppo":
        return PPOLoss(*args, **kwargs)
    else:
        raise ValueError(f"Invalid loss type: {loss_type}")
