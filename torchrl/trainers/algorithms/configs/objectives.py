# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping

import torch

from torchrl.objectives.ppo import ClipPPOLoss, PPOLoss
from torchrl.trainers.algorithms.configs.common import ConfigBase
from torchrl.trainers.algorithms.configs.modules import ModelConfig


@dataclass
class PPOLossConfig(ConfigBase):
    actor_network_cfg: ModelConfig
    critic_network_cfg: ModelConfig

    ppo_cls: type[PPOLoss] = ClipPPOLoss
    entropy_bonus: bool = True
    samples_mc_entropy: int = 1
    entropy_coef: float | Mapping[str, float] = 0.01
    critic_coef: float | None = None
    loss_critic_type: str = "smooth_l1"
    normalize_advantage: bool = False
    normalize_advantage_exclude_dims: tuple[int, ...] = ()
    gamma: float | None = None
    separate_losses: bool = False
    advantage_key: str = None
    value_target_key: str = None
    value_key: str = None
    functional: bool = True
    reduction: str = None
    clip_value: float = None
    device: torch.device = None

    def make(self) -> PPOLoss:
        kwargs = asdict(self)
        del kwargs["ppo_cls"]
        del kwargs["actor_network_cfg"]
        del kwargs["critic_network_cfg"]
        return self.ppo_cls(
            self.actor_network_cfg.make(), self.critic_network_cfg.make(), **kwargs
        )


@dataclass
class LossConfig(ConfigBase):
    pass
