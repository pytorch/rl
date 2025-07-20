# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass

from torchrl.trainers.algorithms.configs.collectors import DataCollectorConfig
from torchrl.trainers.algorithms.configs.common import ConfigBase
from torchrl.trainers.algorithms.configs.data import ReplayBufferConfig
from torchrl.trainers.algorithms.configs.objectives import PPOLossConfig


@dataclass
class TrainerConfig(ConfigBase):
    pass


@dataclass
class PPOConfig(TrainerConfig):
    loss_cfg: PPOLossConfig
    collector_cfg: DataCollectorConfig
    replay_buffer_cfg: ReplayBufferConfig

    optim_steps_per_batch: int

    _target_: str = "torchrl.trainers.algorithms.ppo.PPOTrainer"
