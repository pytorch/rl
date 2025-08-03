# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import pathlib

from typing import Callable

from tensordict import TensorDictBase
from torch import optim

from torchrl.collectors.collectors import DataCollectorBase

from torchrl.objectives.common import LossModule
from torchrl.record.loggers import Logger
from torchrl.trainers.trainers import Trainer

try:
    pass

    _has_tqdm = True
except ImportError:
    _has_tqdm = False

try:
    pass

    _has_ts = True
except ImportError:
    _has_ts = False


class PPOTrainer(Trainer):
    """PPO (Proximal Policy Optimization) trainer implementation.

    This trainer implements the PPO algorithm for training reinforcement learning agents.
    It extends the base Trainer class with PPO-specific functionality including
    policy optimization, value function learning, and entropy regularization.
    """

    def __init__(
        self,
        *,
        collector: DataCollectorBase,
        total_frames: int,
        frame_skip: int,
        optim_steps_per_batch: int,
        loss_module: LossModule | Callable[[TensorDictBase], TensorDictBase],
        optimizer: optim.Optimizer | None = None,
        logger: Logger | None = None,
        clip_grad_norm: bool = True,
        clip_norm: float | None = None,
        progress_bar: bool = True,
        seed: int | None = None,
        save_trainer_interval: int = 10000,
        log_interval: int = 10000,
        save_trainer_file: str | pathlib.Path | None = None,
        replay_buffer=None,
    ) -> None:
        super().__init__(
            collector=collector,
            total_frames=total_frames,
            frame_skip=frame_skip,
            optim_steps_per_batch=optim_steps_per_batch,
            loss_module=loss_module,
            optimizer=optimizer,
            logger=logger,
            clip_grad_norm=clip_grad_norm,
            clip_norm=clip_norm,
            progress_bar=progress_bar,
            seed=seed,
            save_trainer_interval=save_trainer_interval,
            log_interval=log_interval,
            save_trainer_file=save_trainer_file,
        )
        self.replay_buffer = replay_buffer
