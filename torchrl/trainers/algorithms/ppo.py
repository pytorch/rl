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
from torchrl.trainers.algorithms.configs.data import (
    LazyTensorStorageConfig,
    ReplayBufferConfig,
)
from torchrl.trainers.algorithms.configs.envs import GymEnvConfig
from torchrl.trainers.algorithms.configs.modules import MLPConfig, TanhNormalModelConfig
from torchrl.trainers.algorithms.configs.objectives import PPOLossConfig
from torchrl.trainers.algorithms.configs.utils import AdamConfig
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

    @classmethod
    def default_config(cls) -> PPOTrainerConfig:  # type: ignore # noqa: F821
        """Creates a default config for the PPO trainer.

        The task is the Pendulum-v1 environment in Gym, with a 2-layer MLP actor and critic.

        """
        from torchrl.trainers.algorithms.configs.collectors import (
            SyncDataCollectorConfig,
        )
        from torchrl.trainers.algorithms.configs.modules import TensorDictModuleConfig
        from torchrl.trainers.algorithms.configs.trainers import PPOTrainerConfig

        env_cfg = GymEnvConfig(env_name="Pendulum-v1")
        actor_network = TanhNormalModelConfig(
            network=MLPConfig(in_features=3, out_features=2, depth=2, num_cells=128),
            in_keys=["observation"],
            out_keys=["action"],
            return_log_prob=True,
        )
        critic_network = TensorDictModuleConfig(
            module=MLPConfig(in_features=3, out_features=1, depth=2, num_cells=128),
            in_keys=["observation"],
            out_keys=["state_value"],
        )
        collector_cfg = SyncDataCollectorConfig(
            total_frames=1_000_000, frames_per_batch=1000, _partial_=True
        )
        loss_cfg = PPOLossConfig(_partial_=True)
        optimizer_cfg = AdamConfig(_partial_=True)
        replay_buffer_cfg = ReplayBufferConfig(
            storage=LazyTensorStorageConfig(max_size=100_000, device="cpu"),
            batch_size=256,
        )
        return PPOTrainerConfig(
            collector=collector_cfg,
            total_frames=1_000_000,
            frame_skip=1,
            optim_steps_per_batch=1,
            loss_module=loss_cfg,
            optimizer=optimizer_cfg,
            logger=None,
            clip_grad_norm=True,
            clip_norm=1.0,
            progress_bar=True,
            seed=1,
            save_trainer_interval=10000,
            log_interval=10000,
            save_trainer_file=None,
            replay_buffer=replay_buffer_cfg,
            create_env_fn=env_cfg,
            actor_network=actor_network,
            critic_network=critic_network,
        )
