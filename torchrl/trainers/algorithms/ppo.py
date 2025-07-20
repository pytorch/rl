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

from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.envs.batched_envs import ParallelEnv
from torchrl.objectives.common import LossModule
from torchrl.record.loggers import Logger
from torchrl.trainers.algorithms.configs.collectors import DataCollectorConfig
from torchrl.trainers.algorithms.configs.data import ReplayBufferConfig
from torchrl.trainers.algorithms.configs.envs import BatchedEnvConfig, GymEnvConfig
from torchrl.trainers.algorithms.configs.modules import MLPConfig, TanhNormalModelConfig
from torchrl.trainers.algorithms.configs.objectives import PPOLossConfig
from torchrl.trainers.algorithms.configs.trainers import PPOConfig
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
    def from_config(cls, cfg: PPOConfig, **kwargs):
        return cfg.make()

    @property
    def default_config(self):
        inference_batch_size = 1024

        inference_env_cfg = BatchedEnvConfig(
            batched_env_type=ParallelEnv,
            env_config=GymEnvConfig(env_name="Pendulum-v1"),
            num_envs=4,
        )
        specs = inference_env_cfg.specs
        # TODO: maybe an MLPConfig.from_env ?
        # input /output features
        in_features = specs[
            "output_spec", "full_observation_spec", "observation"
        ].shape[-1]
        out_features = specs["output_spec", "full_action_spec", "action"].shape[-1]
        network_config = MLPConfig(
            in_features=in_features,
            out_features=2 * out_features,
            num_cells=[128, 128, 128],
        )

        inference_policy_config = TanhNormalModelConfig(network_config=network_config)

        rb_config = ReplayBufferConfig(
            storage=lambda: LazyTensorStorage(max_size=inference_batch_size)
        )

        collector_cfg = DataCollectorConfig(
            env_cfg=inference_env_cfg,
            policy_cfg=inference_policy_config,
            frames_per_batch=inference_batch_size,
        )

        critic_network_config = MLPConfig(
            in_features=in_features,
            out_features=1,
            num_cells=[128, 128, 128],
            as_tensordict_module=True,
            in_keys=["observation"],
            out_keys=["state_value"],
        )

        ppo_loss_cfg = PPOLossConfig(
            # We use the same config for the inference and training policies
            actor_network_cfg=inference_policy_config,
            critic_network_cfg=critic_network_config,
        )

        return PPOConfig(
            loss_cfg=ppo_loss_cfg,
            collector_cfg=collector_cfg,
            replay_buffer_cfg=rb_config,
            optim_steps_per_batch=1,
        )
