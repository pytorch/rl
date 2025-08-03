# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from torchrl.collectors.collectors import DataCollectorBase
from torchrl.objectives.common import LossModule
from torchrl.trainers.algorithms.configs.common import ConfigBase


@dataclass
class TrainerConfig(ConfigBase):
    pass


@dataclass
class PPOTrainerConfig(TrainerConfig):
    collector: Any
    total_frames: int
    frame_skip: int
    optim_steps_per_batch: int
    loss_module: Any
    optimizer: Any
    logger: Any
    clip_grad_norm: bool
    clip_norm: float | None
    progress_bar: bool
    seed: int | None
    save_trainer_interval: int
    log_interval: int
    save_trainer_file: Any
    replay_buffer: Any
    create_env_fn: Any = None
    actor_network: Any = None
    critic_network: Any = None

    _target_: str = "torchrl.trainers.algorithms.configs.trainers._make_ppo_trainer"


def _make_ppo_trainer(*args, **kwargs) -> PPOTrainer:
    from torchrl.trainers.algorithms.ppo import PPOTrainer
    from torchrl.trainers.trainers import Logger

    collector = kwargs.pop("collector")
    total_frames = kwargs.pop("total_frames")
    if total_frames is None:
        total_frames = collector.total_frames
    frame_skip = kwargs.pop("frame_skip", 1)
    optim_steps_per_batch = kwargs.pop("optim_steps_per_batch", 1)
    loss_module = kwargs.pop("loss_module")
    optimizer = kwargs.pop("optimizer")
    logger = kwargs.pop("logger")
    clip_grad_norm = kwargs.pop("clip_grad_norm", True)
    clip_norm = kwargs.pop("clip_norm")
    progress_bar = kwargs.pop("progress_bar", True)
    replay_buffer = kwargs.pop("replay_buffer")
    save_trainer_interval = kwargs.pop("save_trainer_interval", 10000)
    log_interval = kwargs.pop("log_interval", 10000)
    save_trainer_file = kwargs.pop("save_trainer_file")
    seed = kwargs.pop("seed")
    actor_network = kwargs.pop("actor_network")
    critic_network = kwargs.pop("critic_network")
    create_env_fn = kwargs.pop("create_env_fn")

    # Instantiate networks first
    if actor_network is not None:
        actor_network = actor_network()
    if critic_network is not None:
        critic_network = critic_network()

    if not isinstance(collector, DataCollectorBase):
        # then it's a partial config
        collector = collector(create_env_fn=create_env_fn, policy=actor_network)
    if not isinstance(loss_module, LossModule):
        # then it's a partial config
        loss_module = loss_module(
            actor_network=actor_network, critic_network=critic_network
        )
    if not isinstance(optimizer, torch.optim.Optimizer):
        assert callable(optimizer)
        # then it's a partial config
        optimizer = optimizer(params=loss_module.parameters())

    # Quick instance checks
    if not isinstance(collector, DataCollectorBase):
        raise ValueError("collector must be a DataCollectorBase")
    if not isinstance(loss_module, LossModule):
        raise ValueError("loss_module must be a LossModule")
    if not isinstance(optimizer, torch.optim.Optimizer):
        raise ValueError("optimizer must be a torch.optim.Optimizer")
    if not isinstance(logger, Logger) and logger is not None:
        raise ValueError("logger must be a Logger")

    return PPOTrainer(
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
        replay_buffer=replay_buffer,
    )
