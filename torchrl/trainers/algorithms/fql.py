# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import pathlib
from collections.abc import Callable, Mapping
from typing import Any

import torch
from torchrl.checkpoint import Checkpoint, CheckpointRotation
from torchrl.collectors import BaseCollector
from torchrl.data import ReplayBuffer
from torchrl.objectives import FQLLoss
from torchrl.objectives.utils import TargetNetUpdater
from torchrl.record.loggers import Logger
from torchrl.trainers import Trainer
from torchrl.trainers.algorithms.offline_to_online import OfflineToOnlineTrainer


class FQLTrainer(OfflineToOnlineTrainer):
    """Train FQL with shared offline-to-online replay and Trainer lifecycle hooks.

    See :class:`~torchrl.trainers.algorithms.configs.FQLTrainerConfig` for the
    Hydra configuration counterpart.

    The standard optimizer hook updates all networks before the target-network
    hook runs. Unlike reference FQL, the target EMA uses the updated critic.
    Allocate replay capacity for the offline dataset and online transitions.
    Online collection performs one update per batch, so use one frame per batch
    for one update per interaction. ``update()`` performs a single replay update.

    ``collector`` may be omitted for offline-only training or provided as a
    factory initialized after pretraining. ``compile_loss`` compiles the loss
    without changing its checkpoint keys. See :class:`OfflineToOnlineTrainer`
    for logging, checkpointing and hook options.

    Args:
        loss_module (FQLLoss): flow, actor and critic objective.
        optimizer (torch.optim.Optimizer): optimizer for the loss parameters.
        replay_buffer (ReplayBuffer): offline dataset and online replay storage.
        target_net_updater (TargetNetUpdater): critic target update rule.
        offline_steps (int): gradient updates before online collection.

    Keyword Args:
        logger (Logger, optional): scalar logger. Defaults to ``None``.
        checkpoint (Checkpoint, optional): checkpoint component registry.
        checkpoint_rotation (CheckpointRotation, optional): checkpoint retention
            policy. Use with ``checkpoint``; alternatively set ``save_trainer_file``.
    """

    def __init__(
        self,
        *,
        loss_module: FQLLoss,
        optimizer: torch.optim.Optimizer,
        replay_buffer: ReplayBuffer,
        target_net_updater: TargetNetUpdater,
        offline_steps: int,
        collector: BaseCollector | Callable[[], BaseCollector] | None = None,
        total_frames: int = 0,
        device: torch.device | str | None = None,
        batch_size: int | None = None,
        compile_loss: bool = False,
        logger: Logger | None = None,
        clip_grad_norm: bool = True,
        clip_norm: float | None = None,
        progress_bar: bool = False,
        seed: int | None = None,
        save_trainer_interval: int = 10000,
        log_interval: int = 10000,
        save_trainer_file: str | pathlib.Path | None = None,
        checkpoint: Checkpoint | None = None,
        checkpoint_rotation: CheckpointRotation | None = None,
        checkpoint_metadata: Callable[[Trainer], Mapping[str, Any]] | None = None,
        log_timings: bool = False,
        auto_log_optim_steps: bool = True,
    ) -> None:
        super().__init__(
            loss_module=loss_module,
            optimizer=optimizer,
            replay_buffer=replay_buffer,
            target_net_updater=target_net_updater,
            offline_steps=offline_steps,
            collector=collector,
            total_frames=total_frames,
            device=device,
            batch_size=batch_size,
            compile_loss=compile_loss,
            logger=logger,
            clip_grad_norm=clip_grad_norm,
            clip_norm=clip_norm,
            progress_bar=progress_bar,
            seed=seed,
            save_trainer_interval=save_trainer_interval,
            log_interval=log_interval,
            save_trainer_file=save_trainer_file,
            checkpoint=checkpoint,
            checkpoint_rotation=checkpoint_rotation,
            checkpoint_metadata=checkpoint_metadata,
            log_timings=log_timings,
            auto_log_optim_steps=auto_log_optim_steps,
            enable_logging=False,
        )
