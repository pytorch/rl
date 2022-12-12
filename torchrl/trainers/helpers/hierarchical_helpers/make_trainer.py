# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import List, Optional, Union
from warnings import warn

import torch
from tensordict.nn import TensorDictModuleWrapper
from torch import optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from torchrl.collectors.collectors import _DataCollector
from torchrl.data import ReplayBuffer
from torchrl.envs.common import EnvBase
from torchrl.modules import reset_noise, SafeModule
from torchrl.objectives.common import LossModule
from torchrl.objectives.utils import TargetNetUpdater
from torchrl.trainers.loggers import Logger
from torchrl.trainers.trainers import (
    BatchSubSampler,
    ClearCudaCache,
    CountFramesLog,
    LogReward,
    Recorder,
    ReplayBufferTrainer,
    RewardNormalizer,
    SelectKeys,
    Trainer,
    UpdateWeights,
)

OPTIMIZERS = {
    "adam": optim.Adam,
    "sgd": optim.SGD,
    "adamax": optim.Adamax,
}


def make_trainer(
    collector: _DataCollector,
    loss_module: LossModule,
    recorder: Optional[EnvBase] = None,
    target_net_updater: Optional[TargetNetUpdater] = None,
    policy_exploration: Optional[Union[TensorDictModuleWrapper, SafeModule]] = None,
    replay_buffer: Optional[ReplayBuffer] = None,
    logger: Optional[Logger] = None,
    optim_steps_per_batch: int = 500,
    optimizer: str = "adam",
    lr_scheduler: str = "cosine",
    selected_keys: Optional[List] = None,
    batch_size: int = 256,
    log_interval: int = 10000,
    lr: float = 3e-4,
    weight_decay: float = 0.0,
    clip_norm: float = 1000.0,
    clip_grad_norm: bool = False,
    normalize_rewards_online: bool = False,
    normalize_rewards_online_scale: float = 1.0,
    normalize_rewards_online_decay: float = 0.9999,
    sub_traj_len: int = -1,
    total_frames: int = 1000,
    record_frames: int = 10,
    record_interval: int = 10,
    frame_skip: int = 0,
    frames_per_batch: int = 1000,
) -> Trainer:

    device = next(loss_module.parameters()).device

    # Define optimizier
    optimizer_kwargs = {} if optimizer != "adam" else {"betas": (0.0, 0.9)}
    optimizer = OPTIMIZERS[optimizer](
        loss_module.parameters(),
        lr=lr,
        weight_decay=weight_decay,
        **optimizer_kwargs,
    )

    # Define learning rate scheduler
    if lr_scheduler == "cosine":
        optim_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=int(
                total_frames / frames_per_batch * optim_steps_per_batch
            ),
        )
    elif lr_scheduler == "":
        optim_scheduler = None
    else:
        raise NotImplementedError(f"lr scheduler {lr_scheduler}")

    # Define trainer
    trainer = Trainer(
        collector=collector,
        frame_skip=frame_skip,
        total_frames=total_frames * frame_skip,
        loss_module=loss_module,
        optimizer=optimizer,
        logger=logger,
        optim_steps_per_batch=optim_steps_per_batch,
        clip_grad_norm=clip_grad_norm,
        clip_norm=clip_norm,
    )

    # Define hooks applied to data batches after collection
    if selected_keys:
        # Select specific keys
        trainer.register_op("batch_process", SelectKeys(selected_keys))

    # Move data to cpu
    trainer.register_op("batch_process", lambda batch: batch.cpu())

    if replay_buffer:
        rb_trainer = ReplayBufferTrainer(
            replay_buffer, batch_size, memmap=False, device=device)

        # Add data to replay buffer
        trainer.register_op("batch_process", rb_trainer.extend)

    # Define hooks applied before optimization
    if torch.cuda.device_count() > 0:
        # Clear cuda memory
        trainer.register_op("pre_optim_steps", ClearCudaCache(1))

    # if noisy:
    #     # Reset noise
    #     trainer.register_op("pre_optim_steps", lambda: loss_module.apply(reset_noise))

    # Define hooks applied to sub-batches before optimization
    if replay_buffer:
        trainer.register_op("process_optim_batch", rb_trainer.sample)
    else:
        # Generate mini-batches
        trainer.register_op(
            "process_optim_batch",
            BatchSubSampler(batch_size=batch_size, sub_traj_len=sub_traj_len))
        # Move mini-batches to device
        trainer.register_op("process_optim_batch", lambda batch: batch.to(device))

    if replay_buffer is not None:
        # replay buffer is used 2 or 3 times: to register data, to sample
        # data and to update priorities
        trainer.register_op("post_loss", rb_trainer.update_priority)

    if optim_scheduler is not None:
        trainer.register_op("post_optim", optim_scheduler.step)

    if target_net_updater is not None:
        trainer.register_op("post_optim", target_net_updater.step)

    if normalize_rewards_online:
        # if used the running statistics of the rewards are computed and the
        # rewards used for training will be normalized based on these.
        reward_normalizer = RewardNormalizer(
            scale=normalize_rewards_online_scale,
            decay=normalize_rewards_online_decay,
        )
        trainer.register_op("batch_process", reward_normalizer.update_reward_stats)
        trainer.register_op("process_optim_batch", reward_normalizer.normalize_reward)

    if policy_exploration is not None and hasattr(policy_exploration, "step"):
        trainer.register_op(
            "post_steps", policy_exploration.step, frames=frames_per_batch
        )

    trainer.register_op(
        "post_steps_log", lambda lr: optimizer.param_groups[0]["lr"]
    )

    if recorder is not None:
        trainer.register_op(
            "post_steps_log",
            recorder,
        )

    trainer.register_op(
        "post_steps", UpdateWeights(collector, update_weights_interval=1)
    )

    trainer.register_op("pre_steps_log", LogReward())
    trainer.register_op("pre_steps_log", CountFramesLog(frame_skip=frame_skip))

    return trainer
