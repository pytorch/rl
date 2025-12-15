# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import pathlib
import warnings

from collections.abc import Callable

from functools import partial

from tensordict import TensorDict, TensorDictBase
from torch import optim

from torchrl.collectors import BaseCollector

from torchrl.data.replay_buffers.replay_buffers import ReplayBuffer
from torchrl.objectives.common import LossModule
from torchrl.objectives.utils import TargetNetUpdater
from torchrl.record.loggers import Logger
from torchrl.trainers.trainers import (
    LogScalar,
    ReplayBufferTrainer,
    TargetNetUpdaterHook,
    Trainer,
    UpdateWeights,
    UTDRHook,
)


class SACTrainer(Trainer):
    """A trainer class for Soft Actor-Critic (SAC) algorithm.

    This trainer implements the SAC algorithm, an off-policy actor-critic method that
    optimizes a stochastic policy in an off-policy way, forming a bridge between
    stochastic policy optimization and DDPG-style approaches. SAC incorporates the
    entropy measure of the policy into the reward to encourage exploration.

    The trainer handles:
    - Replay buffer management for off-policy learning
    - Target network updates with configurable update frequency
    - Policy weight updates to the data collector
    - Comprehensive logging of training metrics
    - Gradient clipping and optimization steps

    Args:
        collector (BaseCollector): The data collector used to gather environment interactions.
        total_frames (int): Total number of frames to collect during training.
        frame_skip (int): Number of frames to skip between policy updates.
        optim_steps_per_batch (int): Number of optimization steps per collected batch.
        loss_module (LossModule | Callable): The SAC loss module or a callable that computes losses.
        optimizer (optim.Optimizer, optional): The optimizer for training. If None, must be configured elsewhere.
        logger (Logger, optional): Logger for recording training metrics. Defaults to None.
        clip_grad_norm (bool, optional): Whether to clip gradient norms. Defaults to True.
        clip_norm (float, optional): Maximum gradient norm for clipping. Defaults to None.
        progress_bar (bool, optional): Whether to show a progress bar during training. Defaults to True.
        seed (int, optional): Random seed for reproducibility. Defaults to None.
        save_trainer_interval (int, optional): Interval for saving trainer state. Defaults to 10000.
        log_interval (int, optional): Interval for logging metrics. Defaults to 10000.
        save_trainer_file (str | pathlib.Path, optional): File path for saving trainer state. Defaults to None.
        replay_buffer (ReplayBuffer, optional): Replay buffer for storing and sampling experiences. Defaults to None.
        batch_size (int, optional): Batch size for sampling from replay buffer. Defaults to None.
        enable_logging (bool, optional): Whether to enable metric logging. Defaults to True.
        log_rewards (bool, optional): Whether to log reward statistics. Defaults to True.
        log_actions (bool, optional): Whether to log action statistics. Defaults to True.
        log_observations (bool, optional): Whether to log observation statistics. Defaults to False.
        target_net_updater (TargetNetUpdater, optional): Target network updater for soft updates. Defaults to None.

    Example:
        >>> from torchrl.collectors import Collector
        >>> from torchrl.objectives import SACLoss
        >>> from torchrl.data import ReplayBuffer, LazyTensorStorage
        >>> from torch import optim
        >>>
        >>> # Set up collector, loss, and replay buffer
        >>> collector = Collector(env, policy, frames_per_batch=1000)
        >>> loss_module = SACLoss(actor_network, qvalue_network)
        >>> optimizer = optim.Adam(loss_module.parameters(), lr=3e-4)
        >>> replay_buffer = ReplayBuffer(storage=LazyTensorStorage(100000))
        >>>
        >>> # Create and run trainer
        >>> trainer = SACTrainer(
        ...     collector=collector,
        ...     total_frames=1000000,
        ...     frame_skip=1,
        ...     optim_steps_per_batch=100,
        ...     loss_module=loss_module,
        ...     optimizer=optimizer,
        ...     replay_buffer=replay_buffer,
        ... )
        >>> trainer.train()

    Note:
        This is an experimental/prototype feature. The API may change in future versions.
        SAC is particularly effective for continuous control tasks and environments where
        exploration is crucial due to its entropy regularization.

    """

    def __init__(
        self,
        *,
        collector: BaseCollector,
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
        replay_buffer: ReplayBuffer | None = None,
        batch_size: int | None = None,
        enable_logging: bool = True,
        log_rewards: bool = True,
        log_actions: bool = True,
        log_observations: bool = False,
        target_net_updater: TargetNetUpdater | None = None,
        async_collection: bool = False,
        log_timings: bool = False,
    ) -> None:
        warnings.warn(
            "SACTrainer is an experimental/prototype feature. The API may change in future versions. "
            "Please report any issues or feedback to help improve this implementation.",
            UserWarning,
            stacklevel=2,
        )
        # try to get the action spec
        self._pass_action_spec_from_collector_to_loss(collector, loss_module)

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
            async_collection=async_collection,
            log_timings=log_timings,
        )
        self.replay_buffer = replay_buffer
        self.async_collection = async_collection

        # Note: SAC can use any sampler type, unlike PPO which requires SamplerWithoutReplacement

        if replay_buffer is not None:
            rb_trainer = ReplayBufferTrainer(
                replay_buffer,
                batch_size=None,
                flatten_tensordicts=True,
                memmap=False,
                device=getattr(replay_buffer.storage, "device", "cpu"),
                iterate=True,
            )
            if not self.async_collection:
                self.register_op("pre_epoch", rb_trainer.extend)
            self.register_op("process_optim_batch", rb_trainer.sample)
            self.register_op("post_loss", rb_trainer.update_priority)
        self.register_op("post_optim", TargetNetUpdaterHook(target_net_updater))

        policy_weights_getter = partial(
            TensorDict.from_module, self.loss_module.actor_network
        )
        update_weights = UpdateWeights(
            self.collector, 1, policy_weights_getter=policy_weights_getter
        )
        self.register_op("post_steps", update_weights)

        # Store logging configuration
        self.enable_logging = enable_logging
        self.log_rewards = log_rewards
        self.log_actions = log_actions
        self.log_observations = log_observations

        # Set up comprehensive logging for SAC training
        if self.enable_logging:
            self._setup_sac_logging()

    def _pass_action_spec_from_collector_to_loss(
        self, collector: BaseCollector, loss: LossModule
    ):
        """Pass the action specification from the collector's environment to the loss module.

        This method extracts the action specification from the collector's environment
        and assigns it to the loss module if the loss module doesn't already have one.
        This is necessary for SAC loss computation which requires knowledge of the
        action space bounds for proper entropy calculation and action clipping.

        Args:
            collector (BaseCollector): The data collector containing the environment.
            loss (LossModule): The loss module that needs the action specification.
        """
        if hasattr(loss, "_action_spec") and loss._action_spec is None:
            action_spec = collector.getattr_env("full_action_spec_unbatched").cpu()
            loss._action_spec = action_spec

    def _setup_sac_logging(self):
        """Set up logging hooks for SAC-specific metrics.

        This method configures logging for common SAC metrics including:
        - Training rewards (mean, max, total, and std)
        - Action statistics (action norms)
        - Episode completion rates (done percentage)
        - Observation statistics (when enabled)
        - Q-value and policy loss metrics (handled by loss module)
        """
        # Always log done states as percentage (episode completion rate)
        log_done_percentage = LogScalar(
            key=("next", "done"),
            logname="done_percentage",
            log_pbar=True,
            include_std=False,  # No std for binary values
            reduction="mean",
        )
        if not self.async_collection:
            self.register_op("pre_steps_log", log_done_percentage)
        else:
            self.register_op("post_optim_log", log_done_percentage)

        # Log rewards if enabled
        if self.log_rewards:
            # 1. Log training rewards (most important metric for SAC)
            log_rewards = LogScalar(
                key=("next", "reward"),
                logname="r_training",
                log_pbar=True,  # Show in progress bar
                include_std=True,
                reduction="mean",
            )
            if not self.async_collection:
                self.register_op("pre_steps_log", log_rewards)
            else:
                # In the async case, use the batch passed to the optimizer
                self.register_op("post_optim_log", log_rewards)

            # 2. Log maximum reward in batch (for monitoring best performance)
            log_max_reward = LogScalar(
                key=("next", "reward"),
                logname="r_max",
                log_pbar=False,
                include_std=False,
                reduction="max",
            )
            if not self.async_collection:
                self.register_op("pre_steps_log", log_max_reward)
            else:
                self.register_op("post_optim_log", log_max_reward)

            # 3. Log total reward in batch (for monitoring cumulative performance)
            log_total_reward = LogScalar(
                key=("next", "reward_sum"),
                logname="r_total",
                log_pbar=False,
                include_std=False,
                reduction="max",
            )
            if not self.async_collection:
                self.register_op("pre_steps_log", log_total_reward)
            else:
                self.register_op("post_optim_log", log_total_reward)

        # Log actions if enabled
        if self.log_actions:
            # 4. Log action norms (useful for monitoring policy behavior)
            log_action_norm = LogScalar(
                key="action",
                logname="action_norm",
                log_pbar=False,
                include_std=True,
                reduction="mean",
            )
            if not self.async_collection:
                self.register_op("pre_steps_log", log_action_norm)
            else:
                self.register_op("post_optim_log", log_action_norm)

        # Log observations if enabled
        if self.log_observations:
            # 5. Log observation statistics (for monitoring state distributions)
            log_obs_norm = LogScalar(
                key="observation",
                logname="obs_norm",
                log_pbar=False,
                include_std=True,
                reduction="mean",
            )
            if not self.async_collection:
                self.register_op("pre_steps_log", log_obs_norm)
            else:
                self.register_op("post_optim_log", log_obs_norm)

        self.register_op("pre_steps_log", UTDRHook(self))
