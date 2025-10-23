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

from torchrl.collectors import DataCollectorBase

from torchrl.data.replay_buffers.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.objectives.common import LossModule
from torchrl.objectives.value.advantages import GAE
from torchrl.record.loggers import Logger
from torchrl.trainers.trainers import (
    LogScalar,
    ReplayBufferTrainer,
    Trainer,
    UpdateWeights,
)

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

    .. warning::
        This is an experimental/prototype feature. The API may change in future versions.
        Please report any issues or feedback to help improve this implementation.

    This trainer implements the PPO algorithm for training reinforcement learning agents.
    It extends the base Trainer class with PPO-specific functionality including
    policy optimization, value function learning, and entropy regularization.

    PPO typically uses multiple epochs of optimization on the same batch of data.
    This trainer defaults to 4 epochs, which is a common choice for PPO implementations.

    The trainer includes comprehensive logging capabilities for monitoring training progress:
    - Training rewards (mean, std, max, total)
    - Action statistics (norms)
    - Episode completion rates
    - Observation statistics (optional)

    Logging can be configured via constructor parameters to enable/disable specific metrics.

    Examples:
        >>> # Basic usage with manual configuration
        >>> from torchrl.trainers.algorithms.ppo import PPOTrainer
        >>> from torchrl.trainers.algorithms.configs import PPOTrainerConfig
        >>> from hydra import instantiate
        >>> config = PPOTrainerConfig(...)  # Configure with required parameters
        >>> trainer = instantiate(config)
        >>> trainer.train()

    .. note::
        This trainer requires a configurable environment setup. See the
        :class:`~torchrl.trainers.algorithms.configs` module for configuration options.

    .. warning::
        This is an experimental feature. The API may change in future versions.
        We welcome feedback and contributions to help improve this implementation!
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
        num_epochs: int = 4,
        replay_buffer: ReplayBuffer | None = None,
        batch_size: int | None = None,
        gamma: float = 0.9,
        lmbda: float = 0.99,
        enable_logging: bool = True,
        log_rewards: bool = True,
        log_actions: bool = True,
        log_observations: bool = False,
        async_collection: bool = False,
    ) -> None:
        warnings.warn(
            "PPOTrainer is an experimental/prototype feature. The API may change in future versions. "
            "Please report any issues or feedback to help improve this implementation.",
            UserWarning,
            stacklevel=2,
        )
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
            num_epochs=num_epochs,
            async_collection=async_collection,
        )
        self.replay_buffer = replay_buffer
        self.async_collection = async_collection

        gae = GAE(
            gamma=gamma,
            lmbda=lmbda,
            value_network=self.loss_module.critic_network,
            average_gae=True,
        )
        self.register_op("pre_epoch", gae)

        if (
            not self.async_collection
            and replay_buffer is not None
            and not isinstance(replay_buffer.sampler, SamplerWithoutReplacement)
        ):
            warnings.warn(
                "Sampler is not a SamplerWithoutReplacement, which is required for PPO."
            )

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

        # Set up comprehensive logging for PPO training
        if self.enable_logging:
            self._setup_ppo_logging()

    def _setup_ppo_logging(self):
        """Set up logging hooks for PPO-specific metrics.

        This method configures logging for common PPO metrics including:
        - Training rewards (mean and std)
        - Action statistics (norms, entropy)
        - Episode completion rates
        - Value function statistics
        - Advantage statistics
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
            # 1. Log training rewards (most important metric for PPO)
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
                key=("next", "reward"),
                logname="r_total",
                log_pbar=False,
                include_std=False,
                reduction="sum",
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
