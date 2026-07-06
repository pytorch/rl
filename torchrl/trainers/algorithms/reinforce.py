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
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.objectives.common import LossModule
from torchrl.objectives.value.advantages import GAE
from torchrl.record.loggers import Logger
from torchrl.trainers.trainers import (
    LogScalar,
    LRSchedulerHook,
    ReplayBufferTrainer,
    Trainer,
    UpdateWeights,
    ValueEstimatorHook,
)


class ReinforceTrainer(Trainer):
    """REINFORCE (policy gradient with baseline) trainer implementation.

    .. warning::
        This is an experimental/prototype feature. The API may change in future versions.
        Please report any issues or feedback to help improve this implementation.

    This trainer implements the REINFORCE algorithm for training reinforcement
    learning agents. It extends the base Trainer class with policy-gradient
    functionality, using a critic network as a baseline for advantage
    estimation (GAE by default, matching
    :class:`~torchrl.objectives.ReinforceLoss`).

    REINFORCE is a single-pass on-policy algorithm: each collected batch is
    consumed once. This trainer therefore defaults to 1 epoch per batch.

    The trainer includes comprehensive logging capabilities for monitoring training progress:
    - Training rewards (mean, std, max, total)
    - Action statistics (norms)
    - Episode completion rates
    - Observation statistics (optional)

    Logging can be configured via constructor parameters to enable/disable specific metrics.

    Args:
        collector (BaseCollector): The data collector for gathering training data.
        total_frames (int): Total number of frames to train for.
        frame_skip (int): Frame skip value for the environment.
        optim_steps_per_batch (int): Number of optimization steps per batch.
        loss_module (LossModule): The loss module for computing policy and value losses.
        optimizer (optim.Optimizer, optional): The optimizer for training.
        lr_scheduler (optim.lr_scheduler.LRScheduler, optional): Learning-rate scheduler,
            stepped once per collected batch via :class:`~torchrl.trainers.LRSchedulerHook`.
        logger (Logger, optional): Logger for tracking training metrics.
        clip_grad_norm (bool, optional): Whether to clip gradient norms. Default: True.
        clip_norm (float, optional): Maximum gradient norm value.
        progress_bar (bool, optional): Whether to show a progress bar. Default: True.
        seed (int, optional): Random seed for reproducibility.
        save_trainer_interval (int, optional): Interval for saving trainer state. Default: 10000.
        log_interval (int, optional): Interval for logging metrics. Default: 10000.
        save_trainer_file (str | pathlib.Path, optional): File path for saving trainer state.
        num_epochs (int, optional): Number of epochs per batch. Default: 1.
        replay_buffer (ReplayBuffer, optional): Replay buffer for storing data.
        batch_size (int, optional): Batch size for optimization.
        gamma (float, optional): Discount factor for GAE. Default: 0.9.
        lmbda (float, optional): Lambda parameter for GAE. Default: 0.99.
        enable_logging (bool, optional): Whether to enable logging. Default: True.
        log_rewards (bool, optional): Whether to log rewards. Default: True.
        log_actions (bool, optional): Whether to log actions. Default: True.
        log_observations (bool, optional): Whether to log observations. Default: False.
        async_collection (bool, optional): Whether to use async collection. Default: False.
        add_gae (bool, optional): Whether to add GAE computation. Default: True.
        gae (Callable, optional): Custom GAE module. If None and add_gae is True, a default GAE will be created.
        weight_update_map (dict[str, str], optional): Mapping from collector destination paths (keys in
            collector's weight_sync_schemes) to trainer source paths. Required if collector has
            weight_sync_schemes configured. Example: {"policy": "loss_module.actor_network",
            "replay_buffer.transforms[0]": "loss_module.critic_network"}
        log_timings (bool, optional): If True, automatically register a LogTiming hook to log
            timing information for all hooks to the logger (e.g., wandb, tensorboard).
            Timing metrics will be logged with prefix "time/" (e.g., "time/hook/UpdateWeights").
            Default is False.

    Examples:
        >>> # Basic usage with manual configuration
        >>> from torchrl.trainers.algorithms.reinforce import ReinforceTrainer
        >>> from torchrl.trainers.algorithms.configs import ReinforceTrainerConfig
        >>> from hydra import instantiate
        >>> config = ReinforceTrainerConfig(...)  # Configure with required parameters
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
        collector: BaseCollector,
        total_frames: int,
        frame_skip: int,
        optim_steps_per_batch: int,
        loss_module: LossModule | Callable[[TensorDictBase], TensorDictBase],
        optimizer: optim.Optimizer | None = None,
        lr_scheduler: optim.lr_scheduler.LRScheduler | None = None,
        logger: Logger | None = None,
        clip_grad_norm: bool = True,
        clip_norm: float | None = None,
        progress_bar: bool = True,
        seed: int | None = None,
        save_trainer_interval: int = 10000,
        log_interval: int = 10000,
        save_trainer_file: str | pathlib.Path | None = None,
        num_epochs: int = 1,
        replay_buffer: ReplayBuffer | None = None,
        batch_size: int | None = None,
        gamma: float = 0.9,
        lmbda: float = 0.99,
        enable_logging: bool = True,
        log_rewards: bool = True,
        log_actions: bool = True,
        log_observations: bool = False,
        async_collection: bool = False,
        add_gae: bool = True,
        gae: Callable[[TensorDictBase], TensorDictBase] | None = None,
        weight_update_map: dict[str, str] | None = None,
        log_timings: bool = False,
    ) -> None:
        warnings.warn(
            "ReinforceTrainer is an experimental/prototype feature. The API may change in future versions. "
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
            log_timings=log_timings,
        )
        self.replay_buffer = replay_buffer
        self.async_collection = async_collection

        if not add_gae and gae is not None:
            raise ValueError("gae must not be provided if add_gae is False")
        if add_gae:
            if gae is None:
                gae = GAE(
                    gamma=gamma,
                    lmbda=lmbda,
                    value_network=self.loss_module.critic_network,
                    average_gae=True,
                )
            ValueEstimatorHook(gae).register(self)

        if lr_scheduler is not None:
            LRSchedulerHook(lr_scheduler).register(self)

        if (
            not self.async_collection
            and replay_buffer is not None
            and not isinstance(replay_buffer.sampler, SamplerWithoutReplacement)
        ):
            warnings.warn(
                "Sampler is not a SamplerWithoutReplacement, which is required for REINFORCE."
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

        if (
            hasattr(self.collector, "_weight_sync_schemes")
            and self.collector._weight_sync_schemes
        ):
            if weight_update_map is None:
                raise ValueError(
                    "Collector has weight_sync_schemes configured, but weight_update_map was not provided. "
                    f"Please provide a mapping for all destinations: {list(self.collector._weight_sync_schemes.keys())}"
                )

            scheme_destinations = set(self.collector._weight_sync_schemes.keys())
            map_destinations = set(weight_update_map.keys())

            if scheme_destinations != map_destinations:
                missing = scheme_destinations - map_destinations
                extra = map_destinations - scheme_destinations
                error_msg = "weight_update_map does not match collector's weight_sync_schemes.\n"
                if missing:
                    error_msg += f"  Missing destinations: {missing}\n"
                if extra:
                    error_msg += f"  Extra destinations: {extra}\n"
                raise ValueError(error_msg)

            update_weights = UpdateWeights(
                self.collector,
                1,
                weight_update_map=weight_update_map,
                trainer=self,
            )
        else:
            if weight_update_map is not None:
                warnings.warn(
                    "weight_update_map was provided but collector has no weight_sync_schemes. "
                    "Ignoring weight_update_map and using legacy policy_weights_getter.",
                    UserWarning,
                    stacklevel=2,
                )

            policy_weights_getter = partial(
                TensorDict.from_module, self.loss_module.actor_network
            )
            update_weights = UpdateWeights(
                self.collector, 1, policy_weights_getter=policy_weights_getter
            )

        self.register_op("post_steps", update_weights)

        self.enable_logging = enable_logging
        self.log_rewards = log_rewards
        self.log_actions = log_actions
        self.log_observations = log_observations

        if self.enable_logging:
            self._setup_reinforce_logging()

    def _setup_reinforce_logging(self):
        """Set up logging hooks for REINFORCE-specific metrics.

        This method configures logging for common REINFORCE metrics including:
        - Training rewards (mean and std)
        - Action statistics (norms)
        - Episode completion rates
        - Observation statistics (optional)
        """
        log_done_percentage = LogScalar(
            key=("next", "done"),
            logname="done_percentage",
            log_pbar=True,
            include_std=False,
            reduction="mean",
        )
        if not self.async_collection:
            self.register_op("pre_steps_log", log_done_percentage)
        else:
            self.register_op("post_optim_log", log_done_percentage)

        if self.log_rewards:
            log_rewards = LogScalar(
                key=("next", "reward"),
                logname="r_training",
                log_pbar=True,
                include_std=True,
                reduction="mean",
            )
            if not self.async_collection:
                self.register_op("pre_steps_log", log_rewards)
            else:
                self.register_op("post_optim_log", log_rewards)

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

        if self.log_actions:
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

        if self.log_observations:
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
