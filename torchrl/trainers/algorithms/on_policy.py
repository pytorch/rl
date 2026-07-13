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
from tensordict.utils import NestedKey
from torch import optim

from torchrl.checkpoint import Checkpoint
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


class OnPolicyTrainer(Trainer):
    """Shared implementation for on-policy trainers (PPO, A2C, REINFORCE).

    .. warning::
        This is an experimental/prototype feature. The API may change in future versions.
        Please report any issues or feedback to help improve this implementation.

    This class hosts the training-loop wiring common to on-policy algorithms:
    advantage estimation (GAE by default, registered through
    :class:`~torchrl.trainers.ValueEstimatorHook`), replay-buffer handling,
    collector weight synchronization, optional learning-rate scheduling
    (through :class:`~torchrl.trainers.LRSchedulerHook`) and standard logging
    hooks. Concrete algorithms (:class:`~torchrl.trainers.algorithms.PPOTrainer`,
    :class:`~torchrl.trainers.algorithms.A2CTrainer`,
    :class:`~torchrl.trainers.algorithms.ReinforceTrainer`) subclass it and only
    override class-level defaults such as the number of epochs per batch.

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
        num_epochs (int, optional): Number of epochs per batch. Defaults to the
            algorithm-specific class default (e.g. 4 for PPO, 1 for A2C and REINFORCE).
        replay_buffer (ReplayBuffer, optional): Replay buffer for storing data.
        batch_size (int, optional): Unused; on-policy sub-batch sizes are driven by
            the replay buffer's own ``batch_size``. Passing a value emits a warning.
        gamma (float, optional): Discount factor for GAE. Default: 0.99.
        lmbda (float, optional): Lambda parameter for GAE. Default: 0.95.
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
        auto_log_optim_steps (bool, optional): If True, log the number of optimization
            steps after each optimization loop. Default: True.
        done_key (NestedKey, optional): Done key used by GAE, losses, and logging. Default: "done".
        terminated_key (NestedKey, optional): Terminated key used by GAE, losses, and logging.
            Default: "terminated".
        reward_key (NestedKey, optional): Reward key used by GAE, losses, and logging. Default: "reward".
        episode_reward_key (NestedKey, optional): Episode reward key used for cumulative reward logging.
            Default: "reward".
        action_key (NestedKey, optional): Action key used by losses and logging. Default: "action".
        observation_key (NestedKey, optional): Observation key used for logging. Default: "observation".
    """

    # Overridden by subclasses: name used in warnings and number of epochs used
    # when ``num_epochs`` is not provided.
    _algo_name: str = "on-policy"
    _default_num_epochs: int = 1

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
        checkpoint: Checkpoint | None = None,
        num_epochs: int | None = None,
        replay_buffer: ReplayBuffer | None = None,
        batch_size: int | None = None,
        gamma: float = 0.99,
        lmbda: float = 0.95,
        enable_logging: bool = True,
        log_rewards: bool = True,
        log_actions: bool = True,
        log_observations: bool = False,
        async_collection: bool = False,
        add_gae: bool = True,
        gae: Callable[[TensorDictBase], TensorDictBase] | None = None,
        weight_update_map: dict[str, str] | None = None,
        log_timings: bool = False,
        auto_log_optim_steps: bool = True,
        done_key: NestedKey = "done",
        terminated_key: NestedKey = "terminated",
        reward_key: NestedKey = "reward",
        episode_reward_key: NestedKey = "reward",
        action_key: NestedKey = "action",
        observation_key: NestedKey = "observation",
    ) -> None:
        warnings.warn(
            f"{type(self).__name__} is an experimental/prototype feature. The API may "
            "change in future versions. Please report any issues or feedback to help "
            "improve this implementation.",
            UserWarning,
            stacklevel=2,
        )
        if num_epochs is None:
            num_epochs = self._default_num_epochs
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
            checkpoint=checkpoint,
            num_epochs=num_epochs,
            async_collection=async_collection,
            log_timings=log_timings,
            auto_log_optim_steps=auto_log_optim_steps,
        )
        self.replay_buffer = replay_buffer
        self.async_collection = async_collection

        if batch_size is not None:
            warnings.warn(
                "batch_size is unused by on-policy trainers: sub-batch sizes are "
                "driven by the replay buffer's own batch_size. Set the batch size "
                "on the replay buffer instead.",
                UserWarning,
                stacklevel=2,
            )

        if add_gae and gae is None:
            gae = GAE(
                gamma=gamma,
                lmbda=lmbda,
                value_network=self.loss_module.critic_network,
                average_gae=True,
            )
        elif not add_gae and gae is not None:
            raise ValueError("gae must not be provided if add_gae is False")

        if add_gae:
            if hasattr(gae, "set_keys"):
                gae.set_keys(
                    reward=reward_key,
                    done=done_key,
                    terminated=terminated_key,
                )
            ValueEstimatorHook(gae).register(self)

        if lr_scheduler is not None:
            LRSchedulerHook(lr_scheduler).register(self)

        if hasattr(self.loss_module, "set_keys"):
            self.loss_module.set_keys(
                reward=reward_key,
                done=done_key,
                terminated=terminated_key,
                action=action_key,
            )

        if (
            not self.async_collection
            and replay_buffer is not None
            and not isinstance(replay_buffer.sampler, SamplerWithoutReplacement)
        ):
            warnings.warn(
                "Sampler is not a SamplerWithoutReplacement, which is required "
                f"for {self._algo_name}."
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
                # rb has been extended by the collector
                self.register_op("pre_epoch", rb_trainer.extend)
            self.register_op("process_optim_batch", rb_trainer.sample)
            self.register_op("post_loss", rb_trainer.update_priority)

        # Set up weight updates
        # Validate weight_update_map if collector has weight_sync_schemes
        if (
            hasattr(self.collector, "_weight_sync_schemes")
            and self.collector._weight_sync_schemes
        ):
            if weight_update_map is None:
                raise ValueError(
                    "Collector has weight_sync_schemes configured, but weight_update_map was not provided. "
                    f"Please provide a mapping for all destinations: {list(self.collector._weight_sync_schemes.keys())}"
                )

            # Validate that all scheme destinations are covered in the map
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

            # Use the weight_update_map approach
            update_weights = UpdateWeights(
                self.collector,
                1,
                weight_update_map=weight_update_map,
                trainer=self,
            )
        else:
            # Fall back to legacy approach for backward compatibility
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

        # Store logging configuration
        self.enable_logging = enable_logging
        self.log_rewards = log_rewards
        self.log_actions = log_actions
        self.log_observations = log_observations
        self.done_key = done_key
        self.terminated_key = terminated_key
        self.reward_key = reward_key
        self.episode_reward_key = episode_reward_key
        self.action_key = action_key
        self.observation_key = observation_key

        # Set up comprehensive logging for on-policy training
        if self.enable_logging:
            self._setup_logging()

    def _setup_logging(self):
        """Set up logging hooks for on-policy training metrics.

        This method configures logging for common on-policy metrics including:
        - Training rewards (mean and std)
        - Action statistics (norms)
        - Episode completion rates
        - Observation statistics (optional)
        """
        # Logging hooks read the collected batch, which is only available at
        # the pre_steps_log stage in synchronous mode; in async mode the batch
        # is None there, so hooks run on the optimization sub-batches instead.
        log_dest = "pre_steps_log" if not self.async_collection else "post_optim_log"

        # Always log done states as percentage (episode completion rate)
        log_done_percentage = LogScalar(
            key=("next", self.done_key),
            logname="done_percentage",
            log_pbar=True,
            include_std=False,  # No std for binary values
            reduction="mean",
        )
        self.register_op(log_dest, log_done_percentage)

        # Log rewards if enabled
        if self.log_rewards:
            # 1. Log training rewards (most important on-policy metric)
            log_rewards = LogScalar(
                key=("next", self.reward_key),
                logname="r_training",
                log_pbar=True,  # Show in progress bar
                include_std=True,
                reduction="mean",
            )
            self.register_op(log_dest, log_rewards)

            # 2. Log maximum reward in batch (for monitoring best performance)
            log_max_reward = LogScalar(
                key=("next", self.reward_key),
                logname="r_max",
                log_pbar=False,
                include_std=False,
                reduction="max",
            )
            self.register_op(log_dest, log_max_reward)

            # 3. Log total reward in batch (for monitoring cumulative performance)
            log_total_reward = LogScalar(
                key=("next", self.episode_reward_key),
                logname="r_total",
                log_pbar=False,
                include_std=False,
                reduction="sum",
            )
            self.register_op(log_dest, log_total_reward)

        # Log actions if enabled
        if self.log_actions:
            # 4. Log action norms (useful for monitoring policy behavior)
            log_action_norm = LogScalar(
                key=self.action_key,
                logname="action_norm",
                log_pbar=False,
                include_std=True,
                reduction="mean",
            )
            self.register_op(log_dest, log_action_norm)

        # Log observations if enabled
        if self.log_observations:
            # 5. Log observation statistics (for monitoring state distributions)
            log_obs_norm = LogScalar(
                key=self.observation_key,
                logname="obs_norm",
                log_pbar=False,
                include_std=True,
                reduction="mean",
            )
            self.register_op(log_dest, log_obs_norm)
