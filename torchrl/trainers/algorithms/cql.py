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
    DefaultOptimizationStepper,
    LogScalar,
    ReplayBufferTrainer,
    TargetNetUpdaterHook,
    Trainer,
    UpdateWeights,
    UTDRHook,
)


class CQLTrainer(Trainer):
    """A trainer class for Conservative Q-Learning (CQL) algorithm.

    This trainer implements the CQL algorithm, an off-policy actor-critic method
    that adds a conservative penalty to Q-value estimates to prevent overestimation
    of out-of-distribution actions. CQL is particularly effective for offline RL.

    The trainer handles:
    - Replay buffer management for off-policy learning
    - Target network updates (SoftUpdate) for stable training
    - Policy weight updates to the data collector
    - Action spec passthrough from collector environment to loss module
    - Comprehensive logging of training metrics

    Args:
        collector (BaseCollector): The data collector used to gather environment interactions.
        total_frames (int): Total number of frames to collect during training.
        frame_skip (int): Number of frames to skip between policy updates.
        optim_steps_per_batch (int): Number of optimization steps per collected batch.
        loss_module (LossModule | Callable): The CQL loss module.
        optimizer (optim.Optimizer, optional): The optimizer for training.
        logger (Logger, optional): Logger for recording training metrics. Defaults to None.
        clip_grad_norm (bool, optional): Whether to clip gradient norms. Defaults to True.
        clip_norm (float, optional): Maximum gradient norm for clipping. Defaults to None.
        progress_bar (bool, optional): Whether to show a progress bar. Defaults to True.
        seed (int, optional): Random seed for reproducibility. Defaults to None.
        save_trainer_interval (int, optional): Interval for saving trainer state. Defaults to 10000.
        log_interval (int, optional): Interval for logging metrics. Defaults to 10000.
        save_trainer_file (str | pathlib.Path, optional): File path for saving trainer state.
        replay_buffer (ReplayBuffer, optional): Replay buffer for storing experiences. Defaults to None.
        enable_logging (bool, optional): Whether to enable metric logging. Defaults to True.
        log_rewards (bool, optional): Whether to log reward statistics. Defaults to True.
        log_actions (bool, optional): Whether to log action statistics. Defaults to True.
        log_observations (bool, optional): Whether to log observation statistics. Defaults to False.
        target_net_updater (TargetNetUpdater, optional): Target network updater (typically SoftUpdate).
        async_collection (bool, optional): Whether to use async data collection. Defaults to False.
        log_timings (bool, optional): Whether to log timing information for hooks. Defaults to False.

    Note:
        This is an experimental/prototype feature. The API may change in future versions.
        CQL is designed for offline RL but also works online. The conservative penalty
        prevents overestimation of Q-values for unseen actions.
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
        enable_logging: bool = True,
        log_rewards: bool = True,
        log_actions: bool = True,
        log_observations: bool = False,
        target_net_updater: TargetNetUpdater | None = None,
        async_collection: bool = False,
        log_timings: bool = False,
    ) -> None:
        warnings.warn(
            "CQLTrainer is an experimental/prototype feature. The API may change in future versions. "
            "Please report any issues or feedback to help improve this implementation.",
            UserWarning,
            stacklevel=2,
        )
        self._pass_action_spec_from_collector_to_loss(collector, loss_module)

        loss_components = [
            "loss_actor",
            "loss_qvalue",
            "loss_cql",
            "loss_alpha",
        ]
        if getattr(loss_module, "with_lagrange", False):
            loss_components.append("loss_alpha_prime")
        optimization_stepper = DefaultOptimizationStepper(
            loss_components=loss_components,
        )

        super().__init__(
            collector=collector,
            total_frames=total_frames,
            frame_skip=frame_skip,
            optim_steps_per_batch=optim_steps_per_batch,
            loss_module=loss_module,
            optimizer=optimizer,
            optimization_stepper=optimization_stepper,
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

        self.enable_logging = enable_logging
        self.log_rewards = log_rewards
        self.log_actions = log_actions
        self.log_observations = log_observations

        if self.enable_logging:
            self._setup_cql_logging()

    def _pass_action_spec_from_collector_to_loss(
        self, collector: BaseCollector, loss: LossModule
    ):
        """Pass the action specification from the collector's environment to the loss module."""
        if hasattr(loss, "_action_spec") and loss._action_spec is None:
            action_spec = collector.getattr_env("full_action_spec_unbatched").cpu()
            loss._action_spec = action_spec

    def _setup_cql_logging(self):
        """Set up logging hooks for CQL-specific metrics."""
        hook_dest = "pre_steps_log" if not self.async_collection else "post_optim_log"

        log_done_percentage = LogScalar(
            key=("next", "done"),
            logname="done_percentage",
            log_pbar=True,
            include_std=False,
            reduction="mean",
        )
        self.register_op(hook_dest, log_done_percentage)

        if self.log_rewards:
            log_rewards = LogScalar(
                key=("next", "reward"),
                logname="r_training",
                log_pbar=True,
                include_std=True,
                reduction="mean",
            )
            log_max_reward = LogScalar(
                key=("next", "reward"),
                logname="r_max",
                log_pbar=False,
                include_std=False,
                reduction="max",
            )
            log_total_reward = LogScalar(
                key=("next", "reward_sum"),
                logname="r_total",
                log_pbar=False,
                include_std=False,
                reduction="max",
            )
            self.register_op(hook_dest, log_rewards)
            self.register_op(hook_dest, log_max_reward)
            self.register_op(hook_dest, log_total_reward)

        if self.log_actions:
            log_action_norm = LogScalar(
                key="action",
                logname="action_norm",
                log_pbar=False,
                include_std=True,
                reduction="mean",
            )
            self.register_op(hook_dest, log_action_norm)

        if self.log_observations:
            log_obs_norm = LogScalar(
                key="observation",
                logname="obs_norm",
                log_pbar=False,
                include_std=True,
                reduction="mean",
            )
            self.register_op(hook_dest, log_obs_norm)

        self.register_op("pre_steps_log", UTDRHook(self))
