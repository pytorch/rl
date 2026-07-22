# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import pathlib
import warnings

from collections.abc import Callable

from functools import partial
from typing import Any, Literal

from tensordict import TensorDict, TensorDictBase
from tensordict.nn import TensorDictSequential
from tensordict.utils import NestedKey
from torch import nn, optim

from torchrl.checkpoint import Checkpoint
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


class DDPGTrainer(Trainer):
    """A trainer class for Deep Deterministic Policy Gradient (DDPG) algorithm.

    See also :class:`~torchrl.trainers.algorithms.configs.DDPGTrainerConfig` for the
    Hydra configuration counterpart.

    This trainer implements the DDPG algorithm, an off-policy actor-critic method
    that learns a deterministic policy for continuous action spaces.

    The trainer handles:
    - Replay buffer management for off-policy learning
    - Target network updates (typically SoftUpdate) for stable training
    - Policy weight updates to the data collector
    - Comprehensive logging of training metrics

    Args:
        collector (BaseCollector): The data collector used to gather environment interactions.
        total_frames (int): Total number of frames to collect during training.
        frame_skip (int): Number of frames to skip between policy updates.
        optim_steps_per_batch (int): Number of optimization steps per collected batch.
        loss_module (LossModule | Callable): The DDPG loss module.
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
        batch_size (int, optional): Global learner batch size. Defaults to the
            replay buffer batch size.
        learner_backend (str): Optimization placement, ``"local"`` or ``"ray"``.
        learner_backend_options (dict, optional): Ray world size and resources.
        learner_poll_interval (float): Remote replay polling interval.
        enable_logging (bool, optional): Whether to enable metric logging. Defaults to True.
        log_rewards (bool, optional): Whether to log reward statistics. Defaults to True.
        log_actions (bool, optional): Whether to log action statistics. Defaults to True.
        log_observations (bool, optional): Whether to log observation statistics. Defaults to False.
        target_net_updater (TargetNetUpdater): Target network updater (typically
            :class:`~torchrl.objectives.utils.SoftUpdate`).
        exploration_module (nn.Module, optional): Exploration module appended to
            the deterministic actor for collection. The actor without this module
            is used for the DDPG loss. Defaults to ``None``.
        async_collection (bool, optional): Whether to use async data collection. Defaults to False.
        log_timings (bool, optional): Whether to log timing information for hooks. Defaults to False.
        done_key (NestedKey, optional): Done key used by losses and logging. Defaults to "done".
        terminated_key (NestedKey, optional): Terminated key used by losses and logging. Defaults to "terminated".
        reward_key (NestedKey, optional): Reward key used by losses and logging. Defaults to "reward".
        episode_reward_key (NestedKey, optional): Episode reward key used for cumulative reward logging.
            Defaults to "reward_sum".
        action_key (NestedKey, optional): Action key used by losses and logging. Defaults to "action".
        observation_key (NestedKey, optional): Observation key used for logging. Defaults to "observation".

    Note:
        This is an experimental/prototype feature. The API may change in future versions.
        DDPG is designed for continuous action spaces. For discrete actions, use DQNTrainer.
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
        checkpoint: Checkpoint | None = None,
        replay_buffer: ReplayBuffer | None = None,
        batch_size: int | None = None,
        learner_backend: Literal["local", "ray"] = "local",
        learner_backend_options: dict[str, Any] | None = None,
        learner_poll_interval: float = 0.05,
        enable_logging: bool = True,
        log_rewards: bool = True,
        log_actions: bool = True,
        log_observations: bool = False,
        target_net_updater: TargetNetUpdater | None = None,
        async_collection: bool = False,
        log_timings: bool = False,
        auto_log_optim_steps: bool = True,
        done_key: NestedKey = "done",
        terminated_key: NestedKey = "terminated",
        reward_key: NestedKey = "reward",
        episode_reward_key: NestedKey = "reward_sum",
        action_key: NestedKey = "action",
        observation_key: NestedKey = "observation",
        exploration_module: nn.Module | None = None,
    ) -> None:
        warnings.warn(
            "DDPGTrainer is an experimental/prototype feature. The API may change in future versions. "
            "Please report any issues or feedback to help improve this implementation.",
            UserWarning,
            stacklevel=2,
        )
        if target_net_updater is None:
            raise ValueError("DDPGTrainer requires a target_net_updater.")
        if learner_backend == "ray" and async_collection and enable_logging:
            raise ValueError(
                "DDPGTrainer cannot run batch logging hooks with asynchronous "
                "collection and learner_backend='ray'; set enable_logging=False."
            )
        super().__init__(
            collector=collector,
            total_frames=total_frames,
            frame_skip=frame_skip,
            optim_steps_per_batch=optim_steps_per_batch,
            loss_module=loss_module,
            optimizer=optimizer,
            replay_buffer=replay_buffer,
            target_net_updater=target_net_updater,
            batch_size=batch_size,
            learner_backend=learner_backend,
            learner_backend_options=learner_backend_options,
            learner_poll_interval=learner_poll_interval,
            logger=logger,
            clip_grad_norm=clip_grad_norm,
            clip_norm=clip_norm,
            progress_bar=progress_bar,
            seed=seed,
            save_trainer_interval=save_trainer_interval,
            log_interval=log_interval,
            save_trainer_file=save_trainer_file,
            checkpoint=checkpoint,
            async_collection=async_collection,
            log_timings=log_timings,
            auto_log_optim_steps=auto_log_optim_steps,
        )
        self.replay_buffer = replay_buffer
        self.async_collection = async_collection

        if replay_buffer is not None and learner_backend == "local":
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

        self.target_net_updater = target_net_updater
        if learner_backend == "local":
            self.register_op("post_optim", TargetNetUpdaterHook(target_net_updater))

        self.exploration_module = exploration_module

        if learner_backend == "local":
            weights_source = self.loss_module.actor_network
            if exploration_module is not None:
                weights_source = TensorDictSequential(
                    weights_source, exploration_module
                )
            policy_weights_getter = partial(TensorDict.from_module, weights_source)
            update_weights = UpdateWeights(
                self.collector, 1, policy_weights_getter=policy_weights_getter
            )
            self.register_op("post_steps", update_weights)

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

        if hasattr(self.loss_module, "set_keys"):
            self.loss_module.set_keys(
                reward=reward_key,
                done=done_key,
                terminated=terminated_key,
            )

        if self.enable_logging:
            self._setup_ddpg_logging()

    def _execution_weight_publication(
        self,
    ) -> tuple[NestedKey | None, TensorDictBase | None]:
        return self._compose_execution_weight_publication(self.exploration_module)

    def _setup_ddpg_logging(self):
        """Set up logging hooks for DDPG-specific metrics."""
        hook_dest = "pre_steps_log" if not self.async_collection else "post_optim_log"

        log_done_percentage = LogScalar(
            key=("next", self.done_key),
            logname="done_percentage",
            log_pbar=True,
            include_std=False,
            reduction="mean",
        )
        self.register_op(hook_dest, log_done_percentage)

        if self.log_rewards:
            log_rewards = LogScalar(
                key=("next", self.reward_key),
                logname="r_training",
                log_pbar=True,
                include_std=True,
                reduction="mean",
            )
            log_max_reward = LogScalar(
                key=("next", self.reward_key),
                logname="r_max",
                log_pbar=False,
                include_std=False,
                reduction="max",
            )
            log_total_reward = LogScalar(
                key=("next", self.episode_reward_key),
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
                key=self.action_key,
                logname="action_norm",
                log_pbar=False,
                include_std=True,
                reduction="mean",
            )
            self.register_op(hook_dest, log_action_norm)

        if self.log_observations:
            log_obs_norm = LogScalar(
                key=self.observation_key,
                logname="obs_norm",
                log_pbar=False,
                include_std=True,
                reduction="mean",
            )
            self.register_op(hook_dest, log_obs_norm)

        self.register_op("pre_steps_log", UTDRHook(self))
