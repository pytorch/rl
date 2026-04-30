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

from torchrl.collectors import BaseCollector

from torchrl.data.replay_buffers.replay_buffers import ReplayBuffer
from torchrl.modules import EGreedyModule
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


class DQNTrainer(Trainer):
    """A trainer class for Deep Q-Network (DQN) algorithm.

    See also :class:`~torchrl.trainers.algorithms.configs.DQNTrainerConfig` for the
    Hydra configuration counterpart.

    This trainer implements the DQN algorithm, a value-based method for discrete
    action spaces that learns a Q-function and derives a greedy policy from it.

    The trainer handles:
    - Replay buffer management for off-policy learning
    - Target network updates (typically HardUpdate) with configurable update frequency
    - Policy weight updates to the data collector
    - Comprehensive logging of training metrics

    Args:
        collector (BaseCollector): The data collector used to gather environment interactions.
        total_frames (int): Total number of frames to collect during training.
        frame_skip (int): Number of frames to skip between policy updates.
        optim_steps_per_batch (int): Number of optimization steps per collected batch.
        loss_module (LossModule | Callable): The DQN loss module or a callable that computes losses.
        optimizer (optim.Optimizer, optional): The optimizer for training.
        logger (Logger, optional): Logger for recording training metrics. Defaults to None.
        clip_grad_norm (bool, optional): Whether to clip gradient norms. Defaults to True.
        clip_norm (float, optional): Maximum gradient norm for clipping. Defaults to None.
        progress_bar (bool, optional): Whether to show a progress bar during training. Defaults to True.
        seed (int, optional): Random seed for reproducibility. Defaults to None.
        save_trainer_interval (int, optional): Interval for saving trainer state. Defaults to 10000.
        log_interval (int, optional): Interval for logging metrics. Defaults to 10000.
        save_trainer_file (str | pathlib.Path, optional): File path for saving trainer state. Defaults to None.
        replay_buffer (ReplayBuffer, optional): Replay buffer for storing and sampling experiences. Defaults to None.
        enable_logging (bool, optional): Whether to enable metric logging. Defaults to True.
        log_rewards (bool, optional): Whether to log reward statistics. Defaults to True.
        log_observations (bool, optional): Whether to log observation statistics. Defaults to False.
        target_net_updater (TargetNetUpdater, optional): Target network updater (typically HardUpdate). Defaults to None.
        greedy_module (EGreedyModule, optional): Epsilon-greedy exploration module. When provided,
            the module's epsilon is annealed during training. Defaults to None.
        async_collection (bool, optional): Whether to use async data collection. Defaults to False.
        log_timings (bool, optional): Whether to log timing information for hooks. Defaults to False.
        mixing_strategy (str, optional): Multi-agent mixing strategy. Accepted values are ``"qmix"`` and
            ``"vdn"`` for mixed-value training, ``"iql"`` for independent Q-learning, or None for standard
            DQN. Defaults to None.
        done_key (NestedKey, optional): Key for the done signal used by logging. Defaults to ``"done"``.
        terminated_key (NestedKey, optional): Key for the terminated signal. Defaults to ``"terminated"``.
        reward_key (NestedKey, optional): Source reward key used by logging and reward aggregation.
            Defaults to ``"reward"``.
        episode_reward_key (NestedKey, optional): Source episode reward key used by logging and reward
            aggregation. Defaults to ``"reward_sum"``.
        aggregated_reward_key (NestedKey, optional): Destination key for rewards averaged over the agent
            dimension when using QMIX or VDN. The source is ``reward_key``. Set this to ``reward_key`` to
            overwrite the source reward in-place. Required when ``mixing_strategy`` is ``"qmix"`` or
            ``"vdn"``. Defaults to None.
        aggregated_episode_reward_key (NestedKey, optional): Destination key for episode rewards averaged over
            the agent dimension when using QMIX or VDN. The source is ``episode_reward_key``. Set this to
            ``episode_reward_key`` to overwrite the source reward in-place. Required when ``mixing_strategy``
            is ``"qmix"`` or ``"vdn"``. Defaults to None.
        action_key (NestedKey, optional): Key for actions used by the exploration module and policy specs.
            Defaults to ``"action"``.
        observation_key (NestedKey, optional): Key for observations used by logging. Defaults to
            ``"observation"``.

    Example:
        >>> from torchrl.collectors import Collector
        >>> from torchrl.objectives import DQNLoss
        >>> from torchrl.data import ReplayBuffer, LazyTensorStorage
        >>> from torchrl.objectives.utils import HardUpdate
        >>> from torch import optim
        >>>
        >>> # Set up collector, loss, and replay buffer
        >>> collector = Collector(env, policy, frames_per_batch=128)
        >>> loss_module = DQNLoss(value_network, delay_value=True)
        >>> optimizer = optim.Adam(loss_module.parameters(), lr=2.5e-4)
        >>> replay_buffer = ReplayBuffer(storage=LazyTensorStorage(100000))
        >>> target_net_updater = HardUpdate(loss_module, value_network_update_interval=50)
        >>>
        >>> trainer = DQNTrainer(
        ...     collector=collector,
        ...     total_frames=500000,
        ...     frame_skip=1,
        ...     optim_steps_per_batch=10,
        ...     loss_module=loss_module,
        ...     optimizer=optimizer,
        ...     replay_buffer=replay_buffer,
        ...     target_net_updater=target_net_updater,
        ... )
        >>> trainer.train()

    Note:
        This is an experimental/prototype feature. The API may change in future versions.
        DQN is designed for discrete action spaces (e.g., CartPole, Atari).
        For continuous control, consider using SACTrainer or DDPGTrainer instead.
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
        log_observations: bool = False,
        target_net_updater: TargetNetUpdater | None = None,
        greedy_module: EGreedyModule | None = None,
        async_collection: bool = False,
        log_timings: bool = False,
        auto_log_optim_steps: bool = True,
        mixing_strategy: str | None = None,
        done_key: NestedKey = "done",
        terminated_key: NestedKey = "terminated",
        reward_key: NestedKey = "reward",
        episode_reward_key: NestedKey = "reward_sum",
        aggregated_reward_key: NestedKey | None = None,
        aggregated_episode_reward_key: NestedKey | None = None,
        action_key: NestedKey = "action",
        observation_key: NestedKey = "observation",
    ) -> None:
        warnings.warn(
            "DQNTrainer is an experimental/prototype feature. The API may change in future versions. "
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
            async_collection=async_collection,
            log_timings=log_timings,
            auto_log_optim_steps=auto_log_optim_steps,
        )
        self.replay_buffer = replay_buffer
        self.async_collection = async_collection
        self.mixing_strategy = mixing_strategy
        self.done_key = done_key
        self.terminated_key = terminated_key
        self.reward_key = reward_key
        self.episode_reward_key = episode_reward_key
        self.aggregated_reward_key = aggregated_reward_key
        self.aggregated_episode_reward_key = aggregated_episode_reward_key
        self.action_key = action_key
        self.observation_key = observation_key

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

        self.greedy_module = greedy_module
        if hasattr(self.loss_module, "value_network"):
            weights_source = self.loss_module.value_network
        elif hasattr(self.loss_module, "local_value_network"):
            weights_source = self.loss_module.local_value_network
        else:
            raise AttributeError(
                "loss_module must expose either `value_network` or `local_value_network` "
                "to sync policy weights with the collector."
            )
        if greedy_module is not None:
            from tensordict.nn import TensorDictSequential

            weights_source = TensorDictSequential(weights_source, greedy_module)
            self._greedy_last_frames = 0
            self.register_op("post_steps", self._step_greedy)

        policy_weights_getter = partial(TensorDict.from_module, weights_source)
        update_weights = UpdateWeights(
            self.collector, 1, policy_weights_getter=policy_weights_getter
        )
        self.register_op("post_steps", update_weights)

        self.enable_logging = enable_logging
        self.log_rewards = log_rewards
        self.log_observations = log_observations
        if self.mixing_strategy in ("qmix", "vdn"):
            if (
                self.aggregated_reward_key is None
                or self.aggregated_episode_reward_key is None
            ):
                raise ValueError(
                    "aggregated_reward_key and aggregated_episode_reward_key must be "
                    f"provided when mixing_strategy is {self.mixing_strategy}."
                )
            self.register_op("batch_process", self._aggregate_agent_rewards)

        if self.enable_logging:
            self._setup_dqn_logging()

    def _step_greedy(self):
        """Advance epsilon-greedy annealing by the number of frames collected since last call."""
        delta = self.collected_frames - self._greedy_last_frames
        if delta > 0:
            self.greedy_module.step(delta)
            self._greedy_last_frames = self.collected_frames

    def _aggregate_agent_rewards(self, batch: TensorDictBase) -> TensorDictBase:
        for key, aggregated_key in (
            (self.reward_key, self.aggregated_reward_key),
            (self.episode_reward_key, self.aggregated_episode_reward_key),
        ):
            value = batch.get(("next", key), None)
            if value is not None:
                batch.set(("next", aggregated_key), value.mean(-2))
        return batch

    def _setup_dqn_logging(self):
        """Set up logging hooks for DQN-specific metrics."""
        log_done_percentage = LogScalar(
            key=("next", self.done_key),
            logname="done_percentage",
            log_pbar=True,
            include_std=False,
            reduction="mean",
        )
        hook_dest = "pre_steps_log" if not self.async_collection else "post_optim_log"
        self.register_op(hook_dest, log_done_percentage)

        if self.log_rewards:
            if self.mixing_strategy in ("qmix", "vdn"):
                reward_log_key = self.aggregated_reward_key
                episode_reward_log_key = self.aggregated_episode_reward_key
            else:
                reward_log_key = self.reward_key
                episode_reward_log_key = self.episode_reward_key
            log_rewards = LogScalar(
                key=("next", reward_log_key),
                logname="r_training",
                log_pbar=True,
                include_std=True,
                reduction="mean",
            )
            log_max_reward = LogScalar(
                key=("next", reward_log_key),
                logname="r_max",
                log_pbar=False,
                include_std=False,
                reduction="max",
            )
            log_total_reward = LogScalar(
                key=("next", episode_reward_log_key),
                logname="r_total",
                log_pbar=False,
                include_std=False,
                reduction="max",
            )
            self.register_op(hook_dest, log_rewards)
            self.register_op(hook_dest, log_max_reward)
            self.register_op(hook_dest, log_total_reward)

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
