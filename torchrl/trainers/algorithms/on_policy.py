# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import pathlib
import warnings

from collections.abc import Callable, Mapping

from functools import partial
from typing import Any, Literal

import torch
from tensordict import TensorDict, TensorDictBase
from tensordict.utils import NestedKey
from torch import optim

from torchrl._utils import timeit
from torchrl.checkpoint import Checkpoint, CheckpointRotation
from torchrl.collectors import BaseCollector

from torchrl.data.replay_buffers.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.objectives.common import LossModule
from torchrl.objectives.utils import TargetNetUpdater
from torchrl.objectives.value.advantages import GAE
from torchrl.record.loggers import Logger
from torchrl.trainers.trainers import (
    LogScalar,
    LRSchedulerHook,
    ReplayBufferTrainer,
    TargetNetUpdaterHook,
    Trainer,
    UpdateWeights,
    ValueEstimatorHook,
)


def _next_key(key: NestedKey) -> NestedKey:
    if isinstance(key, tuple):
        return ("next", *key)
    return ("next", key)


def _sibling_key(key: NestedKey, sibling: str) -> NestedKey:
    if isinstance(key, tuple):
        return (*key[:-1], sibling)
    return sibling


class _OnPolicyTelemetry:
    """Compute optional on-policy diagnostics outside the minimal logging path."""

    def __init__(self, trainer: OnPolicyTrainer):
        self.trainer = trainer
        self._last_collected_frames = trainer.collected_frames
        self._optim_start_count = trainer._optim_count
        self._collection_timer = timeit("on_policy/collection").start()
        self._optimization_timer = timeit("on_policy/optimization").start()

    def setup(self) -> None:
        self._last_collected_frames = self.trainer.collected_frames
        self._optim_start_count = self.trainer._optim_count
        self._collection_timer.start()

    def start_collection(self) -> None:
        self._collection_timer.start()

    @staticmethod
    def _masked(batch: TensorDictBase, key: NestedKey) -> torch.Tensor | None:
        if key not in batch.keys(True):
            return None
        value = batch.get(key)
        mask = batch.get(("collector", "mask"), None)
        if mask is not None:
            value = value[mask]
        return value

    @staticmethod
    def _scalar_per_transition(
        batch: TensorDictBase, value: torch.Tensor
    ) -> torch.Tensor | None:
        while value.ndim > batch.ndim and value.shape[-1] == 1:
            value = value.squeeze(-1)
        if value.ndim != batch.ndim:
            return None
        return value

    @staticmethod
    def _summary(prefix: str, value: torch.Tensor) -> dict[str, torch.Tensor]:
        value = value.float()
        return {
            f"{prefix}/min": value.min(),
            f"{prefix}/mean": value.mean(),
            f"{prefix}/std": value.std(unbiased=False),
            f"{prefix}/max": value.max(),
        }

    def _complete_episode_metrics(
        self,
        batch: TensorDictBase,
        reward: torch.Tensor,
        done: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        traj_ids = batch.get(("collector", "traj_ids"), None)
        is_init = batch.get("is_init", None)
        if traj_ids is None or is_init is None:
            return {}
        reward = self._scalar_per_transition(batch, reward)
        done = self._scalar_per_transition(batch, done)
        traj_ids = self._scalar_per_transition(batch, traj_ids)
        is_init = self._scalar_per_transition(batch, is_init)
        if reward is None or done is None or traj_ids is None or is_init is None:
            return {}
        mask = batch.get(("collector", "mask"), None)
        if mask is not None:
            reward = reward[mask]
            done = done[mask]
            traj_ids = traj_ids[mask]
            is_init = is_init[mask]
        reward = reward.reshape(-1)
        done = done.reshape(-1).bool()
        traj_ids = traj_ids.reshape(-1)
        is_init = is_init.reshape(-1).bool()
        episode_start = torch.ones_like(is_init)
        episode_start[1:] = is_init[1:] | (traj_ids[1:] != traj_ids[:-1])
        episode_ids = episode_start.cumsum(0) - 1
        episodes, inverse = episode_ids.unique_consecutive(return_inverse=True)
        num_trajectories = episodes.numel()
        returns = reward.new_zeros(num_trajectories).scatter_add_(0, inverse, reward)
        lengths = torch.zeros(
            num_trajectories, dtype=torch.long, device=inverse.device
        ).scatter_add_(0, inverse, torch.ones_like(inverse))
        starts = torch.zeros_like(lengths).scatter_add_(0, inverse, is_init.long())
        ends = torch.zeros_like(lengths).scatter_add_(0, inverse, done.long())
        complete = starts.bool() & ends.bool()
        if not complete.any():
            return {}
        returns = returns[complete]
        lengths = lengths[complete]
        return {
            **self._summary("episodes/return", returns),
            **self._summary("episodes/length", lengths),
        }

    @staticmethod
    def _flatten_stats(
        prefix: str, stats: Mapping[str, Any], metrics: dict[str, Any]
    ) -> None:
        for key, value in stats.items():
            name = f"{prefix}/{key}"
            if isinstance(value, Mapping):
                _OnPolicyTelemetry._flatten_stats(name, value, metrics)
            elif isinstance(value, torch.Tensor):
                if value.numel() == 1:
                    metrics.setdefault(name, value.detach())
            elif isinstance(value, (bool, int, float)):
                metrics.setdefault(name, value)

    def _target_stats(self, target: Any, prefix: str) -> dict[str, Any]:
        stats = getattr(target, "stats", None)
        if not callable(stats):
            return {}
        try:
            snapshot = stats()
        except (AttributeError, RuntimeError, TypeError):
            return {}
        if not isinstance(snapshot, Mapping):
            return {}
        metrics: dict[str, Any] = {}
        self._flatten_stats(prefix, snapshot, metrics)
        return metrics

    def batch_metrics(self, batch: TensorDictBase | None) -> None:
        trainer = self.trainer
        batch_frames = max(0, trainer.collected_frames - self._last_collected_frames)
        self._last_collected_frames = trainer.collected_frames
        metrics: dict[str, Any] = {
            "frames/collected": trainer.collected_frames,
            "frames/batch": batch_frames,
        }
        elapsed = self._collection_timer.elapsed()
        if elapsed > 0:
            metrics["throughput/collection_frames_per_second"] = batch_frames / elapsed

        if batch is not None:
            done_key = _next_key(trainer.done_key)
            done = self._masked(batch, done_key)
            if done is not None and done.numel():
                metrics["terminals/done_rate"] = done.float().mean()
                done_per_transition = batch.get(done_key)
                while done_per_transition.ndim > batch.ndim:
                    done_per_transition = done_per_transition.any(-1)
                mask = batch.get(("collector", "mask"), None)
                if mask is not None:
                    done_per_transition = done_per_transition[mask]
                metrics["episodes/completed"] = done_per_transition.sum()
            terminated = self._masked(batch, _next_key(trainer.terminated_key))
            if terminated is not None and terminated.numel():
                metrics["terminals/terminated_rate"] = terminated.float().mean()
            truncated = self._masked(
                batch,
                _next_key(_sibling_key(trainer.terminated_key, "truncated")),
            )
            if truncated is not None and truncated.numel():
                metrics["terminals/truncated_rate"] = truncated.float().mean()

            reward = self._masked(batch, _next_key(trainer.reward_key))
            if trainer.log_rewards and reward is not None and reward.numel():
                metrics.update(self._summary("rewards", reward))
                if done is not None:
                    unmasked_reward = batch.get(_next_key(trainer.reward_key))
                    unmasked_done = batch.get(done_key)
                    metrics.update(
                        self._complete_episode_metrics(
                            batch, unmasked_reward, unmasked_done
                        )
                    )

        metrics.update(self._target_stats(trainer.collector, "collector"))
        if trainer.replay_buffer is not None:
            replay_metrics = self._target_stats(trainer.replay_buffer, "replay")
            if "replay/size" not in replay_metrics:
                try:
                    replay_metrics["replay/size"] = len(trainer.replay_buffer)
                except (AttributeError, RuntimeError, TypeError):
                    pass
            storage = getattr(trainer.replay_buffer, "storage", None)
            capacity = getattr(storage, "max_size", None)
            if capacity is not None:
                replay_metrics.setdefault("replay/capacity", capacity)
            metrics.update(replay_metrics)

        self._optim_start_count = trainer._optim_count
        self._optimization_timer.start()
        trainer._log_standard(metrics)

    def replay_reward_metrics(self, batch: TensorDictBase) -> None:
        # Async collection has no learner-side collected batch. Summarize the
        # sampled rewards, but do not interpret replay slice ends as episode ends.
        reward = self._masked(batch, _next_key(self.trainer.reward_key))
        if reward is not None and reward.numel():
            self.trainer._log_standard(self._summary("rewards", reward))

    def optimization_metrics(
        self, optim_steps: int, average_losses: TensorDictBase | None
    ) -> None:
        metrics: dict[str, Any] = {}
        optimizer = self.trainer.optimizer
        if optimizer is not None and optimizer.param_groups:
            metrics["optimizer/learning_rate"] = optimizer.param_groups[0]["lr"]
            if len(optimizer.param_groups) > 1:
                for index, group in enumerate(optimizer.param_groups):
                    name = group.get("name", f"group_{index}")
                    metrics[f"optimizer/learning_rate/{name}"] = group["lr"]
        if average_losses is not None:
            grad_norms = [
                value.float().mean()
                for key, value in average_losses.flatten_keys(".").items()
                if str(key).split(".")[-1].startswith("grad_norm")
            ]
            if grad_norms:
                metrics["optimizer/gradient_norm"] = torch.stack(grad_norms).mean()
        elapsed = self._optimization_timer.elapsed()
        updates = optim_steps - self._optim_start_count
        if elapsed > 0 and updates > 0:
            metrics["throughput/optimizer_updates_per_second"] = updates / elapsed
        self.trainer._log_standard(metrics)

    def register(self) -> None:
        self.trainer.register_op("setup", self.setup)
        self.trainer.register_op("pre_steps_log", self.batch_metrics)
        if self.trainer.async_collection and self.trainer.log_rewards:
            self.trainer.register_op("post_optim_log", self.replay_reward_metrics)
        self.trainer.register_op("post_optim_complete_log", self.optimization_metrics)
        self.trainer.register_op("post_steps", self.start_collection)


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
        target_net_updater (TargetNetUpdater, optional): Target-parameter updater, stepped
            after every optimizer step via :class:`~torchrl.trainers.TargetNetUpdaterHook`.
            Pair it with a loss built with ``delay_actor=True`` (see
            :class:`~torchrl.objectives.ClipPPOLoss`) to maintain the proximal policy of
            PPO-EWMA: a :class:`~torchrl.objectives.SoftUpdate` turns it into an
            exponentially-weighted moving average of the policy. Default: ``None``.
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
        telemetry ("minimal" or "standard", optional): Diagnostic telemetry level.
            ``"minimal"`` preserves the legacy logging set and performs no
            additional metric collection. ``"standard"`` records frame,
            episode, terminal, reward, optimizer, throughput, collector and replay
            diagnostics under the ``training/`` logger namespace. Missing optional
            fields are omitted. Legacy reward and terminal metric aliases are
            emitted only in minimal mode. In async mode, reward summaries use
            replay samples; episode and terminal metrics require a collected
            batch and are omitted. Default: ``"standard"``.
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
        target_net_updater: TargetNetUpdater | None = None,
        logger: Logger | None = None,
        clip_grad_norm: bool = True,
        clip_norm: float | None = None,
        progress_bar: bool = True,
        seed: int | None = None,
        save_trainer_interval: int = 10000,
        log_interval: int = 10000,
        save_trainer_file: str | pathlib.Path | None = None,
        checkpoint: Checkpoint | None = None,
        checkpoint_rotation: CheckpointRotation | None = None,
        checkpoint_metadata: Callable[[Trainer], Mapping[str, Any]] | None = None,
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
        telemetry: Literal["minimal", "standard"] = "standard",
    ):
        warnings.warn(
            f"{type(self).__name__} is an experimental/prototype feature. The API may "
            "change in future versions. Please report any issues or feedback to help "
            "improve this implementation.",
            UserWarning,
            stacklevel=2,
        )
        if num_epochs is None:
            num_epochs = self._default_num_epochs
        if telemetry not in ("minimal", "standard"):
            raise ValueError(
                f"telemetry must be 'minimal' or 'standard', got {telemetry!r}."
            )
        super().__init__(
            collector=collector,
            total_frames=total_frames,
            frame_skip=frame_skip,
            optim_steps_per_batch=optim_steps_per_batch,
            loss_module=loss_module,
            optimizer=optimizer,
            target_net_updater=target_net_updater,
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

        if target_net_updater is not None:
            # stepped after every optimizer step, as the PPO-EWMA proximal
            # policy requires (a post_steps registration would only step it
            # once per collected batch)
            self.register_op("post_optim", TargetNetUpdaterHook(target_net_updater))

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
        self.telemetry = telemetry
        self._training_logger = (
            self.logger.with_prefix("training")
            if self.logger is not None and telemetry == "standard"
            else None
        )

        # Set up comprehensive logging for on-policy training
        if self.enable_logging:
            self._setup_logging()

    def _log_standard(self, metrics: Mapping[str, Any]) -> None:
        """Record standard metrics and forward due values to the training view."""
        due = {}
        for key, value in metrics.items():
            history_key = f"training/{key}"
            self._log_dict[history_key].append(value)
            if self.progress_bar and key in (
                "rewards/mean",
                "rewards/std",
                "terminals/done_rate",
            ):
                self._pbar_str[key] = (
                    value.item() if isinstance(value, torch.Tensor) else value
                )
            if (
                self.collected_frames - self._last_log.get(history_key, 0)
                > self._log_interval
            ):
                self._last_log[history_key] = self.collected_frames
                due[key] = value
        if due and self._training_logger is not None:
            self._training_logger.log_metrics(due, step=self.collected_frames)

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

        # Standard telemetry supplies canonical reward and terminal metrics.
        if self.telemetry == "minimal":
            log_done_percentage = LogScalar(
                key=_next_key(self.done_key),
                logname="done_percentage",
                log_pbar=True,
                include_std=False,  # No std for binary values
                reduction="mean",
            )
            self.register_op(log_dest, log_done_percentage)

        # Log rewards if enabled
        if self.log_rewards and self.telemetry == "minimal":
            # 1. Log training rewards (most important on-policy metric)
            log_rewards = LogScalar(
                key=_next_key(self.reward_key),
                logname="r_training",
                log_pbar=True,  # Show in progress bar
                include_std=True,
                reduction="mean",
            )
            self.register_op(log_dest, log_rewards)

            # 2. Log maximum reward in batch (for monitoring best performance)
            log_max_reward = LogScalar(
                key=_next_key(self.reward_key),
                logname="r_max",
                log_pbar=False,
                include_std=False,
                reduction="max",
            )
            self.register_op(log_dest, log_max_reward)

            # 3. Log total reward in batch (for monitoring cumulative performance)
            log_total_reward = LogScalar(
                key=_next_key(self.episode_reward_key),
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

        if self.telemetry == "standard":
            self._standard_telemetry = _OnPolicyTelemetry(self)
            self._standard_telemetry.register()
