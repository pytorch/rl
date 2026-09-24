# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import pathlib

from collections.abc import Callable, Iterator, Mapping
from typing import Any

from tensordict import TensorDict, TensorDictBase
from tensordict.utils import NestedKey
from torch import optim

from torchrl.checkpoint import Checkpoint, CheckpointRotation
from torchrl.collectors import BaseCollector
from torchrl.data import ReplayBuffer
from torchrl.data.replay_buffers.offline_to_online import OfflineToOnlineReplayBuffer
from torchrl.data.utils import DEVICE_TYPING
from torchrl.objectives.common import LossModule
from torchrl.objectives.utils import TargetNetUpdater
from torchrl.record.loggers import Logger
from torchrl.trainers.algorithms.sac import SACTrainer
from torchrl.trainers.trainers import Trainer, TrainerHookBase

__all__ = [
    "OfflineToOnlineReplayBufferHook",
    "OfflineToOnlineAnnealHook",
    "OfflineToOnlineTrainer",
]


class OfflineToOnlineReplayBufferHook(TrainerHookBase):
    """Trainer hook driving an :class:`~torchrl.data.OfflineToOnlineReplayBuffer`.

    Routes freshly collected experience to the online buffer on ``pre_epoch`` and
    draws a mixed offline/online batch on ``process_optim_batch``. Online
    transitions are projected onto the offline dataset's key schema before being
    stored, so the offline/online concat in
    :meth:`OfflineToOnlineReplayBuffer.sample` does not raise on the policy
    outputs (``loc``/``scale``/``log_prob``) and ``collector`` subtree the
    offline dataset lacks.

    Keyword Args:
        batch_size (int, optional): batch size for :meth:`sample`; falls back to
            the buffer's configured ``batch_size``.
        device (device, optional): device the sampled batch is moved to.
        align_to_offline_keys (bool, optional): project stored online
            transitions onto the offline schema (default ``True``).
    """

    def __init__(
        self,
        replay_buffer: ReplayBuffer | OfflineToOnlineReplayBuffer,
        *,
        batch_size: int | None = None,
        device: DEVICE_TYPING | None = None,
        align_to_offline_keys: bool = True,
    ) -> None:
        self.replay_buffer = replay_buffer
        self.batch_size = batch_size
        self.device = device
        self.align_to_offline_keys = align_to_offline_keys
        self._offline_keys = None

    def aligned_keys(self) -> list | None:
        if not self.align_to_offline_keys:
            return None
        if self._offline_keys is None:
            mixed = isinstance(self.replay_buffer, OfflineToOnlineReplayBuffer)
            offline = self.replay_buffer.offline_buffer if mixed else self.replay_buffer
            if not len(offline):
                return None
            probe = offline[0]
            self._offline_keys = [
                key for key in probe.keys(True, True) if key != "index"
            ]
        return self._offline_keys

    def extend(self, batch: TensorDictBase | None) -> TensorDictBase | None:
        if batch is None:
            return None
        if ("collector", "mask") in batch.keys(True):
            batch = batch[batch.get(("collector", "mask"))]
        else:
            batch = batch.reshape(-1)
        keys = self.aligned_keys()
        if keys is not None:
            batch = batch.select(*keys, strict=False)
        elif "collector" in batch.keys():
            batch = batch.exclude("collector")
        batch = batch.cpu()
        self.replay_buffer.extend(batch)
        return batch

    def sample(self, batch: TensorDictBase) -> TensorDictBase:
        sample = self.replay_buffer.sample(self.batch_size)
        return sample.to(self.device) if self.device is not None else sample

    def state_dict(self) -> dict:
        if not isinstance(self.replay_buffer, OfflineToOnlineReplayBuffer):
            return self.replay_buffer.state_dict()
        return {
            "online_buffer": self.replay_buffer.online_buffer.state_dict(),
            "offline_fraction": self.replay_buffer._offline_fraction,
            "base_offline_fraction": self.replay_buffer._base_offline_fraction,
        }

    def load_state_dict(self, state_dict: dict) -> None:
        if not isinstance(self.replay_buffer, OfflineToOnlineReplayBuffer):
            self.replay_buffer.load_state_dict(state_dict)
            return
        self.replay_buffer.online_buffer.load_state_dict(state_dict["online_buffer"])
        self.replay_buffer._offline_fraction = state_dict.get(
            "offline_fraction", self.replay_buffer._offline_fraction
        )
        self.replay_buffer._base_offline_fraction = state_dict.get(
            "base_offline_fraction", self.replay_buffer._base_offline_fraction
        )

    def register(self, trainer, name: str = "replay_buffer") -> None:
        trainer.register_op("pre_epoch", self.extend)
        trainer.register_op("process_optim_batch", self.sample)
        trainer.register_module(name, self)


class OfflineToOnlineAnnealHook(TrainerHookBase):
    """Linearly decays the buffer's offline sampling fraction during training.

    Once per collected batch (``post_steps``) it calls
    :meth:`OfflineToOnlineReplayBuffer.anneal` with the trainer's current
    ``collected_frames``, so sampling shifts from offline-dominant to purely
    online over ``anneal_frames`` frames.
    """

    def __init__(
        self,
        trainer,
        replay_buffer: OfflineToOnlineReplayBuffer,
        anneal_frames: int,
    ) -> None:
        self.trainer = trainer
        self.replay_buffer = replay_buffer
        self.anneal_frames = anneal_frames

    def __call__(self) -> None:
        self.replay_buffer.anneal(self.trainer.collected_frames, self.anneal_frames)

    def state_dict(self) -> dict:
        return {}

    def load_state_dict(self, state_dict: dict) -> None:
        pass

    def register(self, trainer, name: str = "offline_to_online_anneal") -> None:
        trainer.register_op("post_steps", self)
        trainer.register_module(name, self)


class OfflinePretrainingCollector:
    """Run offline updates before online collection, deferring collector creation."""

    def __init__(
        self,
        trainer: OfflineToOnlineTrainer,
        collector: BaseCollector | Callable[[], BaseCollector] | None,
    ) -> None:
        self.trainer = trainer
        self.collector = collector if isinstance(collector, BaseCollector) else None
        self.factory = collector if self.collector is None else None
        self.seed = None

    @property
    def init_random_frames(self) -> int:
        return self.collector.init_random_frames

    def __iter__(self) -> Iterator[TensorDictBase]:
        trainer = self.trainer
        while (
            trainer.completed_steps < trainer.offline_steps
            and not trainer._stop_training
        ):
            trainer.update()
            if (
                trainer.completed_steps < trainer.offline_steps
                and not trainer._stop_training
                and trainer.completed_steps % max(trainer.save_trainer_interval, 1) == 0
            ):
                trainer.save_trainer(force_save=True)
        trainer.save_trainer(force_save=True)
        if trainer._stop_training or trainer.collected_frames >= trainer.total_frames:
            return
        collector = self.get_collector()
        collector.update_policy_weights_(
            TensorDict.from_module(trainer.loss_module.actor_network)
        )
        for batch in collector:
            if ("collector", "mask") in batch.keys(True):
                batch = batch[batch["collector", "mask"]]
            if batch.batch_size.numel():
                yield batch

    def get_collector(self) -> BaseCollector:
        """Create the collector when collection or environment metadata requires it."""
        if self.collector is None:
            self.collector = self.factory()
            if self.seed is not None:
                seed, kwargs = self.seed
                self.collector.set_seed(seed, **kwargs)
        return self.collector

    def update_policy_weights_(self, weights: TensorDictBase | None = None) -> None:
        if self.collector is not None:
            if weights is None:
                weights = TensorDict.from_module(self.trainer.loss_module.actor_network)
            self.collector.update_policy_weights_(weights)

    def set_seed(self, seed: int, **kwargs) -> int:
        if self.collector is not None:
            return self.collector.set_seed(seed, **kwargs)
        self.seed = seed, kwargs
        return seed

    def getattr_env(self, name: str) -> Any:
        return self.get_collector().getattr_env(name)

    def shutdown(self) -> None:
        if self.collector is not None:
            self.collector.shutdown()

    def state_dict(self) -> dict[str, Any]:
        return {
            "collector": self.collector.state_dict()
            if self.collector is not None
            else None,
            "seed": self.seed,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.seed = state.get("seed", self.seed)
        if state["collector"] is not None:
            self.get_collector().load_state_dict(state["collector"])


class OfflineToOnlineTrainer(SACTrainer):
    """Train from offline replay before optional online fine-tuning.

    See also :class:`~torchrl.trainers.algorithms.configs.OfflineToOnlineTrainerConfig`
    for the Hydra configuration counterpart.

    Builds on :class:`~torchrl.trainers.algorithms.SACTrainer` target updates,
    collector weight synchronization and logging. Offline updates precede online
    collection within the standard Trainer lifecycle. With a mixed buffer,
    online collection anneals the offline sampling fraction over ``anneal_frames``.

    Args:
        collector (BaseCollector or callable, optional): online collector or a
            factory called after offline training, unless loss initialization
            requires environment metadata first. Defaults to ``None``.
        total_frames (int): online frames to collect. Defaults to zero.
        frame_skip (int): frames skipped between policy updates. Defaults to one.
        optim_steps_per_batch (int): updates per collected batch. Defaults to one.
        loss_module (LossModule): actor-critic objective with an ``actor_network``.
        replay_buffer (ReplayBuffer or OfflineToOnlineReplayBuffer): regular
            replay storage or independently sampled offline and online buffers.

    Keyword Args:
        anneal_frames (int, optional): frames over which ``offline_fraction``
            decays to 0. Defaults to ``total_frames``; pass ``<= 0`` to keep the
            fraction fixed.
        batch_size (int, optional): replay-buffer sampling batch size.
        offline_steps (int): gradient updates before online collection (default zero).
        device (device, optional): device for sampled training batches.
        compile_loss (bool): compile the loss module, retaining its checkpoint keys.

    A regular replay buffer retains offline and online transitions together. A
    mixed offline-to-online buffer instead controls their sampling fractions.
    The collector may be omitted for offline-only training or supplied as a
    factory, created when collection or environment metadata requires it.
    Losses are logged by update count
    during pretraining and by collected frames during online training.

    See :class:`~torchrl.trainers.algorithms.SACTrainer` for the remaining
    keyword arguments.

    .. note:: Experimental/prototype feature; the API may change.
    """

    def __init__(
        self,
        *,
        collector: BaseCollector | Callable[[], BaseCollector] | None = None,
        total_frames: int = 0,
        frame_skip: int = 1,
        optim_steps_per_batch: int = 1,
        loss_module: LossModule | Callable[[TensorDictBase], TensorDictBase],
        replay_buffer: ReplayBuffer | OfflineToOnlineReplayBuffer,
        offline_steps: int = 0,
        device: DEVICE_TYPING | None = None,
        compile_loss: bool = False,
        anneal_frames: int | None = None,
        batch_size: int | None = None,
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
        checkpoint_rotation: CheckpointRotation | None = None,
        checkpoint_metadata: Callable[[Trainer], Mapping[str, Any]] | None = None,
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
    ) -> None:
        if offline_steps < 0 or total_frames < 0:
            raise ValueError("Training budgets must be nonnegative.")
        if total_frames and collector is None:
            raise ValueError("Online training requires a collector.")
        self.offline_steps = offline_steps
        if (
            offline_steps
            or total_frames == 0
            or not isinstance(collector, BaseCollector)
        ):
            collector = OfflinePretrainingCollector(self, collector)
        if async_collection:
            raise ValueError(
                "OfflineToOnlineTrainer does not support async_collection."
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
            checkpoint=checkpoint,
            checkpoint_rotation=checkpoint_rotation,
            checkpoint_metadata=checkpoint_metadata,
            replay_buffer=None,
            enable_logging=enable_logging,
            log_rewards=log_rewards,
            log_actions=log_actions,
            log_observations=log_observations,
            target_net_updater=target_net_updater,
            async_collection=False,
            log_timings=log_timings,
            auto_log_optim_steps=auto_log_optim_steps,
            done_key=done_key,
            terminated_key=terminated_key,
            reward_key=reward_key,
            episode_reward_key=episode_reward_key,
            action_key=action_key,
            observation_key=observation_key,
        )

        self.replay_buffer = replay_buffer
        self.anneal_frames = total_frames if anneal_frames is None else anneal_frames

        if device is None and isinstance(replay_buffer, OfflineToOnlineReplayBuffer):
            device = getattr(replay_buffer.online_buffer.storage, "device", "cpu")
        self.device = device
        self.batch_size = batch_size
        OfflineToOnlineReplayBufferHook(
            replay_buffer, batch_size=batch_size, device=device
        ).register(self)

        if (
            isinstance(replay_buffer, OfflineToOnlineReplayBuffer)
            and self.anneal_frames > 0
        ):
            OfflineToOnlineAnnealHook(self, replay_buffer, self.anneal_frames).register(
                self
            )

        self.register_module("target_net_updater", target_net_updater)
        self.register_op("post_optim_complete_log", self.record_update)
        if compile_loss:
            self.loss_module.compile(fullgraph=True)

    @property
    def completed_steps(self) -> int:
        """Number of completed offline and online updates."""
        return self._optim_count

    @property
    def checkpoint_step(self) -> int:
        """Use update counts across both phases when pretraining is configured."""
        return self.completed_steps if self.offline_steps else super().checkpoint_step

    def record_update(self, step: int, losses: TensorDictBase) -> None:
        self.metrics = losses.detach()
        if (
            self.auto_log_optim_steps
            and self.collected_frames == 0
            and self.logger is not None
        ):
            if step % max(self._log_interval, 1) == 0:
                for key, value in self.metrics.items():
                    if value.ndim == 0:
                        self.logger.log_scalar(key, value.item(), step=step)

    def update(self) -> TensorDictBase:
        """Run one replay update through the standard optimization hooks."""
        self.optim_steps(None, optim_steps_per_batch=1)
        return self.metrics

    def shutdown(self) -> None:
        """Release the online collector, if it was initialized."""
        self.collector.shutdown()
