# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import pathlib

from collections.abc import Callable

from tensordict import TensorDictBase
from tensordict.utils import NestedKey
from torch import optim

from torchrl.collectors import BaseCollector
from torchrl.data.replay_buffers.offline_to_online import OfflineToOnlineReplayBuffer
from torchrl.objectives.common import LossModule
from torchrl.objectives.utils import TargetNetUpdater
from torchrl.record.loggers import Logger
from torchrl.trainers.algorithms.sac import SACTrainer
from torchrl.trainers.trainers import TrainerHookBase

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
        replay_buffer: OfflineToOnlineReplayBuffer,
        *,
        batch_size: int | None = None,
        device=None,
        align_to_offline_keys: bool = True,
    ) -> None:
        self.replay_buffer = replay_buffer
        self.batch_size = batch_size
        self.device = device
        self.align_to_offline_keys = align_to_offline_keys
        self._offline_keys = None

    def _aligned_keys(self) -> list | None:
        if not self.align_to_offline_keys:
            return None
        if self._offline_keys is None:
            offline = self.replay_buffer.offline_buffer
            if not len(offline):
                return None
            probe = offline.sample(1)
            self._offline_keys = list(probe.keys(include_nested=True, leaves_only=True))
        return self._offline_keys

    def extend(self, batch: TensorDictBase) -> TensorDictBase:
        if ("collector", "mask") in batch.keys(True):
            batch = batch[batch.get(("collector", "mask"))]
        else:
            batch = batch.reshape(-1)
        keys = self._aligned_keys()
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
        return {
            "online_buffer": self.replay_buffer.online_buffer.state_dict(),
            "offline_fraction": self.replay_buffer._offline_fraction,
            "base_offline_fraction": self.replay_buffer._base_offline_fraction,
        }

    def load_state_dict(self, state_dict: dict) -> None:
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


class OfflineToOnlineTrainer(SACTrainer):
    """A SAC trainer for the offline-pretrain -> online-finetune transition.

    See also
    :class:`~torchrl.trainers.algorithms.configs.OfflineToOnlineTrainerConfig`
    for the Hydra configuration counterpart.

    Builds on :class:`~torchrl.trainers.algorithms.SACTrainer`, swapping the
    plain replay buffer for an :class:`~torchrl.data.OfflineToOnlineReplayBuffer`.
    Each collected batch is routed to the online buffer while optimization
    samples a mixed batch whose offline fraction is linearly annealed to zero
    over ``anneal_frames`` frames -- warm-starting the policy on offline data
    and smoothly handing it over to its own online experience. All other SAC
    behaviour (target-net updates, weight sync, logging) is inherited.

    Args:
        collector (BaseCollector): the data collector for online interactions.
        total_frames (int): total number of frames to collect.
        frame_skip (int): frames skipped between policy updates.
        optim_steps_per_batch (int): optimization steps per collected batch.
        loss_module (LossModule): the SAC loss module.
        replay_buffer (OfflineToOnlineReplayBuffer): the offline-to-online buffer.

    Keyword Args:
        anneal_frames (int, optional): frames over which ``offline_fraction``
            decays to 0. Defaults to ``total_frames``; pass ``<= 0`` to keep the
            fraction fixed.
        batch_size (int, optional): replay-buffer sampling batch size.

    See :class:`~torchrl.trainers.algorithms.SACTrainer` for the remaining
    keyword arguments.

    .. note:: Experimental/prototype feature; the API may change.
    """

    def __init__(
        self,
        *,
        collector: BaseCollector,
        total_frames: int,
        frame_skip: int,
        optim_steps_per_batch: int,
        loss_module: LossModule | Callable[[TensorDictBase], TensorDictBase],
        replay_buffer: OfflineToOnlineReplayBuffer,
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
        if not isinstance(replay_buffer, OfflineToOnlineReplayBuffer):
            raise TypeError(
                "OfflineToOnlineTrainer requires an OfflineToOnlineReplayBuffer, "
                f"got {type(replay_buffer).__name__}."
            )
        if async_collection:
            raise ValueError(
                "OfflineToOnlineTrainer does not support async_collection."
            )

        # Let SACTrainer wire up everything except the replay buffer (its
        # ReplayBufferTrainer assumes a sampler/priority API the offline-to-online
        # buffer does not expose); we register our own RB + annealing hooks below.
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

        device = getattr(replay_buffer.online_buffer.storage, "device", "cpu")
        OfflineToOnlineReplayBufferHook(
            replay_buffer, batch_size=batch_size, device=device
        ).register(self)

        if self.anneal_frames > 0:
            OfflineToOnlineAnnealHook(self, replay_buffer, self.anneal_frames).register(
                self
            )
