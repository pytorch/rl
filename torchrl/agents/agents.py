# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import pathlib
import warnings
from collections import OrderedDict
from textwrap import indent
from typing import Callable, Dict, Optional, Union, Sequence

import numpy as np
import torch.nn
from torch import nn, optim

try:
    from tqdm import tqdm

    _has_tqdm = True
except ImportError:
    _has_tqdm = False

from torchrl.collectors.collectors import _DataCollector
from torchrl.data import (
    ReplayBuffer,
    TensorDictPrioritizedReplayBuffer,
    TensorDictReplayBuffer,
)
from torchrl.data.tensordict.tensordict import _TensorDict
from torchrl.data.utils import expand_right
from torchrl.envs.common import _EnvClass
from torchrl.envs.transforms import TransformedEnv
from torchrl.envs.utils import set_exploration_mode
from torchrl.modules import reset_noise, TDModuleWrapper
from torchrl.objectives.costs.common import _LossModule
from torchrl.objectives.costs.utils import _TargetNetUpdate

REPLAY_BUFFER_CLASS = {
    "prioritized": TensorDictPrioritizedReplayBuffer,
    "circular": TensorDictReplayBuffer,
}

WRITER_METHODS = {
    "grad_norm": "add_scalar",
    "loss": "add_scalar",
}

__all__ = ["Agent"]


class Agent:
    """A generic Agent class.

    An agent is responsible of collecting data and training the model.
    To keep the class as versatile as possible, Agent does not construct any
    of its components: they all must be provided as argument when
    initializing the object.
    To build an Agent, one needs a iterable data source (a `collector`), a
    loss module, an optimizer. Optionally, a recorder (i.e. an environment
    instance used for testing purposes) and a policy can be provided for
    evaluating the training progress.

    Args:
        collector (Sequence[_TensorDict]): An iterable returning batches of
            data in a TensorDict form of shape [batch x time steps].
        total_frames (int): Total number of frames to be collected during
            training.
        loss_module (_LossModule): A module that reads TensorDict batches
            (possibly sampled from a replay buffer) and return a loss
            TensorDict where every key points to a different loss component.
        optimizer (optim.Optimizer): An optimizer that trains the parameters
            of the model.
        recorder (_EnvClass, optional): An environment instance to be used
            for testing.
        optim_scheduler (optim.lr_scheduler._LRScheduler, optional):
            learning rate scheduler.
        target_net_updater (_TargetNetUpdate, optional):
            a target network updater.
        policy_exploration (ProbabilisticTDModule, optional): a policy
            instance used for

            (1) updating the exploration noise schedule;

            (2) testing the policy on the recorder.

            Given that this instance is supposed to both explore and render
            the performance of the policy, it should be possible to turn off
            the explorative behaviour by calling the
            `set_exploration_mode('mode')` context manager.
        replay_buffer (ReplayBuffer, optional): a replay buffer for offline
            learning.
        writer (SummaryWriter, optional): a Tensorboard summary writer for
            logging purposes.
        update_weights_interval (int, optional): interval between two updates
            of the weights of a model living on another device. By default,
            the weights will be updated after every collection of data.
        record_interval (int, optional): total number of optimisation steps
            between two calls to the recorder for testing. Default is 10000.
        record_frames (int, optional): number of frames to be recorded during
            testing. Default is 1000.
        frame_skip (int, optional): frame_skip used in the environment. It is
            important to let the agent know the number of frames skipped at
            each iteration, otherwise the frame count can be underestimated.
            For logging, this parameter is important to normalize the reward.
            Finally, to compare different runs with different frame_skip,
            one must normalize the frame count and rewards. Default is 1.
        optim_steps_per_batch (int, optional): number of optimization steps
            per collection of data. An agent works as follows: a main loop
            collects batches of data (epoch loop), and a sub-loop (training
            loop) performs model updates in between two collections of data.
            Default is 500
        batch_size (int, optional): batch size when sampling data from the
            latest collection or from the replay buffer, if it is present.
            If no replay buffer is present, the sub-sampling will be
            achieved over the latest collection with a resulting batch of
            size (batch_size x sub_traj_len).
            Default is 256
        clip_grad_norm (bool, optional): If True, the gradients will be clipped
            based on the total norm of the model parameters. If False,
            all the partial derivatives will be clamped to
            (-clip_norm, clip_norm). Default is `True`.
        clip_norm (Number, optional): value to be used for clipping gradients.
            Default is 100.0.
        progress_bar (bool, optional): If True, a progress bar will be
            displayed using tqdm. If tqdm is not installed, this option
            won't have any effect. Default is `True`
        seed (int, optional): Seed to be used for the collector, pytorch and
            numpy. Default is 42.
        save_agent_interval (int, optional): How often the agent should be
            saved to disk. Default is 10000.
        save_agent_file (path, optional): path where to save the agent.
            Default is None (no saving)
        normalize_rewards_online (bool, optional): if True, the running
            statistics of the rewards are computed and the rewards used for
            training will be normalized based on these.
            Default is `False`
        sub_traj_len (int, optional): length of the trajectories that
            sub-samples must have in online settings. Default is -1 (i.e.
            takes the full length of the trajectory)
        min_sub_traj_len (int, optional): minimum value of `sub_traj_len`, in
            case some elements of the batch contain few steps.
            Default is -1 (i.e. no minimum value)
        selected_keys (iterable of str, optional): a list of strings that
            indicate the data that should be kept from the data collector.
            Since storing and retrieving information from the replay buffer
            does not come for free, limiting the amount of data passed to
            it can improve the algorithm performance. Default is None,
            i.e. all keys are kept.

    """

    # trackers
    _optim_count: int = 0
    _collected_frames: int = 0
    _last_log: dict = {}
    _last_save: int = 0
    _log_interval: int = 10000
    _reward_stats: dict = {"decay": 0.999}

    def __init__(
        self,
        collector: _DataCollector,
        total_frames: int,
        loss_module: Union[_LossModule, Callable[[_TensorDict], _TensorDict]],
        optimizer: optim.Optimizer,
        recorder: Optional[_EnvClass] = None,
        optim_scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
        target_net_updater: Optional[_TargetNetUpdate] = None,
        policy_exploration: Optional[TDModuleWrapper] = None,
        replay_buffer: Optional[ReplayBuffer] = None,
        writer: Optional["SummaryWriter"] = None,
        update_weights_interval: int = -1,
        record_interval: int = 10000,
        record_frames: int = 1000,
        frame_skip: int = 1,
        optim_steps_per_batch: int = 500,
        batch_size: int = 256,
        clip_grad_norm: bool = True,
        clip_norm: float = 100.0,
        progress_bar: bool = True,
        seed: int = 42,
        save_agent_interval: int = 10000,
        save_agent_file: Optional[Union[str, pathlib.Path]] = None,
        normalize_rewards_online: bool = False,
        sub_traj_len: int = -1,
        min_sub_traj_len: int = -1,
        selected_keys: Optional[Sequence[str]] = None,
    ) -> None:

        # objects
        self.collector = collector
        self.loss_module = loss_module
        self.recorder = recorder
        self.optimizer = optimizer
        self.optim_scheduler = optim_scheduler
        self.replay_buffer = replay_buffer
        self.policy_exploration = policy_exploration
        self.target_net_updater = target_net_updater
        self.writer = writer
        self._params = []
        for p in self.optimizer.param_groups:
            self._params += p["params"]

        # seeding
        self.seed = seed
        self.set_seed()

        # constants
        self.update_weights_interval = update_weights_interval
        self.optim_steps_per_batch = optim_steps_per_batch
        self.batch_size = batch_size
        self.total_frames = total_frames
        self.frame_skip = frame_skip
        self.clip_grad_norm = clip_grad_norm
        self.clip_norm = clip_norm
        if progress_bar and not _has_tqdm:
            warnings.warn(
                "tqdm library not found. Consider installing tqdm to use the Agent progress bar."
            )
        self.progress_bar = progress_bar and _has_tqdm
        self.record_interval = record_interval
        self.record_frames = record_frames
        self.save_agent_interval = save_agent_interval
        self.save_agent_file = save_agent_file
        self.normalize_rewards_online = normalize_rewards_online
        self.sub_traj_len = sub_traj_len
        self.min_sub_traj_len = min_sub_traj_len
        self.selected_keys = selected_keys

    def save_agent(self) -> None:
        _save = False
        if self.save_agent_file is not None:
            if (self._collected_frames - self._last_save) > self.save_agent_interval:
                self._last_save = self._collected_frames
                _save = True
        if _save:
            torch.save(self.state_dict(), self.save_agent_file)

    def load_from_file(self, file: Union[str, pathlib.Path]) -> Agent:
        loaded_dict: OrderedDict = torch.load(file)

        # checks that keys match
        expected_keys = {
            "env",
            "loss_module",
            "_collected_frames",
            "_last_log",
            "_last_save",
            "_optim_count",
        }
        actual_keys = set(loaded_dict.keys())
        if len(actual_keys.difference(expected_keys)) or len(
            expected_keys.difference(actual_keys)
        ):
            raise RuntimeError(
                f"Expected keys {expected_keys} in the loaded file but got"
                f" {actual_keys}"
            )
        self.collector.load_state_dict(loaded_dict["env"])
        self.model.load_state_dict(loaded_dict["model"])
        for key in [
            "_collected_frames",
            "_last_log",
            "_last_save",
            "_optim_count",
        ]:
            setattr(self, key, loaded_dict[key])
        return self

    def set_seed(self):
        seed = self.collector.set_seed(self.seed)
        torch.manual_seed(seed)
        np.random.seed(seed)

    def state_dict(self) -> Dict:
        state_dict = OrderedDict(
            env=self.collector.state_dict(),
            loss_module=self.loss_module.state_dict(),
            _collected_frames=self._collected_frames,
            _last_log=self._last_log,
            _last_save=self._last_save,
            _optim_count=self._optim_count,
        )
        return state_dict

    def load_state_dict(self, state_dict: Dict) -> None:
        model_state_dict = state_dict["loss_module"]
        env_state_dict = state_dict["env"]
        self.loss_module.load_state_dict(model_state_dict)
        self.collector.load_state_dict(env_state_dict)

    @property
    def collector(self) -> _DataCollector:
        return self._collector

    @collector.setter
    def collector(self, collector: _DataCollector) -> None:
        self._collector = collector

    def train(self):
        if self.progress_bar:
            self._pbar = tqdm(total=self.total_frames)
            self._pbar_str = OrderedDict()

        collected_frames = 0
        for i, batch in enumerate(self.collector):
            if self.selected_keys:
                batch = batch.select(*self.selected_keys, "mask")

            if "mask" in batch.keys():
                current_frames = batch.get("mask").sum().item() * self.frame_skip
            else:
                current_frames = batch.numel() * self.frame_skip
            collected_frames += current_frames
            self._collected_frames = collected_frames

            if self.replay_buffer is not None:
                if "mask" in batch.keys():
                    batch = batch[batch.get("mask").squeeze(-1)]
                else:
                    batch = batch.reshape(-1)
                reward_training = batch.get("reward").mean().item()
                batch = batch.cpu()
                self.replay_buffer.extend(batch)
            else:
                if "mask" in batch.keys():
                    reward_training = batch.get("reward")
                    mask = batch.get("mask").squeeze(-1)
                    reward_training = reward_training[mask].mean().item()
                else:
                    reward_training = batch.get("reward").mean().item()

            if self.normalize_rewards_online:
                reward = batch.get("reward")
                self._update_reward_stats(reward)

            if collected_frames > self.collector.init_random_frames:
                self.steps(batch)
            self._collector_scheduler_step(i, current_frames)

            self._log(reward_training=reward_training)
            if self.progress_bar:
                self._pbar.update(current_frames)
                self._pbar_description()

            if collected_frames > self.total_frames:
                break

        self.collector.shutdown()

    @torch.no_grad()
    def _update_reward_stats(self, reward: torch.Tensor) -> None:
        decay = self._reward_stats.get("decay", 0.999)
        sum = self._reward_stats["sum"] = (
            decay * self._reward_stats.get("sum", 0.0) + reward.sum()
        )
        ssq = self._reward_stats["ssq"] = (
            decay * self._reward_stats.get("ssq", 0.0) + reward.pow(2).sum()
        )
        count = self._reward_stats["count"] = (
            decay * self._reward_stats.get("count", 0.0) + reward.numel()
        )

        mean = self._reward_stats["mean"] = sum / count
        var = self._reward_stats["var"] = ssq / count - mean.pow(2)
        self._reward_stats["std"] = var.clamp_min(1e-6).sqrt()

    def _normalize_reward(self, tensordict: _TensorDict) -> None:
        reward = tensordict.get("reward")
        reward = reward - self._reward_stats["mean"]
        reward = reward / self._reward_stats["std"]
        tensordict.set_("reward", reward)

    def _collector_scheduler_step(self, step: int, current_frames: int):
        """Runs entropy annealing steps for exploration, policy weights update
        across workers etc.

        """

        if self.policy_exploration is not None and hasattr(
            self.policy_exploration, "step"
        ):
            self.policy_exploration.step(current_frames)

        if step % self.update_weights_interval == 0:
            self.collector.update_policy_weights_()

    def steps(self, batch: _TensorDict) -> None:
        average_grad_norm = 0.0
        average_losses = None

        self.loss_module.apply(reset_noise)  # TODO: group in loss_module.reset?
        self.loss_module.reset()

        for j in range(self.optim_steps_per_batch):
            self._optim_count += 1
            if self.replay_buffer is not None:
                sub_batch = self.replay_buffer.sample(self.batch_size)
            else:
                sub_batch = self._sub_sample_batch(batch)

            if self.normalize_rewards_online:
                self._normalize_reward(sub_batch)

            sub_batch_device = sub_batch.to(self.loss_module.device)
            losses_td = self.loss_module(sub_batch_device)
            if isinstance(self.replay_buffer, TensorDictPrioritizedReplayBuffer):
                self.replay_buffer.update_priority(sub_batch_device)

            # sum all keys that start with 'loss_'
            loss = sum(
                [item for key, item in losses_td.items() if key.startswith("loss")]
            )
            loss.backward()
            if average_losses is None:
                average_losses: _TensorDict = losses_td.detach()
            else:
                for key, item in losses_td.items():
                    val = average_losses.get(key)
                    average_losses.set(key, val * j / (j + 1) + item / (j + 1))

            grad_norm = self._grad_clip()
            average_grad_norm = average_grad_norm * j / (j + 1) + grad_norm / (j + 1)
            self.optimizer.step()
            self.optimizer.zero_grad()

            self._optim_schedule_step()

            if self._optim_count % self.record_interval == 0:
                self.record()

        if self.optim_steps_per_batch > 0:
            self._log(
                grad_norm=average_grad_norm,
                optim_steps=self._optim_count,
                **average_losses,
            )

    def _optim_schedule_step(self) -> None:
        """Runs scheduler steps, target network update steps etc.
        Returns:
        """
        if self.optim_scheduler is not None:
            self.optim_scheduler.step()
        if self.target_net_updater is not None:
            self.target_net_updater.step()

    def _sub_sample_batch(self, batch: _TensorDict) -> _TensorDict:
        """Sub-sampled part of a batch randomly.

        If the batch has one dimension, a random subsample of length
        self.bach_size will be returned. If the batch has two or more
        dimensions, it is assumed that the first dimension represents the
        batch, and the second the time. If so, the resulting subsample will
        contain consecutive samples across time.
        """

        if batch.ndimension() == 1:
            return batch[torch.randperm(batch.shape[0])[: self.batch_size]]

        sub_traj_len = self.sub_traj_len if self.sub_traj_len > 0 else batch.shape[1]
        if "mask" in batch.keys():
            # if a valid mask is present, it's important to sample only
            # valid steps
            traj_len = batch.get("mask").sum(1).squeeze()
            sub_traj_len = max(
                self.min_sub_traj_len,
                min(sub_traj_len, traj_len.min().int().item()),
            )
        else:
            traj_len = (
                torch.ones(batch.shape[0], device=batch.device, dtype=torch.bool)
                * batch.shape[1]
            )
        len_mask = traj_len >= sub_traj_len
        valid_trajectories = torch.arange(batch.shape[0])[len_mask]

        batch_size = self.batch_size // sub_traj_len
        traj_idx = valid_trajectories[
            torch.randint(
                valid_trajectories.numel(), (batch_size,), device=batch.device
            )
        ]

        if sub_traj_len < batch.shape[1]:
            _traj_len = traj_len[traj_idx]
            seq_idx = (
                torch.rand_like(_traj_len, dtype=torch.float)
                * (_traj_len - sub_traj_len)
            ).int()
            seq_idx = seq_idx.unsqueeze(-1).expand(-1, sub_traj_len)
        elif sub_traj_len == batch.shape[1]:
            seq_idx = torch.zeros(
                batch_size, sub_traj_len, device=batch.device, dtype=torch.long
            )
        else:
            raise ValueError(
                f"sub_traj_len={sub_traj_len} is not allowed. Accepted values "
                f"are in the range [1, {batch.shape[1]}]."
            )

        seq_idx = seq_idx + torch.arange(sub_traj_len, device=seq_idx.device)
        td = batch[traj_idx].clone()
        td = td.apply(
            lambda t: t.gather(
                dim=1,
                index=expand_right(seq_idx, (batch_size, sub_traj_len, *t.shape[2:])),
            ),
            batch_size=(batch_size, sub_traj_len),
        )
        if "mask" in batch.keys() and not td.get("mask").all():
            raise RuntimeError("Sampled invalid steps")
        return td

    def _grad_clip(self) -> float:
        if self.clip_grad_norm:
            gn = nn.utils.clip_grad_norm_(self._params, self.clip_norm)
        else:
            gn = sum([p.grad.pow(2).sum() for p in self._params]).sqrt()
            nn.utils.clip_grad_value_(self._params, self.clip_norm)
        return float(gn)

    def _log(self, **kwargs) -> None:
        collected_frames = self._collected_frames
        for key, item in kwargs.items():
            if (collected_frames - self._last_log.get(key, 0)) > self._log_interval:
                self._last_log[key] = collected_frames
                _log = True
            else:
                _log = False
            method = WRITER_METHODS.get(key, "add_scalar")
            if _log and self.writer is not None:
                getattr(self.writer, method)(key, item, global_step=collected_frames)
            if method == "add_scalar" and self.progress_bar:
                self._pbar_str[key] = float(item)

    def _pbar_description(self) -> None:
        if self.progress_bar:
            self._pbar.set_description(
                ", ".join(
                    [
                        f"{key}: {float(item):4.4f}"
                        for key, item in self._pbar_str.items()
                    ]
                )
            )

    @torch.no_grad()
    @set_exploration_mode("mode")
    def record(self) -> None:
        if self.recorder is not None:
            self.policy_exploration.eval()
            self.recorder.eval()
            if isinstance(self.recorder, TransformedEnv):
                self.recorder.transform.eval()
            td_record = self.recorder.rollout(
                policy=self.policy_exploration,
                n_steps=self.record_frames,
            )
            self.policy_exploration.train()
            self.recorder.train()
            reward = td_record.get("reward").mean() / self.frame_skip
            self._log(reward_evaluation=reward)
            self.recorder.transform.dump()

    def __repr__(self) -> str:
        loss_str = indent(f"loss={self.loss_module}", 4 * " ")
        policy_str = indent(f"policy_exploration={self.policy_exploration}", 4 * " ")
        collector_str = indent(f"collector={self.collector}", 4 * " ")
        buffer_str = indent(f"buffer={self.replay_buffer}", 4 * " ")
        optimizer_str = indent(f"optimizer={self.optimizer}", 4 * " ")
        target_net_updater = indent(
            f"target_net_updater={self.target_net_updater}", 4 * " "
        )
        writer = indent(f"writer={self.writer}", 4 * " ")

        string = "\n".join(
            [
                loss_str,
                policy_str,
                collector_str,
                buffer_str,
                optimizer_str,
                target_net_updater,
                writer,
            ]
        )
        string = f"Agent(\n{string})"
        return string
