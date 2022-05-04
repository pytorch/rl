# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import pathlib
import warnings
from collections import OrderedDict, defaultdict
from textwrap import indent
from typing import Callable, Dict, Optional, Union, Sequence, Tuple, Type

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
from torchrl.modules import TDModuleWrapper, TDModule
from torchrl.objectives.costs.common import _LossModule

REPLAY_BUFFER_CLASS = {
    "prioritized": TensorDictPrioritizedReplayBuffer,
    "circular": TensorDictReplayBuffer,
}

WRITER_METHODS = {
    "grad_norm": "add_scalar",
    "loss": "add_scalar",
}

__all__ = [
    "Trainer",
    "BatchSubSampler",
    "CountFramesLog",
    "LogReward",
    "Recorder",
    "ReplayBuffer",
    "RewardNormalizer",
    "SelectKeys",
    "UpdateWeights",
]

TYPE_DESCR = {float: "4.4f", int: ""}


class Trainer:
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
        policy_exploration (ProbabilisticTDModule, optional): a policy
            instance used for

            (1) updating the exploration noise schedule;

            (2) testing the policy on the recorder.

            Given that this instance is supposed to both explore and render
            the performance of the policy, it should be possible to turn off
            the explorative behaviour by calling the
            `set_exploration_mode('mode')` context manager.
        writer (SummaryWriter, optional): a Tensorboard summary writer for
            logging purposes.
        optim_steps_per_batch (int, optional): number of optimization steps
            per collection of data. An agent works as follows: a main loop
            collects batches of data (epoch loop), and a sub-loop (training
            loop) performs model updates in between two collections of data.
            Default is 500
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
        frame_skip: int,
        loss_module: Union[_LossModule, Callable[[_TensorDict], _TensorDict]],
        optimizer: optim.Optimizer,
        policy_exploration: Optional[TDModuleWrapper] = None,
        writer: Optional["SummaryWriter"] = None,
        optim_steps_per_batch: int = 500,
        clip_grad_norm: bool = True,
        clip_norm: float = 100.0,
        progress_bar: bool = True,
        seed: int = 42,
        save_agent_interval: int = 10000,
        save_agent_file: Optional[Union[str, pathlib.Path]] = None,
    ) -> None:

        # objects
        self.frame_skip = frame_skip
        self.collector = collector
        self.loss_module = loss_module
        self.optimizer = optimizer
        self.policy_exploration = policy_exploration
        self.writer = writer
        self._params = []
        for p in self.optimizer.param_groups:
            self._params += p["params"]

        # seeding
        self.seed = seed
        self.set_seed()

        # constants
        self.optim_steps_per_batch = optim_steps_per_batch
        self.total_frames = total_frames
        self.clip_grad_norm = clip_grad_norm
        self.clip_norm = clip_norm
        if progress_bar and not _has_tqdm:
            warnings.warn(
                "tqdm library not found. Consider installing tqdm to use the Agent progress bar."
            )
        self.progress_bar = progress_bar and _has_tqdm
        self.save_agent_interval = save_agent_interval
        self.save_agent_file = save_agent_file

        self._log_dict = defaultdict(lambda: [])

        self._batch_process_ops = []
        self._post_steps_ops = []
        self._post_steps_log_ops = []
        self._pre_steps_log_ops = []
        self._post_optim_log_ops = []
        self._pre_optim_ops = []
        self._post_loss_ops = []
        self._process_optim_batch_ops = []
        self._post_optim_ops = []

    def save_agent(self) -> None:
        _save = False
        if self.save_agent_file is not None:
            if (self._collected_frames - self._last_save) > self.save_agent_interval:
                self._last_save = self._collected_frames
                _save = True
        if _save:
            torch.save(self.state_dict(), self.save_agent_file)

    def load_from_file(self, file: Union[str, pathlib.Path]) -> Trainer:
        loaded_dict: OrderedDict = torch.load(file)

        # checks that keys match
        expected_keys = {
            "env",
            "loss_module",
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

    def register_op(self, dest: str, op: Callable) -> None:
        if dest == "batch_process":
            _check_input_output_typehint(op, input=_TensorDict, output=_TensorDict)
            self._batch_process_ops.append(op)

        elif dest == "pre_optim_steps":
            _check_input_output_typehint(op, input=None, output=None)
            self._pre_optim_ops.append(op)

        elif dest == "process_optim_batch":
            _check_input_output_typehint(op, input=_TensorDict, output=_TensorDict)
            self._process_optim_batch_ops.append(op)

        elif dest == "post_loss":
            _check_input_output_typehint(op, input=_TensorDict, output=_TensorDict)
            self._post_loss_ops.append(op)

        elif dest == "post_steps":
            _check_input_output_typehint(op, input=None, output=None)
            self._post_steps_ops.append(op)

        elif dest == "post_optim":
            _check_input_output_typehint(op, input=None, output=None)
            self._post_optim_ops.append(op)

        elif dest == "pre_steps_log":
            _check_input_output_typehint(
                op, input=_TensorDict, output=Tuple[str, float]
            )
            self._pre_steps_log_ops.append(op)

        elif dest == "post_steps_log":
            _check_input_output_typehint(
                op, input=_TensorDict, output=Tuple[str, float]
            )
            self._post_steps_log_ops.append(op)

        elif dest == "post_optim_log":
            _check_input_output_typehint(
                op, input=_TensorDict, output=Tuple[str, float]
            )
            self._post_optim_log_ops.append(op)

        else:
            raise RuntimeError(
                f"The hook collection {dest} is not recognised. Choose from:"
                f"(batch_process, pre_steps, pre_step, post_loss, post_steps, "
                f"post_steps_log, post_optim_log)"
            )

    # Process batch
    def _process_batch_hook(self, batch: _TensorDict) -> _TensorDict:
        for op in self._batch_process_ops:
            out = op(batch)
            if isinstance(out, _TensorDict):
                batch = out
        return batch

    def _post_steps_hook(self) -> None:
        for op in self._post_steps_ops:
            op()

    def _post_optim_log(self, batch: _TensorDict) -> None:
        for op in self._post_optim_log_ops:
            result = op(batch)
            if result is not None:
                key, value = result
                self._log(key=value)

    def _pre_optim_hook(self):
        for op in self._pre_optim_ops:
            op()

    def _process_optim_batch_hook(self, batch):
        for op in self._process_optim_batch_ops:
            out = op(batch)
            if isinstance(out, _TensorDict):
                batch = out
        return batch

    def _post_loss_hook(self, batch):
        for op in self._post_loss_ops:
            out = op(batch)
            if isinstance(out, _TensorDict):
                batch = out
        return batch

    def _post_optim_hook(self):
        for op in self._post_optim_ops:
            op()

    def _pre_steps_log_hook(self, batch: _TensorDict) -> None:
        for op in self._pre_steps_log_ops:
            result = op(batch)
            if result is not None:
                key, value = result
                kwargs = {key: value}
                self._log(**kwargs)

    def _post_steps_log_hook(self, batch: _TensorDict) -> None:
        for op in self._post_steps_log_ops:
            result = op(batch)
            if result is not None:
                key, value = result
                kwargs = {key: value}
                self._log(**kwargs)

    def train(self):
        if self.progress_bar:
            self._pbar = tqdm(total=self.total_frames)
            self._pbar_str = OrderedDict()

        self.collected_frames = 0
        for i, batch in enumerate(self.collector):
            batch = self._process_batch_hook(batch)
            self._pre_steps_log_hook(batch)
            current_frames = (
                batch.get("mask", torch.tensor(batch.numel())).sum().item()
                * self.frame_skip
            )
            self.collected_frames += current_frames

            if self.collected_frames > self.collector.init_random_frames:
                self.optim_steps(batch)
            self._post_steps_hook()

            self._post_steps_log_hook(batch)

            if self.progress_bar:
                self._pbar.update(current_frames)
                self._pbar_description()

            if self.collected_frames > self.total_frames:
                break

        print("shutting down collector")
        self.collector.shutdown()

    def _optimizer_step(self, losses_td: _TensorDict) -> None:
        # sum all keys that start with 'loss_'
        loss = sum([item for key, item in losses_td.items() if key.startswith("loss")])
        loss.backward()

        grad_norm = self._grad_clip()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return losses_td.detach().set("grad_norm", grad_norm)

    def optim_steps(self, batch: _TensorDict) -> None:
        # average_grad_norm = 0.0
        average_losses = None

        self._pre_optim_hook()

        for j in range(self.optim_steps_per_batch):
            self._optim_count += 1

            sub_batch = self._process_optim_batch_hook(batch)
            sub_batch_device = sub_batch.to(self.loss_module.device)
            losses_td = self.loss_module(sub_batch_device)
            self._post_loss_hook(sub_batch_device)

            losses_detached = self._optimizer_step(losses_td)
            self._post_optim_hook()

            if average_losses is None:
                average_losses: _TensorDict = losses_detached
            else:
                for key, item in losses_detached.items():
                    val = average_losses.get(key)
                    average_losses.set(key, val * j / (j + 1) + item / (j + 1))

        if self.optim_steps_per_batch > 0:
            self._log(
                optim_steps=self._optim_count,
                **average_losses,
            )

    def _grad_clip(self) -> float:
        if self.clip_grad_norm:
            gn = nn.utils.clip_grad_norm_(self._params, self.clip_norm)
        else:
            gn = sum([p.grad.pow(2).sum() for p in self._params]).sqrt()
            nn.utils.clip_grad_value_(self._params, self.clip_norm)
        return float(gn)

    def _log(self, **kwargs) -> None:
        collected_frames = self.collected_frames
        for key, item in kwargs.items():
            self._log_dict[key].append(item)

            if (collected_frames - self._last_log.get(key, 0)) > self._log_interval:
                self._last_log[key] = collected_frames
                _log = True
            else:
                _log = False
            method = WRITER_METHODS.get(key, "add_scalar")
            if _log and self.writer is not None:
                getattr(self.writer, method)(key, item, global_step=collected_frames)
            if method == "add_scalar" and self.progress_bar:
                if isinstance(item, torch.Tensor):
                    item = item.item()
                self._pbar_str[key] = item

    def _pbar_description(self) -> None:
        if self.progress_bar:
            self._pbar.set_description(
                ", ".join(
                    [
                        f"{key}: {item:{TYPE_DESCR.get(type(item), '4.4f')}}"
                        for key, item in self._pbar_str.items()
                    ]
                )
            )

    def __repr__(self) -> str:
        loss_str = indent(f"loss={self.loss_module}", 4 * " ")
        policy_str = indent(f"policy_exploration={self.policy_exploration}", 4 * " ")
        collector_str = indent(f"collector={self.collector}", 4 * " ")
        optimizer_str = indent(f"optimizer={self.optimizer}", 4 * " ")
        writer = indent(f"writer={self.writer}", 4 * " ")

        string = "\n".join(
            [
                loss_str,
                policy_str,
                collector_str,
                optimizer_str,
                writer,
            ]
        )
        string = f"Agent(\n{string})"
        return string


class SelectKeys:
    def __init__(self, keys: Sequence[str]):
        if isinstance(keys, str):
            raise RuntimeError(
                "Expected keys to be an iterable of str, got str instead"
            )
        self.keys = keys

    def __call__(self, batch: _TensorDict) -> _TensorDict:
        return batch.select(*self.keys)


class ReplayBufferTrainer:
    """
    Args:
        replay_buffer (ReplayBuffer): replay buffer to be used.
        batch_size (int): batch size when sampling data from the
            latest collection or from the replay buffer.
    """

    def __init__(self, replay_buffer: ReplayBuffer, batch_size: int) -> None:
        self.replay_buffer = replay_buffer
        self.batch_size = batch_size

    def extend(self, batch: _TensorDict) -> _TensorDict:
        if "mask" in batch.keys():
            batch = batch[batch.get("mask").squeeze(-1)]
        else:
            batch = batch.reshape(-1)
        # reward_training = batch.get("reward").mean().item()
        batch = batch.cpu()
        self.replay_buffer.extend(batch)

    def sample(self, batch: _TensorDict) -> _TensorDict:
        return self.replay_buffer.sample(self.batch_size)

    def update_priority(self, batch: _TensorDict) -> None:
        if isinstance(self.replay_buffer, TensorDictPrioritizedReplayBuffer):
            self.replay_buffer.update_priority(batch)


class LogReward:
    def __init__(self, logname="reward_training"):
        self.logname = logname

    def __call__(self, batch: _TensorDict) -> Tuple[str, torch.Tensor]:
        if "mask" in batch.keys():
            return (
                self.logname,
                batch.get("reward")[batch.get("mask").squeeze(-1)].mean().item(),
            )
        return self.logname, batch.get("reward").mean().item()


class RewardNormalizer:
    def __init__(self, decay: float = 0.999):
        self._normalize_has_been_called = False
        self._update_has_been_called = False
        self._reward_stats = OrderedDict()
        self._reward_stats["decay"] = decay
        pass

    @torch.no_grad()
    def update_reward_stats(self, batch: _TensorDict) -> None:
        reward = batch.get("reward")
        if "mask" in batch.keys():
            reward = reward[batch.get("mask").squeeze(-1)]
        if self._update_has_been_called and not self._normalize_has_been_called:
            raise RuntimeError(
                "There have been two consecutive calls to update_reward_stats without a call to normalize_reward. "
                "Check that normalize_reward has been registered in the agent."
            )
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

        self._reward_stats["mean"] = sum / count
        if count > 1:
            var = self._reward_stats["var"] = (ssq - sum.pow(2) / count) / (count - 1)
        else:
            var = self._reward_stats["var"] = torch.zeros_like(sum)

        self._reward_stats["std"] = var.clamp_min(1e-6).sqrt()
        self._update_has_been_called = True

    def normalize_reward(self, tensordict: _TensorDict) -> _TensorDict:
        reward = tensordict.get("reward")
        reward = reward - self._reward_stats["mean"]
        reward = reward / self._reward_stats["std"]
        tensordict.set_("reward", reward)
        self._normalize_has_been_called = True
        return tensordict


def mask_batch(batch: _TensorDict) -> _TensorDict:
    if "mask" in batch.keys():
        mask = batch.get("mask")
        return batch[mask.squeeze(-1)]


class BatchSubSampler:
    """
    Args:
        batch_size (int): sub-batch size to collect. The provided batch size
            must be equal to the total number of items in the output tensordict,
            which will have size [batch_size // sub_traj_len, sub_traj_len].
        sub_traj_len (int, optional): length of the trajectories that
            sub-samples must have in online settings. Default is -1 (i.e.
            takes the full length of the trajectory)
        min_sub_traj_len (int, optional): minimum value of `sub_traj_len`, in
            case some elements of the batch contain few steps.
            Default is -1 (i.e. no minimum value)

    """

    def __init__(
        self, batch_size: int, sub_traj_len: int = 0, min_sub_traj_len: int = 0
    ) -> None:
        self.batch_size = batch_size
        self.sub_traj_len = sub_traj_len
        self.min_sub_traj_len = min_sub_traj_len

    def __call__(self, batch: _TensorDict) -> _TensorDict:
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
        if batch_size == 0:
            raise RuntimeError(
                "Resulting batch size is zero. The batch size given to "
                "BatchSubSampler must be equal to the total number of elements "
                "that will result in a batch provided to the loss function."
            )
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


class Recorder:
    """
    Args:
        record_interval (int): total number of optimisation steps
            between two calls to the recorder for testing.
        record_frames (int): number of frames to be recorded during
            testing.
        frame_skip (int): frame_skip used in the environment. It is
            important to let the agent know the number of frames skipped at
            each iteration, otherwise the frame count can be underestimated.
            For logging, this parameter is important to normalize the reward.
            Finally, to compare different runs with different frame_skip,
            one must normalize the frame count and rewards. Default is 1.
        policy_exploration (ProbabilisticTDModule): a policy
            instance used for

            (1) updating the exploration noise schedule;

            (2) testing the policy on the recorder.

            Given that this instance is supposed to both explore and render
            the performance of the policy, it should be possible to turn off
            the explorative behaviour by calling the
            `set_exploration_mode('mode')` context manager.
        recorder (_EnvClass): An environment instance to be used
            for testing.
        exploration_mode (str, optional): exploration mode to use for the
            policy. By default, no exploration is used and the value used is
            "mode". Set to "random" to enable exploration
        out_key (str, optional): reward key to set to the logger. Default is
            `"reward_evaluation"`.

    """

    def __init__(
        self,
        record_interval: int,
        record_frames: int,
        frame_skip: int,
        policy_exploration: TDModule,
        recorder: _EnvClass,
        exploration_mode: str = "mode",
        out_key: str = "reward_evaluation",
    ) -> None:

        self.policy_exploration = policy_exploration
        self.recorder = recorder
        self.record_frames = record_frames
        self.frame_skip = frame_skip
        self._count = 0
        self.record_interval = record_interval
        self.exploration_mode = exploration_mode
        self.out_key = out_key

    @torch.no_grad()
    def __call__(self, batch: _TensorDict) -> Tuple[str, torch.Tensor]:
        out = None
        if self._count % self.record_interval == 0:
            with set_exploration_mode(self.exploration_mode):
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
                self.recorder.transform.dump()
                out = self.out_key, reward
        self._count += 1
        return out


class UpdateWeights:
    def __init__(self, collector: _DataCollector, update_weights_interval: int):
        self.collector = collector
        self.update_weights_interval = update_weights_interval
        self.counter = 0

    def __call__(self):
        self.counter += 1
        if self.counter % self.update_weights_interval == 0:
            self.collector.update_policy_weights_()


class CountFramesLog:
    def __init__(self, frame_skip: int):
        self.frame_count = 0
        self.frame_skip = frame_skip

    def __call__(self, batch: _TensorDict) -> Tuple[str, int]:
        if "mask" in batch.keys():
            current_frames = batch.get("mask").sum().item() * self.frame_skip
        else:
            current_frames = batch.numel() * self.frame_skip
        self.frame_count += current_frames
        return "collected_frames", current_frames


def _check_input_output_typehint(func: Callable, input: Type, output: Type):
    # Placeholder for a function that checks the types input / output against expectations
    return
