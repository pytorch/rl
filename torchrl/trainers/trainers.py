# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import abc
import pathlib
import warnings
from collections import defaultdict, OrderedDict
from copy import deepcopy
from textwrap import indent
from typing import Any, Callable, Sequence, Tuple

import numpy as np
import torch.nn
from tensordict import pad, TensorDictBase
from tensordict.nn import TensorDictModule
from tensordict.utils import expand_right
from torch import nn, optim

from torchrl._utils import (
    _CKPT_BACKEND,
    KeyDependentDefaultDict,
    logger as torchrl_logger,
    VERBOSE,
)
from torchrl.collectors.collectors import DataCollectorBase
from torchrl.collectors.utils import split_trajectories
from torchrl.data.replay_buffers import (
    TensorDictPrioritizedReplayBuffer,
    TensorDictReplayBuffer,
)
from torchrl.data.utils import DEVICE_TYPING
from torchrl.envs.common import EnvBase
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.objectives.common import LossModule
from torchrl.record.loggers import Logger

try:
    from tqdm import tqdm

    _has_tqdm = True
except ImportError:
    _has_tqdm = False

try:
    from torchsnapshot import Snapshot, StateDict

    _has_ts = True
except ImportError:
    _has_ts = False

REPLAY_BUFFER_CLASS = {
    "prioritized": TensorDictPrioritizedReplayBuffer,
    "circular": TensorDictReplayBuffer,
}

LOGGER_METHODS = {
    "grad_norm": "log_scalar",
    "loss": "log_scalar",
}

TYPE_DESCR = {float: "4.4f", int: ""}
REWARD_KEY = ("next", "reward")


class TrainerHookBase:
    """An abstract hooking class for torchrl Trainer class."""

    @abc.abstractmethod
    def state_dict(self) -> dict[str, Any]:
        raise NotImplementedError

    @abc.abstractmethod
    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def register(self, trainer: Trainer, name: str):
        """Registers the hook in the trainer at a default location.

        Args:
            trainer (Trainer): the trainer where the hook must be registered.
            name (str): the name of the hook.

        .. note::
          To register the hook at another location than the default, use
          :meth:`~torchrl.trainers.Trainer.register_op`.

        """
        raise NotImplementedError


class Trainer:
    """A generic Trainer class.

    A trainer is responsible for collecting data and training the model.
    To keep the class as versatile as possible, Trainer does not construct any
    of its specific operations: they all must be hooked at specific points in
    the training loop.

    To build a Trainer, one needs an iterable data source (a :obj:`collector`), a
    loss module and an optimizer.

    Args:
        collector (Sequence[TensorDictBase]): An iterable returning batches of
            data in a TensorDict form of shape [batch x time steps].
        total_frames (int): Total number of frames to be collected during
            training.
        loss_module (LossModule): A module that reads TensorDict batches
            (possibly sampled from a replay buffer) and return a loss
            TensorDict where every key points to a different loss component.
        optimizer (optim.Optimizer): An optimizer that trains the parameters
            of the model.
        logger (Logger, optional): a Logger that will handle the logging.
        optim_steps_per_batch (int): number of optimization steps
            per collection of data. An trainer works as follows: a main loop
            collects batches of data (epoch loop), and a sub-loop (training
            loop) performs model updates in between two collections of data.
        clip_grad_norm (bool, optional): If True, the gradients will be clipped
            based on the total norm of the model parameters. If False,
            all the partial derivatives will be clamped to
            (-clip_norm, clip_norm). Default is ``True``.
        clip_norm (Number, optional): value to be used for clipping gradients.
            Default is None (no clip norm).
        progress_bar (bool, optional): If True, a progress bar will be
            displayed using tqdm. If tqdm is not installed, this option
            won't have any effect. Default is ``True``
        seed (int, optional): Seed to be used for the collector, pytorch and
            numpy. Default is ``None``.
        save_trainer_interval (int, optional): How often the trainer should be
            saved to disk, in frame count. Default is 10000.
        log_interval (int, optional): How often the values should be logged,
            in frame count. Default is 10000.
        save_trainer_file (path, optional): path where to save the trainer.
            Default is None (no saving)
    """

    @classmethod
    def __new__(cls, *args, **kwargs):
        # trackers
        cls._optim_count: int = 0
        cls._collected_frames: int = 0
        cls._last_log: dict[str, Any] = {}
        cls._last_save: int = 0
        cls.collected_frames = 0
        cls._app_state = None
        return super().__new__(cls)

    def __init__(
        self,
        *,
        collector: DataCollectorBase,
        total_frames: int,
        frame_skip: int,
        optim_steps_per_batch: int,
        loss_module: LossModule | Callable[[TensorDictBase], TensorDictBase],
        optimizer: optim.Optimizer | None = None,
        logger: Logger | None = None,
        clip_grad_norm: bool = True,
        clip_norm: float = None,
        progress_bar: bool = True,
        seed: int = None,
        save_trainer_interval: int = 10000,
        log_interval: int = 10000,
        save_trainer_file: str | pathlib.Path | None = None,
    ) -> None:

        # objects
        self.frame_skip = frame_skip
        self.collector = collector
        self.loss_module = loss_module
        self.optimizer = optimizer
        self.logger = logger

        self._log_interval = log_interval

        # seeding
        self.seed = seed
        if seed is not None:
            self.set_seed()

        # constants
        self.optim_steps_per_batch = optim_steps_per_batch
        self.total_frames = total_frames
        self.clip_grad_norm = clip_grad_norm
        self.clip_norm = clip_norm
        if progress_bar and not _has_tqdm:
            warnings.warn(
                "tqdm library not found. "
                "Consider installing tqdm to use the Trainer progress bar."
            )
        self.progress_bar = progress_bar and _has_tqdm
        self.save_trainer_interval = save_trainer_interval
        self.save_trainer_file = save_trainer_file

        self._log_dict = defaultdict(lambda: [])

        self._batch_process_ops = []
        self._post_steps_ops = []
        self._post_steps_log_ops = []
        self._pre_steps_log_ops = []
        self._post_optim_log_ops = []
        self._pre_optim_ops = []
        self._post_loss_ops = []
        self._optimizer_ops = []
        self._process_optim_batch_ops = []
        self._post_optim_ops = []
        self._modules = {}

        if self.optimizer is not None:
            optimizer_hook = OptimizerHook(self.optimizer)
            optimizer_hook.register(self)

    def register_module(self, module_name: str, module: Any) -> None:
        if module_name in self._modules:
            raise RuntimeError(
                f"{module_name} is already registered, choose a different name."
            )
        self._modules[module_name] = module

    def _get_state(self):
        if _CKPT_BACKEND == "torchsnapshot":
            state = StateDict(
                collected_frames=self.collected_frames,
                _last_log=self._last_log,
                _last_save=self._last_save,
                _optim_count=self._optim_count,
            )
        else:
            state = OrderedDict(
                collected_frames=self.collected_frames,
                _last_log=self._last_log,
                _last_save=self._last_save,
                _optim_count=self._optim_count,
            )
        return state

    @property
    def app_state(self):
        self._app_state = {
            "state": StateDict(**self._get_state()),
            "collector": self.collector,
            "loss_module": self.loss_module,
            **{k: item for k, item in self._modules.items()},
        }
        return self._app_state

    def state_dict(self) -> dict:
        state = self._get_state()
        state_dict = OrderedDict(
            collector=self.collector.state_dict(),
            loss_module=self.loss_module.state_dict(),
            state=state,
            **{k: item.state_dict() for k, item in self._modules.items()},
        )
        return state_dict

    def load_state_dict(self, state_dict: dict) -> None:
        model_state_dict = state_dict["loss_module"]
        collector_state_dict = state_dict["collector"]

        self.loss_module.load_state_dict(model_state_dict)
        self.collector.load_state_dict(collector_state_dict)
        for key, item in self._modules.items():
            item.load_state_dict(state_dict[key])

        self.collected_frames = state_dict["state"]["collected_frames"]
        self._last_log = state_dict["state"]["_last_log"]
        self._last_save = state_dict["state"]["_last_save"]
        self._optim_count = state_dict["state"]["_optim_count"]

    def _save_trainer(self) -> None:
        if _CKPT_BACKEND == "torchsnapshot":
            if not _has_ts:
                raise ImportError(
                    "torchsnapshot not found. Consider installing torchsnapshot or "
                    "using the torch checkpointing backend (`CKPT_BACKEND=torch`)"
                )
            Snapshot.take(app_state=self.app_state, path=self.save_trainer_file)
        elif _CKPT_BACKEND == "torch":
            torch.save(self.state_dict(), self.save_trainer_file)
        else:
            raise NotImplementedError(
                f"CKPT_BACKEND should be one of {_CKPT_BACKEND.backends}, got {_CKPT_BACKEND}."
            )

    def save_trainer(self, force_save: bool = False) -> None:
        _save = force_save
        if self.save_trainer_file is not None:
            if (self.collected_frames - self._last_save) > self.save_trainer_interval:
                self._last_save = self.collected_frames
                _save = True
        if _save and self.save_trainer_file:
            self._save_trainer()

    def load_from_file(self, file: str | pathlib.Path, **kwargs) -> Trainer:
        """Loads a file and its state-dict in the trainer.

        Keyword arguments are passed to the :func:`~torch.load` function.

        """
        if _CKPT_BACKEND == "torchsnapshot":
            snapshot = Snapshot(path=file)
            snapshot.restore(app_state=self.app_state)
        elif _CKPT_BACKEND == "torch":
            loaded_dict: OrderedDict = torch.load(file, **kwargs)
            self.load_state_dict(loaded_dict)
        return self

    def set_seed(self):
        seed = self.collector.set_seed(self.seed, static_seed=False)
        torch.manual_seed(seed)
        np.random.seed(seed)

    @property
    def collector(self) -> DataCollectorBase:
        return self._collector

    @collector.setter
    def collector(self, collector: DataCollectorBase) -> None:
        self._collector = collector

    def register_op(self, dest: str, op: Callable, **kwargs) -> None:
        if dest == "batch_process":
            _check_input_output_typehint(
                op, input=TensorDictBase, output=TensorDictBase
            )
            self._batch_process_ops.append((op, kwargs))

        elif dest == "pre_optim_steps":
            _check_input_output_typehint(op, input=None, output=None)
            self._pre_optim_ops.append((op, kwargs))

        elif dest == "process_optim_batch":
            _check_input_output_typehint(
                op, input=TensorDictBase, output=TensorDictBase
            )
            self._process_optim_batch_ops.append((op, kwargs))

        elif dest == "post_loss":
            _check_input_output_typehint(
                op, input=TensorDictBase, output=TensorDictBase
            )
            self._post_loss_ops.append((op, kwargs))

        elif dest == "optimizer":
            _check_input_output_typehint(
                op, input=[TensorDictBase, bool, float, int], output=TensorDictBase
            )
            self._optimizer_ops.append((op, kwargs))

        elif dest == "post_steps":
            _check_input_output_typehint(op, input=None, output=None)
            self._post_steps_ops.append((op, kwargs))

        elif dest == "post_optim":
            _check_input_output_typehint(op, input=None, output=None)
            self._post_optim_ops.append((op, kwargs))

        elif dest == "pre_steps_log":
            _check_input_output_typehint(
                op, input=TensorDictBase, output=Tuple[str, float]
            )
            self._pre_steps_log_ops.append((op, kwargs))

        elif dest == "post_steps_log":
            _check_input_output_typehint(
                op, input=TensorDictBase, output=Tuple[str, float]
            )
            self._post_steps_log_ops.append((op, kwargs))

        elif dest == "post_optim_log":
            _check_input_output_typehint(
                op, input=TensorDictBase, output=Tuple[str, float]
            )
            self._post_optim_log_ops.append((op, kwargs))

        else:
            raise RuntimeError(
                f"The hook collection {dest} is not recognised. Choose from:"
                f"(batch_process, pre_steps, pre_step, post_loss, post_steps, "
                f"post_steps_log, post_optim_log)"
            )

    # Process batch
    def _process_batch_hook(self, batch: TensorDictBase) -> TensorDictBase:
        for op, kwargs in self._batch_process_ops:
            out = op(batch, **kwargs)
            if isinstance(out, TensorDictBase):
                batch = out
        return batch

    def _post_steps_hook(self) -> None:
        for op, kwargs in self._post_steps_ops:
            op(**kwargs)

    def _post_optim_log(self, batch: TensorDictBase) -> None:
        for op, kwargs in self._post_optim_log_ops:
            result = op(batch, **kwargs)
            if result is not None:
                self._log(**result)

    def _pre_optim_hook(self):
        for op, kwargs in self._pre_optim_ops:
            op(**kwargs)

    def _process_optim_batch_hook(self, batch):
        for op, kwargs in self._process_optim_batch_ops:
            out = op(batch, **kwargs)
            if isinstance(out, TensorDictBase):
                batch = out
        return batch

    def _post_loss_hook(self, batch):
        for op, kwargs in self._post_loss_ops:
            out = op(batch, **kwargs)
            if isinstance(out, TensorDictBase):
                batch = out
        return batch

    def _optimizer_hook(self, batch):
        for i, (op, kwargs) in enumerate(self._optimizer_ops):
            out = op(batch, self.clip_grad_norm, self.clip_norm, i, **kwargs)
            if isinstance(out, TensorDictBase):
                batch = out
        return batch.detach()

    def _post_optim_hook(self):
        for op, kwargs in self._post_optim_ops:
            op(**kwargs)

    def _pre_steps_log_hook(self, batch: TensorDictBase) -> None:
        for op, kwargs in self._pre_steps_log_ops:
            result = op(batch, **kwargs)
            if result is not None:
                self._log(**result)

    def _post_steps_log_hook(self, batch: TensorDictBase) -> None:
        for op, kwargs in self._post_steps_log_ops:
            result = op(batch, **kwargs)
            if result is not None:
                self._log(**result)

    def train(self):
        if self.progress_bar:
            self._pbar = tqdm(total=self.total_frames)
            self._pbar_str = {}

        for batch in self.collector:
            batch = self._process_batch_hook(batch)
            current_frames = (
                batch.get(("collector", "mask"), torch.tensor(batch.numel()))
                .sum()
                .item()
                * self.frame_skip
            )
            self.collected_frames += current_frames
            self._pre_steps_log_hook(batch)

            if self.collected_frames > self.collector.init_random_frames:
                self.optim_steps(batch)
            self._post_steps_hook()

            self._post_steps_log_hook(batch)

            if self.progress_bar:
                self._pbar.update(current_frames)
                self._pbar_description()

            if self.collected_frames >= self.total_frames:
                self.save_trainer(force_save=True)
                break
            self.save_trainer()

        self.collector.shutdown()

    def __del__(self):
        try:
            self.collector.shutdown()
        except Exception:
            pass

    def shutdown(self):
        if VERBOSE:
            torchrl_logger.info("shutting down collector")
        self.collector.shutdown()

    def optim_steps(self, batch: TensorDictBase) -> None:
        average_losses = None

        self._pre_optim_hook()

        for j in range(self.optim_steps_per_batch):
            self._optim_count += 1

            sub_batch = self._process_optim_batch_hook(batch)
            losses_td = self.loss_module(sub_batch)
            self._post_loss_hook(sub_batch)

            losses_detached = self._optimizer_hook(losses_td)
            self._post_optim_hook()
            self._post_optim_log(sub_batch)

            if average_losses is None:
                average_losses: TensorDictBase = losses_detached
            else:
                for key, item in losses_detached.items():
                    val = average_losses.get(key)
                    average_losses.set(key, val * j / (j + 1) + item / (j + 1))
            del sub_batch, losses_td, losses_detached

        if self.optim_steps_per_batch > 0:
            self._log(
                optim_steps=self._optim_count,
                **average_losses,
            )

    def _log(self, log_pbar=False, **kwargs) -> None:
        collected_frames = self.collected_frames
        for key, item in kwargs.items():
            self._log_dict[key].append(item)
            if (collected_frames - self._last_log.get(key, 0)) > self._log_interval:
                self._last_log[key] = collected_frames
                _log = True
            else:
                _log = False
            method = LOGGER_METHODS.get(key, "log_scalar")
            if _log and self.logger is not None:
                getattr(self.logger, method)(key, item, step=collected_frames)
            if method == "log_scalar" and self.progress_bar and log_pbar:
                if isinstance(item, torch.Tensor):
                    item = item.item()
                self._pbar_str[key] = item

    def _pbar_description(self) -> None:
        if self.progress_bar:
            self._pbar.set_description(
                ", ".join(
                    [
                        f"{key}: {self._pbar_str[key] :{TYPE_DESCR.get(type(self._pbar_str[key]), '4.4f')}}"
                        for key in sorted(self._pbar_str.keys())
                    ]
                )
            )

    def __repr__(self) -> str:
        loss_str = indent(f"loss={self.loss_module}", 4 * " ")
        collector_str = indent(f"collector={self.collector}", 4 * " ")
        optimizer_str = indent(f"optimizer={self.optimizer}", 4 * " ")
        logger = indent(f"logger={self.logger}", 4 * " ")

        string = "\n".join(
            [
                loss_str,
                collector_str,
                optimizer_str,
                logger,
            ]
        )
        string = f"Trainer(\n{string})"
        return string


def _get_list_state_dict(hook_list):
    out = []
    for item, kwargs in hook_list:
        if hasattr(item, "state_dict"):
            out.append((item.state_dict(), kwargs))
        else:
            out.append((None, kwargs))
    return out


def _load_list_state_dict(list_state_dict, hook_list):
    for i, ((state_dict_item, kwargs), (item, _)) in enumerate(
        zip(list_state_dict, hook_list)
    ):
        if state_dict_item is not None:
            item.load_state_dict(state_dict_item)
            hook_list[i] = (item, kwargs)


class SelectKeys(TrainerHookBase):
    """Selects keys in a TensorDict batch.

    Args:
        keys (iterable of strings): keys to be selected in the tensordict.

    Examples:
        >>> trainer = make_trainer()
        >>> key1 = "first key"
        >>> key2 = "second key"
        >>> td = TensorDict(
        ...     {
        ...         key1: torch.randn(3),
        ...         key2: torch.randn(3),
        ...     },
        ...     [],
        ... )
        >>> trainer.register_op("batch_process", SelectKeys([key1]))
        >>> td_out = trainer._process_batch_hook(td)
        >>> assert key1 in td_out.keys()
        >>> assert key2 not in td_out.keys()

    """

    def __init__(self, keys: Sequence[str]):
        if isinstance(keys, str):
            raise RuntimeError(
                "Expected keys to be an iterable of str, got str instead"
            )
        self.keys = keys

    def __call__(self, batch: TensorDictBase) -> TensorDictBase:
        return batch.select(*self.keys)

    def state_dict(self) -> dict[str, Any]:
        return {}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        pass

    def register(self, trainer, name="select_keys") -> None:
        trainer.register_op("batch_process", self)
        trainer.register_module(name, self)


class ReplayBufferTrainer(TrainerHookBase):
    """Replay buffer hook provider.

    Args:
        replay_buffer (TensorDictReplayBuffer): replay buffer to be used.
        batch_size (int, optional): batch size when sampling data from the
            latest collection or from the replay buffer. If none is provided,
            the replay buffer batch-size will be used (preferred option for
            unchanged batch-sizes).
        memmap (bool, optional): if ``True``, a memmap tensordict is created.
            Default is ``False``.
        device (device, optional): device where the samples must be placed.
            Default to ``None``.
        flatten_tensordicts (bool, optional): if ``True``, the tensordicts will be
            flattened (or equivalently masked with the valid mask obtained from
            the collector) before being passed to the replay buffer. Otherwise,
            no transform will be achieved other than padding (see :obj:`max_dims` arg below).
            Defaults to ``False``.
        max_dims (sequence of int, optional): if :obj:`flatten_tensordicts` is set to False,
            this will be a list of the length of the batch_size of the provided
            tensordicts that represent the maximum size of each. If provided,
            this list of sizes will be used to pad the tensordict and make their shape
            match before they are passed to the replay buffer. If there is no
            maximum value, a -1 value should be provided.

    Examples:
        >>> rb_trainer = ReplayBufferTrainer(replay_buffer=replay_buffer, batch_size=N)
        >>> trainer.register_op("batch_process", rb_trainer.extend)
        >>> trainer.register_op("process_optim_batch", rb_trainer.sample)
        >>> trainer.register_op("post_loss", rb_trainer.update_priority)

    """

    def __init__(
        self,
        replay_buffer: TensorDictReplayBuffer,
        batch_size: int | None = None,
        memmap: bool = False,
        device: DEVICE_TYPING | None = None,
        flatten_tensordicts: bool = False,
        max_dims: Sequence[int] | None = None,
    ) -> None:
        self.replay_buffer = replay_buffer
        self.batch_size = batch_size
        self.memmap = memmap
        self.device = device
        self.flatten_tensordicts = flatten_tensordicts
        self.max_dims = max_dims

    def extend(self, batch: TensorDictBase) -> TensorDictBase:
        if self.flatten_tensordicts:
            if ("collector", "mask") in batch.keys(True):
                batch = batch[batch.get(("collector", "mask"))]
            else:
                batch = batch.reshape(-1)
        else:
            if self.max_dims is not None:
                pads = []
                for d in range(batch.ndimension()):
                    pad_value = (
                        0
                        if self.max_dims[d] == -1
                        else self.max_dims[d] - batch.batch_size[d]
                    )
                    pads += [0, pad_value]
                batch = pad(batch, pads)
        batch = batch.cpu()
        self.replay_buffer.extend(batch)

    def sample(self, batch: TensorDictBase) -> TensorDictBase:
        sample = self.replay_buffer.sample(batch_size=self.batch_size)
        return sample.to(self.device) if self.device is not None else sample

    def update_priority(self, batch: TensorDictBase) -> None:
        self.replay_buffer.update_tensordict_priority(batch)

    def state_dict(self) -> dict[str, Any]:
        return {
            "replay_buffer": self.replay_buffer.state_dict(),
        }

    def load_state_dict(self, state_dict) -> None:
        self.replay_buffer.load_state_dict(state_dict["replay_buffer"])

    def register(self, trainer: Trainer, name: str = "replay_buffer"):
        trainer.register_op("batch_process", self.extend)
        trainer.register_op("process_optim_batch", self.sample)
        trainer.register_op("post_loss", self.update_priority)
        trainer.register_module(name, self)


class OptimizerHook(TrainerHookBase):
    """Add an optimizer for one or more loss components.

    Args:
        optimizer (optim.Optimizer): An optimizer to apply to the loss_components.
        loss_components (Sequence[str], optional): The keys in the loss TensorDict
            for which the optimizer should be appled to the respective values.
            If omitted, the optimizer is applied to all components with the
            names starting with `loss_`.

    Examples:
        >>> optimizer_hook = OptimizerHook(optimizer, ["loss_actor"])
        >>> trainer.register_op("optimizer", optimizer_hook)

    """

    def __init__(
        self,
        optimizer: optim.Optimizer,
        loss_components: Sequence[str] | None = None,
    ):
        if loss_components is not None and not loss_components:
            raise ValueError(
                "loss_components list cannot be empty. "
                "Set to None to act on all components of the loss."
            )

        self.optimizer = optimizer
        self.loss_components = loss_components
        if self.loss_components is not None:
            self.loss_components = set(self.loss_components)

    def _grad_clip(self, clip_grad_norm: bool, clip_norm: float) -> float:
        params = []
        for param_group in self.optimizer.param_groups:
            params += param_group["params"]

        if clip_grad_norm and clip_norm is not None:
            gn = nn.utils.clip_grad_norm_(params, clip_norm)
        else:
            gn = sum([p.grad.pow(2).sum() for p in params if p.grad is not None]).sqrt()
            if clip_norm is not None:
                nn.utils.clip_grad_value_(params, clip_norm)

        return float(gn)

    def __call__(
        self,
        losses_td: TensorDictBase,
        clip_grad_norm: bool,
        clip_norm: float,
        index: int,
    ) -> TensorDictBase:
        loss_components = (
            [item for key, item in losses_td.items() if key in self.loss_components]
            if self.loss_components is not None
            else [item for key, item in losses_td.items() if key.startswith("loss")]
        )
        loss = sum(loss_components)
        loss.backward()

        grad_norm = self._grad_clip(clip_grad_norm, clip_norm)
        losses_td[f"grad_norm_{index}"] = torch.tensor(grad_norm)

        self.optimizer.step()
        self.optimizer.zero_grad()

        return losses_td

    def state_dict(self) -> dict[str, Any]:
        return {}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        pass

    def register(self, trainer, name="optimizer") -> None:
        trainer.register_op("optimizer", self)
        trainer.register_module(name, self)


class ClearCudaCache(TrainerHookBase):
    """Clears cuda cache at a given interval.

    Examples:
        >>> clear_cuda = ClearCudaCache(100)
        >>> trainer.register_op("pre_optim_steps", clear_cuda)

    """

    def __init__(self, interval: int):
        self.interval = interval
        self.count = 0

    def __call__(self, *args, **kwargs):
        self.count += 1
        if self.count % self.interval == 0:
            torch.cuda.empty_cache()


class LogScalar(TrainerHookBase):
    """Reward logger hook.

    Args:
        logname (str, optional): name of the rewards to be logged. Default is :obj:`"r_training"`.
        log_pbar (bool, optional): if ``True``, the reward value will be logged on
            the progression bar. Default is ``False``.
        reward_key (str or tuple, optional): the key where to find the reward
            in the input batch. Defaults to ``("next", "reward")``

    Examples:
        >>> log_reward = LogScalar(("next", "reward"))
        >>> trainer.register_op("pre_steps_log", log_reward)

    """

    def __init__(
        self,
        logname="r_training",
        log_pbar: bool = False,
        reward_key: str | tuple = None,
    ):
        self.logname = logname
        self.log_pbar = log_pbar
        if reward_key is None:
            reward_key = REWARD_KEY
        self.reward_key = reward_key

    def __call__(self, batch: TensorDictBase) -> dict:
        if ("collector", "mask") in batch.keys(True):
            return {
                self.logname: batch.get(self.reward_key)[
                    batch.get(("collector", "mask"))
                ]
                .mean()
                .item(),
                "log_pbar": self.log_pbar,
            }
        return {
            self.logname: batch.get(self.reward_key).mean().item(),
            "log_pbar": self.log_pbar,
        }

    def register(self, trainer: Trainer, name: str = "log_reward"):
        trainer.register_op("pre_steps_log", self)
        trainer.register_module(name, self)


class LogReward(LogScalar):
    """Deprecated class. Use LogScalar instead."""

    def __init__(
        self,
        logname="r_training",
        log_pbar: bool = False,
        reward_key: str | tuple = None,
    ):
        warnings.warn(
            "The 'LogReward' class is deprecated and will be removed in v0.9. Please use 'LogScalar' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(logname=logname, log_pbar=log_pbar, reward_key=reward_key)


class RewardNormalizer(TrainerHookBase):
    """Reward normalizer hook.

    Args:
        decay (:obj:`float`, optional): exponential moving average decay parameter.
            Default is 0.999
        scale (:obj:`float`, optional): the scale used to multiply the reward once
            normalized. Defaults to 1.0.
        eps (:obj:`float`, optional): the epsilon jitter used to prevent numerical
            underflow. Defaults to ``torch.finfo(DEFAULT_DTYPE).eps``
            where ``DEFAULT_DTYPE=torch.get_default_dtype()``.
        reward_key (str or tuple, optional): the key where to find the reward
            in the input batch. Defaults to ``("next", "reward")``

    Examples:
        >>> reward_normalizer = RewardNormalizer()
        >>> trainer.register_op("batch_process", reward_normalizer.update_reward_stats)
        >>> trainer.register_op("process_optim_batch", reward_normalizer.normalize_reward)

    """

    def __init__(
        self,
        decay: float = 0.999,
        scale: float = 1.0,
        eps: float = None,
        log_pbar: bool = False,
        reward_key=None,
    ):
        self._normalize_has_been_called = False
        self._update_has_been_called = False
        self._reward_stats = OrderedDict()
        self._reward_stats["decay"] = decay
        self.scale = scale
        if eps is None:
            eps = torch.finfo(torch.get_default_dtype()).eps
        self.eps = eps
        if reward_key is None:
            reward_key = REWARD_KEY
        self.reward_key = reward_key

    @torch.no_grad()
    def update_reward_stats(self, batch: TensorDictBase) -> None:
        reward = batch.get(self.reward_key)
        if ("collector", "mask") in batch.keys(True):
            reward = reward[batch.get(("collector", "mask"))]
        if self._update_has_been_called and not self._normalize_has_been_called:
            # We'd like to check that rewards are normalized. Problem is that the trainer can collect data without calling steps...
            # raise RuntimeError(
            #     "There have been two consecutive calls to update_reward_stats without a call to normalize_reward. "
            #     "Check that normalize_reward has been registered in the trainer."
            # )
            pass
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

        self._reward_stats["std"] = var.clamp_min(self.eps).sqrt()
        self._update_has_been_called = True

    def normalize_reward(self, tensordict: TensorDictBase) -> TensorDictBase:
        tensordict = tensordict.to_tensordict()  # make sure it is not a SubTensorDict
        reward = tensordict.get(self.reward_key)

        if reward.device is not None:
            reward = reward - self._reward_stats["mean"].to(reward.device)
            reward = reward / self._reward_stats["std"].to(reward.device)
        else:
            reward = reward - self._reward_stats["mean"]
            reward = reward / self._reward_stats["std"]

        tensordict.set(self.reward_key, reward * self.scale)
        self._normalize_has_been_called = True
        return tensordict

    def state_dict(self) -> dict[str, Any]:
        return {
            "_reward_stats": deepcopy(self._reward_stats),
            "scale": self.scale,
            "_normalize_has_been_called": self._normalize_has_been_called,
            "_update_has_been_called": self._update_has_been_called,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        for key, value in state_dict.items():
            setattr(self, key, value)

    def register(self, trainer: Trainer, name: str = "reward_normalizer"):
        trainer.register_op("batch_process", self.update_reward_stats)
        trainer.register_op("process_optim_batch", self.normalize_reward)
        trainer.register_module(name, self)


def mask_batch(batch: TensorDictBase) -> TensorDictBase:
    """Batch masking hook.

    If a tensordict contained padded trajectories but only single events are
    needed, this hook can be used to select the valid events from the original
    tensordict.

    Args:
        batch:

    Examples:
        >>> trainer = mocking_trainer()
        >>> trainer.register_op("batch_process", mask_batch)

    """
    if ("collector", "mask") in batch.keys(True):
        mask = batch.get(("collector", "mask"))
        return batch[mask]
    return batch


class BatchSubSampler(TrainerHookBase):
    """Data subsampler for online RL sota-implementations.

    This class subsamples a part of a whole batch of data just collected from the
    environment.

    Args:
        batch_size (int): sub-batch size to collect. The provided batch size
            must be equal to the total number of items in the output tensordict,
            which will have size [batch_size // sub_traj_len, sub_traj_len].
        sub_traj_len (int, optional): length of the trajectories that
            sub-samples must have in online settings. Default is -1 (i.e.
            takes the full length of the trajectory)
        min_sub_traj_len (int, optional): minimum value of :obj:`sub_traj_len`, in
            case some elements of the batch contain few steps.
            Default is -1 (i.e. no minimum value)

    Examples:
        >>> td = TensorDict(
        ...     {
        ...         key1: torch.stack([torch.arange(0, 10), torch.arange(10, 20)], 0),
        ...         key2: torch.stack([torch.arange(0, 10), torch.arange(10, 20)], 0),
        ...     },
        ...     [2, 10],
        ... )
        >>> trainer.register_op(
        ...     "process_optim_batch",
        ...     BatchSubSampler(batch_size=batch_size, sub_traj_len=sub_traj_len),
        ... )
        >>> td_out = trainer._process_optim_batch_hook(td)
        >>> assert td_out.shape == torch.Size([batch_size // sub_traj_len, sub_traj_len])

    """

    def __init__(
        self, batch_size: int, sub_traj_len: int = 0, min_sub_traj_len: int = 0
    ) -> None:
        self.batch_size = batch_size
        self.sub_traj_len = sub_traj_len
        self.min_sub_traj_len = min_sub_traj_len

    def __call__(self, batch: TensorDictBase) -> TensorDictBase:
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
        if ("collector", "mask") in batch.keys(True):
            # if a valid mask is present, it's important to sample only
            # valid steps
            traj_len = batch.get(("collector", "mask")).sum(-1)
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
        valid_trajectories = torch.arange(batch.shape[0], device=batch.device)[len_mask]

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
        if ("collector", "mask") in batch.keys(True) and not td.get(
            ("collector", "mask")
        ).all():
            raise RuntimeError("Sampled invalid steps")
        return td

    def state_dict(self) -> dict[str, Any]:
        return {}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        pass

    def register(self, trainer: Trainer, name: str = "batch_subsampler"):
        trainer.register_op(
            "process_optim_batch",
            self,
        )
        trainer.register_module(name, self)


class LogValidationReward(TrainerHookBase):
    """Recorder hook for :class:`~torchrl.trainers.Trainer`.

    Args:
        record_interval (int): total number of optimization steps
            between two calls to the recorder for testing.
        record_frames (int): number of frames to be recorded during
            testing.
        frame_skip (int): frame_skip used in the environment. It is
            important to let the trainer know the number of frames skipped at
            each iteration, otherwise the frame count can be underestimated.
            For logging, this parameter is important to normalize the reward.
            Finally, to compare different runs with different frame_skip,
            one must normalize the frame count and rewards. Defaults to ``1``.
        policy_exploration (ProbabilisticTDModule): a policy
            instance used for

            (1) updating the exploration noise schedule;

            (2) testing the policy on the recorder.

            Given that this instance is supposed to both explore and render
            the performance of the policy, it should be possible to turn off
            the explorative behavior by calling the
            `set_exploration_type(ExplorationType.DETERMINISTIC)` context manager.
        environment (EnvBase): An environment instance to be used
            for testing.
        exploration_type (ExplorationType, optional): exploration mode to use for the
            policy. By default, no exploration is used and the value used is
            ``ExplorationType.DETERMINISTIC``. Set to ``ExplorationType.RANDOM`` to enable exploration
        log_keys (sequence of str or tuples or str, optional): keys to read in the tensordict
            for logging. Defaults to ``[("next", "reward")]``.
        out_keys (Dict[str, str], optional): a dictionary mapping the ``log_keys``
            to their name in the logs. Defaults to ``{("next", "reward"): "r_evaluation"}``.
        suffix (str, optional): suffix of the video to be recorded.
        log_pbar (bool, optional): if ``True``, the reward value will be logged on
            the progression bar. Default is `False`.

    """

    ENV_DEPREC = (
        "the environment should be passed under the 'environment' key"
        " and not the 'recorder' key."
    )

    def __init__(
        self,
        *,
        record_interval: int,
        record_frames: int,
        frame_skip: int = 1,
        policy_exploration: TensorDictModule,
        environment: EnvBase = None,
        exploration_type: ExplorationType = ExplorationType.RANDOM,
        log_keys: list[str | tuple[str]] | None = None,
        out_keys: dict[str | tuple[str], str] | None = None,
        suffix: str | None = None,
        log_pbar: bool = False,
        recorder: EnvBase = None,
    ) -> None:
        if environment is None and recorder is not None:
            warnings.warn(self.ENV_DEPREC)
            environment = recorder
        elif environment is not None and recorder is not None:
            raise ValueError("environment and recorder conflict.")
        self.policy_exploration = policy_exploration
        self.environment = environment
        self.record_frames = record_frames
        self.frame_skip = frame_skip
        self._count = 0
        self.record_interval = record_interval
        self.exploration_type = exploration_type
        if log_keys is None:
            log_keys = [("next", "reward")]
        if out_keys is None:
            out_keys = KeyDependentDefaultDict(lambda x: x)
            out_keys[("next", "reward")] = "r_evaluation"
        self.log_keys = log_keys
        self.out_keys = out_keys
        self.suffix = suffix
        self.log_pbar = log_pbar

    @torch.inference_mode()
    def __call__(self, batch: TensorDictBase) -> dict:
        out = None
        if self._count % self.record_interval == 0:
            with set_exploration_type(self.exploration_type):
                if isinstance(self.policy_exploration, torch.nn.Module):
                    self.policy_exploration.eval()
                self.environment.eval()
                td_record = self.environment.rollout(
                    policy=self.policy_exploration,
                    max_steps=self.record_frames,
                    auto_reset=True,
                    auto_cast_to_device=True,
                    break_when_any_done=False,
                ).clone()
                td_record = split_trajectories(td_record)
                if isinstance(self.policy_exploration, torch.nn.Module):
                    self.policy_exploration.train()
                self.environment.train()
                self.environment.transform.dump(suffix=self.suffix)

                out = {}
                for key in self.log_keys:
                    value = td_record.get(key).float()
                    if key == ("next", "reward"):
                        mask = td_record["mask"]
                        mean_value = value[mask].mean() / self.frame_skip
                        total_value = value.sum(dim=td_record.ndim - 1).mean()
                        out[self.out_keys[key]] = mean_value
                        out["total_" + self.out_keys[key]] = total_value
                        continue
                    out[self.out_keys[key]] = value
                out["log_pbar"] = self.log_pbar
        self._count += 1
        self.environment.close()
        return out

    def state_dict(self) -> dict:
        return {
            "_count": self._count,
            "recorder_state_dict": self.environment.state_dict(),
        }

    def load_state_dict(self, state_dict: dict) -> None:
        self._count = state_dict["_count"]
        self.environment.load_state_dict(state_dict["recorder_state_dict"])

    def register(self, trainer: Trainer, name: str = "recorder"):
        trainer.register_module(name, self)
        trainer.register_op(
            "post_steps_log",
            self,
        )


class Recorder(LogValidationReward):
    """Deprecated class. Use LogValidationReward instead."""

    def __init__(
        self,
        *,
        record_interval: int,
        record_frames: int,
        frame_skip: int = 1,
        policy_exploration: TensorDictModule,
        environment: EnvBase = None,
        exploration_type: ExplorationType = ExplorationType.RANDOM,
        log_keys: list[str | tuple[str]] | None = None,
        out_keys: dict[str | tuple[str], str] | None = None,
        suffix: str | None = None,
        log_pbar: bool = False,
        recorder: EnvBase = None,
    ) -> None:
        warnings.warn(
            "The 'Recorder' class is deprecated and will be removed in v0.9. Please use 'LogValidationReward' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(
            record_interval=record_interval,
            record_frames=record_frames,
            frame_skip=frame_skip,
            policy_exploration=policy_exploration,
            environment=environment,
            exploration_type=exploration_type,
            log_keys=log_keys,
            out_keys=out_keys,
            suffix=suffix,
            log_pbar=log_pbar,
            recorder=recorder,
        )


class UpdateWeights(TrainerHookBase):
    """A collector weights update hook class.

    This hook must be used whenever the collector policy weights sit on a
    different device than the policy weights being trained by the Trainer.
    In that case, those weights must be synced across devices at regular
    intervals. If the devices match, this will result in a no-op.

    Args:
        collector (DataCollectorBase): A data collector where the policy weights
            must be synced.
        update_weights_interval (int): Interval (in terms of number of batches
            collected) where the sync must take place.

    Examples:
        >>> update_weights = UpdateWeights(trainer.collector, T)
        >>> trainer.register_op("post_steps", update_weights)

    """

    def __init__(self, collector: DataCollectorBase, update_weights_interval: int):
        self.collector = collector
        self.update_weights_interval = update_weights_interval
        self.counter = 0

    def __call__(self):
        self.counter += 1
        if self.counter % self.update_weights_interval == 0:
            self.collector.update_policy_weights_()

    def register(self, trainer: Trainer, name: str = "update_weights"):
        trainer.register_module(name, self)
        trainer.register_op(
            "post_steps",
            self,
        )

    def state_dict(self) -> dict:
        return {}

    def load_state_dict(self, state_dict) -> None:
        return


class CountFramesLog(TrainerHookBase):
    """A frame counter hook.

    Args:
        frame_skip (int): frame skip of the environment. This argument is
            important to keep track of the total number of frames, not the
            apparent one.
        log_pbar (bool, optional): if ``True``, the reward value will be logged on
            the progression bar. Default is `False`.

    Examples:
        >>> count_frames = CountFramesLog(frame_skip=frame_skip)
        >>> trainer.register_op("pre_steps_log", count_frames)


    """

    @classmethod
    def __new__(cls, *args, **kwargs):
        cls.frame_count = 0
        return super().__new__(cls)

    def __init__(self, frame_skip: int, log_pbar: bool = False):
        self.frame_skip = frame_skip
        self.log_pbar = log_pbar

    def __call__(self, batch: TensorDictBase) -> dict:
        if ("collector", "mask") in batch.keys(True):
            current_frames = (
                batch.get(("collector", "mask")).sum().item() * self.frame_skip
            )
        else:
            current_frames = batch.numel() * self.frame_skip
        self.frame_count += current_frames
        return {"n_frames": self.frame_count, "log_pbar": self.log_pbar}

    def register(self, trainer: Trainer, name: str = "count_frames_log"):
        trainer.register_module(name, self)
        trainer.register_op(
            "pre_steps_log",
            self,
        )

    def state_dict(self) -> dict:
        return {"frame_count": self.frame_count}

    def load_state_dict(self, state_dict) -> None:
        self.frame_count = state_dict["frame_count"]


def _check_input_output_typehint(
    func: Callable, input: type | list[type], output: type
):
    # Placeholder for a function that checks the types input / output against expectations
    return


def flatten_dict(d):
    """Flattens a dictionary with sub-dictionaries accessed through point-separated (:obj:`"var1.var2"`) fields."""
    out = {}
    for key, item in d.items():
        if isinstance(item, dict):
            item = flatten_dict(item)
            for _key, _item in item.items():
                out[".".join([key, _key])] = _item
        else:
            out[key] = item
    return out
