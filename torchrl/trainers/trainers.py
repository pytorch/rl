# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import abc
import itertools
import pathlib
import time
import warnings
from collections import defaultdict, OrderedDict
from collections.abc import Callable, Sequence
from copy import deepcopy
from textwrap import indent
from typing import Any, Literal

import numpy as np
import torch.nn
from tensordict import NestedKey, pad, TensorDictBase
from tensordict._tensorcollection import TensorCollection
from tensordict.nn import TensorDictModule
from tensordict.utils import expand_right
from torch import nn, optim

from torchrl._utils import (
    _CKPT_BACKEND,
    KeyDependentDefaultDict,
    logger as torchrl_logger,
    RL_WARNINGS,
    VERBOSE,
)
from torchrl.collectors import DataCollectorBase
from torchrl.collectors.utils import split_trajectories
from torchrl.data.replay_buffers import (
    PrioritizedSampler,
    TensorDictPrioritizedReplayBuffer,
    TensorDictReplayBuffer,
)
from torchrl.data.utils import DEVICE_TYPING
from torchrl.envs.common import EnvBase
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.objectives.common import LossModule
from torchrl.objectives.utils import TargetNetUpdater
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

# Mapping of metric names to logger methods - controls how different metrics are logged
LOGGER_METHODS = {
    "grad_norm": "log_scalar",
    "loss": "log_scalar",
}

# Format strings for different data types in progress bar display
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
        optim_steps_per_batch (int, optional): number of optimization steps
            per collection of data. An trainer works as follows: a main loop
            collects batches of data (epoch loop), and a sub-loop (training
            loop) performs model updates in between two collections of data.
            If `None`, the trainer will use the number of workers as the number of optimization steps.
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
        async_collection (bool, optional): Whether to collect data asynchronously.
            This will only work if the replay buffer is registed within the data collector.
            If using this, the UTD ratio (Update to Data) will be logged under the key "utd_ratio".
            Default is False.
    """

    @classmethod
    def __new__(cls, *args, **kwargs):
        # Training state trackers (used for logging and checkpointing)
        cls._optim_count: int = 0  # Total number of optimization steps completed
        cls._collected_frames: int = 0  # Total number of frames collected (deprecated)
        cls._last_log: dict[
            str, Any
        ] = {}  # Tracks when each metric was last logged (for log_interval control)
        cls._last_save: int = (
            0  # Tracks when trainer was last saved (for save_interval control)
        )
        cls.collected_frames = 0  # Total number of frames collected (current)
        cls._app_state = None  # Application state for checkpointing
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
        clip_norm: float | None = None,
        progress_bar: bool = True,
        seed: int | None = None,
        save_trainer_interval: int = 10000,
        log_interval: int = 10000,
        save_trainer_file: str | pathlib.Path | None = None,
        num_epochs: int = 1,
        async_collection: bool = False,
    ) -> None:

        # objects
        self.frame_skip = frame_skip
        self.collector = collector
        self.loss_module = loss_module
        self.optimizer = optimizer
        self.logger = logger

        # Logging frequency control - how often to log each metric (in frames)
        self._log_interval = log_interval

        # seeding
        self.seed = seed
        if seed is not None:
            self.set_seed()

        # constants
        self.optim_steps_per_batch = optim_steps_per_batch
        self.total_frames = total_frames
        self.num_epochs = num_epochs
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

        self._log_dict = defaultdict(list)

        # Hook collections for different stages of the training loop
        self._batch_process_ops = (
            []
        )  # Process collected batches (e.g., reward normalization)
        self._post_steps_ops = []  # After optimization steps (e.g., weight updates)

        # Logging hook collections - different points in training loop where logging can occur
        self._post_steps_log_ops = (
            []
        )  # After optimization steps (e.g., validation rewards)
        self._pre_steps_log_ops = (
            []
        )  # Before optimization steps (e.g., rewards, frame counts)
        self._post_optim_log_ops = (
            []
        )  # After each optimization step (e.g., gradient norms)
        self._pre_epoch_log_ops = (
            []
        )  # Before each epoch logging (e.g., epoch-specific metrics)
        self._post_epoch_log_ops = (
            []
        )  # After each epoch logging (e.g., epoch completion metrics)

        # Regular hook collections for non-logging operations
        self._pre_epoch_ops = (
            []
        )  # Before each epoch (e.g., epoch setup, cache clearing)
        self._post_epoch_ops = (
            []
        )  # After each epoch (e.g., epoch cleanup, weight syncing)

        # Optimization-related hook collections
        self._pre_optim_ops = []  # Before optimization steps (e.g., cache clearing)
        self._post_loss_ops = []  # After loss computation (e.g., priority updates)
        self._optimizer_ops = []  # During optimization (e.g., gradient clipping)
        self._process_optim_batch_ops = (
            []
        )  # Process batches for optimization (e.g., subsampling)
        self._post_optim_ops = []  # After optimization (e.g., weight syncing)

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

    def register_op(
        self,
        dest: Literal[
            "batch_process",
            "pre_optim_steps",
            "process_optim_batch",
            "post_loss",
            "optimizer",
            "post_steps",
            "post_optim",
            "pre_steps_log",
            "post_steps_log",
            "post_optim_log",
            "pre_epoch_log",
            "post_epoch_log",
            "pre_epoch",
            "post_epoch",
        ],
        op: Callable,
        **kwargs,
    ) -> None:
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
                op, input=TensorDictBase, output=tuple[str, float]
            )
            self._pre_steps_log_ops.append((op, kwargs))

        elif dest == "post_steps_log":
            _check_input_output_typehint(
                op, input=TensorDictBase, output=tuple[str, float]
            )
            self._post_steps_log_ops.append((op, kwargs))

        elif dest == "post_optim_log":
            _check_input_output_typehint(
                op, input=TensorDictBase, output=tuple[str, float]
            )
            self._post_optim_log_ops.append((op, kwargs))

        elif dest == "pre_epoch_log":
            _check_input_output_typehint(
                op, input=TensorDictBase, output=tuple[str, float]
            )
            self._pre_epoch_log_ops.append((op, kwargs))

        elif dest == "post_epoch_log":
            _check_input_output_typehint(
                op, input=TensorDictBase, output=tuple[str, float]
            )
            self._post_epoch_log_ops.append((op, kwargs))

        elif dest == "pre_epoch":
            _check_input_output_typehint(op, input=None, output=None)
            self._pre_epoch_ops.append((op, kwargs))

        elif dest == "post_epoch":
            _check_input_output_typehint(op, input=None, output=None)
            self._post_epoch_ops.append((op, kwargs))

        else:
            raise RuntimeError(
                f"The hook collection {dest} is not recognised. Choose from:"
                f"(batch_process, pre_steps, pre_step, post_loss, post_steps, "
                f"post_steps_log, post_optim_log, pre_epoch_log, post_epoch_log, "
                f"pre_epoch, post_epoch)"
            )

    register_hook = register_op

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
        """Execute logging hooks that run AFTER EACH optimization step.

        These hooks log metrics that are computed after each individual optimization step,
        such as gradient norms, individual loss components, or step-specific metrics.
        Called after each optimization step within the optimization loop.
        """
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

    def _pre_epoch_log_hook(self, batch: TensorDictBase) -> None:
        """Execute logging hooks that run BEFORE each epoch of optimization.

        These hooks log metrics that should be computed before starting a new epoch
        of optimization steps. Called once per epoch within the optimization loop.
        """
        for op, kwargs in self._pre_epoch_log_ops:
            result = op(batch, **kwargs)
            if result is not None:
                self._log(**result)

    def _pre_epoch_hook(self, batch: TensorDictBase, **kwargs) -> None:
        """Execute regular hooks that run BEFORE each epoch of optimization.

        These hooks perform non-logging operations before starting a new epoch
        of optimization steps. Called once per epoch within the optimization loop.
        """
        for op, kwargs in self._pre_epoch_ops:
            batch = op(batch, **kwargs)
        return batch

    def _post_epoch_log_hook(self, batch: TensorDictBase) -> None:
        """Execute logging hooks that run AFTER each epoch of optimization.

        These hooks log metrics that should be computed after completing an epoch
        of optimization steps. Called once per epoch within the optimization loop.
        """
        for op, kwargs in self._post_epoch_log_ops:
            result = op(batch, **kwargs)
            if result is not None:
                self._log(**result)

    def _post_epoch_hook(self) -> None:
        """Execute regular hooks that run AFTER each epoch of optimization.

        These hooks perform non-logging operations after completing an epoch
        of optimization steps. Called once per epoch within the optimization loop.
        """
        for op, kwargs in self._post_epoch_ops:
            op(**kwargs)

    def _pre_steps_log_hook(self, batch: TensorDictBase) -> None:
        """Execute logging hooks that run BEFORE optimization steps.

        These hooks typically log metrics from the collected batch data,
        such as rewards, frame counts, or other batch-level statistics.
        Called once per batch collection, before any optimization occurs.
        """
        for op, kwargs in self._pre_steps_log_ops:
            result = op(batch, **kwargs)
            if result is not None:
                self._log(**result)

    def _post_steps_log_hook(self, batch: TensorDictBase) -> None:
        """Execute logging hooks that run AFTER optimization steps.

        These hooks typically log metrics that depend on the optimization results,
        such as validation rewards, evaluation metrics, or post-training statistics.
        Called once per batch collection, after all optimization steps are complete.
        """
        for op, kwargs in self._post_steps_log_ops:
            result = op(batch, **kwargs)
            if result is not None:
                self._log(**result)

    def train(self):
        if self.progress_bar:
            self._pbar = tqdm(total=self.total_frames)
            self._pbar_str = {}

        if self.async_collection:
            self.collector.start()
            while not self.collector.getattr_rb("write_count"):
                time.sleep(0.1)

            # Create async iterator that monitors write_count progress
            iterator = self._async_iterator()
        else:
            iterator = self.collector

        for batch in iterator:
            if not self.async_collection:
                batch = self._process_batch_hook(batch)
                current_frames = (
                    batch.get(("collector", "mask"), torch.tensor(batch.numel()))
                    .sum()
                    .item()
                    * self.frame_skip
                )
                self.collected_frames += current_frames
            else:
                # In async mode, batch is None and we track frames via write_count
                batch = None
                cf = self.collected_frames
                self.collected_frames = self.collector.getattr_rb("write_count")
                current_frames = self.collected_frames - cf

            # LOGGING POINT 1: Pre-optimization logging (e.g., rewards, frame counts)
            self._pre_steps_log_hook(batch)

            if self.collected_frames >= self.collector.init_random_frames:
                self.optim_steps(batch)
            self._post_steps_hook()

            # LOGGING POINT 2: Post-optimization logging (e.g., validation rewards, evaluation metrics)
            self._post_steps_log_hook(batch)

            if self.progress_bar:
                self._pbar.update(current_frames)
                self._pbar_description()

            if self.collected_frames >= self.total_frames:
                self.save_trainer(force_save=True)
                break
            self.save_trainer()

        self.collector.shutdown()

    def _async_iterator(self):
        """Create an iterator for async collection that monitors replay buffer write_count.

        This iterator yields None batches and terminates when total_frames is reached
        based on the replay buffer's write_count rather than using a fixed range.
        This ensures the training loop properly consumes the entire collector output.
        """
        while True:
            current_write_count = self.collector.getattr_rb("write_count")
            # Check if we've reached the target frames
            if current_write_count >= self.total_frames:
                break
            else:
                yield None

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
        optim_steps_per_batch = self.optim_steps_per_batch
        j = -1

        for _ in range(self.num_epochs):
            # LOGGING POINT 3: Pre-epoch logging (e.g., epoch-specific metrics)
            self._pre_epoch_log_hook(batch)
            # Regular pre-epoch operations (e.g., epoch setup)
            batch_processed = self._pre_epoch_hook(batch)

            if optim_steps_per_batch is None:
                prog = itertools.count()
            else:
                prog = range(optim_steps_per_batch)

            for j in prog:
                self._optim_count += 1
                try:
                    sub_batch = self._process_optim_batch_hook(batch_processed)
                except StopIteration:
                    break
                if sub_batch is None:
                    break
                losses_td = self.loss_module(sub_batch)
                self._post_loss_hook(sub_batch)

                losses_detached = self._optimizer_hook(losses_td)
                self._post_optim_hook()

                # LOGGING POINT 4: Post-optimization step logging (e.g., gradient norms, step-specific metrics)
                self._post_optim_log(sub_batch)

                if average_losses is None:
                    average_losses: TensorDictBase = losses_detached
                else:
                    for key, item in losses_detached.items():
                        val = average_losses.get(key)
                        average_losses.set(key, val * j / (j + 1) + item / (j + 1))
                del sub_batch, losses_td, losses_detached

            # LOGGING POINT 5: Post-epoch logging (e.g., epoch completion metrics)
            self._post_epoch_log_hook(batch)
            # Regular post-epoch operations (e.g., epoch cleanup)
            self._post_epoch_hook()

        if j >= 0:
            # Log optimization statistics and average losses after completing all optimization steps
            # This is the main logging point for training metrics like loss values and optimization step count
            self._log(
                optim_steps=self._optim_count,
                **average_losses,
            )

    def _log(self, log_pbar=False, **kwargs) -> None:
        """Main logging method that handles both logger output and progress bar updates.

        This method is called from various hooks throughout the training loop to log metrics.
        It maintains a history of logged values and controls logging frequency based on log_interval.

        Args:
            log_pbar: If True, the value will also be displayed in the progress bar
            **kwargs: Key-value pairs to log, where key is the metric name and value is the metric value
        """
        collected_frames = self.collected_frames
        for key, item in kwargs.items():
            # Store all values in history regardless of logging frequency
            self._log_dict[key].append(item)

            # Check if enough frames have passed since last logging for this key
            if (collected_frames - self._last_log.get(key, 0)) > self._log_interval:
                self._last_log[key] = collected_frames
                _log = True
            else:
                _log = False

            # Determine logging method (defaults to "log_scalar")
            method = LOGGER_METHODS.get(key, "log_scalar")

            # Log to external logger (e.g., tensorboard, wandb) if conditions are met
            if _log and self.logger is not None:
                getattr(self.logger, method)(key, item, step=collected_frames)

            # Update progress bar if requested and method is scalar
            if method == "log_scalar" and self.progress_bar and log_pbar:
                if isinstance(item, torch.Tensor):
                    item = item.item()
                self._pbar_str[key] = item

    def _pbar_description(self) -> None:
        """Update the progress bar description with current metric values.

        This method formats and displays the current values of metrics that have
        been marked for progress bar display (log_pbar=True) in the logging hooks.
        """
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
        iterate (bool, optional): if ``True``, the replay buffer will be iterated over
            in a loop. Defaults to ``False`` (call to :meth:`~torchrl.data.ReplayBuffer.sample` will be used).

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
        iterate: bool = False,
    ) -> None:
        self.replay_buffer = replay_buffer
        if hasattr(replay_buffer, "update_tensordict_priority"):
            self._update_priority = self.replay_buffer.update_tensordict_priority
        else:
            if isinstance(replay_buffer.sampler, PrioritizedSampler):
                raise ValueError(
                    "Prioritized sampler not supported for replay buffer trainer if not within a TensorDictReplayBuffer"
                )
            self._update_priority = None
        self.batch_size = batch_size
        self.memmap = memmap
        self.device = device
        self.flatten_tensordicts = flatten_tensordicts
        self.max_dims = max_dims
        self.iterate = iterate
        if iterate:
            self.replay_buffer_iter = iter(self.replay_buffer)

    def extend(self, batch: TensorDictBase) -> TensorDictBase:
        if self.flatten_tensordicts:
            if ("collector", "mask") in batch.keys(True):
                batch = batch[batch.get(("collector", "mask"))]
            else:
                if "truncated" in batch["next"]:
                    batch["next", "truncated"][..., -1] = True
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
        return batch

    def sample(self, batch: TensorDictBase) -> TensorDictBase:
        if self.iterate:
            try:
                sample = next(self.replay_buffer_iter)
            except StopIteration:
                # reset the replay buffer
                self.replay_buffer_iter = iter(self.replay_buffer)
                raise
        else:
            sample = self.replay_buffer.sample(batch_size=self.batch_size)
        return sample.to(self.device) if self.device is not None else sample

    def update_priority(self, batch: TensorDictBase) -> None:
        if self._update_priority is not None:
            self._update_priority(batch)

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
    """Generic scalar logger hook for any tensor values in the batch.

    This hook can log any scalar values from the collected batch data, including
    rewards, action norms, done states, and any other metrics. It automatically
    handles masking and computes both mean and standard deviation.

    Args:
        key (NestedKey): the key where to find the value in the input batch.
            Can be a string for simple keys or a tuple for nested keys.
            Default is `torchrl.trainers.trainers.REWARD_KEY` (= `("next", "reward")`).
        logname (str, optional): name of the metric to be logged. If None, will use
            the key as the log name. Default is None.
        log_pbar (bool, optional): if ``True``, the value will be logged on
            the progression bar. Default is ``False``.
        include_std (bool, optional): if ``True``, also log the standard deviation
            of the values. Default is ``True``.
        reduction (str, optional): reduction method to apply. Can be "mean", "sum",
            "min", "max". Default is "mean".

    Examples:
        >>> # Log training rewards
        >>> log_reward = LogScalar(("next", "reward"), "r_training", log_pbar=True)
        >>> trainer.register_op("pre_steps_log", log_reward)

        >>> # Log action norms
        >>> log_action_norm = LogScalar("action", "action_norm", include_std=True)
        >>> trainer.register_op("pre_steps_log", log_action_norm)

        >>> # Log done states (as percentage)
        >>> log_done = LogScalar(("next", "done"), "done_percentage", reduction="mean")
        >>> trainer.register_op("pre_steps_log", log_done)

    """

    def __init__(
        self,
        key: NestedKey = REWARD_KEY,
        logname: str | None = None,
        log_pbar: bool = False,
        include_std: bool = True,
        reduction: str = "mean",
    ):
        self.key = key
        self.logname = logname if logname is not None else str(key)
        self.log_pbar = log_pbar
        self.include_std = include_std
        self.reduction = reduction

        # Validate reduction method
        if reduction not in ["mean", "sum", "min", "max"]:
            raise ValueError(
                f"reduction must be one of ['mean', 'sum', 'min', 'max'], got {reduction}"
            )

    def _apply_reduction(self, tensor: torch.Tensor) -> torch.Tensor:
        """Apply the specified reduction to the tensor."""
        if self.reduction == "mean":
            return tensor.float().mean()
        elif self.reduction == "sum":
            return tensor.sum()
        elif self.reduction == "min":
            return tensor.min()
        elif self.reduction == "max":
            return tensor.max()
        else:
            raise ValueError(f"Unknown reduction: {self.reduction}")

    def __call__(self, batch: TensorDictBase) -> dict:
        # Get the tensor from the batch
        tensor = batch.get(self.key)

        # Apply mask if available
        if ("collector", "mask") in batch.keys(True):
            mask = batch.get(("collector", "mask"))
            tensor = tensor[mask]

        # Compute the main statistic
        main_value = self._apply_reduction(tensor).item()

        # Prepare the result dictionary
        result = {
            self.logname: main_value,
            "log_pbar": self.log_pbar,
        }

        # Add standard deviation if requested
        if self.include_std and tensor.numel() > 1:
            std_value = tensor.std().item()
            result[f"{self.logname}_std"] = std_value

        return result

    def register(self, trainer: Trainer, name: str | None = None):
        if name is None:
            name = f"log_{self.logname}"
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
        # Convert old API to new API
        if reward_key is None:
            reward_key = REWARD_KEY
        super().__init__(key=reward_key, logname=logname, log_pbar=log_pbar)


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
        eps: float | None = None,
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

    def __init__(
        self,
        collector: DataCollectorBase,
        update_weights_interval: int,
        policy_weights_getter: Callable[[Any], Any] | None = None,
    ):
        self.collector = collector
        self.update_weights_interval = update_weights_interval
        self.counter = 0
        self.policy_weights_getter = policy_weights_getter

    def __call__(self):
        self.counter += 1
        if self.counter % self.update_weights_interval == 0:
            weights = (
                self.policy_weights_getter()
                if self.policy_weights_getter is not None
                else None
            )
            if weights is not None:
                self.collector.update_policy_weights_(weights)
            else:
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


class TargetNetUpdaterHook(TrainerHookBase):
    """A hook for target parameters update.

    Examples:
        >>> # define a loss module
        >>> loss_module = SACLoss(actor_network, qvalue_network)
        >>> # define a target network updater
        >>> target_net_updater = SoftUpdate(loss_module)
        >>> # define a target network updater hook
        >>> target_net_updater_hook = TargetNetUpdaterHook(target_net_updater)
        >>> # register the target network updater hook
        >>> trainer.register_op("post_optim", target_net_updater_hook)
    """

    def __init__(self, target_params_updater: TargetNetUpdater):
        if not isinstance(target_params_updater, TargetNetUpdater):
            raise ValueError(
                f"Expected a target network updater, got {type(target_params_updater)=}"
            )
        self.target_params_updater = target_params_updater

    def __call__(self, tensordict: TensorCollection | None = None):
        self.target_params_updater.step()
        return tensordict

    def register(self, trainer: Trainer, name: str):
        trainer.register_op("post_steps", self)


class UTDRHook(TrainerHookBase):
    """Hook for logging Update-to-Data (UTD) ratio during async collection.

    The UTD ratio measures how many optimization steps are performed per
    collected data sample, providing insight into training efficiency during
    asynchronous data collection. This metric is particularly useful for
    off-policy algorithms where data collection and training happen concurrently.

    The UTD ratio is calculated as: (batch_size * update_count) / write_count
    where:
    - batch_size: Size of batches sampled from replay buffer
    - update_count: Total number of optimization steps performed
    - write_count: Total number of samples written to replay buffer

    Args:
        trainer (Trainer): The trainer instance to monitor for UTD calculation.
                          Must have async_collection=True for meaningful results.

    Note:
        This hook is only meaningful when async_collection is enabled, as it
        relies on the replay buffer's write_count to track data collection progress.
    """

    def __init__(self, trainer: Trainer):
        self.trainer = trainer

    def __call__(self, batch: TensorDictBase | None = None) -> dict:
        if (
            hasattr(self.trainer, "replay_buffer")
            and self.trainer.replay_buffer is not None
        ):
            write_count = self.trainer.replay_buffer.write_count
            batch_size = self.trainer.replay_buffer.batch_size
        else:
            write_count = self.trainer.collector.getattr_rb("write_count")
            batch_size = self.trainer.collector.getattr_rb("batch_size")
        if not write_count:
            return {}
        if batch_size is None and RL_WARNINGS:
            warnings.warn("Batch size is not set. Using 1.")
            batch_size = 1
        update_count = self.trainer._optim_count
        utd_ratio = batch_size * update_count / write_count
        return {
            "utd_ratio": utd_ratio,
            "write_count": write_count,
            "update_count": update_count,
            "log_pbar": False,
        }

    def register(self, trainer: Trainer, name: str = "utdr_hook"):
        """Register the UTD ratio hook with the trainer.

        Args:
            trainer (Trainer): The trainer to register with.
            name (str): Name to use when registering the hook module.
        """
        trainer.register_op("pre_steps_log", self)
        trainer.register_module(name, self)

    def state_dict(self) -> dict[str, Any]:
        """Return state dictionary for checkpointing."""
        return {}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Load state from dictionary."""
