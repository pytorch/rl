# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Learner execution and learner-group coordination.

Replay samples, optimization metrics, and model weights use TensorDict. Private
frozen records are used only inside distributed learner groups to validate that
all ranks executed the same command.
"""

from __future__ import annotations

import abc
import random
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol, TYPE_CHECKING

import numpy as np
import torch
from tensordict import TensorDict, TensorDictBase
from torch import nn, optim

from torchrl.data.replay_buffers import DataParallelReplayBufferClient, ReplayBuffer
from torchrl.data.utils import DEVICE_TYPING
from torchrl.distributed import DataParallelContext
from torchrl.objectives.common import LossModule
from torchrl.objectives.utils import TargetNetUpdater
from torchrl.trainers.trainers import DefaultOptimizationStepper, OptimizationStepper
from torchrl.weight_update.weight_sync_schemes import WeightStrategy

if TYPE_CHECKING:
    from typing import Self


@dataclass(frozen=True)
class _LearnerStepCommand:
    round_id: int
    num_steps: int
    global_batch_size: int

    def __post_init__(self) -> None:
        for name, value in (
            ("round_id", self.round_id),
            ("num_steps", self.num_steps),
            ("global_batch_size", self.global_batch_size),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(
                    f"{name} must be an integer, got {type(value).__name__}."
                )
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}.")


@dataclass(frozen=True)
class _LearnerStepReceipt:
    round_id: int
    optim_steps: int
    model_version: int
    metrics: TensorDictBase


class _LearnerBatchSource(Protocol):
    """Internal seam that supplies one learner batch per optimizer step."""

    def next(self, global_batch_size: int) -> TensorDictBase:
        ...


class _ReplayBatchSource:
    def __init__(
        self,
        replay_buffer: ReplayBuffer | DataParallelReplayBufferClient,
        device: torch.device,
    ) -> None:
        self.replay_buffer = replay_buffer
        self.device = device

    def next(self, global_batch_size: int) -> TensorDictBase:
        batch = self.replay_buffer.sample(global_batch_size)
        if not isinstance(batch, TensorDictBase):
            raise TypeError(
                "Learner replay sampling must return a TensorDictBase, got "
                f"{type(batch).__name__}."
            )
        return batch.to(self.device)


class Learner:
    """Actor-local owner of optimization state and bounded update commands.

    A learner samples replay data and owns the loss module, optimizer, target
    updater, and RNG state needed to execute optimization steps. A
    :class:`~torchrl.trainers.LearnerGroup` coordinates one or more learners.

    Args:
        loss_module (LossModule): Actor-local TorchRL loss module.
        replay_buffer (ReplayBuffer or DataParallelReplayBufferClient): Replay
            buffer used to sample optimization batches and update priorities.
        optimizer (Optimizer, optional): Default optimizer.
        optimization_stepper (OptimizationStepper, optional): Stepper used for
            each sampled batch. Defaults to :class:`DefaultOptimizationStepper`
            when an optimizer is provided.
        data_parallel_context (DataParallelContext, optional): Gradient
            collective context. Defaults to a single-process CPU context.
        batch_source (_LearnerBatchSource, optional): Internal custom batch
            source. Defaults to replay-backed sampling.
        target_net_updater (TargetNetUpdater, optional): Updater constructed for
            ``loss_module``. Every data-parallel rank applies the same update to
            its local target-network replica after synchronized optimization.
        clip_grad_norm (bool): If ``clip_norm`` is set, clip by total norm when
            ``True`` and by individual gradient values otherwise. Defaults to
            ``True``.
        clip_norm (float, optional): Gradient clipping threshold. ``None``
            disables clipping. Defaults to ``None``.
        models (mapping, optional): Named modules available to
            :meth:`get_weights`. ``"policy"`` is inferred when possible.
        update_replay_priority (bool): Whether to forward priorities written to
            the sampled batch by the loss module. Defaults to ``True`` so
            prioritized replay works without another hook. Set to ``False`` if
            another component performs the update.

    Example:
        A DQN learner consumes replay data, performs several updates, and
        exports policy weights for a collector:

        >>> import torch
        >>> from tensordict import TensorDict
        >>> from torchrl.data import LazyTensorStorage, TensorDictReplayBuffer
        >>> from torchrl.modules import QValueActor
        >>> from torchrl.objectives import DQNLoss, SoftUpdate
        >>> from torchrl.trainers import Learner
        >>> _ = torch.manual_seed(0)
        >>> policy = QValueActor(
        ...     torch.nn.Linear(4, 2),
        ...     in_keys=["observation"],
        ...     action_space="one-hot",
        ... )
        >>> loss = DQNLoss(policy, action_space="one-hot", delay_value=True)
        >>> replay = TensorDictReplayBuffer(
        ...     storage=LazyTensorStorage(16), batch_size=4
        ... )
        >>> transitions = TensorDict(
        ...     {
        ...         "observation": torch.randn(8, 4),
        ...         "action": torch.nn.functional.one_hot(torch.arange(8) % 2, 2),
        ...         "next": TensorDict(
        ...             {
        ...                 "observation": torch.randn(8, 4),
        ...                 "reward": torch.randn(8, 1),
        ...                 "done": torch.zeros(8, 1, dtype=torch.bool),
        ...                 "terminated": torch.zeros(8, 1, dtype=torch.bool),
        ...             },
        ...             [8],
        ...         ),
        ...     },
        ...     [8],
        ... )
        >>> _ = replay.extend(transitions)
        >>> learner = Learner(
        ...     loss,
        ...     replay,
        ...     optimizer=torch.optim.Adam(loss.parameters(), lr=0.01),
        ...     target_net_updater=SoftUpdate(loss, eps=0.95),
        ... ).initialize()
        >>> weights_before = learner.get_weights().clone()
        >>> metrics = learner.step(num_steps=3)
        >>> learner.model_version
        3
        >>> set(metrics.keys()) == {"loss", "grad_norm"}
        True
        >>> weights_after = learner.get_weights(expected_version=3)
        >>> any(
        ...     not torch.equal(weights_before.get(key), weights_after.get(key))
        ...     for key in weights_before.keys(True, True)
        ... )
        True
        >>> learner.close()
    """

    def __init__(
        self,
        loss_module: LossModule,
        replay_buffer: ReplayBuffer | DataParallelReplayBufferClient,
        *,
        optimizer: optim.Optimizer | None = None,
        optimization_stepper: OptimizationStepper | None = None,
        data_parallel_context: DataParallelContext | None = None,
        batch_source: _LearnerBatchSource | None = None,
        target_net_updater: TargetNetUpdater | None = None,
        clip_grad_norm: bool = True,
        clip_norm: float | None = None,
        models: Mapping[str, nn.Module] | None = None,
        update_replay_priority: bool = True,
    ) -> None:
        if not isinstance(loss_module, LossModule):
            raise TypeError(
                "loss_module must be a torchrl.objectives.LossModule, got "
                f"{type(loss_module).__name__}."
            )
        if not isinstance(
            replay_buffer, (ReplayBuffer, DataParallelReplayBufferClient)
        ):
            raise TypeError(
                "replay_buffer must be a ReplayBuffer or "
                "DataParallelReplayBufferClient, got "
                f"{type(replay_buffer).__name__}."
            )
        if target_net_updater is not None:
            if not isinstance(target_net_updater, TargetNetUpdater):
                raise TypeError(
                    "target_net_updater must be a TargetNetUpdater, got "
                    f"{type(target_net_updater).__name__}."
                )
            if target_net_updater.loss_module is not loss_module:
                raise ValueError(
                    "target_net_updater must be constructed with the learner's "
                    "loss_module."
                )
        if optimization_stepper is None:
            if optimizer is None:
                raise ValueError(
                    "Learner requires either optimizer or optimization_stepper."
                )
            optimization_stepper = DefaultOptimizationStepper()
        self.loss_module = loss_module
        self.replay_buffer = replay_buffer
        self.optimizer = optimizer
        self.optimization_stepper = optimization_stepper
        self.data_parallel_context = data_parallel_context or DataParallelContext(
            device="cpu"
        )
        self.clip_grad_norm = clip_grad_norm
        self.clip_norm = clip_norm
        self.target_net_updater = target_net_updater
        self.update_replay_priority = update_replay_priority
        self.batch_source = batch_source or _ReplayBatchSource(
            replay_buffer, self.data_parallel_context.device
        )
        self._modules: dict[str, Any] = {}
        self._model_version = 0
        self._last_round = 0
        self._initialized = False
        self._closed = False

        if models is None:
            policy = self._infer_policy(loss_module)
            models = {"policy": policy}
        self.models = dict(models)
        self.optimization_stepper.register(self, name="optimization_stepper")

    @property
    def model_version(self) -> int:
        """Current model version."""
        return self._model_version

    @property
    def last_round(self) -> int:
        """Last fully completed optimization round."""
        return self._last_round

    def register_module(self, module_name: str, module: Any) -> None:
        """Register learner-local state for checkpointing."""
        if module_name in self._modules:
            raise RuntimeError(f"{module_name} is already registered.")
        self._modules[module_name] = module

    def sync_gradients(self, optimizer: optim.Optimizer) -> None:
        """Average optimizer gradients across learner ranks."""
        self.data_parallel_context.sync_gradients(optimizer)

    def initialize(self) -> Self:
        """Synchronize initial state and make the learner ready for commands."""
        self._ensure_open()
        if self.data_parallel_context.world_size > 1 and not getattr(
            self.optimization_stepper, "supports_data_parallel", False
        ):
            raise RuntimeError(
                f"{type(self.optimization_stepper).__name__} does not declare "
                "data-parallel support. Set supports_data_parallel=True only "
                "after synchronizing every optimizer before clipping and stepping."
            )
        self.data_parallel_context.broadcast_module(self.loss_module)
        self.data_parallel_context.barrier()
        self._initialized = True
        return self

    def step(
        self, num_steps: int = 1, *, batch_size: int | None = None
    ) -> TensorDictBase:
        """Run optimization steps and return their averaged scalar metrics.

        Args:
            num_steps (int): Number of consecutive optimizer steps. Defaults to
                ``1``.
            batch_size (int, optional): Global batch size. When omitted, use the
                replay buffer's configured batch size.

        Returns:
            TensorDictBase: Detached CPU scalar metrics averaged across steps.
        """
        command = _LearnerStepCommand(
            round_id=self.last_round + 1,
            num_steps=num_steps,
            global_batch_size=self._resolve_batch_size(batch_size),
        )
        return self._execute_round(command).metrics

    def _execute_round(self, command: _LearnerStepCommand) -> _LearnerStepReceipt:
        self._ensure_ready()
        expected_round = self.last_round + 1
        if command.round_id != expected_round:
            raise RuntimeError(
                f"Expected round_id={expected_round}, got {command.round_id}."
            )

        averaged_metrics: TensorDictBase | None = None
        for index in range(command.num_steps):
            batch = self.batch_source.next(command.global_batch_size)
            metrics = self.optimization_stepper.step(self, batch)
            if self.update_replay_priority and hasattr(
                self.replay_buffer, "update_tensordict_priority"
            ):
                self.replay_buffer.update_tensordict_priority(batch)
            if self.target_net_updater is not None:
                self.target_net_updater.step()
            self._model_version += 1
            metrics = self._normalize_metrics(metrics)
            if averaged_metrics is None:
                averaged_metrics = metrics
            else:
                if set(averaged_metrics.keys(True, True)) != set(
                    metrics.keys(True, True)
                ):
                    raise RuntimeError("Learner metric keys changed within a round.")
                for key, value in metrics.items(True, True):
                    previous = averaged_metrics.get(key)
                    averaged_metrics.set(
                        key, previous * index / (index + 1) + value / (index + 1)
                    )

        self._last_round = command.round_id
        if averaged_metrics is None:
            averaged_metrics = TensorDict(device="cpu")
        return _LearnerStepReceipt(
            round_id=command.round_id,
            optim_steps=command.num_steps,
            model_version=self.model_version,
            metrics=averaged_metrics,
        )

    def get_weights(
        self,
        model_id: str = "policy",
        *,
        expected_version: int | None = None,
    ) -> TensorDictBase:
        """Return detached CPU weights for a named model.

        Args:
            model_id (str): Logical model name. Defaults to ``"policy"``.
            expected_version (int, optional): If provided, reject extraction
                unless the learner is at this model version.
        """
        self._ensure_ready()
        if expected_version is not None and expected_version != self.model_version:
            raise RuntimeError(
                f"Expected model_version={expected_version}, got "
                f"{self.model_version}."
            )
        try:
            model = self.models[model_id]
        except KeyError as err:
            raise KeyError(
                f"Unknown model_id {model_id!r}; available models are "
                f"{sorted(self.models)}."
            ) from err
        weights = WeightStrategy(extract_as="tensordict").extract_weights(model)
        if not isinstance(weights, TensorDictBase):
            raise TypeError(
                "The tensordict weight strategy must return TensorDictBase."
            )
        return weights.detach().to("cpu")

    def state_dict(self) -> dict[str, Any]:
        """Return learner optimization and RNG state."""
        state: dict[str, Any] = {
            "loss_module": self.loss_module.state_dict(),
            "optimization_stepper": self.optimization_stepper.state_dict(),
            "model_version": self.model_version,
            "last_round": self.last_round,
            "rng": self._rng_state(),
        }
        if self.optimizer is not None:
            state["optimizer"] = self.optimizer.state_dict()
        if self.target_net_updater is not None:
            state["target_net_updater"] = self.target_net_updater.state_dict()
        return state

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        """Restore learner optimization and RNG state."""
        self.loss_module.load_state_dict(state_dict["loss_module"])
        if self.optimizer is not None and "optimizer" in state_dict:
            self.optimizer.load_state_dict(state_dict["optimizer"])
        self.optimization_stepper.load_state_dict(
            state_dict.get("optimization_stepper", {})
        )
        if self.target_net_updater is not None and "target_net_updater" in state_dict:
            self.target_net_updater.load_state_dict(state_dict["target_net_updater"])
        self._model_version = int(state_dict.get("model_version", 0))
        self._last_round = int(state_dict.get("last_round", 0))
        if "rng" in state_dict:
            self._set_rng_state(state_dict["rng"])

    def close(self) -> None:
        """Close rank-local resources without touching shared services."""
        if self._closed:
            return
        self.data_parallel_context.close()
        self._closed = True

    def _resolve_batch_size(self, batch_size: int | None) -> int:
        if batch_size is None:
            batch_size = getattr(self.replay_buffer, "batch_size", None)
            if batch_size is None:
                raise ValueError(
                    "batch_size must be provided when the replay buffer has no "
                    "configured batch size."
                )
        if isinstance(batch_size, bool) or not isinstance(batch_size, int):
            raise TypeError(
                "batch_size must be an integer, got " f"{type(batch_size).__name__}."
            )
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}.")
        return batch_size

    @staticmethod
    def _infer_policy(loss_module: LossModule) -> nn.Module:
        for name in ("value_network", "local_value_network", "actor_network"):
            policy = getattr(loss_module, name, None)
            if isinstance(policy, nn.Module):
                return policy
        return loss_module

    @staticmethod
    def _normalize_metrics(metrics: TensorDictBase) -> TensorDictBase:
        if not isinstance(metrics, TensorDictBase):
            raise TypeError(
                "OptimizationStepper.step must return TensorDictBase, got "
                f"{type(metrics).__name__}."
            )
        result = TensorDict(device="cpu")
        for key, value in metrics.items(True, True):
            if not isinstance(value, torch.Tensor):
                value = torch.as_tensor(value)
            if value.numel() != 1:
                raise RuntimeError(
                    f"Learner metric {key!r} must be scalar, got shape "
                    f"{tuple(value.shape)}."
                )
            result.set(key, value.detach().reshape(()).to("cpu"))
        return result

    @staticmethod
    def _rng_state() -> dict[str, Any]:
        return {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        }

    @staticmethod
    def _set_rng_state(state: Mapping[str, Any]) -> None:
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
        torch.set_rng_state(state["torch"])
        if torch.cuda.is_available() and state.get("cuda"):
            torch.cuda.set_rng_state_all(state["cuda"])

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("Learner is closed.")

    def _ensure_ready(self) -> None:
        self._ensure_open()
        if not self._initialized:
            raise RuntimeError("Learner must be initialized before use.")


class LearnerGroup(abc.ABC):
    """Collective-safety boundary for synchronized learners.

    The group owns round numbering, model-version validation, and the global
    batch size. A controller only asks it for a number of optimization steps and
    receives TensorDict metrics.

    Example:
        A backend-neutral controller updates a learner group and publishes the
        matching policy weights to a collector:

        >>> def optimize_and_publish(group, collector):
        ...     metrics = group.step(num_steps=2)
        ...     version = group.model_version
        ...     weights = group.get_weights(expected_version=version)
        ...     collector.update_policy_weights_(weights)
        ...     return metrics
    """

    @abc.abstractmethod
    def start(self) -> Self:
        """Start every learner rank and wait for initial synchronization."""

    @abc.abstractmethod
    def step(self, num_steps: int = 1) -> TensorDictBase:
        """Run consecutive optimization steps on every rank."""

    @abc.abstractmethod
    def get_weights(
        self,
        model_id: str = "policy",
        *,
        expected_version: int | None = None,
    ) -> TensorDictBase:
        """Return rank-zero weights for a named model."""

    @abc.abstractmethod
    def state_dict(self) -> dict[str, Any]:
        """Return group and learner state."""

    @abc.abstractmethod
    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        """Restore group and learner state."""

    @abc.abstractmethod
    def shutdown(self, timeout: float | None = None) -> None:
        """Idempotently stop every learner rank."""

    @property
    @abc.abstractmethod
    def is_alive(self) -> bool:
        """Whether the complete learner group is available."""

    @property
    @abc.abstractmethod
    def world_size(self) -> int:
        """Number of learner ranks."""

    @property
    @abc.abstractmethod
    def global_batch_size(self) -> int:
        """Batch size across every learner rank."""

    @property
    @abc.abstractmethod
    def last_round(self) -> int:
        """Last fully completed optimization round."""

    @property
    @abc.abstractmethod
    def model_version(self) -> int:
        """Version after the last completed optimizer step."""

    def __enter__(self) -> Self:
        return self.start()

    def __exit__(self, *exc_info: Any) -> None:
        self.shutdown()


class LocalLearnerGroup(LearnerGroup):
    """Single-process implementation of the learner-group contract.

    Args:
        learner_factory (callable): Factory receiving the replay buffer and
            :class:`~torchrl.distributed.DataParallelContext`, in that order.
        replay_buffer (ReplayBuffer or DataParallelReplayBufferClient): Replay
            buffer used by the local learner.
        global_batch_size (int): Batch size across all learner ranks.
        device (DEVICE_TYPING): Learner device. Defaults to CPU.
        seed (int, optional): Learner seed.

    Example:
        >>> import torch
        >>> from tensordict import TensorDict
        >>> from torchrl.data import LazyTensorStorage, TensorDictReplayBuffer
        >>> from torchrl.objectives import LossModule
        >>> from torchrl.trainers import Learner, LocalLearnerGroup
        >>> replay = TensorDictReplayBuffer(storage=LazyTensorStorage(4), batch_size=2)
        >>> _ = replay.extend(TensorDict({"x": torch.ones(4, 1)}, [4]))
        >>> class Loss(LossModule):
        ...     def __init__(self):
        ...         super().__init__()
        ...         self.weight = torch.nn.Parameter(torch.ones(()))
        ...     def forward(self, batch):
        ...         return TensorDict({"loss": self.weight * batch["x"].mean()})
        >>> def make_learner(replay_buffer, data_parallel_context):
        ...     loss = Loss()
        ...     return Learner(
        ...         loss,
        ...         replay_buffer,
        ...         optimizer=torch.optim.SGD(loss.parameters(), lr=0.1),
        ...         data_parallel_context=data_parallel_context,
        ...     )
        >>> group = LocalLearnerGroup(
        ...     make_learner, replay, global_batch_size=2
        ... ).start()
        >>> metrics = group.step(num_steps=2)
        >>> (group.last_round, group.model_version, metrics["loss"].ndim)
        (1, 2, 0)
        >>> weights = group.get_weights(expected_version=group.model_version)
        >>> torch.testing.assert_close(weights["weight"], torch.tensor(0.8))
        >>> group.shutdown()
    """

    def __init__(
        self,
        learner_factory: Callable[
            [ReplayBuffer | DataParallelReplayBufferClient, DataParallelContext],
            Learner,
        ],
        replay_buffer: ReplayBuffer | DataParallelReplayBufferClient,
        *,
        global_batch_size: int,
        device: DEVICE_TYPING = "cpu",
        seed: int | None = None,
    ) -> None:
        if isinstance(global_batch_size, bool) or not isinstance(
            global_batch_size, int
        ):
            raise TypeError("global_batch_size must be an integer.")
        if global_batch_size <= 0:
            raise ValueError("global_batch_size must be positive.")
        self.learner_factory = learner_factory
        self.replay_buffer = replay_buffer
        self._global_batch_size = global_batch_size
        self.device = torch.device(device)
        self.seed = seed
        self._learner: Learner | None = None

    @property
    def world_size(self) -> int:
        return 1

    @property
    def global_batch_size(self) -> int:
        return self._global_batch_size

    @property
    def last_round(self) -> int:
        return self._require_learner().last_round

    @property
    def model_version(self) -> int:
        return self._require_learner().model_version

    @property
    def is_alive(self) -> bool:
        return self._learner is not None

    def start(self) -> Self:
        if self.is_alive:
            return self
        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)
        context = DataParallelContext(device=self.device)
        replay_buffer = self.replay_buffer
        if hasattr(replay_buffer, "data_parallel"):
            replay_buffer = replay_buffer.data_parallel(rank=0, world_size=1)
        learner = self.learner_factory(replay_buffer, context)
        if not isinstance(learner, Learner):
            context.close()
            raise TypeError(
                "learner_factory must return Learner, got " f"{type(learner).__name__}."
            )
        self._learner = learner.initialize()
        return self

    def step(self, num_steps: int = 1) -> TensorDictBase:
        return self._require_learner().step(
            num_steps=num_steps, batch_size=self.global_batch_size
        )

    def get_weights(
        self,
        model_id: str = "policy",
        *,
        expected_version: int | None = None,
    ) -> TensorDictBase:
        return self._require_learner().get_weights(
            model_id, expected_version=expected_version
        )

    def state_dict(self) -> dict[str, Any]:
        return {
            "format_version": 1,
            "world_size": self.world_size,
            "global_batch_size": self.global_batch_size,
            "learner": self._require_learner().state_dict(),
        }

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        if int(state_dict["world_size"]) != self.world_size:
            raise ValueError("Cannot restore a learner group with a new world size.")
        if int(state_dict["global_batch_size"]) != self.global_batch_size:
            raise ValueError("Checkpoint global_batch_size does not match the group.")
        self._require_learner().load_state_dict(state_dict["learner"])

    def shutdown(self, timeout: float | None = None) -> None:
        del timeout
        if self._learner is None:
            return
        self._learner.close()
        self._learner = None

    def _require_learner(self) -> Learner:
        if self._learner is None:
            raise RuntimeError("LocalLearnerGroup is not started.")
        return self._learner


__all__ = ["Learner", "LearnerGroup", "LocalLearnerGroup"]
