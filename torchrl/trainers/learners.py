# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Learner execution and its control-plane protocol.

The learner data plane represents replay samples, loss metrics, and model
parameters as :class:`~tensordict.TensorDictBase` objects. The four frozen
dataclasses in this module carry lower-volume control-plane messages across four
different boundaries:

#. :class:`LearnerContext` is created once when a group starts a rank and injects
   actor-local resources into the learner factory.
#. :class:`LearnerStepRequest` is one immutable optimization command dispatched
   to every rank.
#. :class:`LearnerStepResult` is the completion receipt used to validate that
   every rank completed the same command before controller counters advance.
#. :class:`LearnerWeights` is requested separately when policy publication is
   due, coupling a serialized snapshot to the model version that produced it.

These records are separate because their lifetimes and destinations do not
overlap. A context contains live process-group and replay handles and never
returns to the controller; a request must remain small; a result must not carry
model weights on every optimizer step; and a weight snapshot is not itself an
optimization result. Combining them into one generic record would permit invalid
states and would make every command pay for fields it cannot use. Plain keyword
arguments or dictionaries would reduce the class count, but would lose the
atomic command boundary and typed validation at the Ray serialization boundary.
``frozen=True`` prevents accidental field rebinding after dispatch; it does not
claim that referenced services or tensors are deeply immutable.
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

from torchrl.data.utils import DEVICE_TYPING
from torchrl.distributed import DataParallelContext
from torchrl.objectives.utils import TargetNetUpdater
from torchrl.trainers.trainers import DefaultOptimizationStepper, OptimizationStepper
from torchrl.weight_update.weight_sync_schemes import WeightStrategy

if TYPE_CHECKING:
    from typing import Self


@dataclass(frozen=True)
class LearnerContext:
    """One-time actor-local dependencies passed to a learner factory.

    A learner group creates one context per rank after it has established the
    process group and rank-aware replay view. User code receives the context in
    its factory and uses those already-owned resources to construct trainable
    state; it should not create or manage the services itself. Bundling these
    dependencies keeps the factory signature stable across local and Ray groups
    and prevents distributed setup details from leaking into algorithm code.

    This record exists only during learner construction. It is separate from a
    :class:`LearnerStepRequest` because it contains live actor-local handles that
    must never be sent back to the controller with each optimization command.

    Args:
        rank (int): Global data-parallel rank.
        local_rank (int): Rank among workers on the same node.
        world_size (int): Number of learner ranks.
        device (torch.device): Device visible to the learner.
        generation (int): Learner-group generation identifier.
        replay_buffer: Rank-aware, lifecycle-free replay client.
        data_parallel_context (DataParallelContext): Collective context.
        seed (int, optional): Rank-specific seed.

    Example:
        A learner factory consumes the actor-local context instead of creating
        process groups or replay clients itself:

        >>> import torch
        >>> from tensordict import TensorDict
        >>> from torchrl.trainers import Learner
        >>> class Loss(torch.nn.Module):
        ...     def __init__(self):
        ...         super().__init__()
        ...         self.weight = torch.nn.Parameter(torch.ones(()))
        ...     def forward(self, batch):
        ...         return TensorDict({"loss": self.weight * batch["x"].mean()}, [])
        >>> def make_learner(context):
        ...     loss = Loss().to(context.device)
        ...     optimizer = torch.optim.SGD(loss.parameters(), lr=0.1)
        ...     return Learner(
        ...         loss,
        ...         context.replay_buffer,
        ...         optimizer=optimizer,
        ...         data_parallel_context=context.data_parallel_context,
        ...     )
    """

    rank: int
    local_rank: int
    world_size: int
    device: torch.device
    generation: int
    replay_buffer: Any
    data_parallel_context: DataParallelContext
    seed: int | None = None


@dataclass(frozen=True)
class LearnerStepRequest:
    """One atomic optimization command sent to every learner rank.

    The controller constructs one request and the group dispatches the same
    immutable value to every rank. ``round_id`` and ``global_batch_size`` repeat
    state known by the group on purpose: each rank validates them before sampling
    or mutating optimizer state, which rejects stale, reordered, or
    misconfigured commands before an ambiguous partial step can occur.

    A request object is used instead of three loose ``step`` arguments so the
    command has one serialization and validation boundary and can be logged as a
    unit when a Ray generation fails. Learner state, metrics, and weights flow
    back through their dedicated result and snapshot records.

    Args:
        round_id (int): Consecutive controller-assigned round, starting at one.
        num_steps (int): Number of optimizer steps in this round.
        global_batch_size (int): Batch size across all ranks.

    Example:
        Controller code uses the group's current state to issue the next
        consecutive, global-batch command:

        >>> from torchrl.trainers import LearnerStepRequest
        >>> def run_two_updates(group):
        ...     request = LearnerStepRequest(
        ...         round_id=group.last_round + 1,
        ...         num_steps=2,
        ...         global_batch_size=group.global_batch_size,
        ...     )
        ...     return group.step(request)
    """

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
class LearnerStepResult:
    """Completion receipt for one learner-group optimization command.

    Every rank produces this receipt after a command. The group first compares
    round, step count, model version, and metric structure across ranks; only a
    consistent set is reduced and returned to the controller. The controller can
    therefore advance authoritative counters without inferring progress from
    metrics or actor liveness.

    This is separate from :class:`LearnerStepRequest` because a request describes
    intended work while a result certifies completed work. It is separate from
    :class:`LearnerWeights` so ordinary optimization rounds return only small
    scalar metrics instead of serializing the model at every step.

    Args:
        round_id (int): Completed controller round.
        optim_steps (int): Number of optimizer steps completed in the round.
        model_version (int): Version after the final completed step.
        metrics (TensorDictBase): Detached CPU scalar metrics.

    Example:
        A controller applies a completed result to its authoritative counters
        and logging state:

        >>> def record_result(result, controller_state):
        ...     expected_round = controller_state["round"] + 1
        ...     if result.round_id != expected_round:
        ...         raise RuntimeError("learner rounds must be consecutive")
        ...     controller_state["round"] = result.round_id
        ...     controller_state["optim_steps"] += result.optim_steps
        ...     controller_state["model_version"] = result.model_version
        ...     controller_state["metrics"] = result.metrics.to_dict()
    """

    round_id: int
    optim_steps: int
    model_version: int
    metrics: TensorDictBase


@dataclass(frozen=True)
class LearnerWeights:
    """Atomic, versioned model snapshot exported for publication.

    A controller requests this record only when collector publication or an
    explicit inspection is due. Keeping ``model_id``, ``model_version``, and the
    serialized TensorDict together prevents an unversioned or incorrectly named
    snapshot from being published after a newer learner round completes.

    Weights are not embedded in :class:`LearnerStepResult`: optimization and
    publication have different cadences, and FSDP-style extraction may itself be
    an expensive collective. The metadata remains ordinary Python control data,
    while ``weights`` stays a TensorDict compatible with TorchRL weight-update
    schemes.

    Args:
        model_id (str): Logical model identifier.
        model_version (int): Learner version that produced the snapshot.
        weights (TensorDictBase): Detached CPU weights.

    Example:
        Weight publication passes the serialized snapshot to the collector and
        records the version that workers received:

        >>> def publish_weights(collector, snapshot):
        ...     collector.update_policy_weights_(snapshot.weights)
        ...     return snapshot.model_version
    """

    model_id: str
    model_version: int
    weights: TensorDictBase


class LearnerBatchSource(Protocol):
    """Internal seam that supplies one learner batch per optimizer step."""

    def next(self, request: LearnerStepRequest) -> TensorDictBase:
        ...


class _ReplayBatchSource:
    def __init__(self, replay_buffer: Any, device: torch.device) -> None:
        self.replay_buffer = replay_buffer
        self.device = device

    def next(self, request: LearnerStepRequest) -> TensorDictBase:
        batch = self.replay_buffer.sample(request.global_batch_size)
        if not isinstance(batch, TensorDictBase):
            raise TypeError(
                "Learner replay sampling must return a TensorDictBase, got "
                f"{type(batch).__name__}."
            )
        return batch.to(self.device)


class Learner:
    """Actor-local owner of optimization state and bounded update commands.

    A learner executes bounded optimization commands inside one process. It owns
    model optimization, replay sampling, target updates, and RNG state. A
    :class:`~torchrl.trainers.LearnerGroup` coordinates one or more learners,
    while a :class:`~torchrl.trainers.Trainer` schedules group commands alongside
    collection, logging, stopping, and shared-service lifecycle management.

    Args:
        loss_module (nn.Module): Actor-local loss module.
        replay_buffer: Rank-aware lifecycle-free replay client.
        optimizer (Optimizer, optional): Default optimizer.
        optimization_stepper (OptimizationStepper, optional): Stepper used for
            each sampled batch. Defaults to :class:`DefaultOptimizationStepper`
            when an optimizer is provided.
        data_parallel_context (DataParallelContext, optional): Gradient
            collective context. Defaults to a single-process CPU context.
        batch_source (LearnerBatchSource, optional): Custom internal batch
            source. Defaults to replay-backed sampling.
        target_net_updater (TargetNetUpdater, optional): Target updater run
            after every completed optimizer step.
        clip_grad_norm (bool): Whether to clip by total norm. Defaults to True.
        clip_norm (float, optional): Gradient clipping threshold.
        models (mapping, optional): Named modules available to
            :meth:`get_weights`. ``"policy"`` is inferred when possible.
        update_replay_priority (bool): Whether to call
            ``update_tensordict_priority`` after a step. Defaults to True.

    Example:
        >>> import torch
        >>> from tensordict import TensorDict
        >>> from torchrl.data import LazyTensorStorage, TensorDictReplayBuffer
        >>> from torchrl.trainers import Learner, LearnerStepRequest
        >>> class Loss(torch.nn.Module):
        ...     def __init__(self):
        ...         super().__init__()
        ...         self.weight = torch.nn.Parameter(torch.ones(()))
        ...     def forward(self, batch):
        ...         return TensorDict({"loss": self.weight * batch["x"].mean()}, [])
        >>> replay = TensorDictReplayBuffer(storage=LazyTensorStorage(4), batch_size=2)
        >>> _ = replay.extend(TensorDict({"x": torch.ones(4, 1)}, [4]))
        >>> loss = Loss()
        >>> learner = Learner(
        ...     loss,
        ...     replay,
        ...     optimizer=torch.optim.SGD(loss.parameters(), lr=0.1),
        ... )
        >>> _ = learner.initialize()
        >>> weight_before = loss.weight.detach().clone()
        >>> result = learner.step(LearnerStepRequest(1, 1, 2))
        >>> result.metrics["loss"].item()
        1.0
        >>> bool(loss.weight < weight_before)
        True
    """

    def __init__(
        self,
        loss_module: nn.Module,
        replay_buffer: Any,
        *,
        optimizer: optim.Optimizer | None = None,
        optimization_stepper: OptimizationStepper | None = None,
        data_parallel_context: DataParallelContext | None = None,
        batch_source: LearnerBatchSource | None = None,
        target_net_updater: TargetNetUpdater | None = None,
        clip_grad_norm: bool = True,
        clip_norm: float | None = None,
        models: Mapping[str, nn.Module] | None = None,
        update_replay_priority: bool = True,
    ) -> None:
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
        self._active_request: LearnerStepRequest | None = None
        self._active_metrics: list[TensorDictBase] = []

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
        """Last fully completed controller round."""
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

    def step(self, request: LearnerStepRequest) -> LearnerStepResult:
        """Execute a consecutive bounded optimization round."""
        self._begin_round(request)
        for _ in range(request.num_steps):
            self._step_round_once()
        return self._finish_round()

    def _begin_round(self, request: LearnerStepRequest) -> None:
        """Begin an ordered round for a learner-group implementation."""
        self._ensure_ready()
        if self._active_request is not None:
            raise RuntimeError(
                f"Round {self._active_request.round_id} is already in progress."
            )
        expected_round = self.last_round + 1
        if request.round_id != expected_round:
            raise RuntimeError(
                f"Expected round_id={expected_round}, got {request.round_id}."
            )
        self._active_request = request
        self._active_metrics = []

    def _step_round_once(self) -> LearnerStepResult:
        """Execute one substep of the active round."""
        request = self._active_request
        if request is None:
            raise RuntimeError("No learner round is active.")
        if len(self._active_metrics) >= request.num_steps:
            raise RuntimeError(f"Round {request.round_id} is already complete.")
        batch = self.batch_source.next(request)
        metrics = self.optimization_stepper.step(self, batch)
        if self.update_replay_priority and hasattr(
            self.replay_buffer, "update_tensordict_priority"
        ):
            self.replay_buffer.update_tensordict_priority(batch)
        if self.target_net_updater is not None:
            self.target_net_updater.step()
        self._model_version += 1
        metrics = self._normalize_metrics(metrics)
        self._active_metrics.append(metrics)
        return LearnerStepResult(
            round_id=request.round_id,
            optim_steps=1,
            model_version=self.model_version,
            metrics=metrics,
        )

    def _finish_round(self) -> LearnerStepResult:
        """Finish the active round after every requested substep completed."""
        request = self._active_request
        if request is None:
            raise RuntimeError("No learner round is active.")
        if len(self._active_metrics) != request.num_steps:
            raise RuntimeError(
                f"Round {request.round_id} completed {len(self._active_metrics)} "
                f"of {request.num_steps} optimizer steps."
            )
        averaged_metrics = self._average_metrics(self._active_metrics)
        self._last_round = request.round_id
        result = LearnerStepResult(
            round_id=request.round_id,
            optim_steps=request.num_steps,
            model_version=self.model_version,
            metrics=averaged_metrics,
        )
        self._active_request = None
        self._active_metrics = []
        return result

    def get_weights(self, model_id: str = "policy") -> LearnerWeights:
        """Return a detached CPU snapshot of a named model."""
        self._ensure_ready()
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
        return LearnerWeights(
            model_id=model_id,
            model_version=self.model_version,
            weights=weights.detach().to("cpu"),
        )

    def state_dict(self) -> dict[str, Any]:
        """Return learner optimization and RNG state."""
        if self._active_request is not None:
            raise RuntimeError("Cannot checkpoint a learner during an active round.")
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

    def _synchronize_after_restore(self) -> dict[str, int]:
        self._ensure_ready()
        self.data_parallel_context.broadcast_module(self.loss_module)
        self.data_parallel_context.barrier()
        return {
            "last_round": self.last_round,
            "model_version": self.model_version,
        }

    def close(self) -> None:
        """Close rank-local resources without touching shared services."""
        if self._closed:
            return
        self.data_parallel_context.close()
        self._closed = True

    @staticmethod
    def _infer_policy(loss_module: nn.Module) -> nn.Module:
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
    def _average_metrics(metrics: list[TensorDictBase]) -> TensorDictBase:
        if not metrics:
            return TensorDict(device="cpu")
        keys = set(metrics[0].keys(True, True))
        if any(set(item.keys(True, True)) != keys for item in metrics[1:]):
            raise RuntimeError("Learner metric keys changed within a round.")
        result = metrics[0].clone()
        for key in keys:
            result.set(
                key,
                torch.stack([item.get(key) for item in metrics]).mean(),
            )
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
    """Collective-safety boundary for one or more synchronized learners.

    Example:
        A backend-neutral controller can command either a local or Ray group,
        then publish the matching versioned weights:

        >>> from torchrl.trainers import LearnerStepRequest
        >>> def optimize_and_publish(group, collector):
        ...     result = group.step(
        ...         LearnerStepRequest(
        ...             group.last_round + 1, 1, group.global_batch_size
        ...         )
        ...     )
        ...     snapshot = group.get_weights()
        ...     if snapshot.model_version != result.model_version:
        ...         raise RuntimeError("stale learner weights")
        ...     collector.update_policy_weights_(snapshot.weights)
        ...     return result.metrics
    """

    @abc.abstractmethod
    def start(self) -> Self:
        """Start every learner rank and wait for initial synchronization."""

    @abc.abstractmethod
    def step(self, request: LearnerStepRequest) -> LearnerStepResult:
        """Run one ordered optimization command on every rank."""

    @abc.abstractmethod
    def get_weights(self, model_id: str = "policy") -> LearnerWeights:
        """Return a versioned rank-zero model snapshot."""

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

    def __enter__(self) -> Self:
        return self.start()

    def __exit__(self, *exc_info: Any) -> None:
        self.shutdown()


class LocalLearnerGroup(LearnerGroup):
    """Single-process implementation of the learner-group contract.

    Args:
        learner_factory (callable): Factory receiving a :class:`LearnerContext`.
        replay_buffer: Replay buffer or lifecycle-free replay client.
        global_batch_size (int): Batch size used by learner requests.
        device (DEVICE_TYPING): Learner device. Defaults to CPU.
        seed (int, optional): Learner seed.

    Example:
        >>> import torch
        >>> from tensordict import TensorDict
        >>> from torchrl.data import LazyTensorStorage, TensorDictReplayBuffer
        >>> from torchrl.trainers import (
        ...     Learner, LearnerStepRequest, LocalLearnerGroup
        ... )
        >>> replay = TensorDictReplayBuffer(storage=LazyTensorStorage(4), batch_size=2)
        >>> _ = replay.extend(TensorDict({"x": torch.ones(4, 1)}, [4]))
        >>> class Loss(torch.nn.Module):
        ...     def __init__(self):
        ...         super().__init__()
        ...         self.weight = torch.nn.Parameter(torch.ones(()))
        ...     def forward(self, batch):
        ...         return TensorDict({"loss": self.weight * batch["x"].mean()}, [])
        >>> def make_learner(context):
        ...     loss = Loss()
        ...     return Learner(loss, context.replay_buffer,
        ...                    optimizer=torch.optim.SGD(loss.parameters(), lr=0.1),
        ...                    data_parallel_context=context.data_parallel_context)
        >>> group = LocalLearnerGroup(make_learner, replay, global_batch_size=2).start()
        >>> result = group.step(LearnerStepRequest(1, 2, 2))
        >>> result.optim_steps
        2
        >>> torch.testing.assert_close(
        ...     group.get_weights().weights["weight"], torch.tensor(0.8)
        ... )
        >>> group.shutdown()
    """

    def __init__(
        self,
        learner_factory: Callable[[LearnerContext], Learner],
        replay_buffer: Any,
        *,
        global_batch_size: int,
        device: DEVICE_TYPING = "cpu",
        seed: int | None = None,
    ) -> None:
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
        factory_context = LearnerContext(
            rank=0,
            local_rank=0,
            world_size=1,
            device=context.device,
            generation=0,
            replay_buffer=replay_buffer,
            data_parallel_context=context,
            seed=self.seed,
        )
        learner = self.learner_factory(factory_context)
        if not isinstance(learner, Learner):
            context.close()
            raise TypeError(
                "learner_factory must return Learner, got " f"{type(learner).__name__}."
            )
        self._learner = learner.initialize()
        return self

    def step(self, request: LearnerStepRequest) -> LearnerStepResult:
        learner = self._require_learner()
        if request.global_batch_size != self.global_batch_size:
            raise ValueError(
                "LearnerStepRequest.global_batch_size must match the group: "
                f"{request.global_batch_size} != {self.global_batch_size}."
            )
        return learner.step(request)

    def get_weights(self, model_id: str = "policy") -> LearnerWeights:
        return self._require_learner().get_weights(model_id)

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


__all__ = [
    "Learner",
    "LearnerContext",
    "LearnerGroup",
    "LearnerStepRequest",
    "LearnerStepResult",
    "LearnerWeights",
    "LocalLearnerGroup",
]
