# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Private optimization execution boundaries used by :class:`Trainer`."""

from __future__ import annotations

import random
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol, TYPE_CHECKING

import numpy as np
import torch
from tensordict import TensorDict, TensorDictBase
from torch import nn, optim

from torchrl.objectives.common import LossModule
from torchrl.objectives.utils import TargetNetUpdater
from torchrl.trainers._distributed import _DDPProcessGroup
from torchrl.trainers.trainers import (
    DefaultOptimizationStepper,
    OptimizationStepper,
    Trainer,
)
from torchrl.weight_update.weight_sync_schemes import WeightStrategy

if TYPE_CHECKING:
    from typing import Self


@dataclass(frozen=True)
class _ExecutionStep:
    """Result of one synchronized optimization command."""

    round_id: int
    optim_steps: int
    model_version: int
    metrics: TensorDictBase


class _TrainerExecutionBackend(Protocol):
    """Private lifecycle and command surface owned by a Trainer."""

    world_size: int
    global_batch_size: int
    model_version: int
    last_round: int
    generation: int

    def start(self) -> None:
        ...

    def step(self, num_steps: int) -> _ExecutionStep:
        ...

    def get_weights(
        self, model_id: str = "policy", *, expected_version: int | None = None
    ) -> TensorDictBase:
        ...

    def state_dict(self) -> dict[str, Any]:
        ...

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        ...

    def is_alive(self) -> bool:
        ...

    def shutdown(self, timeout: float | None = None) -> None:
        ...


class _LossForwarder(nn.Module):
    """Give DDP one forward boundary for all loss entry points."""

    def __init__(self, loss_module: LossModule) -> None:
        super().__init__()
        self.loss_module = loss_module

    def forward(
        self, batch: TensorDictBase, method: str | None = None
    ) -> tuple[TensorDictBase | tuple[Any, ...], tuple[torch.Tensor, ...]]:
        if method is None:
            result = self.loss_module(batch)
        else:
            result = getattr(self.loss_module, method)(batch)
        # DDP's unused-parameter traversal does not inspect TensorDict leaves.
        # Expose the differentiable leaves in an ordinary tuple while preserving
        # the LossModule result as the first element returned to the stepper.
        return result, self._graph_tensors(result)

    @classmethod
    def _graph_tensors(cls, value: Any) -> tuple[torch.Tensor, ...]:
        if isinstance(value, torch.Tensor):
            return (value,)
        if isinstance(value, TensorDictBase):
            return tuple(
                item
                for item in value.values(include_nested=True, leaves_only=True)
                if isinstance(item, torch.Tensor)
            )
        if isinstance(value, Mapping):
            return tuple(
                tensor for item in value.values() for tensor in cls._graph_tensors(item)
            )
        if isinstance(value, (tuple, list)):
            return tuple(
                tensor for item in value for tensor in cls._graph_tensors(item)
            )
        return ()


class _Learner:
    """Rank-local optimization state hidden behind a Trainer backend."""

    def __init__(
        self,
        *,
        loss_module: LossModule,
        replay_buffer: Any,
        local_batch_size: int,
        optimizer: optim.Optimizer | None,
        optimization_stepper: OptimizationStepper | None,
        target_net_updater: TargetNetUpdater | None,
        process_group: _DDPProcessGroup | None = None,
        clip_grad_norm: bool = True,
        clip_norm: float | None = None,
        update_replay_priority: bool = True,
    ) -> None:
        if not isinstance(loss_module, LossModule):
            raise TypeError(
                "loss_module must be a torchrl.objectives.LossModule, got "
                f"{type(loss_module).__name__}."
            )
        if isinstance(local_batch_size, bool) or not isinstance(local_batch_size, int):
            raise TypeError("local_batch_size must be an integer.")
        if local_batch_size <= 0:
            raise ValueError("local_batch_size must be positive.")
        if optimization_stepper is None:
            if optimizer is None:
                raise ValueError(
                    "Remote optimization requires an optimizer or an "
                    "OptimizationStepper."
                )
            optimization_stepper = DefaultOptimizationStepper()
        if target_net_updater is not None:
            if not isinstance(target_net_updater, TargetNetUpdater):
                raise TypeError("target_net_updater must be a TargetNetUpdater.")
            if target_net_updater.loss_module is not loss_module:
                raise ValueError(
                    "target_net_updater must reference the learner loss_module."
                )

        self.loss_module = loss_module
        self.replay_buffer = replay_buffer
        self.local_batch_size = local_batch_size
        self.optimizer = optimizer
        self.optimization_stepper = optimization_stepper
        self.target_net_updater = target_net_updater
        self.process_group = process_group
        self.clip_grad_norm = clip_grad_norm
        self.clip_norm = clip_norm
        self.update_replay_priority = update_replay_priority
        self._modules: dict[str, Any] = {}
        self._ddp_loss = None
        self._model_version = 0
        self._last_round = 0
        self._initialized = False
        self._closed = False
        self.models = {"policy": self._infer_policy(loss_module)}
        self.optimization_stepper.register(
            self, name="optimization_stepper"  # type: ignore[arg-type]
        )

    @property
    def device(self) -> torch.device:
        if self.process_group is None:
            return torch.device("cpu")
        return self.process_group.device

    @property
    def model_version(self) -> int:
        return self._model_version

    @property
    def last_round(self) -> int:
        return self._last_round

    def register_module(self, module_name: str, module: Any) -> None:
        if module_name in self._modules:
            raise RuntimeError(f"{module_name} is already registered.")
        self._modules[module_name] = module

    def initialize(self) -> Self:
        if self._closed:
            raise RuntimeError("Cannot initialize a closed learner.")
        if self.process_group is not None:
            self._ddp_loss = self.process_group.wrap(_LossForwarder(self.loss_module))
            self.process_group.barrier()
        self._initialized = True
        return self

    def compute_loss(
        self, batch: TensorDictBase, method: str | None = None
    ) -> TensorDictBase | tuple[Any, ...]:
        if self._ddp_loss is not None:
            result, _ = self._ddp_loss(batch, method)
            return result
        if method is None:
            return self.loss_module(batch)
        return getattr(self.loss_module, method)(batch)

    def sync_gradients(self, optimizer: optim.Optimizer) -> None:
        # DDP synchronizes during backward. This method remains part of the
        # OptimizationStepper context so the same stepper runs locally.
        del optimizer

    def step(self, num_steps: int, round_id: int) -> _ExecutionStep:
        self._ensure_ready()
        if round_id != self.last_round + 1:
            raise RuntimeError(
                f"Expected learner round {self.last_round + 1}, got {round_id}."
            )
        if isinstance(num_steps, bool) or not isinstance(num_steps, int):
            raise TypeError("num_steps must be an integer.")
        if num_steps <= 0:
            raise ValueError("num_steps must be positive.")
        metrics = []
        for _ in range(num_steps):
            batch = self.replay_buffer.sample(self.local_batch_size)
            if not isinstance(batch, TensorDictBase):
                raise TypeError(
                    "Replay sampling must return TensorDictBase, got "
                    f"{type(batch).__name__}."
                )
            batch = batch.to(self.device)
            result = self.optimization_stepper._step(self, batch)
            if self.update_replay_priority and hasattr(
                self.replay_buffer, "update_tensordict_priority"
            ):
                self.replay_buffer.update_tensordict_priority(batch)
            if self.target_net_updater is not None:
                self.target_net_updater.step()
            metrics.append(self._normalize_metrics(result))
            self._model_version += 1
        self._last_round = round_id
        return _ExecutionStep(
            round_id=round_id,
            optim_steps=num_steps,
            model_version=self.model_version,
            metrics=self._average_metrics(metrics),
        )

    def get_weights(
        self, model_id: str = "policy", *, expected_version: int | None = None
    ) -> TensorDictBase:
        self._ensure_ready()
        if expected_version is not None and expected_version != self.model_version:
            raise RuntimeError(
                f"Expected model version {expected_version}, got {self.model_version}."
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
            raise TypeError("Weight extraction must return TensorDictBase.")
        return weights.detach().to("cpu")

    def state_dict(self) -> dict[str, Any]:
        state = {
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

    def synchronize_after_restore(self) -> None:
        if self.process_group is not None:
            self.process_group.barrier()

    def close(self) -> None:
        if self._closed:
            return
        self._ddp_loss = None
        if self.process_group is not None:
            self.process_group.close()
        self._closed = True

    def _ensure_ready(self) -> None:
        if self._closed:
            raise RuntimeError("Learner is closed.")
        if not self._initialized:
            raise RuntimeError("Learner is not initialized.")

    @staticmethod
    def _infer_policy(loss_module: LossModule) -> nn.Module:
        for name in ("actor_network", "value_network", "local_value_network"):
            policy = getattr(loss_module, name, None)
            if isinstance(policy, nn.Module):
                return policy
        return loss_module

    @staticmethod
    def _normalize_metrics(metrics: TensorDictBase) -> TensorDictBase:
        if not isinstance(metrics, TensorDictBase):
            raise TypeError(
                "OptimizationStepper must return TensorDictBase, got "
                f"{type(metrics).__name__}."
            )
        result = TensorDict(device="cpu")
        for key, value in metrics.items(True, True):
            if not isinstance(value, torch.Tensor):
                value = torch.as_tensor(value)
            if value.numel() != 1:
                continue
            result.set(key, value.detach().reshape(()).to("cpu"))
        return result

    @staticmethod
    def _average_metrics(metrics: list[TensorDictBase]) -> TensorDictBase:
        if not metrics:
            return TensorDict(device="cpu")
        common_keys = set(metrics[0].keys(True, True))
        for item in metrics[1:]:
            common_keys.intersection_update(item.keys(True, True))
        result = TensorDict(device="cpu")
        for key in common_keys:
            result.set(key, torch.stack([item.get(key) for item in metrics]).mean())
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


class _LocalTrainerExecution:
    """Lifecycle marker for the existing in-process Trainer loop."""

    generation = 0
    world_size = 1

    def __init__(self, trainer: Trainer) -> None:
        self._trainer = trainer

    @property
    def global_batch_size(self) -> int:
        batch_size = getattr(self._trainer, "batch_size", None)
        return 0 if batch_size is None else int(batch_size)

    def is_alive(self) -> bool:
        return True

    def shutdown(self, timeout: float | None = None) -> None:
        del timeout
