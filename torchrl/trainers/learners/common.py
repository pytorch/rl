# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

import torch
from tensordict import TensorDictBase
from torch import nn

from torchrl.objectives.common import LossModule

if TYPE_CHECKING:
    from torchrl.trainers.trainers import OptimizationStepper


def _clone_tensors(obj: Any) -> Any:
    """Clone tensors in a state-dict-like structure."""
    if isinstance(obj, torch.Tensor):
        return obj.clone()
    if isinstance(obj, dict):
        return {key: _clone_tensors(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [_clone_tensors(value) for value in obj]
    if isinstance(obj, tuple):
        return tuple(_clone_tensors(value) for value in obj)
    return obj


@dataclass(frozen=True)
class LearnerCapabilities:
    """Placement capabilities exposed by a :class:`Learner` backend.

    Attributes:
        sharded (bool): Whether parameters are sharded across processes.
        remote (bool): Whether updates execute in another process.
    """

    sharded: bool = False
    remote: bool = False


class Learner(nn.Module):
    r"""Backend-agnostic execution boundary for policy optimization.

    A learner owns the model whose weights are published, the loss module, and
    an :class:`~torchrl.trainers.OptimizationStepper`. The learner decides
    *where* an update runs and how weights are materialized; the stepper is the
    single owner of *how* optimization runs, including loss reduction,
    backward, gradient accumulation, clipping, mixed precision, and optimizer
    stepping. This is the same stepper contract used by
    :class:`~torchrl.trainers.Trainer`.

    Algorithm code therefore calls ``learner.update(batch)`` without carrying
    a second optimization policy or knowing whether the model is local or
    sharded.

    Args:
        model (torch.nn.Module): Module whose parameters are published by
            :meth:`get_weights`.
        loss_module (LossModule): Loss evaluated for each update.
        optimization_stepper (OptimizationStepper): Optimization policy. The
            standard choice is
            :class:`~torchrl.trainers.MixedPrecisionOptimizationStepper`,
            including for full-precision training.

    .. note::
        ``checkpoint`` / ``load_checkpoint`` compose model, loss-module, and
        stepper state. ``state_dict`` keeps its ordinary ``nn.Module`` meaning
        so a learner remains safe to nest inside another module.
    """

    capabilities: LearnerCapabilities = LearnerCapabilities()

    def __init__(
        self,
        model: nn.Module,
        loss_module: LossModule,
        optimization_stepper: OptimizationStepper,
    ):
        super().__init__()
        self.model = model
        self.loss_module = loss_module
        self.optimization_stepper = optimization_stepper

    @property
    def optimizer(self) -> torch.optim.Optimizer:
        """Optimizer owned by the configured stepper.

        FSDP2 checkpointing currently requires a single optimizer. Custom
        multi-optimizer steppers remain usable for updates but must provide
        their own distributed checkpoint implementation.
        """
        optimizer = getattr(self.optimization_stepper, "optimizer", None)
        if not isinstance(optimizer, torch.optim.Optimizer):
            raise RuntimeError(
                f"{type(self).__name__} requires an optimizer-owning "
                "OptimizationStepper for this operation."
            )
        return optimizer

    def _optimized_parameters(self) -> list[torch.nn.Parameter]:
        return [
            parameter
            for group in self.optimizer.param_groups
            for parameter in group["params"]
        ]

    def _extra_optimized_parameters(self) -> list[torch.nn.Parameter]:
        model_parameters = {id(parameter) for parameter in self.model.parameters()}
        return [
            parameter
            for parameter in self._optimized_parameters()
            if id(parameter) not in model_parameters
        ]

    def _restore_extra_params(self, checkpoint: dict[str, Any]) -> None:
        parameters = self._extra_optimized_parameters()
        saved = checkpoint.get("extra_params", [])
        if len(saved) != len(parameters):
            raise RuntimeError(
                f"This checkpoint carries {len(saved)} optimized parameter(s) "
                f"outside the published model, but the current stepper has "
                f"{len(parameters)}."
            )
        with torch.no_grad():
            for parameter, value in zip(parameters, saved):
                parameter.copy_(value)

    def compute_loss(
        self, batch: TensorDictBase, method: str | None = None
    ) -> TensorDictBase | tuple[Any, ...]:
        """Evaluate the learner-owned loss through the stepper context."""
        if method is None:
            return self.loss_module(batch)
        return getattr(self.loss_module, method)(batch)

    def _set_requires_gradient_sync(self, requires_sync: bool) -> None:
        setter = getattr(self.model, "set_requires_gradient_sync", None)
        if setter is not None:
            setter(requires_sync)

    def update(self, batch: TensorDictBase) -> TensorDictBase:
        """Update the learner from one TensorDict micro-batch.

        The returned TensorDict contains detached scalar metrics, matching the
        :class:`~torchrl.trainers.OptimizationStepper` contract used by
        :class:`~torchrl.trainers.Trainer`.
        """
        return self.optimization_stepper._step(self, batch)

    def get_weights(self) -> TensorDictBase:
        """Return a detached, plain-tensor snapshot of published weights."""
        raise NotImplementedError

    def checkpoint(self) -> dict[str, Any]:
        """Return independent model, loss, and optimization state."""
        return {
            "model": _clone_tensors(self.model.state_dict()),
            "loss_module": _clone_tensors(self.loss_module.state_dict()),
            "optimization_stepper": _clone_tensors(
                self.optimization_stepper.state_dict()
            ),
        }

    def load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        """Restore a checkpoint produced by :meth:`checkpoint`."""
        self.model.load_state_dict(checkpoint["model"])
        self.loss_module.load_state_dict(checkpoint["loss_module"])
        self.optimization_stepper.load_state_dict(checkpoint["optimization_stepper"])
