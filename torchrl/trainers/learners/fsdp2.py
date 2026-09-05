# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import contextlib
import functools
from typing import Any

import torch
from tensordict import TensorDict, TensorDictBase
from torch import nn

from torchrl.objectives.common import LossModule
from torchrl.trainers.learners.common import (
    _clone_tensors,
    Learner,
    LearnerCapabilities,
)
from torchrl.trainers.trainers import OptimizationStepper


@functools.cache
def _dist_state_dict():
    from torch.distributed.checkpoint import state_dict

    return state_dict


class FSDP2Learner(Learner):
    """FSDP2 placement backend for :class:`~torchrl.trainers.Learner`.

    The caller applies :func:`torch.distributed._composable.fsdp.fully_shard`
    before constructing the optimizer and stepper. ``FSDP2Learner`` then uses
    the same ``OptimizationStepper`` contract as ``Trainer`` and
    ``LocalLearner``; it only adds FSDP gradient-sync control, full-weight
    gathering, and distributed checkpoint translation.

    Args:
        model (torch.nn.Module): Model already wrapped with ``fully_shard``.
        loss_module (LossModule): Loss evaluated by :meth:`update`.
        optimization_stepper (OptimizationStepper): Optimizer-owning stepper.

    .. warning::
        With ``cpu_offload=True``, :meth:`get_weights` returns full weights on
        rank 0 and an empty TensorDict on other ranks.

    Examples:
        >>> import torch
        >>> import torch.distributed as dist
        >>> from tensordict import TensorDict
        >>> from torch import nn
        >>> from torch.distributed._composable.fsdp import fully_shard
        >>> from torch.distributed.device_mesh import init_device_mesh
        >>> from torchrl.objectives.common import LossModule
        >>> from torchrl.trainers import FSDP2Learner, MixedPrecisionOptimizationStepper
        >>> dist.init_process_group(backend="gloo", rank=0, world_size=1)
        >>> model = nn.Linear(4, 1)
        >>> fully_shard(model, mesh=init_device_mesh("cpu", (1,)))
        >>> class ToyLoss(LossModule):
        ...     def __init__(self, model):
        ...         super().__init__()
        ...         self.model = model
        ...     def forward(self, batch):
        ...         prediction = self.model(batch["x"])
        ...         return TensorDict({"loss_mse": (prediction - batch["y"]).pow(2).mean()})
        >>> loss_module = ToyLoss(model)
        >>> stepper = MixedPrecisionOptimizationStepper(
        ...     torch.optim.Adam(loss_module.parameters())
        ... )
        >>> learner = FSDP2Learner(model, loss_module, stepper)
        >>> batch = TensorDict({"x": torch.randn(8, 4), "y": torch.randn(8, 1)}, [8])
        >>> learner.update(batch)["loss_mse"].ndim
        0
        >>> dist.destroy_process_group()
    """

    def __init__(
        self,
        model: nn.Module,
        loss_module: LossModule,
        optimization_stepper: OptimizationStepper,
    ):
        try:
            _dist_state_dict()
        except ImportError as err:
            raise RuntimeError(
                "FSDP2Learner requires torch.distributed.checkpoint.state_dict."
            ) from err
        super().__init__(model, loss_module, optimization_stepper)
        self.capabilities = LearnerCapabilities(sharded=True, remote=False)

    def get_weights(self, *, cpu_offload: bool = True) -> TensorDictBase:
        """Gather a detached plain-tensor snapshot of the sharded model."""
        dsd = _dist_state_dict()
        options = dsd.StateDictOptions(full_state_dict=True, cpu_offload=cpu_offload)
        state_dict = dsd.get_model_state_dict(self.model, options=options)
        return TensorDict(state_dict).unflatten_keys(".").detach().clone()

    @contextlib.contextmanager
    def _extras_hidden_from_optimizer(self):
        """Hide non-model parameters from DCP and carry them explicitly."""
        extras = self._extra_optimized_parameters()
        extra_ids = {id(parameter) for parameter in extras}
        saved_groups = [group["params"] for group in self.optimizer.param_groups]
        saved_state = {
            parameter: self.optimizer.state.pop(parameter)
            for parameter in extras
            if parameter in self.optimizer.state
        }
        for group in self.optimizer.param_groups:
            group["params"] = [
                parameter
                for parameter in group["params"]
                if id(parameter) not in extra_ids
            ]
        try:
            yield extras, saved_state
        finally:
            for group, parameters in zip(self.optimizer.param_groups, saved_groups):
                group["params"] = parameters
            self.optimizer.state.update(saved_state)

    def checkpoint(self) -> dict[str, Any]:
        """Gather model and optimizer state while preserving stepper state."""
        stepper_state = _clone_tensors(self.optimization_stepper.state_dict())
        stepper_state.pop("optimizer", None)

        dsd = _dist_state_dict()
        options = dsd.StateDictOptions(full_state_dict=True, cpu_offload=True)
        with self._extras_hidden_from_optimizer() as (extras, extra_state):
            model_state, optimizer_state = dsd.get_state_dict(
                self.model, self.optimizer, options=options
            )
        return {
            "model": _clone_tensors(model_state),
            "optimizer": _clone_tensors(optimizer_state),
            "optimization_stepper": stepper_state,
            "extra_params": [parameter.detach().clone() for parameter in extras],
            "extra_optim_state": [
                _clone_tensors(extra_state.get(parameter, {})) for parameter in extras
            ],
        }

    def load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        """Restore and reshard a checkpoint produced by :meth:`checkpoint`."""
        dsd = _dist_state_dict()
        options = dsd.StateDictOptions(full_state_dict=True, broadcast_from_rank0=True)
        with self._extras_hidden_from_optimizer() as (extras, _):
            dsd.set_state_dict(
                self.model,
                self.optimizer,
                model_state_dict=checkpoint["model"],
                optim_state_dict=checkpoint["optimizer"],
                options=options,
            )

        if not extras or checkpoint.get("extra_params"):
            self._restore_extra_params(checkpoint)
            for parameter, state in zip(
                extras, checkpoint.get("extra_optim_state", [{}] * len(extras))
            ):
                if state:
                    self.optimizer.state[parameter] = _clone_tensors(state)
        if extras and torch.distributed.is_initialized():
            for parameter in extras:
                torch.distributed.broadcast(parameter.data, src=0)
                for value in self.optimizer.state.get(parameter, {}).values():
                    if isinstance(value, torch.Tensor):
                        torch.distributed.broadcast(value, src=0)

        stepper_state = {
            "optimizer": self.optimizer.state_dict(),
            **checkpoint["optimization_stepper"],
        }
        self.optimization_stepper.load_state_dict(stepper_state)
