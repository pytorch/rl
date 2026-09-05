# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

from tensordict import TensorDict, TensorDictBase
from torch import nn

from torchrl.objectives.common import LossModule
from torchrl.trainers.learners.common import Learner, LearnerCapabilities
from torchrl.trainers.trainers import OptimizationStepper


class LocalLearner(Learner):
    """Single-process :class:`~torchrl.trainers.Learner` backend.

    ``LocalLearner`` supplies placement and weight publication. Optimization
    behavior comes entirely from the same ``OptimizationStepper`` used by
    :class:`~torchrl.trainers.Trainer`.

    Args:
        model (torch.nn.Module): Module whose weights are published.
        loss_module (LossModule): Loss evaluated by :meth:`update`.
        optimization_stepper (OptimizationStepper): Optimization policy.

    Examples:
        >>> import torch
        >>> from tensordict import TensorDict
        >>> from torch import nn
        >>> from torchrl.objectives.common import LossModule
        >>> from torchrl.trainers import LocalLearner, MixedPrecisionOptimizationStepper
        >>> class ToyLoss(LossModule):
        ...     def __init__(self, model):
        ...         super().__init__()
        ...         self.model = model
        ...     def forward(self, batch):
        ...         prediction = self.model(batch["x"])
        ...         return TensorDict({"loss_mse": (prediction - batch["y"]).pow(2).mean()})
        >>> model = nn.Linear(4, 1)
        >>> loss_module = ToyLoss(model)
        >>> stepper = MixedPrecisionOptimizationStepper(
        ...     torch.optim.Adam(loss_module.parameters())
        ... )
        >>> learner = LocalLearner(model, loss_module, stepper)
        >>> batch = TensorDict({"x": torch.randn(8, 4), "y": torch.randn(8, 1)}, [8])
        >>> metrics = learner.update(batch)
        >>> metrics["loss_mse"].ndim
        0
    """

    def __init__(
        self,
        model: nn.Module,
        loss_module: LossModule,
        optimization_stepper: OptimizationStepper,
    ):
        super().__init__(model, loss_module, optimization_stepper)
        self.capabilities = LearnerCapabilities(sharded=False, remote=False)

    def get_weights(self) -> TensorDictBase:
        return TensorDict.from_module(self.model).data.detach().clone()
