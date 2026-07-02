# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import torch
from tensordict import is_tensorclass, TensorDict, TensorDictBase
from torch import nn

from torchrl.objectives.common import LossModule
from torchrl.trainers.learners.common import Learner, LearnerCapabilities


class LocalLearner(Learner):
    """A single-process, single-device :class:`~torchrl.trainers.learners.Learner`.

    Wraps a model and an optimizer and performs the update in-process: forward
    through ``loss_module``, sum the outputs' ``"loss"``-prefixed entries,
    backward, optionally clip gradients, and step the optimizer. This is the
    reference implementation the :class:`~torchrl.trainers.learners.Learner`
    contract is designed around -- a sharded (e.g. FSDP2-backed) or remote
    learner implements the same two methods, :meth:`update` and
    :meth:`get_weights`, so that algorithm code written against
    ``LocalLearner`` runs unchanged against those backends.

    Args:
        model (torch.nn.Module): the trainable module. Also the source for
            :meth:`get_weights`.
        optimizer (torch.optim.Optimizer): the optimizer stepping ``model``'s
            parameters.

    Keyword Args:
        clip_grad_norm (float, optional): if set, gradients are clipped to
            this max norm via :func:`torch.nn.utils.clip_grad_norm_` before
            the optimizer step, and the resulting norm is written to the
            output tensordict under ``"grad_norm"``. Defaults to ``None``.
        grad_accum_steps (int, optional): number of :meth:`update` calls to
            accumulate gradients over before stepping the optimizer and
            zeroing gradients. Defaults to ``1`` (step on every call).

    Examples:
        >>> import torch
        >>> from tensordict import TensorDict
        >>> from torch import nn
        >>> from torchrl.trainers.learners import LocalLearner
        >>> from torchrl.objectives.common import LossModule
        >>>
        >>> class ToyLoss(LossModule):
        ...     def forward(self, batch):
        ...         pred = self.actor(batch["x"])
        ...         return TensorDict({"loss_mse": (pred - batch["y"]).pow(2).mean()})
        >>>
        >>> model = nn.Linear(4, 1)
        >>> loss_module = ToyLoss()
        >>> loss_module.actor = model
        >>> learner = LocalLearner(model, torch.optim.Adam(model.parameters()))
        >>> batch = TensorDict({"x": torch.randn(8, 4), "y": torch.randn(8, 1)}, [8])
        >>> out = learner.update(batch, loss_module)
        >>> out.get("loss_mse").item() > 0
        True
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        *,
        clip_grad_norm: float | None = None,
        grad_accum_steps: int = 1,
    ) -> None:
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.clip_grad_norm = clip_grad_norm
        if grad_accum_steps < 1:
            raise ValueError(f"grad_accum_steps must be >= 1, got {grad_accum_steps}.")
        self.grad_accum_steps = grad_accum_steps
        self._accum_step = 0
        self.capabilities = LearnerCapabilities(sharded=False, remote=False)

    def update(self, batch: TensorDictBase, loss_module: LossModule) -> TensorDictBase:
        if self._accum_step == 0:
            self.optimizer.zero_grad(set_to_none=True)

        loss_td = loss_module(batch)
        loss_keys = [k for k in loss_td.keys() if k.startswith("loss")]
        if not loss_keys:
            raise ValueError(
                "loss_module returned no keys starting with 'loss': "
                f"{list(loss_td.keys())}. LossModule.forward must return at "
                "least one 'loss'-prefixed entry."
            )
        total_loss = sum(loss_td.get(k) for k in loss_keys) / self.grad_accum_steps
        total_loss.backward()

        self._accum_step += 1
        if self._accum_step < self.grad_accum_steps:
            return loss_td
        self._accum_step = 0

        if self.clip_grad_norm is not None:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.clip_grad_norm
            )
            # loss_module may return a strict TensorClass (e.g. RewardModelLossOutput)
            # that rejects undeclared keys; convert to a writable TensorDict first.
            if is_tensorclass(loss_td):
                loss_td = loss_td.to_tensordict()
            loss_td.set("grad_norm", grad_norm)

        self.optimizer.step()
        return loss_td

    def get_weights(self) -> TensorDictBase:
        return TensorDict.from_module(self.model)
