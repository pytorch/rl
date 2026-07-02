# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

from dataclasses import dataclass

import torch
from tensordict import is_tensorclass, TensorDictBase
from torch import nn

from torchrl.objectives.common import LossModule


@dataclass
class LearnerCapabilities:
    """Declares what a :class:`Learner` implementation supports.

    Algorithm code is not expected to branch on these (that would defeat the
    point of the abstraction); they exist for logging, validation, and for
    orchestration code that needs to know, e.g., whether a learner can be
    checkpointed independently on every rank.

    Attributes:
        sharded (bool): whether the learner's parameters are sharded across
            multiple devices/processes (e.g. FSDP2). Defaults to ``False``.
        remote (bool): whether :meth:`~Learner.update` dispatches to a
            separate process rather than running in-line. Defaults to
            ``False``.
    """

    sharded: bool = False
    remote: bool = False


class Learner(nn.Module):
    """Base class for the trainable-policy role.

    A :class:`Learner` owns a trainable model and an optimizer and exposes a
    single, backend-agnostic entry point, :meth:`update`, for taking one
    optimization step on a :class:`~tensordict.TensorDictBase` batch with a
    given :class:`~torchrl.objectives.common.LossModule`. Algorithm code calls
    ``learner.update(batch, loss_module)`` without knowing whether the update
    runs locally on one device, under sharded (e.g. FSDP2) training, or on a
    separate remote training process -- that placement is the ``Learner``
    subclass's responsibility, not the algorithm's.

    This mirrors the role :class:`~torchrl.collectors.Collector` plays for
    data collection and :class:`~torchrl.modules.llm.LLMWrapperBase` plays for
    generation/scoring: a fixed, TensorDict-native contract with multiple
    interchangeable backends.

    :meth:`update` is implemented once, here, and is intentionally backend
    agnostic: it only touches ``self.model``, ``self.optimizer``,
    ``self.clip_grad_norm``, and ``self.grad_accum_steps``, all of which a
    subclass sets in its constructor. This is what lets
    :class:`~torchrl.trainers.learners.FSDP2Learner` reuse the exact same
    training step as :class:`~torchrl.trainers.learners.LocalLearner`: sharded
    training only changes how the model is constructed (wrapped with
    ``fully_shard``) and how :meth:`get_weights` gathers the result, not how a
    step is taken.

    .. note::
        ``update`` requires ``loss_module.forward`` to follow the
        :class:`~torchrl.objectives.common.LossModule` convention: every
        differentiable loss term is returned under a key starting with
        ``"loss"`` (these are summed and used for the backward pass); any
        other returned entry (e.g. ``accuracy``, a KL for logging) is treated
        as a non-differentiable metric and left untouched.

    .. seealso:: :class:`~torchrl.weight_update.WeightSyncScheme` consumes
        :meth:`get_weights` to synchronize a learner's parameters to remote
        inference workers, so a ``Learner`` composes with the existing
        weight-sync machinery without changes on either side.
    """

    capabilities: LearnerCapabilities = LearnerCapabilities()

    model: nn.Module
    optimizer: torch.optim.Optimizer
    clip_grad_norm: float | None
    grad_accum_steps: int
    _accum_step: int

    def update(self, batch: TensorDictBase, loss_module: LossModule) -> TensorDictBase:
        """Take one optimization step on ``batch`` using ``loss_module``.

        Args:
            batch (TensorDictBase): a batch, in the format expected by
                ``loss_module``.
            loss_module (LossModule): computes the loss(es) for ``batch``.
                Its output's ``"loss"``-prefixed keys are summed and
                backpropagated; other keys are passed through for logging.

        Returns:
            TensorDictBase: the tensordict returned by ``loss_module``,
            augmented with a ``"grad_norm"`` entry when gradient clipping is
            enabled.
        """
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
        """Return the learner's current parameters as a tensordict.

        The returned tensordict holds plain (fully materialized) tensors, even
        when the learner's parameters are internally sharded, so it is
        accepted as-is by :meth:`~torchrl.weight_update.WeightSyncScheme.send`.
        This is the seam between the training role and the weight-sync /
        inference roles.
        """
        raise NotImplementedError
