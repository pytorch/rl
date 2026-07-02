# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

from dataclasses import dataclass

from tensordict import TensorDictBase
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

    A :class:`Learner` owns a trainable model and exposes a single,
    backend-agnostic entry point, :meth:`update`, for taking one optimization
    step on a :class:`~tensordict.TensorDictBase` batch with a given
    :class:`~torchrl.objectives.common.LossModule`. Algorithm code calls
    ``learner.update(batch, loss_module)`` without knowing whether the update
    runs locally on one device, under sharded (e.g. FSDP2) training, or on a
    separate remote training process -- that placement is the ``Learner``
    subclass's responsibility, not the algorithm's.

    This mirrors the role :class:`~torchrl.collectors.Collector` plays for
    data collection and :class:`~torchrl.modules.llm.LLMWrapperBase` plays for
    generation/scoring: a fixed, TensorDict-native contract with multiple
    interchangeable backends.

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
        raise NotImplementedError

    def get_weights(self) -> TensorDictBase:
        """Return the learner's current parameters as a tensordict.

        The returned tensordict is accepted as-is by
        :meth:`~torchrl.weight_update.WeightSyncScheme.send`, so it is the
        seam between the training role and the weight-sync/inference roles.
        """
        raise NotImplementedError
