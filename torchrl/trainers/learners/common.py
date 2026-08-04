# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import warnings
from dataclasses import dataclass

import torch
from tensordict import is_tensorclass, TensorDictBase
from torch import nn

from torchrl.objectives.common import LossModule


def _clone_tensors(obj):
    """Recursively clone every tensor in a (possibly nested) state-dict-like structure.

    ``nn.Module.state_dict()`` and ``Optimizer.state_dict()`` both return
    views onto their live tensors, not independent copies, so holding onto
    their output as a "checkpoint" while training continues silently mutates
    that checkpoint too.
    """
    if isinstance(obj, torch.Tensor):
        return obj.clone()
    if isinstance(obj, dict):
        return {k: _clone_tensors(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clone_tensors(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_clone_tensors(v) for v in obj)
    return obj


@dataclass(frozen=True)
class LearnerCapabilities:
    """Declares what a :class:`Learner` implementation supports.

    Frozen, so the class-level default on :attr:`Learner.capabilities` can be
    shared between instances without one learner's mutation leaking into every
    other.

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

    During gradient accumulation, if ``self.model`` exposes
    ``set_requires_gradient_sync`` (as an FSDP2 ``fully_shard``-wrapped module
    does), :meth:`update` disables cross-rank gradient synchronization on
    every accumulation step except the last: gradients still accumulate
    correctly (verified against a non-sharded reference), but the
    communication (reduce-scatter) only happens once per accumulation window
    instead of once per micro-batch. This is a no-op for
    :class:`~torchrl.trainers.learners.LocalLearner`, whose model has no such
    method.

    .. note::
        ``update`` requires ``loss_module.forward`` to follow the
        :class:`~torchrl.objectives.common.LossModule` convention: every
        differentiable loss term is returned under a key starting with
        ``"loss"`` (these are summed and used for the backward pass); any
        other returned entry (e.g. ``accuracy``, a KL for logging) is treated
        as a non-differentiable metric and left untouched.

    .. warning::
        **The optimizer, not** ``model``\\ **, defines what is trained.**
        ``model`` is the weight-sync source (:meth:`get_weights`) and the
        gradient-sync handle; the parameters that are clipped and stepped are
        the ones in ``optimizer.param_groups``. Several TorchRL losses hold
        their trainable parameters on the *loss module* as
        :class:`~tensordict.nn.TensorDictParams` -- and for losses that expand
        their networks (``SACLoss``, ``REDQLoss``, ...) those are copies, not
        the modules you passed in -- so the canonical construction is
        ``optimizer = Adam(loss_module.parameters())``. Building the optimizer
        from ``model.parameters()`` in that case leaves the critics untrained.
        :meth:`update` checks this before its first optimizer step and raises
        if a parameter received a gradient that no param group covers.

    .. note::
        A single ``Learner`` owns a single optimizer, so algorithms that
        deliberately use several (separate actor / critic / entropy-temperature
        optimizers, different learning rates per network) are not expressible
        as one ``Learner`` today. Pass a single optimizer over
        ``loss_module.parameters()``, or use one ``Learner`` per optimizer.

    .. seealso:: :class:`~torchrl.weight_update.WeightSyncScheme` consumes
        :meth:`get_weights` to synchronize a learner's parameters to remote
        inference workers, so a ``Learner`` composes with the existing
        weight-sync machinery without changes on either side.

    .. warning::
        Checkpointing is :meth:`checkpoint` / :meth:`load_checkpoint`, *not*
        ``state_dict`` / ``load_state_dict``. A bare
        :class:`~torch.optim.Optimizer` is not an ``nn.Module``, so
        ``nn.Module.state_dict()`` silently omits its state and a training
        resume would reset Adam's moments; but overriding ``state_dict`` to
        return the optimizer too would break the ``nn.Module`` contract
        (``destination`` / ``prefix`` / ``keep_vars`` are positional there, and
        a parent module calling ``child.state_dict(destination=...)`` discards
        the return value -- so nesting a ``Learner`` inside any other module
        would silently drop all of its state). Separate names keep both
        contracts intact: ``state_dict`` behaves exactly as ``nn.Module``'s,
        and :meth:`checkpoint` covers model + optimizer + accumulation state.
    """

    #: Frozen, so this class-level default is safe to share between instances.
    capabilities: LearnerCapabilities = LearnerCapabilities()

    model: nn.Module
    optimizer: torch.optim.Optimizer
    clip_grad_norm: float | None
    grad_accum_steps: int
    _accum_step: int
    _checked_optimizer_coverage: bool = False

    def _optimized_parameters(self) -> list[torch.nn.Parameter]:
        """Every parameter the optimizer will step -- what to clip, too."""
        return [p for group in self.optimizer.param_groups for p in group["params"]]

    def _check_optimizer_coverage(self, loss_module: LossModule) -> None:
        """Fail loudly if a parameter got a gradient no param group will step.

        Best effort by construction: it can only see the parameters that have a
        gradient at the first optimizer step, so a loss term that only becomes
        active later is not covered. It catches the common and otherwise silent
        mistake of building the optimizer from a bare model while the loss
        module owns (or expands) the trainable parameters.
        """
        optimized = {id(p) for p in self._optimized_parameters()}
        missing = [
            name
            for name, param in loss_module.named_parameters()
            if param.grad is not None and id(param) not in optimized
        ]
        if missing:
            shown = ", ".join(missing[:5])
            if len(missing) > 5:
                shown += f", ... (+{len(missing) - 5} more)"
            raise RuntimeError(
                f"{type(self).__name__}'s optimizer does not cover "
                f"{len(missing)} parameter(s) of the loss module that received "
                f"a gradient: {shown}. Those parameters are differentiated but "
                "never stepped, so they would stay frozen for the whole run. "
                "Build the optimizer over the loss module's parameters "
                "(e.g. `Adam(loss_module.parameters())`) rather than over a "
                "bare model: losses that expand their networks hold copies of "
                "them, not the modules you passed in."
            )

    def update(self, batch: TensorDictBase, loss_module: LossModule) -> TensorDictBase:
        """Take one optimization step on ``batch`` using ``loss_module``.

        Args:
            batch (TensorDictBase): a batch, in the format expected by
                ``loss_module``.
            loss_module (LossModule): computes the loss(es) for ``batch``.
                Its output's ``"loss"``-prefixed keys are summed and
                backpropagated; other keys are passed through for logging.

        Returns:
            TensorDictBase: the tensordict returned by ``loss_module``. On the
            calls that actually step the optimizer (every call unless
            ``grad_accum_steps > 1``) it additionally carries ``"grad_norm"``
            when ``clip_grad_norm`` is set -- so with gradient accumulation the
            output key set differs between accumulation calls and step calls.

        Raises:
            ValueError: if ``loss_module`` returns no ``"loss"``-prefixed key,
                or if the summed loss is not a scalar (which is what a loss
                built with ``reduction="none"`` produces).
            RuntimeError: if a parameter received a gradient that the optimizer
                does not cover -- see the class-level warning.
        """
        if self._accum_step == 0:
            self.optimizer.zero_grad(set_to_none=True)

        # Sharded (e.g. FSDP2) models can defer the cross-rank gradient
        # reduction until the last micro-batch of an accumulation window;
        # LocalLearner's plain model has no such method, so this is a no-op.
        set_grad_sync = getattr(self.model, "set_requires_gradient_sync", None)
        if set_grad_sync is not None:
            set_grad_sync(self._accum_step == self.grad_accum_steps - 1)

        loss_td = loss_module(batch)
        # "loss" (no underscore) is a real out_key -- DQNLoss, GAILLoss and
        # OnlineDTLoss all use it -- so this prefix cannot be tightened to
        # "loss_" without silently dropping their entire loss.
        loss_keys = [k for k in loss_td.keys() if k.startswith("loss")]
        if not loss_keys:
            raise ValueError(
                "loss_module returned no keys starting with 'loss': "
                f"{list(loss_td.keys())}. LossModule.forward must return at "
                "least one 'loss'-prefixed entry."
            )
        total_loss = sum(loss_td.get(k) for k in loss_keys) / self.grad_accum_steps
        if total_loss.numel() != 1:
            raise ValueError(
                f"The summed loss must be a scalar to be backpropagated, but "
                f"{loss_keys} summed to shape {tuple(total_loss.shape)}. This "
                "is what a loss module built with `reduction='none'` returns; "
                "use reduction='mean' (or 'sum') for the loss driving a "
                "Learner, and compute per-sample losses separately if you need "
                "them for logging."
            )
        total_loss.backward()

        self._accum_step += 1
        if self._accum_step < self.grad_accum_steps:
            return loss_td
        self._accum_step = 0

        if not self._checked_optimizer_coverage:
            self._check_optimizer_coverage(loss_module)
            self._checked_optimizer_coverage = True

        if self.clip_grad_norm is not None:
            # Clip what is about to be stepped, which is not necessarily
            # ``self.model.parameters()`` -- see the class-level warning.
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self._optimized_parameters(), self.clip_grad_norm
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

    def checkpoint(self) -> dict:
        """Return a checkpoint covering the model, optimizer and accumulation state.

        Deliberately *not* ``state_dict``: see the class-level warning.
        ``state_dict`` keeps its :class:`~torch.nn.Module` meaning (so a
        ``Learner`` can be nested inside another module without losing state),
        and this method adds what ``state_dict`` structurally cannot carry --
        the optimizer, which is not an ``nn.Module``.

        The returned tensors are independent clones: both
        ``nn.Module.state_dict()`` and ``Optimizer.state_dict()`` return views
        onto the live parameters/state, so an in-place update after
        checkpointing would otherwise silently corrupt the "saved" checkpoint
        too.

        Returns:
            dict: with keys ``"model"``, ``"optimizer"`` and ``"accum_step"``.
        """
        return {
            "model": _clone_tensors(self.model.state_dict()),
            "optimizer": _clone_tensors(self.optimizer.state_dict()),
            "accum_step": self._accum_step,
        }

    def load_checkpoint(self, checkpoint: dict) -> None:
        """Restore a checkpoint produced by :meth:`checkpoint`.

        Args:
            checkpoint (dict): as returned by :meth:`checkpoint`.

        .. note::
            Gradients are not part of the checkpoint, so a checkpoint taken
            mid-accumulation-window cannot be resumed mid-window: the
            accumulation counter is reset to 0 (with a warning) rather than
            resuming from a non-zero step with empty gradients, which would
            step the optimizer after fewer micro-batches than
            ``grad_accum_steps`` and therefore under-scale that update.
        """
        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self._restore_accum_step(checkpoint.get("accum_step", 0))

    def _restore_accum_step(self, accum_step: int) -> None:
        if accum_step:
            warnings.warn(
                f"This checkpoint was taken {accum_step} step(s) into a "
                f"{self.grad_accum_steps}-step gradient accumulation window. "
                "Gradients are not checkpointed, so the partial window is "
                "discarded and accumulation restarts from 0.",
                UserWarning,
                stacklevel=3,
            )
            self.optimizer.zero_grad(set_to_none=True)
        self._accum_step = 0
