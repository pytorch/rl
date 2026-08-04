# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import functools

import torch
from tensordict import TensorDict, TensorDictBase
from torch import nn

from torchrl.trainers.learners.common import (
    _clone_tensors,
    Learner,
    LearnerCapabilities,
)


@functools.lru_cache(maxsize=None)
def _dist_state_dict():
    """Return ``torch.distributed.checkpoint.state_dict``, imported on demand.

    Deliberately lazy rather than a module-top import guarded by
    ``importlib.util.find_spec``: this is a *torch submodule* whose presence
    depends on the local torch build, and ``find_spec`` on a dotted name
    eagerly imports the parent packages (and raises, rather than returning
    ``None``, when a parent is missing). Importing it on first use keeps
    ``import torchrl`` free of ``torch.distributed.checkpoint``.
    """
    from torch.distributed.checkpoint import state_dict

    return state_dict


class FSDP2Learner(Learner):
    """A :class:`~torchrl.trainers.learners.Learner` for FSDP2-sharded models.

    Accepts a model that the caller has already wrapped with
    :func:`torch.distributed._composable.fsdp.fully_shard` (typically
    per-submodule, then on the root module), and an optimizer built on that
    (sharded) model's parameters. ``fully_shard`` must be applied *before* the
    optimizer is constructed, so the optimizer holds the sharded
    (:class:`~torch.distributed.tensor.DTensor`) parameters rather than the
    original ones.

    :meth:`~torchrl.trainers.learners.Learner.update` is inherited unchanged
    from :class:`~torchrl.trainers.learners.Learner`: FSDP2's sharding is
    transparent to the training step (forward/backward/optimizer-step all
    dispatch through ``DTensor`` the same way they would through a regular
    tensor). Only two things differ from
    :class:`~torchrl.trainers.learners.LocalLearner`: how the model arrives
    (already sharded, by the caller) and how :meth:`get_weights` reports it.

    :meth:`get_weights` reassembles the sharded parameters into plain tensors
    with
    :func:`torch.distributed.checkpoint.state_dict.get_model_state_dict`, so
    the returned tensordict holds regular tensors and is consumable by
    :meth:`~torchrl.weight_update.WeightSyncScheme.send` exactly like
    :class:`~torchrl.trainers.learners.LocalLearner`'s, with no changes needed
    on the receiving (inference) side. This gather is the seam between
    sharded training and (typically replicated) inference, and is the one
    place sharding is not transparent.

    .. warning::
        With the default ``cpu_offload=True``, :meth:`get_weights` returns the
        full weights **on rank 0 only**; every other rank gets an *empty*
        tensordict. That is deliberate -- gathering the full model onto every
        rank replicates it in every rank's memory for no benefit and does not
        scale -- but it means ``scheme.send(learner.get_weights())`` called
        unconditionally on all ranks sends nothing from the non-zero ones.
        Either send from rank 0 only, or pass ``cpu_offload=False`` to gather
        full copies everywhere.

    Args:
        model (torch.nn.Module): a model already wrapped with ``fully_shard``.
        optimizer (torch.optim.Optimizer): an optimizer constructed on
            ``model``'s (already-sharded) parameters.

    Keyword Args:
        clip_grad_norm (float, optional): as in
            :class:`~torchrl.trainers.learners.LocalLearner`. Gradient-norm
            clipping dispatches through ``DTensor`` transparently. Defaults to
            ``None``.
        grad_accum_steps (int, optional): as in
            :class:`~torchrl.trainers.learners.LocalLearner`. Defaults to
            ``1``.

    .. warning::
        ``FSDP2Learner`` does not itself decide what to shard, at what
        granularity, or on what device mesh -- those are model-specific
        choices that belong in the caller's model-construction code, exactly
        as they would with bare ``fully_shard``. This keeps
        ``FSDP2Learner`` a thin adapter rather than a second place those
        decisions are made.

    Examples:
        >>> import torch
        >>> import torch.distributed as dist
        >>> from torch import nn
        >>> from torch.distributed._composable.fsdp import fully_shard
        >>> from torch.distributed.device_mesh import init_device_mesh
        >>> from tensordict import TensorDict
        >>> from torchrl.objectives.common import LossModule
        >>> from torchrl.trainers.learners import FSDP2Learner
        >>>
        >>> dist.init_process_group(backend="gloo", rank=0, world_size=1)
        >>> mesh = init_device_mesh("cpu", (1,))
        >>> model = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 1))
        >>> for layer in model:
        ...     fully_shard(layer, mesh=mesh)
        >>> fully_shard(model, mesh=mesh)
        >>> optimizer = torch.optim.Adam(model.parameters())
        >>>
        >>> class ToyLoss(LossModule):
        ...     def forward(self, batch):
        ...         pred = self.actor(batch["x"])
        ...         return TensorDict({"loss_mse": (pred - batch["y"]).pow(2).mean()})
        >>> loss_module = ToyLoss()
        >>> loss_module.actor = model
        >>>
        >>> learner = FSDP2Learner(model, optimizer)
        >>> batch = TensorDict({"x": torch.randn(8, 4), "y": torch.randn(8, 1)}, [8])
        >>> out = learner.update(batch, loss_module)
        >>> weights = learner.get_weights()  # gathered, plain tensors
        >>> dist.destroy_process_group()
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        *,
        clip_grad_norm: float | None = None,
        grad_accum_steps: int = 1,
    ) -> None:
        try:
            _dist_state_dict()
        except ImportError as err:
            raise RuntimeError(
                "FSDP2Learner requires torch.distributed.checkpoint.state_dict, "
                "which is not available in this torch build."
            ) from err
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.clip_grad_norm = clip_grad_norm
        if grad_accum_steps < 1:
            raise ValueError(f"grad_accum_steps must be >= 1, got {grad_accum_steps}.")
        self.grad_accum_steps = grad_accum_steps
        self._accum_step = 0
        self.capabilities = LearnerCapabilities(sharded=True, remote=False)

    def get_weights(self, *, cpu_offload: bool = True) -> TensorDictBase:
        """Reassemble the sharded model into a plain-tensor tensordict.

        Keyword Args:
            cpu_offload (bool, optional): if ``True`` (the default), the
                gathered weights are returned **only on rank 0** (every other
                rank gets an empty tensordict) and moved to CPU, avoiding an
                all-rank replication of the full model -- see the class-level
                warning. Set to ``False`` to gather full device-resident copies
                onto every rank instead.
        """
        dsd = _dist_state_dict()
        options = dsd.StateDictOptions(full_state_dict=True, cpu_offload=cpu_offload)
        state_dict = dsd.get_model_state_dict(self.model, options=options)
        return TensorDict(state_dict).unflatten_keys(".")

    def checkpoint(self) -> dict:
        """DTensor-aware checkpoint: gathers model + optimizer state to rank 0.

        Overrides :meth:`~torchrl.trainers.learners.Learner.checkpoint`, whose
        plain (non-distributed) ``state_dict()`` calls would return raw,
        per-rank ``DTensor`` shards rather than a portable checkpoint.

        Like the base implementation, the returned tensors are independent
        clones. ``get_state_dict`` does *not* guarantee that: with
        ``cpu_offload=True`` it only copies when the shards are off-CPU, so on a
        CPU mesh (or a single-rank one) the "gathered" tensors alias the live
        shards, and training on would silently mutate the checkpoint.
        """
        dsd = _dist_state_dict()
        options = dsd.StateDictOptions(full_state_dict=True, cpu_offload=True)
        model_state_dict, optim_state_dict = dsd.get_state_dict(
            self.model, self.optimizer, options=options
        )
        return {
            "model": _clone_tensors(model_state_dict),
            "optimizer": _clone_tensors(optim_state_dict),
            "accum_step": self._accum_step,
        }

    def load_checkpoint(self, checkpoint: dict) -> None:
        """Restore a checkpoint produced by :meth:`checkpoint`.

        ``broadcast_from_rank0=True`` lets rank 0 hold the full checkpoint
        (as produced by :meth:`checkpoint`) while every rank reshards it
        according to its local shards -- the counterpart of the
        ``cpu_offload``-gathered save.

        Args:
            checkpoint (dict): as returned by :meth:`checkpoint`.
        """
        dsd = _dist_state_dict()
        options = dsd.StateDictOptions(full_state_dict=True, broadcast_from_rank0=True)
        dsd.set_state_dict(
            self.model,
            self.optimizer,
            model_state_dict=checkpoint["model"],
            optim_state_dict=checkpoint["optimizer"],
            options=options,
        )
        self._restore_accum_step(checkpoint.get("accum_step", 0))
