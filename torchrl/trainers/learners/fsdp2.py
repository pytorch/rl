# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import torch
from tensordict import TensorDict, TensorDictBase
from torch import nn

from torchrl.trainers.learners.common import Learner, LearnerCapabilities

try:
    from torch.distributed.tensor import DTensor

    _has_dtensor = True
except ImportError:  # pragma: no cover - torch without distributed.tensor
    _has_dtensor = False


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

    :meth:`get_weights` gathers every sharded leaf into a plain tensor via
    ``DTensor.full_tensor()``, so the returned tensordict holds regular
    tensors and is consumable by
    :meth:`~torchrl.weight_update.WeightSyncScheme.send` exactly like
    :class:`~torchrl.trainers.learners.LocalLearner`'s, with no changes needed
    on the receiving (inference) side. This gather is the seam between
    sharded training and (typically replicated) inference, and is the one
    place sharding is not transparent.

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
        if not _has_dtensor:
            raise RuntimeError(
                "FSDP2Learner requires torch.distributed.tensor (DTensor), "
                "which is not available in this torch build."
            )
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.clip_grad_norm = clip_grad_norm
        if grad_accum_steps < 1:
            raise ValueError(f"grad_accum_steps must be >= 1, got {grad_accum_steps}.")
        self.grad_accum_steps = grad_accum_steps
        self._accum_step = 0
        self.capabilities = LearnerCapabilities(sharded=True, remote=False)

    def get_weights(self) -> TensorDictBase:
        td = TensorDict.from_module(self.model)
        return td.apply(lambda t: t.full_tensor() if isinstance(t, DTensor) else t)
