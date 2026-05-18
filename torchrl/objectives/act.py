# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass

import torch.nn.functional as F
from tensordict import TensorDict, TensorDictBase, TensorDictParams
from tensordict.nn import TensorDictModule
from tensordict.utils import NestedKey

from torchrl.objectives.common import LossModule


class ACTLoss(LossModule):
    r"""Loss module for Action Chunking with Transformers (ACT).

    Implements the training objective from *Learning Fine-Grained Bimanual
    Manipulation with Low-Cost Hardware* (`Zhao et al., 2023
    <https://arxiv.org/abs/2304.13705>`_), pairing an L1
    chunk-reconstruction term with a KL-divergence penalty on the CVAE
    latent:

    .. math::

        \mathcal{L} = \underbrace{\|a_{\text{pred}} -
        a_{\text{chunk}}\|_1}_{\text{reconstruction}}
        + \beta \cdot
        \underbrace{D_{\mathrm{KL}}\!\left(q(z|o,a)\,\|\,
        \mathcal{N}(0,I)\right)}_{\text{KL}}

    The ``actor_network`` must read ``"observation"`` and ``"action_chunk"``
    and write ``"action_pred"``, ``"mu"``, and ``"log_var"``.  This matches
    the contract of :class:`~torchrl.modules.models.ACTModel` when wrapped
    with a :class:`~tensordict.nn.TensorDictModule`.

    Three values are returned in the output TensorDict:

    * ``"loss_act"`` — the full (differentiable) training loss.
    * ``"loss_reconstruction"`` — detached L1 reconstruction term (for
      logging).
    * ``"loss_kl"`` — detached KL term (for logging).

    Args:
        actor_network (TensorDictModule): ACT policy.  Must expose
            ``in_keys`` containing ``"observation"`` and ``"action_chunk"``
            and write ``"action_pred"``, ``"mu"``, ``"log_var"``.

    Keyword Args:
        kl_weight (float, optional): β — weight on the KL divergence term.
            Defaults to ``10.0`` (as in the original paper).
        reduction (str, optional): ``"none"`` | ``"mean"`` | ``"sum"``.
            Defaults to ``"mean"``.

    Examples:
        >>> import torch
        >>> from tensordict import TensorDict
        >>> from tensordict.nn import TensorDictModule
        >>> from torchrl.modules.models import ACTModel
        >>> from torchrl.objectives import ACTLoss
        >>> model = ACTModel(obs_dim=14, action_dim=7, chunk_size=10)
        >>> actor = TensorDictModule(
        ...     model,
        ...     in_keys=["observation", "action_chunk"],
        ...     out_keys=["action_pred", "mu", "log_var"],
        ... )
        >>> loss_fn = ACTLoss(actor, kl_weight=10.0)
        >>> td = TensorDict(
        ...     {
        ...         "observation": torch.randn(4, 14),
        ...         "action_chunk": torch.randn(4, 10, 7),
        ...     },
        ...     batch_size=[4],
        ... )
        >>> loss_td = loss_fn(td)
        >>> loss_td["loss_act"].backward()
    """

    @dataclass
    class _AcceptedKeys:
        """Configurable tensordict keys for :class:`ACTLoss`.

        Attributes:
            observation (NestedKey): Observation key. Default ``"observation"``.
            action_chunk (NestedKey): Expert action chunk
                ``(..., T, action_dim)``. Default ``"action_chunk"``.
            action_pred (NestedKey): Predicted chunk written by the policy.
                Default ``"action_pred"``.
            mu (NestedKey): CVAE encoder mean. Default ``"mu"``.
            log_var (NestedKey): CVAE encoder log-variance. Default
                ``"log_var"``.
        """

        observation: NestedKey = "observation"
        action_chunk: NestedKey = "action_chunk"
        action_pred: NestedKey = "action_pred"
        mu: NestedKey = "mu"
        log_var: NestedKey = "log_var"

    default_keys = _AcceptedKeys()

    actor_network: TensorDictModule
    actor_network_params: TensorDictParams | None
    target_actor_network_params: TensorDictParams | None

    def _forward_value_estimator_keys(self, **kwargs) -> None:
        pass

    def __init__(
        self,
        actor_network: TensorDictModule,
        *,
        kl_weight: float = 10.0,
        reduction: str = "mean",
    ) -> None:
        self._in_keys = None
        self._out_keys = None
        super().__init__()
        self.convert_to_functional(actor_network, "actor_network")
        self.kl_weight = kl_weight
        self.reduction = reduction

    @property
    def in_keys(self):
        if self._in_keys is None:
            self._in_keys = [
                self.tensor_keys.observation,
                self.tensor_keys.action_chunk,
            ]
        return self._in_keys

    @in_keys.setter
    def in_keys(self, value):
        self._in_keys = value

    @property
    def out_keys(self):
        if self._out_keys is None:
            self._out_keys = ["loss_act", "loss_reconstruction", "loss_kl"]
        return self._out_keys

    @out_keys.setter
    def out_keys(self, value):
        self._out_keys = value

    def forward(self, tensordict: TensorDictBase) -> TensorDict:
        """Compute the ACT loss.

        Args:
            tensordict (TensorDictBase): Input data containing
                ``"observation"`` and ``"action_chunk"``.

        Returns:
            TensorDict with keys ``"loss_act"``, ``"loss_reconstruction"``,
            and ``"loss_kl"``.
        """
        action_chunk = tensordict.get(self.tensor_keys.action_chunk)
        td_in = TensorDict(
            {
                "observation": tensordict.get(self.tensor_keys.observation),
                "action_chunk": action_chunk,
            },
            batch_size=tensordict.batch_size,
            device=tensordict.device,
        )
        with self.actor_network_params.to_module(self.actor_network):
            td_out = self.actor_network(td_in)

        action_pred = td_out.get(self.tensor_keys.action_pred)
        mu = td_out.get(self.tensor_keys.mu)
        log_var = td_out.get(self.tensor_keys.log_var)

        # L1 reconstruction — average over chunk and action dimensions first,
        # then apply the batch reduction. Reducing only the trailing two dims
        # keeps multi-dim batch shapes intact so the per-element loss matches
        # ``tensordict.batch_size`` under ``reduction="none"``.
        loss_recon = F.l1_loss(action_pred, action_chunk, reduction="none")
        loss_recon = loss_recon.mean(dim=(-2, -1))
        if self.reduction == "mean":
            loss_recon = loss_recon.mean()
        elif self.reduction == "sum":
            loss_recon = loss_recon.sum()

        # KL divergence: KL(N(mu, sigma²) || N(0, I))
        kl_per_dim = -0.5 * (1.0 + log_var - mu.pow(2) - log_var.exp())
        kl = kl_per_dim.sum(dim=-1)  # sum over latent dims
        if self.reduction == "mean":
            loss_kl = kl.mean()
        elif self.reduction == "sum":
            loss_kl = kl.sum()
        else:
            loss_kl = kl

        loss_act = loss_recon + self.kl_weight * loss_kl

        return TensorDict(
            {
                "loss_act": loss_act,
                "loss_reconstruction": loss_recon.detach(),
                "loss_kl": loss_kl.detach(),
            },
            batch_size=[],
        )
