# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""RL fine-tuning objectives for token-based Vision-Language-Action policies."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, TYPE_CHECKING

import torch
from tensordict import TensorDict, TensorDictBase
from tensordict.nn import dispatch
from tensordict.utils import NestedKey

from torchrl.data.vla.schema import ACTION_TOKENS_KEY
from torchrl.objectives.common import LossModule
from torchrl.objectives.utils import _reduce

if TYPE_CHECKING:
    from torchrl.modules.vla import VLAWrapperBase

__all__ = ["VLATokenGRPOLoss"]


class VLATokenGRPOLoss(LossModule):
    """GRPO / PPO-clip objective for token-based VLA policies.

    Reinforcement fine-tuning for autoregressive (token) VLAs, following the
    SimpleVLA-RL / RL4VLA recipe: a clipped policy-gradient surrogate over the
    action tokens with group-relative (or any precomputed) advantages, plus an
    optional KL penalty to a reference policy
    (`SimpleVLA-RL <https://arxiv.org/abs/2509.09674>`_,
    `RLVLA <https://arxiv.org/abs/2505.19789>`_). It mirrors the LLM
    :class:`~torchrl.objectives.llm.GRPOLoss` math while operating on a
    fixed-length action-token chunk (no sequence masking).

    The per-sample log-probability is the sum of the per-token log-probs over the
    action chunk. With ``log_weight = current_logp - sample_logp`` (the
    importance weight) and an ``advantage`` per sample, the objective is the
    PPO-clipped ``-min(exp(log_weight) * A, clip(exp(log_weight)) * A)``.

    The actor is a token-head :class:`~torchrl.modules.vla.VLAWrapperBase`; it is
    stored as a plain submodule (so it works with lazily-initialized policies and
    exposes its parameters through ``loss.parameters()``).

    Args:
        actor_network (VLAWrapperBase): a token-head VLA policy (provides
            :meth:`~torchrl.modules.vla.VLAWrapperBase.get_dist`).

    Keyword Args:
        clip_epsilon (float): PPO clipping range. Defaults to ``0.2``.
        kl_to_ref_coeff (float, optional): coefficient of the KL-to-reference
            penalty (requires ``ref_log_probs`` in the input). Defaults to ``None``.
        reduction (str): ``"mean"`` (default), ``"sum"`` or ``"none"``.

    Examples:
        >>> import torch
        >>> from tensordict import NonTensorStack, TensorDict
        >>> from torchrl.modules.vla import TinyVLA
        >>> from torchrl.objectives.vla import VLATokenGRPOLoss
        >>> policy = TinyVLA(action_dim=2, chunk_size=2, action_head="tokens", vocab_size=8)
        >>> loss = VLATokenGRPOLoss(policy)
        >>> obs = TensorDict(
        ...     {
        ...         "observation": {
        ...             "image": torch.zeros(3, 3, 16, 16, dtype=torch.uint8),
        ...             "state": torch.zeros(3, 5),
        ...         },
        ...         "language_instruction": NonTensorStack("a", "b", "c"),
        ...     },
        ...     batch_size=[3],
        ... )
        >>> td = policy(obs.clone())  # roll out: action_tokens + log_probs
        >>> td["advantage"] = torch.randn(3)
        >>> td["log_probs"] = td["log_probs"].detach()
        >>> sorted(loss(td).keys())
        ['clip_fraction', 'loss_objective']

    .. seealso:: :class:`~torchrl.objectives.llm.GRPOLoss`,
        :class:`~torchrl.modules.vla.VLAWrapperBase`.
    """

    @dataclass
    class _AcceptedKeys:
        """Configurable tensordict keys for :class:`VLATokenGRPOLoss`.

        Attributes:
            action_tokens (NestedKey): the sampled action tokens. Defaults to
                ``"action_tokens"``.
            advantage (NestedKey): per-sample advantage. Defaults to ``"advantage"``.
            sample_log_prob (NestedKey): per-token log-probs from the behavior
                (collection) policy. Defaults to ``"log_probs"``.
            ref_log_probs (NestedKey): per-token log-probs of the reference
                policy (for the KL penalty). Defaults to ``"ref_log_probs"``.
        """

        action_tokens: NestedKey = ACTION_TOKENS_KEY
        advantage: NestedKey = "advantage"
        sample_log_prob: NestedKey = "log_probs"
        ref_log_probs: NestedKey = "ref_log_probs"

    tensor_keys: _AcceptedKeys
    default_keys = _AcceptedKeys
    out_keys = ["loss_objective", "clip_fraction"]

    def __init__(
        self,
        actor_network: VLAWrapperBase,
        *,
        clip_epsilon: float = 0.2,
        kl_to_ref_coeff: float | None = None,
        reduction: Literal["mean", "sum", "none"] | None = None,
    ) -> None:
        super().__init__()
        if getattr(actor_network, "action_head", None) != "tokens":
            raise ValueError(
                "VLATokenGRPOLoss requires a token-head VLA policy "
                "(action_head='tokens') exposing get_dist; got action_head="
                f"{getattr(actor_network, 'action_head', None)!r}."
            )
        if not 0 <= clip_epsilon < 1:
            raise ValueError(f"clip_epsilon must be in [0, 1), got {clip_epsilon}.")
        self._in_keys = None
        self.actor_network = actor_network
        self.clip_epsilon = clip_epsilon
        self._clip_bounds = (math.log1p(-clip_epsilon), math.log1p(clip_epsilon))
        self.kl_to_ref_coeff = kl_to_ref_coeff
        self.reduction = "mean" if reduction is None else reduction
        self.out_keys = ["loss_objective", "clip_fraction"]
        if kl_to_ref_coeff is not None and kl_to_ref_coeff > 0:
            self.out_keys = self.out_keys + ["loss_kl_to_ref", "kl_to_ref"]

    def _forward_value_estimator_keys(self, **kwargs) -> None:
        pass

    def _set_in_keys(self) -> None:
        keys = [
            self.tensor_keys.action_tokens,
            self.tensor_keys.advantage,
            self.tensor_keys.sample_log_prob,
            self.tensor_keys.ref_log_probs,
            *self.actor_network.in_keys,
        ]
        self._in_keys = list(set(keys))

    @property
    def in_keys(self):
        if self._in_keys is None:
            self._set_in_keys()
        return self._in_keys

    @in_keys.setter
    def in_keys(self, values):
        self._in_keys = values

    def set_keys(self, **kwargs) -> None:
        super().set_keys(**kwargs)
        self._in_keys = None

    @staticmethod
    def _sum_tokens(per_token: torch.Tensor, batch_ndim: int) -> torch.Tensor:
        # sum per-token log-probs over the trailing (chunk, action_dim) dims
        if per_token.ndim <= batch_ndim:
            return per_token
        return per_token.sum(dim=tuple(range(batch_ndim, per_token.ndim)))

    def _require(self, tensordict: TensorDictBase, key: NestedKey) -> torch.Tensor:
        value = tensordict.get(key, default=None)
        if value is None:
            raise KeyError(f"Required key {key!r} not found in tensordict.")
        return value

    @dispatch
    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        tensordict = tensordict.copy()
        action_tokens = self._require(tensordict, self.tensor_keys.action_tokens)
        advantage = self._require(tensordict, self.tensor_keys.advantage)
        sample_log_prob = self._require(tensordict, self.tensor_keys.sample_log_prob)
        if sample_log_prob.requires_grad:
            raise RuntimeError(
                f"{self.tensor_keys.sample_log_prob!r} requires grad; detach the "
                "behavior-policy log-probs before computing the loss."
            )

        # advantage is per-sample: drop trailing singleton dims (but keep a 1-D
        # batch) to get the batch shape, then sum the per-token log-probs over the
        # remaining (action chunk) dims.
        while advantage.ndim > 1 and advantage.shape[-1] == 1:
            advantage = advantage.squeeze(-1)
        batch_ndim = advantage.ndim

        dist = self.actor_network.get_dist(tensordict)
        cur_per_token = dist.log_prob(action_tokens)
        if sample_log_prob.shape != cur_per_token.shape:
            raise ValueError(
                f"sample_log_prob shape {tuple(sample_log_prob.shape)} must match the "
                f"per-token log-prob shape {tuple(cur_per_token.shape)} (per-token "
                "behavior log-probs are expected)."
            )
        cur_log_prob = self._sum_tokens(cur_per_token, batch_ndim)
        if cur_log_prob.shape != advantage.shape:
            raise ValueError(
                f"advantage shape {tuple(advantage.shape)} must index the leading batch "
                f"dimensions of the action tokens (reduced log-prob shape "
                f"{tuple(cur_log_prob.shape)})."
            )
        sample_log_prob = self._sum_tokens(sample_log_prob, batch_ndim)

        log_weight = cur_log_prob - sample_log_prob
        gain1 = log_weight.exp() * advantage
        log_weight_clip = log_weight.clamp(*self._clip_bounds)
        clip_fraction = (log_weight_clip != log_weight).to(log_weight.dtype).mean()
        gain2 = log_weight_clip.exp() * advantage
        gain = torch.stack([gain1, gain2], -1).min(dim=-1).values
        loss_objective = _reduce(-gain, self.reduction)

        td_out = TensorDict(
            {"loss_objective": loss_objective, "clip_fraction": clip_fraction.detach()}
        )
        if self.kl_to_ref_coeff is not None and self.kl_to_ref_coeff > 0:
            ref_per_token = self._require(tensordict, self.tensor_keys.ref_log_probs)
            ref_log_prob = self._sum_tokens(ref_per_token, batch_ndim)
            # k3 estimator of KL(current || reference): always non-negative
            log_ratio = ref_log_prob - cur_log_prob
            kl = _reduce(log_ratio.exp() - log_ratio - 1, self.reduction)
            td_out.set("loss_kl_to_ref", self.kl_to_ref_coeff * kl)
            td_out.set("kl_to_ref", kl.detach())
        return td_out
