# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch
from tensordict import TensorDict, TensorDictBase, TensorDictParams
from tensordict.nn import dispatch, TensorDictModule, TensorDictModuleBase
from tensordict.utils import NestedKey
from torch.nn import functional as F

from torchrl.objectives.common import LossModule
from torchrl.objectives.utils import _vmap_func

__all__ = ["TransitiveQLoss"]


class TransitiveQLoss(LossModule):
    r"""Transitive RL loss for offline goal-conditioned RL.

    Transitive RL learns a goal-conditioned critic
    :math:`Q(s, a, g) \in (0, 1]`, read as :math:`\gamma^{d(s, g)}` for the
    temporal distance :math:`d`, with a divide-and-conquer update that
    replaces the one-step temporal-difference backup with the triangle
    inequality:

    .. math::

        Q(s_i, a_i, s_j) \leftarrow
        \bar{Q}(s_i, a_i, s_k)\,\bar{Q}(s_k, a_k, s_j),
        \qquad i \le k < j,

    where :math:`s_k` is a state the dataset trajectory visited between
    :math:`s_i` and :math:`s_j` and :math:`\bar{Q}` is a target critic. A
    factor whose horizon is at most one step is replaced by its exact value
    :math:`\gamma^{k - i}` (resp. :math:`\gamma^{j - k}`). The critic is fit
    with an expectile-weighted binary cross-entropy, optionally re-weighted by
    :math:`(1 + \log_\gamma \text{target})^{-\lambda}` to favour short chunks.

    Reference: Park, Oberai, Atreya, Levine. "Transitive RL: Value Learning
    via Divide and Conquer." ICLR 2026. https://arxiv.org/abs/2510.22512

    See also :class:`~torchrl.trainers.algorithms.configs.objectives.TransitiveQLossConfig`.

    The loss reads batches of shape ``[*B, 4]`` whose last dimension holds,
    in this order, the anchor record :math:`s_i`, the subgoal record
    :math:`s_k`, the goal record :math:`s_j` and a policy-goal record. The
    flat ``[4 * B]`` batches sampled with
    :class:`~torchrl.data.replay_buffers.SubgoalTriplet` are accepted as is.
    Each record carries its ``"step_offset"`` from the anchor.

    The goal fed to the critic is read from the goal record at the
    ``achieved_goal`` key (the full observation by default), so that the
    triangle inequality holds in the critic's state space. When the policy is
    conditioned on another goal representation (``policy_goal`` key), an
    ``oracle_qvalue_network`` reading that representation is distilled from
    the critic and used for policy extraction.

    The policy is extracted with reparameterized gradients: the actor
    maximizes the (ensemble minimum of the) critic logits, normalized by their
    mean absolute value, plus ``alpha`` times the behavior log-likelihood of
    the dataset action.

    Args:
        actor_network (TensorDictModule): a probabilistic actor reading the
            ``observation`` and ``goal`` keys and supporting ``get_dist``.
        qvalue_network (TensorDictModule or list of TensorDictModule): the
            critic, reading the ``observation``, ``goal`` and ``action`` keys
            and writing the ``value`` key as a logit. A list provides the
            ensemble members (with their own initialization); a single module
            is expanded ``num_qvalue_nets`` times.

    Keyword Args:
        oracle_qvalue_network (TensorDictModule or list of TensorDictModule, optional):
            a critic with the same keys as ``qvalue_network`` whose ``goal``
            entry carries the ``policy_goal`` representation. It is trained to
            match the critic and replaces it for policy extraction. Defaults to
            ``None`` (the policy is extracted from the critic, and
            ``policy_goal`` must match ``achieved_goal``).
        num_qvalue_nets (int, optional): ensemble size used when a single
            critic module is given. Defaults to ``2``.
        discount (float, optional): the discount :math:`\gamma` defining the
            value scale. Defaults to ``0.99``.
        expectile (float, optional): expectile :math:`\kappa` of the critic
            regression; ``0.5`` gives a behavioral critic. Defaults to ``0.7``.
        distance_weight (float, optional): exponent :math:`\lambda` of the
            distance-based re-weighting; ``0`` disables it. Defaults to ``0``.
        alpha (float, optional): weight of the behavior log-likelihood in the
            actor loss. Defaults to ``1.0``.
        actor_action (Literal["mode", "rsample"], optional): the policy action
            scored by the critic: the distribution mode or a reparameterized
            sample. Actions are clamped to ``[-1, 1]``. Defaults to ``"mode"``.

    Examples:
        >>> import torch
        >>> from torch import nn
        >>> from tensordict import TensorDict
        >>> from tensordict.nn import TensorDictModule
        >>> from torchrl.modules import IndependentNormal, ProbabilisticActor
        >>> from torchrl.objectives import TransitiveQLoss
        >>> class Critic(nn.Module):
        ...     def __init__(self):
        ...         super().__init__()
        ...         self.net = nn.Linear(3 + 3 + 2, 1)
        ...     def forward(self, obs, goal, action):
        ...         return self.net(torch.cat([obs, goal, action], -1))
        >>> class Actor(nn.Module):
        ...     def __init__(self):
        ...         super().__init__()
        ...         self.net = nn.Linear(3 + 3, 2)
        ...     def forward(self, obs, goal):
        ...         loc = self.net(torch.cat([obs, goal], -1))
        ...         return loc, torch.ones_like(loc)
        >>> actor = ProbabilisticActor(
        ...     TensorDictModule(Actor(), ["observation", "goal"], ["loc", "scale"]),
        ...     in_keys=["loc", "scale"],
        ...     distribution_class=IndependentNormal,
        ... )
        >>> critic = TensorDictModule(
        ...     Critic(), ["observation", "goal", "action"], ["state_action_value"]
        ... )
        >>> loss = TransitiveQLoss(actor, critic, discount=0.9)
        >>> batch = TensorDict(
        ...     {
        ...         "observation": torch.randn(8, 4, 3),
        ...         "action": torch.rand(8, 4, 2) * 2 - 1,
        ...         "step_offset": torch.tensor([0, 1, 5, 3]).expand(8, 4),
        ...     },
        ...     batch_size=[8, 4],
        ... )
        >>> out = loss(batch)
        >>> sorted(k for k in out.keys() if k.startswith("loss"))
        ['loss_actor', 'loss_qvalue']
    """

    @dataclass
    class _AcceptedKeys:
        """TensorDict keys configurable through :meth:`set_keys`.

        Attributes:
            observation (NestedKey): the state read by every network.
                Defaults to ``"observation"``.
            action (NestedKey): the dataset action. Defaults to ``"action"``.
            goal (NestedKey): the goal input of the critic and actor networks.
                Defaults to ``"goal"``.
            achieved_goal (NestedKey): the entry of the goal and subgoal
                records used as the critic goal. Defaults to ``"observation"``.
            policy_goal (NestedKey): the entry of the goal records used as the
                actor and oracle-critic goal. Defaults to ``"observation"``.
            value (NestedKey): the critic output logit. Defaults to
                ``"state_action_value"``.
            step_offset (NestedKey): per-record distance to the anchor.
                Defaults to ``"step_offset"``.
            validity_mask (NestedKey): per-record validity flag; missing
                entries count as valid. Defaults to ``"validity_mask"``.
        """

        observation: NestedKey = "observation"
        action: NestedKey = "action"
        goal: NestedKey = "goal"
        achieved_goal: NestedKey = "observation"
        policy_goal: NestedKey = "observation"
        value: NestedKey = "state_action_value"
        step_offset: NestedKey = "step_offset"
        validity_mask: NestedKey = "validity_mask"

    default_keys = _AcceptedKeys
    actor_network: TensorDictModuleBase
    qvalue_network: TensorDictModule
    oracle_qvalue_network: TensorDictModule
    actor_network_params: TensorDictParams
    qvalue_network_params: TensorDictParams
    oracle_qvalue_network_params: TensorDictParams
    target_actor_network_params: TensorDictParams
    target_qvalue_network_params: TensorDictParams
    target_oracle_qvalue_network_params: TensorDictParams

    def __init__(
        self,
        actor_network: TensorDictModuleBase,
        qvalue_network: TensorDictModule | list[TensorDictModule],
        *,
        oracle_qvalue_network: TensorDictModule | list[TensorDictModule] | None = None,
        num_qvalue_nets: int = 2,
        discount: float = 0.99,
        expectile: float = 0.7,
        distance_weight: float = 0.0,
        alpha: float = 1.0,
        actor_action: Literal["mode", "rsample"] = "mode",
    ):
        super().__init__()
        if not 0.0 < discount < 1.0:
            raise ValueError(f"discount must be in (0, 1), got {discount}.")
        if not 0.0 < expectile < 1.0:
            raise ValueError(f"expectile must be in (0, 1), got {expectile}.")
        if actor_action not in ("mode", "rsample"):
            raise ValueError(f"Unknown actor_action {actor_action}.")
        if isinstance(qvalue_network, (list, tuple)):
            num_qvalue_nets = len(qvalue_network)
        self.convert_to_functional(actor_network, "actor_network")
        self.convert_to_functional(
            qvalue_network,
            "qvalue_network",
            num_qvalue_nets,
            create_target_params=True,
        )
        self.use_oracle = oracle_qvalue_network is not None
        if self.use_oracle:
            num_oracle = (
                len(oracle_qvalue_network)
                if isinstance(oracle_qvalue_network, (list, tuple))
                else num_qvalue_nets
            )
            self.convert_to_functional(
                oracle_qvalue_network, "oracle_qvalue_network", num_oracle
            )
        self.discount = discount
        self.log_discount = math.log(discount)
        self.expectile = expectile
        self.distance_weight = distance_weight
        self.alpha = alpha
        self.actor_action = actor_action
        self.qvalue_vmap = _vmap_func(
            self.qvalue_network, in_dims=(None, 0), randomness="different"
        )
        if self.use_oracle:
            self.oracle_qvalue_vmap = _vmap_func(
                self.oracle_qvalue_network, in_dims=(None, 0), randomness="different"
            )

    def _forward_value_estimator_keys(self, **kwargs) -> None:
        pass

    @property
    def in_keys(self):
        keys = self.tensor_keys
        return list(
            dict.fromkeys(
                [
                    keys.observation,
                    keys.action,
                    keys.achieved_goal,
                    keys.policy_goal,
                    keys.step_offset,
                ]
            )
        )

    def _critic_input(
        self, state: TensorDictBase, goal: torch.Tensor, action: torch.Tensor
    ) -> TensorDict:
        keys = self.tensor_keys
        return TensorDict(
            {
                keys.observation: state.get(keys.observation),
                keys.goal: goal,
                keys.action: action,
            },
            batch_size=state.batch_size,
        )

    def _logits(self, vmap_func, td: TensorDictBase, params) -> torch.Tensor:
        return vmap_func(td, params).get(self.tensor_keys.value).squeeze(-1)

    def _bce(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.binary_cross_entropy_with_logits(logits, target, reduction="none")

    def _masked_mean(self, value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask = mask.to(value.dtype).expand_as(value)
        return (value * mask).sum() / mask.sum().clamp_min(1.0)

    def _validity_mask(self, state: TensorDictBase) -> torch.Tensor:
        mask = state.get(self.tensor_keys.validity_mask, None)
        if mask is None:
            return torch.ones(state.batch_size, dtype=torch.bool, device=state.device)
        return mask

    def qvalue_loss(
        self, tensordict: TensorDictBase
    ) -> tuple[torch.Tensor, torch.Tensor | None, dict]:
        """Returns the critic loss, the oracle distillation loss and metadata."""
        keys = self.tensor_keys
        state, subgoal, goal = (tensordict[..., r] for r in range(3))
        mask = self._validity_mask(state)
        action = state.get(keys.action)
        goal_obs = goal.get(keys.achieved_goal)
        logits = self._logits(
            self.qvalue_vmap,
            self._critic_input(state, goal_obs, action),
            self.qvalue_network_params,
        )
        with torch.no_grad():
            first_offset = subgoal.get(keys.step_offset).to(logits.dtype)
            second_offset = goal.get(keys.step_offset).to(logits.dtype) - first_offset
            pairs = torch.stack(
                [
                    self._critic_input(state, subgoal.get(keys.achieved_goal), action),
                    self._critic_input(subgoal, goal_obs, subgoal.get(keys.action)),
                ],
                0,
            )
            target_q = torch.sigmoid(
                self._logits(self.qvalue_vmap, pairs, self.target_qvalue_network_params)
            )
            first_q = torch.where(
                first_offset <= 1, self.discount**first_offset, target_q[:, 0]
            )
            second_q = torch.where(
                second_offset <= 1, self.discount**second_offset, target_q[:, 1]
            )
            target = first_q * second_q
            q = torch.sigmoid(logits)
            weight = torch.where(target >= q, self.expectile, 1 - self.expectile)
            if self.distance_weight:
                distance = target.log() / self.log_discount
                weight = weight * (1 + distance).pow(-self.distance_weight)
        loss_qvalue = self._masked_mean(weight * self._bce(logits, target), mask)
        loss_oracle = None
        if self.use_oracle:
            oracle_logits = self._logits(
                self.oracle_qvalue_vmap,
                self._critic_input(state, goal.get(keys.policy_goal), action),
                self.oracle_qvalue_network_params,
            )
            loss_oracle = self._masked_mean(self._bce(oracle_logits, q.detach()), mask)
        metadata = {
            "q_mean": q.mean().detach(),
            "q_min": q.min().detach(),
            "q_max": q.max().detach(),
            "target_mean": target.mean(),
        }
        return loss_qvalue, loss_oracle, metadata

    def actor_loss(self, tensordict: TensorDictBase) -> tuple[torch.Tensor, dict]:
        """Returns the reparameterized-gradient actor loss and metadata."""
        keys = self.tensor_keys
        state, policy_goal = tensordict[..., 0], tensordict[..., 3]
        goal = policy_goal.get(keys.policy_goal)
        actor_td = TensorDict(
            {keys.observation: state.get(keys.observation), keys.goal: goal},
            batch_size=state.batch_size,
        )
        with self.actor_network_params.to_module(self.actor_network):
            dist = self.actor_network.get_dist(actor_td)
        action = dist.mode if self.actor_action == "mode" else dist.rsample()
        action = action.clamp(-1, 1)
        critic_td = self._critic_input(state, goal, action)
        if self.use_oracle:
            logits = self._logits(
                self.oracle_qvalue_vmap,
                critic_td,
                self.oracle_qvalue_network_params.detach(),
            )
        else:
            logits = self._logits(
                self.qvalue_vmap, critic_td, self.qvalue_network_params.detach()
            )
        mask = self._validity_mask(state)
        q = logits.min(0).values
        q_loss = -self._masked_mean(q, mask) / (
            self._masked_mean(q.abs(), mask).detach() + 1e-6
        )
        log_prob = dist.log_prob(state.get(keys.action))
        bc_loss = -self.alpha * self._masked_mean(log_prob, mask)
        squared_error = (dist.mode - state.get(keys.action)).pow(2).mean(-1)
        return q_loss + bc_loss, {
            "actor_q_loss": q_loss.detach(),
            "bc_loss": bc_loss.detach(),
            "bc_mse": self._masked_mean(squared_error, mask).detach(),
        }

    @dispatch
    def forward(self, tensordict: TensorDictBase) -> TensorDict:
        if tensordict.ndim == 1 and tensordict.shape[0] % 4 == 0:
            tensordict = tensordict.unflatten(0, (tensordict.shape[0] // 4, 4))
        if tensordict.ndim < 2 or tensordict.shape[-1] != 4:
            raise ValueError(
                "TransitiveQLoss expects the flat records of SubgoalTriplet or a "
                "tensordict whose last dimension holds the (state, subgoal, goal, "
                f"policy goal) records, got batch size {tensordict.shape}."
            )
        loss_qvalue, loss_oracle, metadata = self.qvalue_loss(tensordict)
        loss_actor, actor_metadata = self.actor_loss(tensordict)
        out = {"loss_qvalue": loss_qvalue, "loss_actor": loss_actor}
        if loss_oracle is not None:
            out["loss_oracle_qvalue"] = loss_oracle
        result = TensorDict({**out, **metadata, **actor_metadata})
        self._clear_weakrefs(
            tensordict,
            result,
            "actor_network_params",
            "qvalue_network_params",
            "target_qvalue_network_params",
        )
        return result
