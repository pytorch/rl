# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from tensordict import TensorDict, TensorDictBase, TensorDictParams
from tensordict.nn import TensorDictModule
from tensordict.utils import NestedKey
from torchrl.objectives.common import LossModule


def _resolve_time_dim(tensordict: TensorDictBase) -> int:
    """Locate the rollout time dimension of ``tensordict``.

    Mirrors :meth:`~torchrl.objectives.value.ValueEstimatorBase._get_time_dim`:
    the dimension named ``"time"`` if the tensordict carries dimension names,
    otherwise the last batch dimension. A ``[T]`` rollout from an unbatched
    collector therefore resolves to dimension 0 and a ``[B, T]`` batch to
    dimension 1, rather than assuming time is always dimension 1.
    """
    if tensordict._has_names():
        for i, name in enumerate(tensordict.names):
            if name == "time":
                return i
    return tensordict.ndim - 1


def _shift_time(
    tensor: torch.Tensor, time_dim: int, fill_value: float | bool = 0
) -> torch.Tensor:
    """Advance ``tensor`` by one step along ``time_dim``, backfilling the freed slot with ``fill_value``."""
    length = tensor.shape[time_dim]
    shifted = torch.full_like(tensor, fill_value)
    if length > 1:
        shifted.narrow(time_dim, 0, length - 1).copy_(
            tensor.narrow(time_dim, 1, length - 1)
        )
    return shifted


class COMALoss(LossModule):
    """Counterfactual multi-agent policy-gradient loss (COMA).

    Reference: Foerster, J. et al. *Counterfactual Multi-Agent Policy
    Gradients.* AAAI 2018. https://arxiv.org/abs/1705.08926

    COMA trains a *decentralised* actor (each agent's policy conditions only
    on its own local observation) together with a *centralised* critic. The
    critic is centralised through its inputs -- it typically conditions on
    the joint observation/action of the team (see :func:`add_joint_observation`,
    :func:`add_masked_joint_action` and :func:`add_action_without_self`) --
    and must emit one action value per possible action of the acting agent,
    under ``("agents", "action_value")``.

    The actor's learning signal is a counterfactual advantage: for agent
    ``i``, the critic's own output is marginalised over agent ``i``'s action
    (holding every other agent's action fixed) to obtain a baseline. The
    advantage, ``chosen_action_value - baseline``, credits agent ``i`` only
    for the part of the outcome its own action choice affected, addressing
    the multi-agent credit-assignment problem without factorising the joint
    reward.

    The critic is trained with an n-step TD target bootstrapped from a target
    network; see :meth:`compute_value_target`.

    Args:
        actor_network (ProbabilisticTensorDictSequential): the decentralised
            policy. Conditions on ``("agents", "observation")`` and outputs a
            distribution over ``("agents", "action")`` -- build with e.g.
            :class:`~torchrl.modules.ProbabilisticActor` wrapping a
            :class:`~torchrl.modules.MultiAgentMLP` with
            ``centralized=False``.
        qvalue_network (TensorDictModule): the centralised critic. Must
            output one action value per possible action of the acting agent
            under ``("agents", "action_value")``.

    Keyword Args:
        gamma (float, optional): discount factor. Defaults to ``0.99``.
        qvalue_loss_coef (float, optional): weight of the critic's MSE loss
            relative to the actor loss. Defaults to ``0.5``.
        entropy_coef (float, optional): weight of the entropy bonus.
            Defaults to ``0.0`` (no bonus).
        n_step (int, optional): number of Bellman backups applied by
            :meth:`compute_value_target`. ``1`` is the TD(0) target; higher
            values match EPyMARL's ``q_nstep``. Defaults to ``1``.
        normalize_advantage (bool, optional): if ``True``, standardises the
            counterfactual advantage (zero mean, unit variance) across the
            batch before scaling the actor loss, MAPPO-style. Defaults to
            ``False``.
        reduction (str, optional): the reduction to apply to the elementwise
            actor / critic / entropy losses. Can be one of ``"mean"``,
            ``"sum"`` or ``"none"``. Padded or otherwise invalid positions
            (``("collector", "mask")``, or the ``"shifted_valid"`` mask
            written by :meth:`compute_value_target`) are excluded from the
            reduction automatically. Defaults to ``"mean"``.

    Examples:
        >>> import torch
        >>> from tensordict import TensorDict
        >>> from tensordict.nn import TensorDictModule
        >>> from torch import nn
        >>> from torchrl.modules import OneHotCategorical, ProbabilisticActor
        >>> from torchrl.objectives.multiagent import COMALoss
        >>> from torchrl.objectives.multiagent.coma import add_action_without_self
        >>> n_agents, obs_dim, n_actions = 3, 4, 5
        >>> actor_net = TensorDictModule(
        ...     nn.Linear(obs_dim, n_actions),
        ...     in_keys=[("agents", "observation")],
        ...     out_keys=[("agents", "logits")],
        ... )
        >>> actor = ProbabilisticActor(
        ...     module=actor_net,
        ...     in_keys=[("agents", "logits")],
        ...     out_keys=[("agents", "action")],
        ...     distribution_class=OneHotCategorical,
        ...     return_log_prob=True,
        ... )
        >>> class Critic(nn.Module):
        ...     def __init__(self):
        ...         super().__init__()
        ...         self.net = nn.Linear(obs_dim + (n_agents - 1) * n_actions, n_actions)
        ...     def forward(self, observation, action_without_self):
        ...         return self.net(torch.cat([observation, action_without_self], dim=-1))
        >>> qvalue_net = TensorDictModule(
        ...     Critic(),
        ...     in_keys=[("agents", "observation"), ("agents", "action_without_self")],
        ...     out_keys=[("agents", "action_value")],
        ... )
        >>> loss = COMALoss(actor, qvalue_net)
        >>> batch, time = 2, 4
        >>> tensordict = TensorDict(
        ...     {
        ...         "agents": TensorDict(
        ...             {"observation": torch.zeros(batch, time, n_agents, obs_dim)},
        ...             [batch, time, n_agents],
        ...         ),
        ...         "next": TensorDict(
        ...             {
        ...                 "agents": TensorDict(
        ...                     {
        ...                         "reward": torch.zeros(batch, time, n_agents, 1),
        ...                         "done": torch.zeros(batch, time, n_agents, 1, dtype=torch.bool),
        ...                         "terminated": torch.zeros(batch, time, n_agents, 1, dtype=torch.bool),
        ...                     },
        ...                     [batch, time, n_agents],
        ...                 ),
        ...             },
        ...             [batch, time],
        ...         ),
        ...     },
        ...     [batch, time],
        ... )
        >>> with torch.no_grad():
        ...     _ = actor(tensordict)
        >>> _ = add_action_without_self(tensordict)
        >>> _ = loss.compute_value_target(tensordict)
        >>> loss(tensordict)
        TensorDict(
            fields={
                advantage: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False),
                entropy: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False),
                loss_actor: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False),
                loss_entropy: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False),
                loss_qvalue: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False),
                pred_value: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False),
                target_value: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False)},
            batch_size=torch.Size([]),
            device=None,
            is_shared=False)
    """

    @dataclass
    class _AcceptedKeys:
        action: NestedKey = ("agents", "action")
        action_value: NestedKey = ("agents", "action_value")
        chosen_action_value: NestedKey = ("agents", "chosen_action_value")
        logits: NestedKey = ("agents", "logits")
        reward: NestedKey = ("agents", "reward")
        done: NestedKey = ("agents", "done")
        terminated: NestedKey = ("agents", "terminated")
        value_target: NestedKey = "value_target"

    tensor_keys: _AcceptedKeys
    default_keys = _AcceptedKeys
    out_keys = [
        "loss_actor",
        "loss_qvalue",
        "loss_entropy",
        "entropy",
        "pred_value",
        "target_value",
        "advantage",
    ]

    actor_network: TensorDictModule
    actor_network_params: TensorDictParams
    target_actor_network_params: TensorDictParams
    qvalue_network: TensorDictModule
    qvalue_network_params: TensorDictParams
    target_qvalue_network_params: TensorDictParams

    def __init__(
        self,
        actor_network: TensorDictModule,
        qvalue_network: TensorDictModule,
        *,
        gamma: float = 0.99,
        qvalue_loss_coef: float = 0.5,
        entropy_coef: float = 0.0,
        n_step: int = 1,
        normalize_advantage: bool = False,
        reduction: str | None = None,
    ) -> None:
        super().__init__()
        if n_step < 1:
            raise ValueError(f"n_step must be >= 1, got {n_step}.")
        if reduction is None:
            reduction = "mean"
        self.convert_to_functional(actor_network, "actor_network")
        self.convert_to_functional(qvalue_network, "qvalue_network", create_target_params=True)
        self.gamma = gamma
        self.qvalue_loss_coef = qvalue_loss_coef
        self.entropy_coef = entropy_coef
        self.n_step = n_step
        self.normalize_advantage = normalize_advantage
        self.reduction = reduction

    def forward(self, tensordict: TensorDictBase) -> TensorDict:
        td_copy = tensordict.clone(False)

        dist = self.actor_network.get_dist(td_copy)
        with self.qvalue_network_params.to_module(self.qvalue_network):
            self.qvalue_network(td_copy)

        chosen_action_value = self._chosen_action_value(td_copy)

        advantage = self._counterfactual_advantage(td_copy, chosen_action_value).detach()

        if self.normalize_advantage:
            # MAPPO-style per-batch standardisation: same counterfactual
            # advantage, rescaled so the actor step size is batch-invariant.
            # Restricted to the effective loss mask so padded/invalid
            # transitions don't skew the statistics; population std
            # (correction=0) avoids NaN when a single valid element remains,
            # which Bessel's correction (the default) divides by zero.
            adv_mean, adv_std = self._advantage_norm_stats(tensordict, advantage)
            advantage = (advantage - adv_mean) / adv_std.clamp_min(1e-6)

        log_prob = dist.log_prob(td_copy.get(self.tensor_keys.action))

        actor_loss = -log_prob * advantage.squeeze(-1)
        loss_actor = self._reduce_loss(actor_loss, tensordict=tensordict)

        target_value = td_copy.get(self.tensor_keys.value_target)
        qvalue_loss = (
            F.mse_loss(chosen_action_value, target_value, reduction="none")
            * self.qvalue_loss_coef
        )
        loss_qvalue = self._reduce_loss(qvalue_loss, tensordict=tensordict)

        entropy = dist.entropy()
        loss_entropy = self._reduce_loss(
            -self.entropy_coef * entropy, tensordict=tensordict
        )

        return TensorDict(
            {
                "loss_actor": loss_actor,
                "loss_qvalue": loss_qvalue,
                "loss_entropy": loss_entropy,
                "entropy": self._reduce_loss(entropy.detach(), tensordict=tensordict),
                "pred_value": self._reduce_loss(
                    chosen_action_value.detach().squeeze(-1), tensordict=tensordict
                ),
                "target_value": self._reduce_loss(
                    target_value.detach().squeeze(-1), tensordict=tensordict
                ),
                "advantage": self._reduce_loss(
                    advantage.detach().squeeze(-1), tensordict=tensordict
                ),
            },
            batch_size=[],
        )

    def compute_value_target(
        self,
        tensordict: TensorDictBase,
        params: TensorDictParams | None = None,
    ) -> TensorDictBase:
        """Write the n-step Q-value target before flattening rollout data.

        Targets follow the recursion ``G_k(t) = r(t) + gamma * (1 - done(t))
        * G_{k-1}(t+1)`` with ``G_0`` the target-network chosen-action
        Q-values, applied ``n_step`` times along the rollout's time
        dimension. The time dimension is resolved via
        :func:`~torchrl.objectives.multiagent.coma._resolve_time_dim` (the
        dimension named ``"time"`` if any, otherwise the last batch
        dimension), so a ``[T]`` rollout from an unbatched collector, a ``[B,
        T]`` batch, and inputs with additional leading batch dimensions all
        shift the correct axis. ``n_step=1`` is the TD(0) target;
        ``n_step=10`` matches EPyMARL's ``q_nstep: 10`` within episodes.

        The shift itself is gated on ``done``, not ``terminated``: whenever a
        rollout ends -- whether by real termination or by truncation --
        row ``t + 1`` in the same window is a fresh, unrelated episode (an
        auto-reset collector writes its first step right after), so reusing
        its chosen-action Q-value as the bootstrap for row ``t`` would silently
        mix the two trajectories. ``terminated`` only decides *why* the
        bootstrap is missing there: for a true terminal state it is
        legitimately zero (an absorbing state has no future value), so the
        row is valid as-is; for a truncation (``done`` but not ``terminated``)
        the real bootstrap -- the target Q at the actual next transition --
        is simply not available in this window, and rather than fabricate one,
        this writes a ``"shifted_valid"`` mask (picked up automatically by
        :meth:`~torchrl.objectives.common.LossModule._reduce_loss`, see
        :data:`~torchrl.objectives.common.AUTO_LOSS_MASK_KEYS`) that excludes
        it -- along with the tail transitions at a non-terminated rollout end,
        up to ``n_step`` of them -- from the loss instead of silently biasing
        the target.
        """
        if params is None:
            params = self.target_qvalue_network_params

        td_copy = tensordict.clone(False)
        with params.to_module(self.qvalue_network):
            self.qvalue_network(td_copy)

        chosen_action_value = self._chosen_action_value(td_copy)
        if chosen_action_value.ndim < 2:
            raise ValueError("COMALoss.compute_value_target expects an environment/time rollout batch.")

        time_dim = _resolve_time_dim(tensordict)

        reward = tensordict.get(("next", self.tensor_keys.reward))
        done = tensordict.get(("next", self.tensor_keys.done)).to(torch.bool)
        terminated = tensordict.get(
            ("next", self.tensor_keys.terminated), default=done
        ).to(torch.bool)
        not_done = (~done).to(chosen_action_value.dtype)

        value_target = chosen_action_value
        valid = torch.ones_like(done, dtype=torch.bool)
        for _ in range(self.n_step):
            next_value = _shift_time(value_target, time_dim, fill_value=0.0)
            next_valid = _shift_time(valid, time_dim, fill_value=False)
            value_target = reward + self.gamma * not_done * next_value
            # terminated: always valid, the zeroed bootstrap above is correct.
            # truncated: never valid, no real bootstrap is available.
            # otherwise: valid iff the shifted-in value itself was.
            valid = terminated | (~done & next_valid)

        tensordict.set(self.tensor_keys.value_target, value_target.detach())
        tensordict.set("shifted_valid", valid)
        return tensordict

    def diagnostics(self, tensordict: TensorDictBase) -> dict[str, torch.Tensor]:
        """Return unreduced COMA quantities for trainer-side observability.

        This deliberately leaves aggregation to a reusable trainer hook: callers
        can log global summaries, per-agent values, or custom histograms without
        changing the loss used for optimization.
        """
        td_copy = tensordict.clone(False)
        with self.qvalue_network_params.to_module(self.qvalue_network):
            self.qvalue_network(td_copy)
        chosen_action_value = self._chosen_action_value(td_copy)
        advantage = self._counterfactual_advantage(td_copy, chosen_action_value)
        target_value = td_copy.get(self.tensor_keys.value_target)

        # Q-contrast measures (flat-critic hypothesis): how differentiated are
        # the Q outputs across own actions, how sensitive are they to the other
        # agents' actions, and how contrasted are the empirical targets by
        # chosen action (the reference the critic should match).
        action_value = td_copy.get(self.tensor_keys.action_value)
        action_value_spread = action_value.std(dim=-1, correction=0)

        permuted = tensordict.clone(False)
        for key in (("agents", "masked_joint_action"), ("agents", "action_without_self")):
            if key in permuted.keys(True):
                values = permuted.get(key)
                index = torch.randperm(values.shape[0], device=values.device)
                permuted.set(key, values[index])
        with self.qvalue_network_params.to_module(self.qvalue_network):
            self.qvalue_network(permuted)
        others_sensitivity = (permuted.get(self.tensor_keys.action_value) - action_value).abs().mean(dim=-1)

        flat_action = tensordict.get(self.tensor_keys.action).to(torch.float).reshape(-1, action_value.shape[-1])
        flat_target = target_value.reshape(-1, 1)
        counts = flat_action.sum(dim=0)
        means = (flat_action * flat_target).sum(dim=0) / counts.clamp_min(1.0)
        taken = counts > 0
        if int(taken.sum()) > 1:
            target_contrast = means[taken].std(correction=0)
        else:
            target_contrast = torch.zeros((), device=action_value.device)

        return {
            "advantage": advantage.detach(),
            "td_error": (target_value - chosen_action_value).detach(),
            "chosen_action_value": chosen_action_value.detach(),
            "target_value": target_value.detach(),
            "action_value_spread": action_value_spread.detach(),
            "others_sensitivity": others_sensitivity.detach(),
            "target_contrast": target_contrast.detach(),
        }

    def _chosen_action_value(self, tensordict: TensorDictBase) -> torch.Tensor:
        action = tensordict.get(self.tensor_keys.action).to(torch.float)
        action_value = tensordict.get(self.tensor_keys.action_value)
        chosen_action_value = (action * action_value).sum(dim=-1, keepdim=True)
        tensordict.set(self.tensor_keys.chosen_action_value, chosen_action_value)
        return chosen_action_value

    def _counterfactual_advantage(
        self,
        tensordict: TensorDictBase,
        chosen_action_value: torch.Tensor,
    ) -> torch.Tensor:
        action_prob = tensordict.get(self.tensor_keys.logits).softmax(dim=-1)
        action_value = tensordict.get(self.tensor_keys.action_value)
        counterfactual_baseline = (action_prob * action_value).sum(dim=-1, keepdim=True)
        return chosen_action_value - counterfactual_baseline

    def _advantage_norm_stats(
        self, tensordict: TensorDictBase, advantage: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Mean/std of ``advantage``, restricted to the effective loss mask.

        Padded or otherwise invalid positions (``("collector", "mask")``,
        ``"shifted_valid")``) must not skew the normalisation statistics.
        ``correction=0`` (population std) avoids ``NaN`` when a single valid
        element remains, which the default Bessel's correction divides by
        zero.
        """
        mask = None
        for mask_key in self._loss_mask_keys():
            tensordict_mask = tensordict.get(mask_key, default=None)
            if tensordict_mask is not None:
                tensordict_mask = self._expand_loss_mask(tensordict_mask, advantage)
                mask = tensordict_mask if mask is None else mask & tensordict_mask
        valid_advantage = advantage[mask] if mask is not None else advantage
        return valid_advantage.mean(), valid_advantage.std(correction=0)


def add_action_without_self(tensordict: TensorDictBase) -> TensorDictBase:
    """Write each agent's other-agent actions under ``action_without_self``."""
    if ("agents", "action") not in tensordict.keys(True):
        return tensordict

    actions = tensordict.get(("agents", "action"))
    n_agents = actions.shape[-2]
    if n_agents == 1:
        tensordict.set(("agents", "action_without_self"), actions.new_empty(*actions.shape[:-2], 1, 0))
        return tensordict

    other_actions = torch.stack(
        [torch.cat([actions[..., :i, :], actions[..., i + 1 :, :]], dim=-2) for i in range(n_agents)],
        dim=-3,
    )
    tensordict.set(("agents", "action_without_self"), other_actions.reshape(*actions.shape[:-2], n_agents, -1))
    return tensordict


def add_joint_observation(tensordict: TensorDictBase) -> TensorDictBase:
    """Write the concatenated team observation under ``joint_observation``.

    Every agent row receives the same concatenation of all agents'
    observations. This mirrors EPyMARL's Gym wrapper, where the global state
    fed to the COMA critic is the concatenation of individual observations.
    """
    observation = tensordict.get(("agents", "observation"))
    n_agents = observation.shape[-2]
    joint = observation.reshape(*observation.shape[:-2], 1, -1).expand(
        *observation.shape[:-2], n_agents, n_agents * observation.shape[-1]
    )
    tensordict.set(("agents", "joint_observation"), joint.clone())
    return tensordict


def add_masked_joint_action(tensordict: TensorDictBase) -> TensorDictBase:
    """Write the joint one-hot action with each agent's own slot zeroed.

    This mirrors EPyMARL's ``COMACritic._build_inputs``: agent ``i`` sees the
    full joint action vector of size ``n_agents * n_actions`` with the block
    corresponding to its own action masked to zero, so its Q-values are not
    conditioned on the action being marginalised by the counterfactual
    baseline.
    """
    actions = tensordict.get(("agents", "action")).to(torch.float)
    n_agents, n_actions = actions.shape[-2], actions.shape[-1]
    joint = actions.reshape(*actions.shape[:-2], 1, -1).expand(*actions.shape[:-2], n_agents, n_agents * n_actions)
    own_block = torch.eye(n_agents, device=actions.device, dtype=actions.dtype).repeat_interleave(n_actions, dim=1)
    tensordict.set(("agents", "masked_joint_action"), joint * (1.0 - own_block))
    return tensordict