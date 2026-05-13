# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Multi-agent PPO objectives.

Implements :class:`MAPPOLoss` (centralised critic) and :class:`IPPOLoss`
(independent / decentralised critic).

References:
    - Yu, C. et al. *The Surprising Effectiveness of PPO in Cooperative
      Multi-Agent Games.* NeurIPS 2022. https://arxiv.org/abs/2103.01955
    - de Witt, C. S. et al. *Is Independent Learning All You Need in the
      StarCraft Multi-Agent Challenge?* 2020. https://arxiv.org/abs/2011.09533
"""
from __future__ import annotations

import contextlib
from typing import Any

from tensordict import TensorDictBase
from tensordict.nn import TensorDictModule
from tensordict.nn.probabilistic import ProbabilisticTensorDictSequential

from torchrl.modules.value_norm import ValueNorm
from torchrl.objectives.ppo import ClipPPOLoss
from torchrl.objectives.utils import (
    _GAMMA_LMBDA_DEPREC_ERROR,
    default_value_kwargs,
    distance_loss,
    ValueEstimators,
)
from torchrl.objectives.value import (
    GAE,
    MultiAgentGAE,
    TD0Estimator,
    TD1Estimator,
    TDLambdaEstimator,
    VTrace,
)


class _MultiAgentPPOMixin:
    """Shared plumbing for :class:`MAPPOLoss` and :class:`IPPOLoss`.

    Two pieces:

    1. Default the value estimator to :class:`MultiAgentGAE` so per-agent value
       outputs broadcast cleanly against team-shared reward / done signals.
    2. Wrap the parent's :meth:`loss_critic` so that, when a
       :class:`~torchrl.modules.ValueNorm` is attached, the running value
       target stats are updated and both target and prediction are
       normalised before the MSE. This stabilises critic-loss magnitude
       when reward scales drift during training (Yu et al. 2022, Table 13).
    """

    default_value_estimator = ValueEstimators.MAGAE

    def make_value_estimator(self, value_type: ValueEstimators = None, **hyperparams):
        if value_type is None:
            value_type = self.default_value_estimator
        if value_type != ValueEstimators.MAGAE:
            # Fall through to the parent for non-multi-agent estimators (the
            # user is opting out of per-agent broadcasting).
            return super().make_value_estimator(value_type, **hyperparams)

        self.value_type = value_type
        hp = dict(default_value_kwargs(value_type))
        if hasattr(self, "gamma"):
            hp["gamma"] = self.gamma
        hp.update(hyperparams)
        self._value_estimator = MultiAgentGAE(value_network=self.critic_network, **hp)
        tensor_keys = {
            "advantage": self.tensor_keys.advantage,
            "value": self.tensor_keys.value,
            "value_target": self.tensor_keys.value_target,
            "reward": self.tensor_keys.reward,
            "done": self.tensor_keys.done,
            "terminated": self.tensor_keys.terminated,
            "sample_log_prob": self.tensor_keys.sample_log_prob,
        }
        self._value_estimator.set_keys(**tensor_keys)

    def loss_critic(self, tensordict: TensorDictBase):
        # Delegate to ClipPPOLoss; if no value_norm is attached this is a
        # no-op wrapper. Otherwise we normalise target and prediction so the
        # MSE lives on a fixed scale across training.
        value_norm: ValueNorm | None = getattr(self, "value_norm", None)
        if value_norm is None:
            return super().loss_critic(tensordict)

        target_return = tensordict.get(self.tensor_keys.value_target, None)
        if target_return is None:
            raise KeyError(
                f"the key {self.tensor_keys.value_target} was not found in the "
                "input tensordict. Make sure the value estimator ran before "
                "computing the loss."
            )

        # Forward the critic ourselves so we can normalise its output.
        with self.critic_network_params.to_module(
            self.critic_network
        ) if self.functional else contextlib.nullcontext():
            state_value_td = self.critic_network(tensordict)
        state_value = state_value_td.get(self.tensor_keys.value)
        if state_value is None:
            raise KeyError(
                f"the key {self.tensor_keys.value} was not found in the critic "
                "output tensordict."
            )

        value_norm.update(target_return)
        normalised_target = value_norm.normalize(target_return.detach())
        normalised_pred = value_norm.normalize(state_value)
        loss_value = distance_loss(
            normalised_target, normalised_pred, loss_function=self.loss_critic_type
        )

        self._clear_weakrefs(
            tensordict,
            "actor_network_params",
            "critic_network_params",
            "target_actor_network_params",
            "target_critic_network_params",
        )
        if self._has_critic:
            return self.critic_coef * loss_value, None, None
        return loss_value, None, None


class MAPPOLoss(_MultiAgentPPOMixin, ClipPPOLoss):
    """Multi-Agent PPO loss with a centralised critic (Yu et al. 2022).

    MAPPO trains a *decentralised actor* (each agent's policy conditions only
    on its local observation) together with a *centralised critic* (single
    value function that conditions on the full team state or concatenated
    observations). The decentralised actor lets policies run independently at
    execution time, while the centralised critic reduces variance during
    training by giving every agent the same value baseline derived from full
    state information.

    This class is a thin specialisation of :class:`ClipPPOLoss`. The
    differences:

    - The default value estimator is :class:`~torchrl.objectives.value.MultiAgentGAE`,
      which broadcasts team-shared rewards / done flags along the agent
      dimension before computing returns.
    - ``normalize_advantage_exclude_dims`` defaults to ``(-2,)`` so the agent
      dim is excluded when standardising advantages.
    - An optional :class:`~torchrl.modules.ValueNorm` can be supplied via
      ``value_norm=PopArtValueNorm(shape=1)`` to stabilise the critic loss;
      the MAPPO paper reports this is load-bearing on SMAC (their Table 13).
      :class:`~torchrl.modules.RunningValueNorm` is a no-decay alternative
      for stationary reward scales.

    Args:
        actor_network (ProbabilisticTensorDictSequential): per-agent policy
            operator. Conventionally built with
            :class:`~torchrl.modules.MultiAgentMLP` using
            ``centralized=False, share_params=True`` for cooperative
            homogeneous teams.
        critic_network (TensorDictModule): centralised value operator. Build
            this with :class:`~torchrl.modules.MultiAgentMLP` and
            ``centralized=True, share_params=True``, or with any module that
            consumes a global ``"state"`` key and returns
            ``("agents", "state_value")`` of shape ``[*B, n_agents, 1]``.

    Keyword Args:
        value_norm (ValueNorm, optional): if supplied, the critic target and
            prediction are normalised by this running normaliser before the
            MSE / smooth-L1 distance. Defaults to ``None`` (no value norm).
        clip_epsilon (float): PPO ratio clip. Defaults to ``0.2``.
        entropy_coeff (float): entropy bonus weight. Defaults to ``0.01``
            (MAPPO default).
        critic_coef (float, optional): critic loss weight. Defaults to ``1.0``.
        normalize_advantage (bool): whether to standardise the advantage.
            Defaults to ``True`` (MAPPO default; differs from base
            :class:`ClipPPOLoss` which defaults to ``False``).
        normalize_advantage_exclude_dims (tuple of int): dimensions to
            exclude from advantage standardisation. Defaults to ``(-2,)``
            (the agent dim).
        **kwargs: forwarded to :class:`ClipPPOLoss`.

    The expected tensordict layout follows the torchrl multi-agent convention
    (see :class:`~torchrl.envs.libs.vmas.VmasEnv`,
    :class:`~torchrl.envs.libs.pettingzoo.PettingZooEnv`):

    - ``("agents", "observation")``: ``[*B, T, n_agents, obs_dim]``
    - ``("agents", "action")``: ``[*B, T, n_agents, action_dim]``
    - Optional ``"state"`` at the root for centralised critics
    - Team-shared ``("next", "reward")``, ``("next", "done")``,
      ``("next", "terminated")`` of shape ``[*B, T, 1]`` (or per-agent under
      ``("next", "agents", "reward")`` for competitive settings).

    Example:
        >>> import torch
        >>> from tensordict import TensorDict
        >>> from tensordict.nn import TensorDictModule
        >>> from torchrl.modules import (
        ...     MultiAgentMLP, PopArtValueNorm, ProbabilisticActor,
        ... )
        >>> from torchrl.modules.distributions import NormalParamExtractor, TanhNormal
        >>> from torchrl.objectives.multiagent import MAPPOLoss
        >>> n_agents, obs_dim, action_dim, state_dim = 3, 6, 2, 12
        >>> # Decentralised actor
        >>> actor_net = torch.nn.Sequential(
        ...     MultiAgentMLP(
        ...         n_agent_inputs=obs_dim, n_agent_outputs=2 * action_dim,
        ...         n_agents=n_agents, centralized=False, share_params=True,
        ...     ),
        ...     NormalParamExtractor(),
        ... )
        >>> actor_module = TensorDictModule(
        ...     actor_net,
        ...     in_keys=[("agents", "observation")],
        ...     out_keys=[("agents", "loc"), ("agents", "scale")],
        ... )
        >>> actor = ProbabilisticActor(
        ...     module=actor_module,
        ...     in_keys=[("agents", "loc"), ("agents", "scale")],
        ...     out_keys=[("agents", "action")],
        ...     distribution_class=TanhNormal,
        ... )
        >>> # Centralised critic — same agent-dim layout as the actor, with
        >>> # centralized=True so each agent's effective input is the full
        >>> # team's observation concatenated.
        >>> critic = TensorDictModule(
        ...     MultiAgentMLP(
        ...         n_agent_inputs=obs_dim, n_agent_outputs=1,
        ...         n_agents=n_agents, centralized=True, share_params=True,
        ...     ),
        ...     in_keys=[("agents", "observation")],
        ...     out_keys=[("agents", "state_value")],
        ... )
        >>> loss = MAPPOLoss(actor, critic, value_norm=PopArtValueNorm(shape=1))
        >>> loss.set_keys(value=("agents", "state_value"), action=("agents", "action"))
    """

    actor_network: TensorDictModule
    critic_network: TensorDictModule

    def __init__(
        self,
        actor_network: ProbabilisticTensorDictSequential | None = None,
        critic_network: TensorDictModule | None = None,
        *,
        value_norm: ValueNorm | None = None,
        entropy_coeff: float | dict[str, float] = 0.01,
        normalize_advantage: bool = True,
        normalize_advantage_exclude_dims: tuple[int, ...] = (-2,),
        **kwargs: Any,
    ) -> None:
        super().__init__(
            actor_network,
            critic_network,
            entropy_coeff=entropy_coeff,
            normalize_advantage=normalize_advantage,
            normalize_advantage_exclude_dims=normalize_advantage_exclude_dims,
            **kwargs,
        )
        # Registered as a submodule so it moves with .to(device) and shows up
        # in state_dict(). None is a valid value — we still keep the
        # attribute for the loss_critic override to query.
        self.value_norm = value_norm
        if value_norm is not None:
            self.add_module("_value_norm_module", value_norm)


class IPPOLoss(_MultiAgentPPOMixin, ClipPPOLoss):
    """Independent PPO loss (de Witt et al. 2020).

    IPPO is the decentralised counterpart of MAPPO: each agent has its *own*
    value function that conditions only on its local observation. There is no
    centralised critic and no global state required. Surprisingly competitive
    with MAPPO on many SMAC scenarios (the de Witt et al. paper is titled
    *Is Independent Learning All You Need...*).

    Structurally this loss is identical to :class:`MAPPOLoss`; the difference
    lives entirely in the critic the user passes in. We expose it as a
    separate class so the API is self-documenting: when you import
    ``IPPOLoss`` it is unambiguous which algorithm you are running, and the
    docstring spells out the critic-construction recipe.

    Args:
        actor_network (ProbabilisticTensorDictSequential): per-agent policy.
            Build with ``MultiAgentMLP(centralized=False, share_params=True)``.
        critic_network (TensorDictModule): per-agent value operator. Build
            with ``MultiAgentMLP(centralized=False, share_params=True)`` so
            each agent values its own observation.

    Keyword Args:
        value_norm (ValueNorm, optional): rarely used with IPPO; defaults to
            ``None``.
        entropy_coeff (float): defaults to ``0.01``.
        normalize_advantage (bool): defaults to ``True``.
        normalize_advantage_exclude_dims (tuple of int): defaults to ``(-2,)``.
        **kwargs: forwarded to :class:`ClipPPOLoss`.
    """

    actor_network: TensorDictModule
    critic_network: TensorDictModule

    def __init__(
        self,
        actor_network: ProbabilisticTensorDictSequential | None = None,
        critic_network: TensorDictModule | None = None,
        *,
        value_norm: ValueNorm | None = None,
        entropy_coeff: float | dict[str, float] = 0.01,
        normalize_advantage: bool = True,
        normalize_advantage_exclude_dims: tuple[int, ...] = (-2,),
        **kwargs: Any,
    ) -> None:
        super().__init__(
            actor_network,
            critic_network,
            entropy_coeff=entropy_coeff,
            normalize_advantage=normalize_advantage,
            normalize_advantage_exclude_dims=normalize_advantage_exclude_dims,
            **kwargs,
        )
        self.value_norm = value_norm
        if value_norm is not None:
            self.add_module("_value_norm_module", value_norm)


# Silence unused-import linters: the imports below are needed in case a user
# overrides ``make_value_estimator`` to fall back to non-multi-agent estimators.
_ = (
    GAE,
    TD0Estimator,
    TD1Estimator,
    TDLambdaEstimator,
    VTrace,
    _GAMMA_LMBDA_DEPREC_ERROR,
)
