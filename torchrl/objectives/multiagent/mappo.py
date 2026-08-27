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

from typing import Any

from tensordict.nn import TensorDictModule
from tensordict.nn.probabilistic import ProbabilisticTensorDictSequential

from torchrl.modules.value_norm import ValueNorm
from torchrl.objectives.ppo import ClipPPOLoss
from torchrl.objectives.utils import ValueEstimators


class _MultiAgentPPOMixin:
    """Shared plumbing for :class:`MAPPOLoss` and :class:`IPPOLoss`.

    Two pieces:

    1. Default the value estimator to :class:`MultiAgentGAE` so per-agent
       value outputs broadcast cleanly against team-shared reward / done
       signals.
    2. When a :class:`~torchrl.modules.ValueNorm` is attached, normalise the
       value-target and critic prediction by the running stats *before* the
       MSE / smooth-L1 distance. This stabilises critic-loss magnitude when
       reward scales drift during training (Yu et al. 2022, Table 13).

       The normalisation is plumbed through
       :meth:`~torchrl.objectives.PPOLoss._critic_loss_inputs`, the hook
       :class:`~torchrl.objectives.PPOLoss` exposes for exactly this purpose,
       so all of the parent's other critic-loss machinery (``clip_value``,
       ``separate_losses``, ``log_explained_variance``, ...) continues to
       work alongside ``value_norm``.
    """

    default_value_estimator = ValueEstimators.MAGAE

    # Subclasses (MAPPOLoss / IPPOLoss) wire this up in ``__init__`` —
    # populated here as a type hint only so static checkers know it exists.
    value_norm: ValueNorm | None

    def _critic_loss_inputs(self, target_return, state_value, old_state_value):
        """Override of :meth:`PPOLoss._critic_loss_inputs` for PopArt normalisation.

        Pushes ``target_return`` / ``state_value`` / ``old_state_value``
        through the attached :class:`ValueNorm` if any, leaving the parent's
        ``clip_value`` / ``log_explained_variance`` / ``separate_losses``
        machinery untouched. ``old_state_value`` (used by the PPO value-clip
        path) is normalised with the same stats so the clip radius stays in
        normalised space — the convention the MAPPO paper assumes.

        The running stats are updated *once* per call, on the un-normalised
        ``target_return`` (which is what PopArt expects: the EMA tracks the
        real return distribution, not the normalised one).
        """
        value_norm = self.value_norm
        if value_norm is None:
            return target_return, state_value, old_state_value
        value_norm.update(target_return)
        normalised_target = value_norm.normalize(target_return.detach())
        normalised_pred = value_norm.normalize(state_value)
        normalised_old = (
            value_norm.normalize(old_state_value)
            if old_state_value is not None
            else None
        )
        return normalised_target, normalised_pred, normalised_old


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
            MSE / smooth-L1 distance. Composes correctly with ``clip_value``
            (the clip radius is applied in normalised space).
            Defaults to ``None`` (no value norm).
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
        >>> from tensordict.nn import TensorDictModule
        >>> from torchrl.modules import (
        ...     MultiAgentMLP, PopArtValueNorm, ProbabilisticActor,
        ... )
        >>> from torchrl.modules.distributions import NormalParamExtractor, TanhNormal
        >>> from torchrl.objectives.multiagent import MAPPOLoss
        >>> n_agents, obs_dim, action_dim = 3, 6, 2
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
        # ``nn.Module.__setattr__`` registers the ``ValueNorm`` as a child
        # module automatically, so ``.to(device)`` / ``state_dict()`` / etc.
        # pick it up without us calling ``add_module`` a second time.
        self.value_norm = value_norm


class IPPOLoss(MAPPOLoss):
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
