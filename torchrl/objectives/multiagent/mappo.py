# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Multi-agent PPO objectives.

Implements :class:`MultiPPOLoss` — the unified multi-agent PPO loss that
covers both MAPPO (centralised critic) and IPPO (independent critic) via a
``critic_type`` field.

:class:`MAPPOLoss` and :class:`IPPOLoss` are retained as deprecated aliases
and will be removed in v0.11.

References:
    - Yu, C. et al. *The Surprising Effectiveness of PPO in Cooperative
      Multi-Agent Games.* NeurIPS 2022. https://arxiv.org/abs/2103.01955
    - de Witt, C. S. et al. *Is Independent Learning All You Need in the
      StarCraft Multi-Agent Challenge?* 2020. https://arxiv.org/abs/2011.09533
"""
from __future__ import annotations

import warnings
from typing import Any, Literal

from tensordict.nn import TensorDictModule
from tensordict.nn.probabilistic import ProbabilisticTensorDictSequential

from torchrl.modules.value_norm import ValueNorm
from torchrl.objectives.ppo import ClipPPOLoss
from torchrl.objectives.utils import ValueEstimators


class _MultiAgentPPOMixin:
    """Shared plumbing for :class:`MultiPPOLoss` and its deprecated aliases.

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

    # Subclasses wire this up in ``__init__`` — populated here as a type hint
    # only so static checkers know it exists.
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


class MultiPPOLoss(_MultiAgentPPOMixin, ClipPPOLoss):
    """Unified multi-agent PPO loss covering MAPPO and IPPO.

    Pass ``critic_type="centralized"`` (default) for **MAPPO** — a
    decentralised actor together with a centralised critic that conditions on
    the full team state, reducing variance at the cost of requiring global
    information during training (Yu et al. 2022).

    Pass ``critic_type="independent"`` for **IPPO** — each agent has its own
    value function conditioned only on its local observation. No global state
    is required, and the approach is surprisingly competitive on many SMAC
    scenarios (de Witt et al. 2020).

    The loss computation is identical in both modes — the only difference is
    the critic network you supply. ``critic_type`` is stored as
    ``self.critic_type`` for introspection and self-documentation; it does not
    alter the gradient computation.

    This class is a thin specialisation of :class:`ClipPPOLoss`. Differences:

    - The default value estimator is :class:`~torchrl.objectives.value.MultiAgentGAE`,
      which broadcasts team-shared rewards / done flags along the agent dim
      before computing returns.
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
        critic_network (TensorDictModule): value operator. For
            ``critic_type="centralized"`` build with
            ``MultiAgentMLP(centralized=True, share_params=True)``; for
            ``critic_type="independent"`` use
            ``MultiAgentMLP(centralized=False, share_params=True)``.

    Keyword Args:
        critic_type (str): ``"centralized"`` (MAPPO, default) or
            ``"independent"`` (IPPO). Stored for introspection only; does not
            alter the loss computation.
        value_norm (ValueNorm, optional): if supplied, the critic target and
            prediction are normalised by this running normaliser before the
            MSE / smooth-L1 distance. Composes correctly with ``clip_value``
            (the clip radius is applied in normalised space).
            Defaults to ``None`` (no value norm).
        clip_epsilon (float): PPO ratio clip. Defaults to ``0.2``.
        entropy_coeff (float): entropy bonus weight. Defaults to ``0.01``.
        critic_coef (float, optional): critic loss weight. Defaults to ``1.0``.
        normalize_advantage (bool): whether to standardise the advantage.
            Defaults to ``True``.
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
        >>> from torchrl.objectives.multiagent import MultiPPOLoss
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
        >>> # Centralised critic (MAPPO)
        >>> critic = TensorDictModule(
        ...     MultiAgentMLP(
        ...         n_agent_inputs=obs_dim, n_agent_outputs=1,
        ...         n_agents=n_agents, centralized=True, share_params=True,
        ...     ),
        ...     in_keys=[("agents", "observation")],
        ...     out_keys=[("agents", "state_value")],
        ... )
        >>> loss = MultiPPOLoss(actor, critic, critic_type="centralized",
        ...                     value_norm=PopArtValueNorm(shape=1))
        >>> loss.set_keys(value=("agents", "state_value"), action=("agents", "action"))
    """

    def __init__(
        self,
        actor_network: ProbabilisticTensorDictSequential | None = None,
        critic_network: TensorDictModule | None = None,
        *,
        critic_type: Literal["centralized", "independent"] = "centralized",
        value_norm: ValueNorm | None = None,
        entropy_coeff: float | dict[str, float] = 0.01,
        normalize_advantage: bool = True,
        normalize_advantage_exclude_dims: tuple[int, ...] = (-2,),
        **kwargs: Any,
    ) -> None:
        if critic_type not in ("centralized", "independent"):
            raise ValueError(
                f"critic_type must be 'centralized' or 'independent', got {critic_type!r}"
            )
        super().__init__(
            actor_network,
            critic_network,
            entropy_coeff=entropy_coeff,
            normalize_advantage=normalize_advantage,
            normalize_advantage_exclude_dims=normalize_advantage_exclude_dims,
            **kwargs,
        )
        self.critic_type = critic_type
        # ``nn.Module.__setattr__`` registers the ``ValueNorm`` as a child
        # module automatically, so ``.to(device)`` / ``state_dict()`` / etc.
        # pick it up without us calling ``add_module`` a second time.
        self.value_norm = value_norm


class MAPPOLoss(MultiPPOLoss):
    """Deprecated alias for ``MultiPPOLoss(critic_type='centralized')``.

    .. deprecated:: 0.9
        Use :class:`MultiPPOLoss` with ``critic_type='centralized'`` instead.
        ``MAPPOLoss`` will be removed in v0.11.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        warnings.warn(
            "MAPPOLoss is deprecated and will be removed in v0.11. "
            "Use MultiPPOLoss(critic_type='centralized') instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        kwargs.setdefault("critic_type", "centralized")
        super().__init__(*args, **kwargs)


class IPPOLoss(MultiPPOLoss):
    """Deprecated alias for ``MultiPPOLoss(critic_type='independent')``.

    .. deprecated:: 0.9
        Use :class:`MultiPPOLoss` with ``critic_type='independent'`` instead.
        ``IPPOLoss`` will be removed in v0.11.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        warnings.warn(
            "IPPOLoss is deprecated and will be removed in v0.11. "
            "Use MultiPPOLoss(critic_type='independent') instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        kwargs.setdefault("critic_type", "independent")
        super().__init__(*args, **kwargs)
