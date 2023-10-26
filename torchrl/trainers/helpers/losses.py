# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import warnings
from dataclasses import dataclass
from typing import Any, Optional, Tuple

from torchrl.modules import ActorCriticOperator, ActorValueOperator
from torchrl.objectives import DistributionalDQNLoss, DQNLoss, HardUpdate, SoftUpdate
from torchrl.objectives.common import LossModule
from torchrl.objectives.deprecated import REDQLoss_deprecated
from torchrl.objectives.utils import TargetNetUpdater


def make_target_updater(
    cfg: "DictConfig", loss_module: LossModule  # noqa: F821
) -> Optional[TargetNetUpdater]:
    """Builds a target network weight update object."""
    if cfg.loss == "double":
        if not cfg.hard_update:
            target_net_updater = SoftUpdate(
                loss_module, eps=1 - 1 / cfg.value_network_update_interval
            )
        else:
            target_net_updater = HardUpdate(
                loss_module,
                value_network_update_interval=cfg.value_network_update_interval,
            )
    else:
        if cfg.hard_update:
            raise RuntimeError(
                "hard/soft-update are supposed to be used with double SAC loss. "
                "Consider using --loss=double or discarding the hard_update flag."
            )
        target_net_updater = None
    return target_net_updater


def make_redq_loss(
    model, cfg
) -> Tuple[REDQLoss_deprecated, Optional[TargetNetUpdater]]:
    """Builds the REDQ loss module."""
    warnings.warn(
        "This helper function will be deprecated in v0.4. Consider using the local helper in the REDQ example.",
        category=DeprecationWarning,
    )
    loss_kwargs = {}
    if hasattr(cfg, "distributional") and cfg.distributional:
        raise NotImplementedError
    else:
        loss_kwargs.update({"loss_function": cfg.loss_function})
        loss_kwargs.update({"delay_qvalue": cfg.loss == "double"})
        loss_class = REDQLoss_deprecated
    if isinstance(model, ActorValueOperator):
        actor_model = model.get_policy_operator()
        qvalue_model = model.get_value_operator()
    elif isinstance(model, ActorCriticOperator):
        raise RuntimeError(
            "Although REDQ Q-value depends upon selected actions, using the"
            "ActorCriticOperator will lead to resampling of the actions when"
            "computing the Q-value loss, which we don't want. Please use the"
            "ActorValueOperator instead."
        )
    else:
        actor_model, qvalue_model = model

    loss_module = loss_class(
        actor_network=actor_model,
        qvalue_network=qvalue_model,
        num_qvalue_nets=cfg.num_q_values,
        gSDE=cfg.gSDE,
        **loss_kwargs,
    )
    loss_module.make_value_estimator(gamma=cfg.gamma)
    target_net_updater = make_target_updater(cfg, loss_module)
    return loss_module, target_net_updater


def make_dqn_loss(model, cfg) -> Tuple[DQNLoss, Optional[TargetNetUpdater]]:
    """Builds the DQN loss module."""
    loss_kwargs = {}
    if cfg.distributional:
        loss_class = DistributionalDQNLoss
    else:
        loss_kwargs.update({"loss_function": cfg.loss_function})
        loss_class = DQNLoss
    if cfg.loss not in ("single", "double"):
        raise NotImplementedError
    loss_kwargs.update({"delay_value": cfg.loss == "double"})
    loss_module = loss_class(model, **loss_kwargs)
    loss_module.make_value_estimator(gamma=cfg.gamma)
    target_net_updater = make_target_updater(cfg, loss_module)
    return loss_module, target_net_updater


@dataclass
class LossConfig:
    """Generic Loss config struct."""

    loss: str = "double"
    # whether double or single SAC loss should be used. Default=double
    hard_update: bool = False
    # whether soft-update should be used with double SAC loss (default) or hard updates.
    loss_function: str = "smooth_l1"
    # loss function for the value network. Either one of l1, l2 or smooth_l1 (default).
    value_network_update_interval: int = 1000
    # how often the target value network weights are updated (in number of updates).
    # If soft-updates are used, the value is translated into a moving average decay by using
    # the formula decay=1-1/cfg.value_network_update_interval. Default=1000
    gamma: float = 0.99
    # Decay factor for return computation. Default=0.99.
    num_q_values: int = 2
    # As suggested in the original SAC paper and in https://arxiv.org/abs/1802.09477, we can
    # use two (or more!) different qvalue networks trained independently and choose the lowest value
    # predicted to predict the state action value. This can be disabled by using this flag.
    # REDQ uses an arbitrary number of Q-value functions to speed up learning in MF contexts.
    target_entropy: Any = None
    # Target entropy for the policy distribution. Default is None (auto calculated as the `target_entropy = -action_dim`)


@dataclass
class A2CLossConfig:
    """A2C Loss config struct."""

    gamma: float = 0.99
    # Decay factor for return computation. Default=0.99.
    entropy_coef: float = 1e-3
    # Entropy factor for the A2C loss
    critic_coef: float = 1.0
    # Critic factor for the A2C loss
    critic_loss_function: str = "smooth_l1"
    # loss function for the value network. Either one of l1, l2 or smooth_l1 (default).


@dataclass
class PPOLossConfig:
    """PPO Loss config struct."""

    loss: str = "clip"
    # PPO loss class, either clip or kl or base/<empty>. Default=clip

    # PPOLoss base parameters:
    gamma: float = 0.99
    # Decay factor for return computation. Default=0.99.
    lmbda: float = 0.95
    # lambda factor in GAE (using 'lambda' as attribute is prohibited in python, hence the misspelling)
    entropy_bonus: bool = True
    # Whether or not to add an entropy term to the PPO loss.
    entropy_coef: float = 1e-3
    # Entropy factor for the PPO loss
    samples_mc_entropy: int = 1
    # Number of samples to use for a Monte-Carlo estimate if the policy distribution has not closed formula.
    loss_function: str = "smooth_l1"
    # loss function for the value network. Either one of l1, l2 or smooth_l1 (default).
    critic_coef: float = 1.0
    # Critic loss multiplier when computing the total loss.

    # ClipPPOLoss parameters:
    clip_epsilon: float = 0.2
    # weight clipping threshold in the clipped PPO loss equation.

    # KLPENPPOLoss parameters:
    dtarg: float = 0.01
    # target KL divergence.
    beta: float = 1.0
    # initial KL divergence multiplier.
    increment: float = 2
    # how much beta should be incremented if KL > dtarg. Valid range: increment >= 1.0
    decrement: float = 0.5
    # how much beta should be decremented if KL < dtarg. Valid range: decrement <= 1.0
    samples_mc_kl: int = 1
    # Number of samples to use for a Monte-Carlo estimate of KL if necessary
