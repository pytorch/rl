# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from torchrl.objectives import (
    ClipPPOLoss,
    CQLLoss,
    DDPGLoss,
    DQNLoss,
    IQLLoss,
    KLPENPPOLoss,
    PPOLoss,
    SACLoss,
)
from torchrl.objectives.iql import DiscreteIQLLoss
from torchrl.objectives.sac import DiscreteSACLoss
from torchrl.trainers.algorithms.configs.common import ConfigBase


@dataclass
class LossConfig(ConfigBase):
    """A class to configure a loss.

    Args:
        loss_type: The type of loss to use.
    """

    _partial_: bool = False

    def __post_init__(self) -> None:
        """Post-initialization hook for loss configurations."""


@dataclass
class SACLossConfig(LossConfig):
    """A class to configure a SAC loss."""

    actor_network: Any = None
    qvalue_network: Any = None
    value_network: Any = None
    discrete: bool = False
    num_qvalue_nets: int = 2
    loss_function: str = "smooth_l1"
    alpha_init: float = 1.0
    min_alpha: float | None = None
    max_alpha: float | None = None
    action_spec: Any = None
    fixed_alpha: bool = False
    target_entropy: str | float = "auto"
    delay_actor: bool = False
    delay_qvalue: bool = True
    delay_value: bool = True
    gamma: float | None = None
    priority_key: str | None = None
    separate_losses: bool = False
    reduction: str | None = None
    skip_done_states: bool = False
    deactivate_vmap: bool = False
    _target_: str = "torchrl.trainers.algorithms.configs.objectives._make_sac_loss"

    def __post_init__(self) -> None:
        """Post-initialization hook for SAC loss configurations."""
        super().__post_init__()


def _make_sac_loss(*args, **kwargs) -> SACLoss:
    discrete_loss_type = kwargs.pop("discrete", False)

    # Instantiate networks if they are config objects
    actor_network = kwargs.get("actor_network")
    qvalue_network = kwargs.get("qvalue_network")
    value_network = kwargs.get("value_network")

    if actor_network is not None and hasattr(actor_network, "_target_"):
        kwargs["actor_network"] = actor_network()
    if qvalue_network is not None and hasattr(qvalue_network, "_target_"):
        kwargs["qvalue_network"] = qvalue_network()
    if value_network is not None and hasattr(value_network, "_target_"):
        kwargs["value_network"] = value_network()

    if discrete_loss_type:
        return DiscreteSACLoss(*args, **kwargs)
    else:
        return SACLoss(*args, **kwargs)


@dataclass
class PPOLossConfig(LossConfig):
    """A class to configure a PPO loss."""

    actor_network: Any = None
    critic_network: Any = None
    loss_type: str = "clip"
    entropy_bonus: bool = True
    samples_mc_entropy: int = 1
    entropy_coeff: float | None = None
    log_explained_variance: bool = True
    critic_coeff: float = 0.25
    loss_critic_type: str = "smooth_l1"
    normalize_advantage: bool = True
    normalize_advantage_exclude_dims: tuple = ()
    gamma: float | None = None
    separate_losses: bool = False
    advantage_key: str | None = None
    value_target_key: str | None = None
    value_key: str | None = None
    functional: bool = True
    actor: Any = None
    critic: Any = None
    reduction: str | None = None
    clip_value: float | None = None
    device: Any = None
    _target_: str = "torchrl.trainers.algorithms.configs.objectives._make_ppo_loss"

    def __post_init__(self) -> None:
        """Post-initialization hook for PPO loss configurations."""
        super().__post_init__()


def _make_ppo_loss(*args, **kwargs) -> PPOLoss:
    loss_type = kwargs.pop("loss_type", "clip")
    if loss_type == "clip":
        return ClipPPOLoss(*args, **kwargs)
    elif loss_type == "kl":
        return KLPENPPOLoss(*args, **kwargs)
    elif loss_type == "ppo":
        return PPOLoss(*args, **kwargs)
    else:
        raise ValueError(f"Invalid loss type: {loss_type}")


@dataclass
class TargetNetUpdaterConfig:
    """An abstract class to configure target net updaters."""

    loss_module: Any
    _partial_: bool = True


@dataclass
class SoftUpdateConfig(TargetNetUpdaterConfig):
    """A class for soft update instantiation."""

    _target_: str = "torchrl.objectives.utils.SoftUpdate"
    eps: float | None = None  # noqa # type-ignore
    tau: float | None = 0.001  # noqa # type-ignore


@dataclass
class HardUpdateConfig(TargetNetUpdaterConfig):
    """A class for hard update instantiation."""

    _target_: str = "torchrl.objectives.utils.HardUpdate"
    value_network_update_interval: int = 1000


@dataclass
class GAEConfig(LossConfig):
    """A class to configure a GAELoss."""

    gamma: float | None = None
    lmbda: float | None = None
    value_network: Any = None
    average_gae: bool = True
    differentiable: bool = False
    vectorized: bool | None = None
    skip_existing: bool | None = None
    advantage_key: str | None = None
    value_target_key: str | None = None
    value_key: str | None = None
    shifted: bool = False
    device: Any = None
    time_dim: int | None = None
    auto_reset_env: bool = False
    deactivate_vmap: bool = False
    _target_: str = "torchrl.objectives.value.GAE"
    _partial_: bool = False

    def __post_init__(self) -> None:
        """Post-initialization hook for GAELoss configurations."""
        super().__post_init__()


@dataclass
class DQNLossConfig(LossConfig):
    """A class to configure a DQN loss."""

    value_network: Any = None
    loss_function: str = "l2"
    delay_value: bool = True
    double_dqn: bool = False
    action_space: Any = None
    gamma: float | None = None
    reduction: str | None = None
    _target_: str = "torchrl.trainers.algorithms.configs.objectives._make_dqn_loss"

    def __post_init__(self) -> None:
        super().__post_init__()


def _make_dqn_loss(*args, **kwargs) -> DQNLoss:
    value_network = kwargs.get("value_network")
    if value_network is not None and hasattr(value_network, "_target_"):
        kwargs["value_network"] = value_network()
    return DQNLoss(*args, **kwargs)


@dataclass
class DDPGLossConfig(LossConfig):
    """A class to configure a DDPG loss."""

    actor_network: Any = None
    value_network: Any = None
    loss_function: str = "l2"
    delay_actor: bool = False
    delay_value: bool = True
    gamma: float | None = None
    separate_losses: bool = False
    reduction: str | None = None
    _target_: str = "torchrl.trainers.algorithms.configs.objectives._make_ddpg_loss"

    def __post_init__(self) -> None:
        super().__post_init__()


def _make_ddpg_loss(*args, **kwargs) -> DDPGLoss:
    actor_network = kwargs.get("actor_network")
    value_network = kwargs.get("value_network")
    if actor_network is not None and hasattr(actor_network, "_target_"):
        kwargs["actor_network"] = actor_network()
    if value_network is not None and hasattr(value_network, "_target_"):
        kwargs["value_network"] = value_network()
    return DDPGLoss(*args, **kwargs)


@dataclass
class IQLLossConfig(LossConfig):
    """A class to configure an IQL loss."""

    actor_network: Any = None
    qvalue_network: Any = None
    value_network: Any = None
    discrete: bool = False
    num_qvalue_nets: int = 2
    loss_function: str = "smooth_l1"
    temperature: float = 1.0
    expectile: float = 0.5
    gamma: float | None = None
    separate_losses: bool = False
    reduction: str | None = None
    deactivate_vmap: bool = False
    _target_: str = "torchrl.trainers.algorithms.configs.objectives._make_iql_loss"

    def __post_init__(self) -> None:
        super().__post_init__()


def _make_iql_loss(*args, **kwargs) -> IQLLoss:
    discrete_loss_type = kwargs.pop("discrete", False)
    actor_network = kwargs.get("actor_network")
    qvalue_network = kwargs.get("qvalue_network")
    value_network = kwargs.get("value_network")
    if actor_network is not None and hasattr(actor_network, "_target_"):
        kwargs["actor_network"] = actor_network()
    if qvalue_network is not None and hasattr(qvalue_network, "_target_"):
        kwargs["qvalue_network"] = qvalue_network()
    if value_network is not None and hasattr(value_network, "_target_"):
        kwargs["value_network"] = value_network()
    if discrete_loss_type:
        return DiscreteIQLLoss(*args, **kwargs)
    return IQLLoss(*args, **kwargs)


@dataclass
class CQLLossConfig(LossConfig):
    """A class to configure a CQL loss."""

    actor_network: Any = None
    qvalue_network: Any = None
    loss_function: str = "smooth_l1"
    alpha_init: float = 1.0
    min_alpha: float | None = None
    max_alpha: float | None = None
    action_spec: Any = None
    fixed_alpha: bool = False
    target_entropy: str | float = "auto"
    delay_actor: bool = False
    delay_qvalue: bool = True
    gamma: float | None = None
    temperature: float = 1.0
    min_q_weight: float = 1.0
    max_q_backup: bool = False
    deterministic_backup: bool = True
    num_random: int = 10
    with_lagrange: bool = False
    lagrange_thresh: float = 0.0
    reduction: str | None = None
    deactivate_vmap: bool = False
    _target_: str = "torchrl.trainers.algorithms.configs.objectives._make_cql_loss"

    def __post_init__(self) -> None:
        super().__post_init__()


def _make_cql_loss(*args, **kwargs) -> CQLLoss:
    actor_network = kwargs.get("actor_network")
    qvalue_network = kwargs.get("qvalue_network")
    if actor_network is not None and hasattr(actor_network, "_target_"):
        kwargs["actor_network"] = actor_network()
    if qvalue_network is not None and hasattr(qvalue_network, "_target_"):
        kwargs["qvalue_network"] = qvalue_network()
    return CQLLoss(*args, **kwargs)
