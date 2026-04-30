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
    TD3Loss,
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
    """Hydra configuration for :class:`~torchrl.objectives.SACLoss` (and :class:`~torchrl.objectives.sac.DiscreteSACLoss` when ``discrete=True``).

    Every kwarg accepted by ``SACLoss.__init__`` is exposed as a field here. The
    ``discrete``/``action_space``/``num_actions``/``target_entropy_weight`` fields
    apply only when the discrete variant is selected.
    """

    actor_network: Any = None
    qvalue_network: Any = None
    value_network: Any = None
    discrete: bool = False
    action_space: Any = None
    num_actions: int | None = None
    num_qvalue_nets: int = 2
    loss_function: str = "smooth_l1"
    alpha_init: float = 1.0
    min_alpha: float | None = None
    max_alpha: float | None = None
    action_spec: Any = None
    fixed_alpha: bool = False
    target_entropy: str | float = "auto"
    target_entropy_weight: float = 0.98
    delay_actor: bool = False
    delay_qvalue: bool = True
    delay_value: bool = True
    gamma: float | None = None
    priority_key: str | None = None
    separate_losses: bool = False
    reduction: str | None = None
    skip_done_states: bool = False
    deactivate_vmap: bool = False
    use_prioritized_weights: str | bool = "auto"
    scalar_output_mode: str | None = None
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
    """Hydra configuration for the PPO loss family.

    Dispatches between :class:`~torchrl.objectives.ClipPPOLoss` (``loss_type='clip'``),
    :class:`~torchrl.objectives.KLPENPPOLoss` (``loss_type='kl'``) and
    :class:`~torchrl.objectives.PPOLoss` (``loss_type='ppo'``). Every kwarg
    accepted by any of those classes is exposed here; only the kwargs relevant
    to the selected ``loss_type`` are forwarded.
    """

    actor_network: Any = None
    critic_network: Any = None
    loss_type: str = "clip"
    entropy_bonus: bool = True
    samples_mc_entropy: int = 1
    entropy_coeff: float | None = None
    log_explained_variance: bool = True
    critic_coeff: float | None = None
    loss_critic_type: str = "smooth_l1"
    normalize_advantage: bool = False
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
    clip_epsilon: float = 0.2
    dtarg: float = 0.01
    beta: float = 1.0
    increment: float = 2.0
    decrement: float = 0.5
    samples_mc_kl: int = 1
    device: Any = None
    _target_: str = "torchrl.trainers.algorithms.configs.objectives._make_ppo_loss"

    def __post_init__(self) -> None:
        """Post-initialization hook for PPO loss configurations."""
        super().__post_init__()


def _make_ppo_loss(*args, **kwargs) -> PPOLoss:
    loss_type = kwargs.pop("loss_type", "clip")
    # Drop kwargs that don't apply to the chosen loss flavor so each class
    # receives only what its __init__ accepts.
    clip_only = {"clip_epsilon"}
    kl_only = {"dtarg", "beta", "increment", "decrement", "samples_mc_kl"}
    ppo_only = {
        "log_explained_variance",
        "advantage_key",
        "value_target_key",
        "value_key",
        "functional",
        "actor",
        "critic",
    }
    if loss_type == "clip":
        for k in kl_only | ppo_only:
            kwargs.pop(k, None)
        return ClipPPOLoss(*args, **kwargs)
    elif loss_type == "kl":
        for k in clip_only | ppo_only:
            kwargs.pop(k, None)
        return KLPENPPOLoss(*args, **kwargs)
    elif loss_type == "ppo":
        for k in clip_only | kl_only:
            kwargs.pop(k, None)
        return PPOLoss(*args, **kwargs)
    else:
        raise ValueError(f"Invalid loss type: {loss_type}")


@dataclass
class TD3LossConfig(LossConfig):
    """Hydra configuration for :class:`~torchrl.objectives.TD3Loss`.

    Every kwarg accepted by ``TD3Loss.__init__`` is exposed as a field here.
    """

    actor_network: Any = None
    qvalue_network: Any = None
    action_spec: Any = None
    bounds: tuple[float] | None = None
    num_qvalue_nets: int = 2
    policy_noise: float = 0.2
    noise_clip: float = 0.5
    loss_function: str = "smooth_l1"
    delay_actor: bool = True
    delay_qvalue: bool = True
    gamma: float | None = None
    priority_key: str | None = None
    separate_losses: bool = False
    reduction: str | None = None
    deactivate_vmap: bool = False
    use_prioritized_weights: str | bool = "auto"
    _target_: str = "torchrl.trainers.algorithms.configs.objectives._make_td3_loss"


def _make_td3_loss(*args, **kwargs) -> TD3Loss:
    # Instantiate networks if they are config objects
    actor_network = kwargs.get("actor_network")
    qvalue_network = kwargs.get("qvalue_network")

    if actor_network is not None and hasattr(actor_network, "_target_"):
        kwargs["actor_network"] = actor_network()
    if qvalue_network is not None and hasattr(qvalue_network, "_target_"):
        kwargs["qvalue_network"] = qvalue_network()

    return TD3Loss(*args, **kwargs)


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
    """Hydra configuration for :class:`~torchrl.objectives.value.GAE`.

    Every kwarg accepted by ``GAE.__init__`` is exposed as a field here.
    """

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
    """Hydra configuration for :class:`~torchrl.objectives.DQNLoss`.

    Every kwarg accepted by ``DQNLoss.__init__`` is exposed as a field here.
    """

    value_network: Any = None
    loss_function: str = "l2"
    delay_value: bool = True
    double_dqn: bool = False
    action_space: Any = None
    gamma: float | None = None
    priority_key: str | None = None
    reduction: str | None = None
    use_prioritized_weights: str | bool = "auto"
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
    """Hydra configuration for :class:`~torchrl.objectives.DDPGLoss`.

    Every kwarg accepted by ``DDPGLoss.__init__`` is exposed as a field here.
    """

    actor_network: Any = None
    value_network: Any = None
    loss_function: str = "l2"
    delay_actor: bool = False
    delay_value: bool = True
    gamma: float | None = None
    separate_losses: bool = False
    reduction: str | None = None
    use_prioritized_weights: str | bool = "auto"
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
    """Hydra configuration for :class:`~torchrl.objectives.IQLLoss` (and :class:`~torchrl.objectives.iql.DiscreteIQLLoss` when ``discrete=True``).

    Every kwarg accepted by ``IQLLoss.__init__`` is exposed as a field here. The
    ``discrete``/``action_space`` fields apply only when the discrete variant is
    selected.
    """

    actor_network: Any = None
    qvalue_network: Any = None
    value_network: Any = None
    discrete: bool = False
    action_space: Any = None
    num_qvalue_nets: int = 2
    loss_function: str = "smooth_l1"
    temperature: float = 1.0
    expectile: float = 0.5
    gamma: float | None = None
    priority_key: str | None = None
    separate_losses: bool = False
    reduction: str | None = None
    deactivate_vmap: bool = False
    scalar_output_mode: str | None = None
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
        # DiscreteIQLLoss has no `deactivate_vmap` kwarg.
        kwargs.pop("deactivate_vmap", None)
        return DiscreteIQLLoss(*args, **kwargs)
    # IQLLoss has no `action_space` kwarg.
    kwargs.pop("action_space", None)
    return IQLLoss(*args, **kwargs)


@dataclass
class CQLLossConfig(LossConfig):
    """Hydra configuration for :class:`~torchrl.objectives.CQLLoss`.

    Every kwarg accepted by ``CQLLoss.__init__`` is exposed as a field here.
    """

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
    scalar_output_mode: str | None = None
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
