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
    QMixerLoss,
    SACLoss,
    TD3Loss,
)
from torchrl.objectives.iql import DiscreteIQLLoss
from torchrl.objectives.sac import DiscreteSACLoss
from torchrl.trainers.algorithms.configs.common import _normalize_hydra_key, ConfigBase


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
    gamma = kwargs.pop("gamma", None)

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
        loss = DiscreteSACLoss(*args, **kwargs)
    else:
        loss = SACLoss(*args, **kwargs)
    if gamma is not None:
        loss.make_value_estimator(gamma=gamma)
    return loss


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
    gamma = kwargs.pop("gamma", None)
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
        loss = ClipPPOLoss(*args, **kwargs)
    elif loss_type == "kl":
        for k in clip_only | ppo_only:
            kwargs.pop(k, None)
        loss = KLPENPPOLoss(*args, **kwargs)
    elif loss_type == "ppo":
        for k in clip_only | kl_only:
            kwargs.pop(k, None)
        loss = PPOLoss(*args, **kwargs)
    else:
        raise ValueError(f"Invalid loss type: {loss_type}")
    if gamma is not None:
        loss.make_value_estimator(gamma=gamma)
    return loss


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
    gamma = kwargs.pop("gamma", None)

    # Instantiate networks if they are config objects
    actor_network = kwargs.get("actor_network")
    qvalue_network = kwargs.get("qvalue_network")

    if actor_network is not None and hasattr(actor_network, "_target_"):
        kwargs["actor_network"] = actor_network()
    if qvalue_network is not None and hasattr(qvalue_network, "_target_"):
        kwargs["qvalue_network"] = qvalue_network()

    loss = TD3Loss(*args, **kwargs)
    if gamma is not None:
        loss.make_value_estimator(gamma=gamma)
    return loss


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
    action_key: Any = None
    action_value_key: Any = None
    value_key: Any = None
    reward_key: Any = None
    done_key: Any = None
    terminated_key: Any = None
    priority_key: Any = None
    priority_weight_key: Any = None
    _target_: str = "torchrl.trainers.algorithms.configs.objectives._make_dqn_loss"

    def __post_init__(self) -> None:
        super().__post_init__()


def _make_dqn_loss(*args, **kwargs) -> DQNLoss:
    tensor_keys = {}
    for key in (
        "action_key",
        "action_value_key",
        "value_key",
        "reward_key",
        "done_key",
        "terminated_key",
        "priority_key",
        "priority_weight_key",
    ):
        if key in kwargs:
            value = kwargs.pop(key)
            if value is not None:
                tensor_keys[key.removesuffix("_key")] = _normalize_hydra_key(value)

    value_network = kwargs.get("value_network")
    gamma = kwargs.pop("gamma", None)

    if value_network is not None and hasattr(value_network, "_target_"):
        kwargs["value_network"] = value_network()
    loss = DQNLoss(*args, **kwargs)
    if tensor_keys:
        loss.set_keys(**tensor_keys)
    if gamma is not None:
        loss.make_value_estimator(gamma=gamma)
    return loss


@dataclass
class QMixerLossConfig(LossConfig):
    """A class to configure a QMixer loss."""

    local_value_network: Any = None
    mixer_network: Any = None
    loss_function: str = "l2"
    delay_value: bool = True
    action_space: Any = None
    gamma: float | None = None
    priority_key: str | None = None
    action_key: Any = None
    action_value_key: Any = None
    local_value_key: Any = None
    global_value_key: Any = None
    reward_key: Any = None
    done_key: Any = None
    terminated_key: Any = None
    _target_: str = "torchrl.trainers.algorithms.configs.objectives._make_qmixer_loss"

    def __post_init__(self) -> None:
        super().__post_init__()


def _make_qmixer_loss(*args, **kwargs) -> QMixerLoss:
    tensor_keys = {}
    for key in (
        "action_key",
        "action_value_key",
        "local_value_key",
        "global_value_key",
        "reward_key",
        "done_key",
        "terminated_key",
        "priority_key",
    ):
        if key in kwargs:
            value = kwargs.pop(key)
            if value is not None:
                tensor_keys[key.removesuffix("_key")] = _normalize_hydra_key(value)
    local_value_network = kwargs.get("local_value_network")
    mixer_network = kwargs.get("mixer_network")
    gamma = kwargs.pop("gamma", None)

    if local_value_network is not None and hasattr(local_value_network, "_target_"):
        kwargs["local_value_network"] = local_value_network()
    if mixer_network is not None and hasattr(mixer_network, "_target_"):
        kwargs["mixer_network"] = mixer_network()

    loss = QMixerLoss(*args, **kwargs)
    loss.set_keys(**tensor_keys)
    if gamma is not None:
        loss.make_value_estimator(gamma=gamma)
    return loss


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
    gamma = kwargs.pop("gamma", None)

    actor_network = kwargs.get("actor_network")
    value_network = kwargs.get("value_network")
    if actor_network is not None and hasattr(actor_network, "_target_"):
        kwargs["actor_network"] = actor_network()
    if value_network is not None and hasattr(value_network, "_target_"):
        kwargs["value_network"] = value_network()
    loss = DDPGLoss(*args, **kwargs)
    if gamma is not None:
        loss.make_value_estimator(gamma=gamma)
    return loss


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
    gamma = kwargs.pop("gamma", None)

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
        loss = DiscreteIQLLoss(*args, **kwargs)
    else:
        # IQLLoss has no `action_space` kwarg.
        kwargs.pop("action_space", None)
        loss = IQLLoss(*args, **kwargs)
    if gamma is not None:
        loss.make_value_estimator(gamma=gamma)
    return loss


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
    gamma = kwargs.pop("gamma", None)

    actor_network = kwargs.get("actor_network")
    qvalue_network = kwargs.get("qvalue_network")
    if actor_network is not None and hasattr(actor_network, "_target_"):
        kwargs["actor_network"] = actor_network()
    if qvalue_network is not None and hasattr(qvalue_network, "_target_"):
        kwargs["qvalue_network"] = qvalue_network()
    loss = CQLLoss(*args, **kwargs)
    if gamma is not None:
        loss.make_value_estimator(gamma=gamma)
    return loss
