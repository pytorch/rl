from numbers import Number
from typing import Type, Optional, Tuple, Iterable

import torch

from torchrl.envs.common import Specs
from torchrl.modules.distributions import (
    Delta,
    TanhNormal,
    TanhDelta,
    Categorical,
)
from . import ActorCriticOperator
from .models.models import DuelingCnnDQNet, DdpgCnnActor, DdpgCnnQNet, DdpgMlpQNet, DdpgMlpActor, MLP
from .probabilistic_operators import QValueActor, DistributionalQValueActor, Actor, ProbabilisticOperator
from ..data import TensorSpec

DISTRIBUTIONS = {
    "delta": Delta,
    "tanh-normal": TanhNormal,
    "categorical": Categorical,
    "tanh-delta": TanhDelta,
}

__all__ = ["make_dqn_actor", "make_ddpg_actor"]


def make_dqn_actor(
        env_specs: Specs,
        net_class: Type = DuelingCnnDQNet,
        net_kwargs: Optional[dict] = None,
        atoms: int = 51,  # for distributional dqn
        vmin: Number = -3,
        vmax: Number = 3,
        actor_kwargs: Optional[dict] = None,
        in_key: str = "observation_pixels",
) -> Actor:
    default_net_kwargs = {"cnn_kwargs": {"depth": 3, "num_cells": 64, "out_features": 64}, }
    net_kwargs = net_kwargs if net_kwargs is not None else dict()
    default_net_kwargs.update(net_kwargs)

    actor_kwargs = actor_kwargs if actor_kwargs is not None else dict()
    distribution_class = DISTRIBUTIONS["delta"]
    out_features = env_specs["action_spec"].shape[0]
    actor_class = QValueActor
    if atoms:
        out_features = (atoms, out_features)
        support = torch.linspace(vmin, vmax, atoms)
        actor_class = DistributionalQValueActor
        actor_kwargs.update({"support": support})
        if issubclass(net_class, DuelingCnnDQNet):
            default_net_kwargs.update({'out_features_value': (atoms, 1)})

    net = net_class(out_features=out_features, **default_net_kwargs, )
    print("policy is ", net)

    actor_kwargs.setdefault(
        "default_interaction_mode",
        "mode",
    )

    return actor_class(
        in_keys=[in_key],
        action_spec=env_specs["action_spec"],
        mapping_operator=net,
        distribution_class=distribution_class,
        # variable_size=variable_size,
        safe=True,
        **actor_kwargs,
    )


def make_ddpg_actor(
        env_specs: Specs,
        from_pixels: bool = True,
        actor_net_kwargs: Optional[dict] = None,
        value_net_kwargs: Optional[dict] = None,
        atoms: int = 0,  # for distributional dqn
        vmin: Number = -3,
        vmax: Number = 3,
        actor_kwargs: Optional[dict] = None,
        value_kwargs: Optional[dict] = None,
) -> Tuple[Actor, ProbabilisticOperator]:
    actor_net_kwargs = actor_net_kwargs if actor_net_kwargs is not None else dict()
    value_net_kwargs = value_net_kwargs if value_net_kwargs is not None else dict()
    actor_kwargs = actor_kwargs if actor_kwargs is not None else dict()
    value_kwargs = value_kwargs if value_kwargs is not None else dict()

    out_features = env_specs["action_spec"].shape[0]
    actor_class = Actor
    if atoms:
        raise NotImplementedError
        # https://arxiv.org/pdf/1804.08617.pdf

    action_distribution_class = DISTRIBUTIONS["tanh-delta"]
    actor_kwargs.setdefault(
        "default_interaction_mode",
        "mode",
    )
    actor_net_default_kwargs = {'action_dim': out_features}
    actor_net_default_kwargs.update(actor_net_kwargs)
    if from_pixels:
        in_keys = ["observation_pixels"]
        actor_net = DdpgCnnActor(**actor_net_default_kwargs)

    else:
        in_keys = ["observation_vector"]
        actor_net = DdpgMlpActor(**actor_net_default_kwargs)

    actor = actor_class(
        in_keys=in_keys,
        action_spec=env_specs["action_spec"],
        mapping_operator=actor_net,
        distribution_class=action_distribution_class,
        safe=True,
        **actor_kwargs,
    )

    state_class = ProbabilisticOperator
    value_distribution_class = DISTRIBUTIONS["delta"]
    value_kwargs.setdefault(
        "default_interaction_mode",
        "mode",
    )
    value_net_default_kwargs = {}
    value_net_default_kwargs.update(value_net_kwargs)
    if from_pixels:
        in_keys = ["observation_pixels", "action"]
        out_keys = ["state_action_value"]
        q_net = DdpgCnnQNet(**value_net_default_kwargs)
    else:
        in_keys = ["observation_vector", "action"]
        out_keys = ["state_action_value"]
        q_net = DdpgMlpQNet(**value_net_default_kwargs)

    value = state_class(
        in_keys=in_keys,
        out_keys=out_keys,
        spec=None,
        mapping_operator=q_net,
        distribution_class=value_distribution_class,
        **value_kwargs,
    )

    return actor, value


def make_actor_critic_model(spec: TensorSpec, in_keys: Optional[Iterable[str]] = None, **kwargs) -> ActorCriticOperator:
    if in_keys is None:
        in_keys = ["observation_vector"]
    common_mapping_operator = MLP(
        num_cells=[400, ],
        out_features=300,
        activate_last_layer=True,
    )
    policy_operator = MLP(
        num_cells=[200],
        out_features=2 * spec.shape[-1],
    )
    value_operator = MLP(
        num_cells=[200],
        out_features=1,
    )
    policy_distribution_class = TanhNormal
    return ActorCriticOperator(spec=spec,
                               in_keys=in_keys,
                               common_mapping_operator=common_mapping_operator,
                               policy_operator=policy_operator,
                               value_operator=value_operator,
                               policy_distribution_class=policy_distribution_class,
                               policy_interaction_mode="random",
                               **kwargs
                               )
