from numbers import Number
from typing import Type, Optional, Tuple, Iterable, Union

import torch
from torch import nn

from torchrl.envs.common import Specs
from torchrl.modules.distributions import (
    Delta,
    TanhNormal,
    TanhDelta,
    OneHotCategorical,
)
from . import ActorValueOperator
from .models.models import DuelingCnnDQNet, DdpgCnnActor, DdpgCnnQNet, DdpgMlpQNet, DdpgMlpActor, MLP
from .probabilistic_operators import QValueActor, DistributionalQValueActor, Actor, ProbabilisticOperator
from ..data import TensorSpec, UnboundedContinuousTensorSpec

DISTRIBUTIONS = {
    "delta": Delta,
    "tanh-normal": TanhNormal,
    "categorical": OneHotCategorical,
    "tanh-delta": TanhDelta,
}

__all__ = ["make_dqn_actor", "make_ddpg_actor", "make_sac_model"]


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
    """
    DQN constructor helper function.

    Args:
        env_specs (Specs): specs container of the environment.
        net_class (Type): class to be used.
            default: DuelingCnnDQNet
        net_kwargs (dict, optional): kwargs of the net_class type
        atoms (int): number of atoms of the support when using distributional DQN
            default: 51
        vmin (scalar): minimum value of the support for distributional DQN
            default: -3
        vmax (scalar): maximum value of the support for distributional DQN
            default: 3
        actor_kwargs (dict, optional): kwargs for the QValueActor class.
        in_key (str): tensordict key to be read by the QValueNetwork.
            default: "observation_pixels"


    Returns: A DQN policy operator.

    """
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
    """
    DDPG constructor helper function.

    Args:
        env_specs (Specs): specs container of the environment.
        from_pixels (bool): if True, the environment observation will be asumed to be pixels and a Conv net will be
        needed. Otherwise, a state vector is assumed and an MLP is created.
            default: True
        actor_net_kwargs (dict, optional): kwargs for the DdpgCnnActor / DdpgMlpActor classes.
        value_net_kwargs (dict, optional): kwargs for the DdpgCnnQNet / DdpgMlpQNet classes.
        actor_kwargs (dict, optional): kwargs for the Actor class, called to instantiate the policy operator.
        value_kwargs (dict, optional): kwargs for the ProbabilisticOperator class, called to instantiate the value
            operator.

    Returns: An actor and a value operators for DDPG.

    For more details on DDPG, refer to "CONTINUOUS CONTROL WITH DEEP REINFORCEMENT LEARNING",
    https://arxiv.org/pdf/1509.02971.pdf.
    """

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
    if from_pixels:
        value_net_default_kwargs = {}
        value_net_default_kwargs.update(value_net_kwargs)

        in_keys = ["observation_pixels", "action"]
        out_keys = ["state_action_value"]
        q_net = DdpgCnnQNet(**value_net_default_kwargs)
    else:
        value_net_default_kwargs1 = {'activation_class': torch.nn.ELU}
        value_net_default_kwargs1.update(value_net_kwargs.get('mlp_net_kwargs_net1', {}))
        value_net_default_kwargs2 = {'num_cells': [400, 300], 'depth': 2, 'activation_class': torch.nn.ELU}
        value_net_default_kwargs2.update(value_net_kwargs.get('mlp_net_kwargs_net2', {}))
        in_keys = ["observation_vector", "action"]
        out_keys = ["state_action_value"]
        q_net = DdpgMlpQNet(mlp_net_kwargs_net1=value_net_default_kwargs1,
                            mlp_net_kwargs_net2=value_net_default_kwargs2)

    value = state_class(
        in_keys=in_keys,
        out_keys=out_keys,
        spec=None,
        mapping_operator=q_net,
        distribution_class=value_distribution_class,
        **value_kwargs,
    )

    return actor, value


def make_actor_value_model(
        spec: TensorSpec,
        in_keys: Optional[Iterable[str]] = None,
        **kwargs) -> ActorValueOperator:
    """
    Actor-value model constructor helper function.
    Currently constructs MLP networks with immutable default arguments as described in "Proximal Policy Optimization
    Algorithms", https://arxiv.org/abs/1707.06347
    Other configurations can easily be implemented by modifying this function at will.

    Args:
        spec (TensorSpec): action_spec descriptor
        in_keys (Iterable[str], optional):
            default: ["observation_vector"]
        **kwargs: kwargs to be passed to the ActorCriticOperator.

    Returns: A joined ActorCriticOperator.

    """
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
    return ActorValueOperator(spec=spec,
                              in_keys=in_keys,
                              common_mapping_operator=common_mapping_operator,
                              policy_operator=policy_operator,
                              value_operator=value_operator,
                              policy_distribution_class=policy_distribution_class,
                              policy_interaction_mode="random",
                              **kwargs
                              )


def make_sac_model(
        action_spec: TensorSpec,
        in_keys: Optional[Iterable[str]] = None,
        actor_net_kwargs=None,
        qvalue_net_kwargs=None,
        value_net_kwargs=None,
        double_qvalue=True,
        **kwargs) -> Union[
    Tuple[Actor, ProbabilisticOperator, ProbabilisticOperator],
    Tuple[Actor, ProbabilisticOperator, ProbabilisticOperator, ProbabilisticOperator]
]:
    """
    Actor, Q-value and value model constructor helper function for SAC.
    Follows default parameters proposed in SAC original paper: https://arxiv.org/pdf/1801.01290.pdf.
    Other configurations can easily be implemented by modifying this function at will.

    Args:
        action_spec (TensorSpec): action_spec descriptor
        in_keys (Iterable[str], optional):
            default: ["observation_vector"]
        **kwargs: kwargs to be passed to the ActorCriticOperator.

    Returns: A joined ActorCriticOperator.

    """
    if actor_net_kwargs is None:
        actor_net_kwargs = {}
    if value_net_kwargs is None:
        value_net_kwargs = {}
    if qvalue_net_kwargs is None:
        qvalue_net_kwargs = {}

    if in_keys is None:
        in_keys = ["observation_vector"]

    actor_net_kwargs_default = {
        'num_cells': [256, 256],
        'out_features': 2 * action_spec.shape[-1],
        'activation_class': nn.ELU,
    }
    actor_net_kwargs_default.update(actor_net_kwargs)
    actor_net = MLP(
        **actor_net_kwargs_default
    )

    qvalue_net_kwargs_default = {
        'num_cells': [256, 256],
        'out_features': 1,
        'activation_class': nn.ELU,
    }
    qvalue_net_kwargs_default.update(qvalue_net_kwargs)
    qvalue_net = MLP(
        **qvalue_net_kwargs_default,
    )
    if double_qvalue:
        qvalue_net_bis = MLP(
            **qvalue_net_kwargs_default
        )

    value_net_kwargs_default = {
        'num_cells': [256, 256],
        'out_features': 1,
        'activation_class': nn.ELU,
    }
    value_net_kwargs_default.update(value_net_kwargs)
    value_net = MLP(
        **value_net_kwargs_default,
    )

    value_spec = UnboundedContinuousTensorSpec()

    actor = Actor(
        action_spec=action_spec,
        in_keys=in_keys,
        mapping_operator=actor_net,
        distribution_class=TanhNormal,
        default_interaction_mode="random",
    )
    qvalue = ProbabilisticOperator(
        spec=value_spec,
        in_keys=["action"] + in_keys,
        out_keys=["state_action_value"],
        mapping_operator=qvalue_net,
        distribution_class=Delta
    )
    if double_qvalue:
        qvalue_bis = ProbabilisticOperator(
            spec=value_spec,
            in_keys=["action"] + in_keys,
            out_keys=["state_action_value"],
            mapping_operator=qvalue_net_bis,
            distribution_class=Delta
        )
    value = ProbabilisticOperator(
        spec=value_spec,
        in_keys=in_keys,
        out_keys=["state_value"],
        mapping_operator=value_net,
        distribution_class=Delta
    )
    if double_qvalue:
        return actor, qvalue, qvalue_bis, value
    return actor, qvalue, value
