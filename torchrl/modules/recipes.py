import torch

from torchrl.envs.common import _EnvWrapper
from torchrl.modules.distributions import (
    Delta,
    TanhNormal,
    TanhDelta,
    Categorical,
)
from . import ActorCriticOperator
from .models.models import DuelingCnnDQNet, DdpgCnnActor, DdpgCnnQNet, DdpgMlpQNet, DdpgMlpActor, MLP
from .probabilistic_operators import QValueActor, DistributionalQValueActor, Actor, ProbabilisticOperator

DISTRIBUTIONS = {
    "delta": Delta,
    "tanh-normal": TanhNormal,
    "categorical": Categorical,
    "tanh-delta": TanhDelta,
}

__all__ = ["make_dqn_actor", "make_ddpg_actor"]


def make_dqn_actor(
        env_specs,
        net_class=DuelingCnnDQNet,
        net_kwargs={},
        atoms=51,  # for distributional dqn
        vmin=-3,
        vmax=3,
        actor_kwargs={},
        in_key="observation_pixels",
):
    default_net_kwargs = {"cnn_kwargs": {"depth": 3, "num_cells": 64, "out_features": 64}, }
    default_net_kwargs.update(net_kwargs)

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
        env_specs,
        from_pixels=True,
        actor_net_kwargs={},
        value_net_kwargs={},
        atoms=0,  # for distributional dqn
        vmin=-3,
        vmax=3,
        actor_kwargs={},
        value_kwargs={},
):
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


def make_tensor_dict(env: _EnvWrapper, actor=None, ):
    """Returns a zeroed-tensordict with fields matching those required for a full step
    (action selection and environment step) in the environment
    
    """
    with torch.no_grad():
        tensor_dict = env.reset()
        if actor is not None:
            tensor_dict = tensor_dict.unsqueeze(0)
            tensor_dict = actor(tensor_dict.to(next(actor.parameters()).device))
            tensor_dict = tensor_dict.squeeze(0)
        else:
            tensor_dict.set("action", env.action_spec.rand(), inplace=False)
        tensor_dict = env.step(tensor_dict.to("cpu"))
        return tensor_dict


def make_actor_critic_model(spec, in_keys=["observation_vector"], **kwargs):
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
