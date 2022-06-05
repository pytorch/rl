from argparse import Namespace
from typing import Optional, Sequence

import torch
from torch import nn

from torchrl.data import DEVICE_TYPING
from torchrl.envs.common import _EnvClass
from torchrl.envs.utils import set_exploration_mode
from torchrl.modules import NoisyLinear, DdpgCnnActor, DdpgCnnQNet, TanhNormal, \
    ValueOperator, ProbabilisticActor, TensorDictModule, \
    NormalParamWrapper, MLP, ActorCriticOperator
from torchrl.trainers.helpers.models import ACTIVATIONS


def make_redq_model_state(
    proof_environment: _EnvClass,
    args: Namespace,
    device: DEVICE_TYPING = "cpu",
    in_keys: Optional[Sequence[str]] = None,
    actor_net_kwargs=None,
    qvalue_net_kwargs=None,
    observation_key=None,
    **kwargs,
) -> nn.ModuleList:

    tanh_loc = args.tanh_loc
    default_policy_scale = args.default_policy_scale
    gSDE = args.gSDE

    action_spec = proof_environment.action_spec

    if actor_net_kwargs is None:
        actor_net_kwargs = {}
    if qvalue_net_kwargs is None:
        qvalue_net_kwargs = {}

    linear_layer_class = torch.nn.Linear if not args.noisy else NoisyLinear

    out_features_actor = (2 - gSDE) * action_spec.shape[-1]
    if in_keys is None:
        in_keys_actor = ["observation_vector"]
    else:
        in_keys_actor = in_keys

    actor_net_kwargs_default = {
        "num_cells": [args.actor_cells, args.actor_cells],
        "out_features": out_features_actor,
        "activation_class": ACTIVATIONS[args.activation],
    }
    actor_net_kwargs_default.update(actor_net_kwargs)
    actor_net = MLP(**actor_net_kwargs_default)
    out_keys_actor = ["param"]

    qvalue_net_kwargs_default = {
        "num_cells": [args.qvalue_cells, args.qvalue_cells],
        "out_features": 1,
        "activation_class": ACTIVATIONS[args.activation],
    }
    qvalue_net_kwargs_default.update(qvalue_net_kwargs)
    qvalue_net = MLP(
        **qvalue_net_kwargs_default,
    )
    in_keys_qvalue = in_keys_actor + ["action"]

    dist_class = TanhNormal
    dist_kwargs = {
        "min": action_spec.space.minimum,
        "max": action_spec.space.maximum,
        "tanh_loc": tanh_loc,
    }

    actor_net = NormalParamWrapper(
        actor_net,
        scale_mapping=f"biased_softplus_{default_policy_scale}",
        scale_lb=args.scale_lb,
    )
    actor_module = TensorDictModule(
        actor_net,
        in_keys=in_keys_actor,
        out_keys=["loc", "scale"] + out_keys_actor[1:],
    )


    actor = ProbabilisticActor(
        spec=action_spec,
        dist_param_keys=["loc", "scale"],
        module=actor_module,
        distribution_class=dist_class,
        distribution_kwargs=dist_kwargs,
        default_interaction_mode="random",
        return_log_prob=True,
    )
    qvalue = ValueOperator(
        in_keys=in_keys_qvalue,
        module=qvalue_net,
    )
    model = nn.ModuleList([actor, qvalue]).to(device)

    # init nets
    with torch.no_grad(), set_exploration_mode("random"):
        td = proof_environment.rollout(1000)
        td = td.to(device)
        for net in model:
            net(td)
    del td
    return model

def make_redq_model_pixels(
    proof_environment: _EnvClass,
    args: Namespace,
    device: DEVICE_TYPING = "cpu",
    in_keys: Optional[Sequence[str]] = None,
    actor_net_kwargs=None,
    qvalue_net_kwargs=None,
    observation_key=None,
    **kwargs,
) -> nn.ModuleList:
    tanh_loc = args.tanh_loc
    default_policy_scale = args.default_policy_scale
    gSDE = args.gSDE

    action_spec = proof_environment.action_spec

    if actor_net_kwargs is None:
        actor_net_kwargs = {}
    if qvalue_net_kwargs is None:
        qvalue_net_kwargs = {}

    linear_layer_class = torch.nn.Linear if not args.noisy else NoisyLinear

    out_features_actor = (2 - gSDE) * action_spec.shape[-1]
    if in_keys is None:
        in_keys_actor = ["pixels"]
    else:
        in_keys_actor = in_keys
    actor_net_kwargs_default = {
        "mlp_net_kwargs": {
            "layer_class": linear_layer_class,
            "activation_class": ACTIVATIONS[args.activation],
        },
        "conv_net_kwargs": {"activation_class": ACTIVATIONS[args.activation]},
    }
    actor_net_kwargs_default.update(actor_net_kwargs)
    actor_net = DdpgCnnActor(out_features_actor, **actor_net_kwargs_default)
    out_keys_actor = ["param", "hidden"]

    value_net_default_kwargs = {
        "mlp_net_kwargs": {
            "layer_class": linear_layer_class,
            "activation_class": ACTIVATIONS[args.activation],
        },
        "conv_net_kwargs": {"activation_class": ACTIVATIONS[args.activation]},
    }
    value_net_default_kwargs.update(qvalue_net_kwargs)

    in_keys_qvalue = ["pixels", "action"]
    qvalue_net = DdpgCnnQNet(**value_net_default_kwargs)

    dist_class = TanhNormal
    dist_kwargs = {
        "min": action_spec.space.minimum,
        "max": action_spec.space.maximum,
        "tanh_loc": tanh_loc,
    }

    actor_net = NormalParamWrapper(
        actor_net,
        scale_mapping=f"biased_softplus_{default_policy_scale}",
        scale_lb=args.scale_lb,
    )
    actor_module = TensorDictModule(
        actor_net,
        in_keys=in_keys_actor,
        out_keys=["loc", "scale"] + out_keys_actor[1:],
    )

    actor = ProbabilisticActor(
        spec=action_spec,
        dist_param_keys=["loc", "scale"],
        module=actor_module,
        distribution_class=dist_class,
        distribution_kwargs=dist_kwargs,
        default_interaction_mode="random",
        return_log_prob=True,
    )
    qvalue = ValueOperator(
        in_keys=in_keys_qvalue,
        module=qvalue_net,
    )
    model = nn.ModuleList([actor, qvalue]).to(device)

    # init nets
    with torch.no_grad(), set_exploration_mode("random"):
        td = proof_environment.rollout(1000)
        td = td.to(device)
        for net in model:
            net(td)
    del td
    return model

def make_redq_model_pixels_shared(
    proof_environment: _EnvClass,
    args: Namespace,
    device: DEVICE_TYPING = "cpu",
    in_keys: Optional[Sequence[str]] = None,
    actor_net_kwargs=None,
    qvalue_net_kwargs=None,
    observation_key=None,
    **kwargs,
) -> nn.ModuleList:
    tanh_loc = args.tanh_loc
    default_policy_scale = args.default_policy_scale
    gSDE = args.gSDE

    action_spec = proof_environment.action_spec

    if actor_net_kwargs is None:
        actor_net_kwargs = {}
    if qvalue_net_kwargs is None:
        qvalue_net_kwargs = {}

    linear_layer_class = torch.nn.Linear if not args.noisy else NoisyLinear

    out_features_actor = (2 - gSDE) * action_spec.shape[-1]
    actor_net_kwargs_default = {
        "mlp_net_kwargs": {
            "layer_class": linear_layer_class,
            "activation_class": ACTIVATIONS[args.activation],
        },
        "conv_net_kwargs": {"activation_class": ACTIVATIONS[args.activation]},
        "use_avg_pooling": True,
    }
    actor_net_kwargs_default.update(actor_net_kwargs)
    actor_net = DdpgCnnActor(out_features_actor, **actor_net_kwargs_default)

    value_net_default_kwargs = {
        "mlp_net_kwargs": {
            "layer_class": linear_layer_class,
            "activation_class": ACTIVATIONS[args.activation],
        },
        "conv_net_kwargs": {"activation_class": ACTIVATIONS[args.activation]},
        "use_avg_pooling": True,
    }
    value_net_default_kwargs.update(qvalue_net_kwargs)

    qvalue_net = DdpgCnnQNet(**value_net_default_kwargs)

    # a bit of surgery
    common_mapper = actor_net.convnet
    common_net = TensorDictModule(
        common_mapper,
        in_keys=["pixels"],
        out_keys=["hidden"],
    )
    actor_mapper = NormalParamWrapper(actor_net.mlp)
    dist_class = TanhNormal
    dist_kwargs = {
        "min": action_spec.space.minimum,
        "max": action_spec.space.maximum,
        "tanh_loc": tanh_loc,
    }
    actor_subnet = TensorDictModule(
        actor_mapper,
        in_keys=["hidden"],
        out_keys=["loc", "scale"]
    )
    actor_subnet = ProbabilisticActor(
        spec=action_spec,
        dist_param_keys=["loc", "scale"],
        module=actor_subnet,
        distribution_class=dist_class,
        distribution_kwargs=dist_kwargs,
        default_interaction_mode="random",
        return_log_prob=True,
    )
    qvalue_mapper = qvalue_net.mlp
    qvalue_subnet = ValueOperator(
        qvalue_mapper,
        in_keys=["hidden"],
    )

    model = ActorCriticOperator(
        common_net,
        actor_subnet,
        qvalue_subnet,
    ).to(device)

    # init nets
    with torch.no_grad(), set_exploration_mode("random"):
        td = proof_environment.rollout(1000)
        td = td.to(device)
        model(td)
    del td
    return model
