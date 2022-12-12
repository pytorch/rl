# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import itertools
from dataclasses import dataclass
from typing import Optional, Sequence

import torch
from torch import distributions as d, nn

from torchrl.data import (
    CompositeSpec,
    DiscreteTensorSpec,
    NdUnboundedContinuousTensorSpec,
)
from torchrl.data.utils import DEVICE_TYPING
from torchrl.envs import TensorDictPrimer, TransformedEnv
from torchrl.envs.common import EnvBase
from torchrl.envs.model_based.dreamer import DreamerEnv
from torchrl.envs.utils import set_exploration_mode
from torchrl.modules import (
    ActorValueOperator,
    NoisyLinear,
    NormalParamWrapper,
    SafeModule,
    SafeProbabilisticModule,
    SafeSequential,
)
from torchrl.modules.distributions import (
    Delta,
    OneHotCategorical,
    TanhDelta,
    TanhNormal,
    TruncatedNormal,
)
from torchrl.modules.distributions.continuous import SafeTanhTransform
from torchrl.modules.models.exploration import LazygSDEModule
from torchrl.modules.models.model_based import (
    DreamerActor,
    ObsDecoder,
    ObsEncoder,
    RSSMPosterior,
    RSSMPrior,
    RSSMRollout,
)
from torchrl.modules.models.models import (
    ConvNet,
    DdpgCnnActor,
    DdpgCnnQNet,
    DdpgMlpActor,
    DdpgMlpQNet,
    DuelingCnnDQNet,
    DuelingMlpDQNet,
    LSTMNet,
    MLP,
)
from torchrl.modules.tensordict_module import (
    Actor,
    DistributionalQValueActor,
    QValueActor,
)
from torchrl.modules.tensordict_module.actors import (
    ActorCriticWrapper,
    ProbabilisticActor,
    ValueOperator,
)
from torchrl.modules.tensordict_module.world_models import WorldModelWrapper
from torchrl.trainers.helpers import transformed_env_constructor

DISTRIBUTIONS = {
    "delta": Delta,
    "tanh-normal": TanhNormal,
    "categorical": OneHotCategorical,
    "tanh-delta": TanhDelta,
}

ACTIVATIONS = {
    "elu": nn.ELU,
    "tanh": nn.Tanh,
    "relu": nn.ReLU,
}


def make_ppo_model(
    proof_environment: EnvBase,
    device,
    gSDE=False,
    distribution="tanh_normal",
    tanh_loc=False,
    shared_mapping=False,
    lstm=False,
    default_policy_scale=1.0,
) -> ActorValueOperator:

    specs = proof_environment.specs
    action_spec = specs["action_spec"]

    if proof_environment.from_pixels:
        in_keys_actor = ["pixels"]
        in_keys_critic = ["pixels"]
    else:
        in_keys_actor = ["observation_vector"]
        in_keys_critic = ["observation_vector"]
    out_keys = ["action"]

    if action_spec.domain == "continuous":
        dist_in_keys = ["loc", "scale"]
        out_features = (2 - gSDE) * action_spec.shape[-1]
        if distribution == "tanh_normal":
            policy_distribution_kwargs = {
                "min": action_spec.space.minimum,
                "max": action_spec.space.maximum,
                "tanh_loc": tanh_loc,
            }
            policy_distribution_class = TanhNormal
        elif distribution == "truncated_normal":
            policy_distribution_kwargs = {
                "min": action_spec.space.minimum,
                "max": action_spec.space.maximum,
                "tanh_loc": tanh_loc,
            }
            policy_distribution_class = TruncatedNormal
    elif action_spec.domain == "discrete":
        out_features = action_spec.shape[-1]
        policy_distribution_kwargs = {}
        policy_distribution_class = OneHotCategorical
        dist_in_keys = ["logits"]
    else:
        raise NotImplementedError(
            f"actions with domain {action_spec.domain} are not supported"
        )

    if shared_mapping:
        hidden_features = 300
        if proof_environment.from_pixels:
            if in_keys_actor is None:
                in_keys_actor = ["pixels"]
            common_module = ConvNet(
                bias_last_layer=True,
                depth=None,
                num_cells=[32, 64, 64],
                kernel_sizes=[8, 4, 3],
                strides=[4, 2, 1],
            )
        else:
            if lstm:
                raise NotImplementedError(
                    "lstm not yet compatible with shared mapping for PPO"
                )
            common_module = MLP(
                num_cells=[
                    400,
                ],
                out_features=hidden_features,
                activate_last_layer=True,
            )
        common_operator = SafeModule(
            spec=None,
            module=common_module,
            in_keys=in_keys_actor,
            out_keys=["hidden"],
        )

        policy_net = MLP(
            num_cells=[200],
            out_features=out_features,
        )

        shared_out_keys = ["hidden"]
        if not gSDE:
            if action_spec.domain == "continuous":
                policy_net = NormalParamWrapper(
                    policy_net,
                    scale_mapping=f"biased_softplus_{default_policy_scale}",
                )
            actor_module = SafeModule(
                policy_net, in_keys=shared_out_keys, out_keys=dist_in_keys
            )
        else:
            gSDE_state_key = "hidden"
            actor_module = SafeModule(
                policy_net,
                in_keys=shared_out_keys,
                out_keys=["action"],  # will be overwritten
            )

            if action_spec.domain == "continuous":
                min = action_spec.space.minimum
                max = action_spec.space.maximum
                transform = SafeTanhTransform()
                if (min != -1).any() or (max != 1).any():
                    transform = d.ComposeTransform(
                        transform,
                        d.AffineTransform(loc=(max + min) / 2, scale=(max - min) / 2),
                    )
            else:
                raise RuntimeError("cannot use gSDE with discrete actions")

            actor_module = SafeSequential(
                actor_module,
                SafeModule(
                    LazygSDEModule(transform=transform),
                    in_keys=["action", gSDE_state_key, "_eps_gSD"],
                    out_keys=["loc", "scale", "action", "_eps_gSDE"],
                ),
            )

        policy_operator = ProbabilisticActor(
            spec=CompositeSpec(action=action_spec),
            module=actor_module,
            dist_in_keys=dist_in_keys,
            default_interaction_mode="random",
            distribution_class=policy_distribution_class,
            distribution_kwargs=policy_distribution_kwargs,
            return_log_prob=True,
        )
        value_net = MLP(
            num_cells=[200],
            out_features=1,
        )
        value_operator = ValueOperator(value_net, in_keys=shared_out_keys)
        actor_value = ActorValueOperator(
            common_operator=common_operator,
            policy_operator=policy_operator,
            value_operator=value_operator,
        ).to(device)
    else:
        if proof_environment.from_pixels:
            raise RuntimeError(
                "PPO learnt from pixels require the shared_mapping to be set to True."
            )
        if lstm:
            policy_net = LSTMNet(
                out_features=out_features,
                lstm_kwargs={"input_size": 256, "hidden_size": 256},
                mlp_kwargs={"num_cells": [256, 256], "out_features": 256},
            )
            in_keys_actor += ["hidden0", "hidden1"]
            out_keys += ["hidden0", "hidden1", ("next", "hidden0"), ("next", "hidden1")]
        else:
            policy_net = MLP(
                num_cells=[400, 300],
                out_features=out_features,
            )

        if not gSDE:
            if action_spec.domain == "continuous":
                policy_net = NormalParamWrapper(
                    policy_net,
                    scale_mapping=f"biased_softplus_{default_policy_scale}",
                )
            actor_module = SafeModule(
                policy_net, in_keys=in_keys_actor, out_keys=dist_in_keys
            )
        else:
            in_keys = in_keys_actor
            gSDE_state_key = in_keys_actor[0]
            actor_module = SafeModule(
                policy_net,
                in_keys=in_keys,
                out_keys=["action"],  # will be overwritten
            )

            if action_spec.domain == "continuous":
                min = action_spec.space.minimum
                max = action_spec.space.maximum
                transform = SafeTanhTransform()
                if (min != -1).any() or (max != 1).any():
                    transform = d.ComposeTransform(
                        transform,
                        d.AffineTransform(loc=(max + min) / 2, scale=(max - min) / 2),
                    )
            else:
                raise RuntimeError("cannot use gSDE with discrete actions")

            actor_module = SafeSequential(
                actor_module,
                SafeModule(
                    LazygSDEModule(transform=transform),
                    in_keys=["action", gSDE_state_key, "_eps_gSDE"],
                    out_keys=["loc", "scale", "action", "_eps_gSDE"],
                ),
            )

        policy_po = ProbabilisticActor(
            actor_module,
            spec=action_spec,
            dist_in_keys=dist_in_keys,
            distribution_class=policy_distribution_class,
            distribution_kwargs=policy_distribution_kwargs,
            return_log_prob=True,
            default_interaction_mode="random",
        )

        value_net = MLP(
            num_cells=[400, 300],
            out_features=1,
        )
        value_po = ValueOperator(
            value_net,
            in_keys=in_keys_critic,
        )
        actor_value = ActorCriticWrapper(policy_po, value_po).to(device)

    with torch.no_grad(), set_exploration_mode("random"):
        td = proof_environment.rollout(max_steps=1000)
        td_device = td.to(device)
        td_device = actor_value(td_device)  # for init

    model = nn.ModuleDict({
        "policy": actor_value.get_policy_operator(), # For the collector
        "actor": actor_value.get_policy_operator(),
        "critic": actor_value.get_value_operator(),
    })

    return model
