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


def make_sac_model(
    proof_environment: EnvBase,
    device: DEVICE_TYPING = "cpu",
    in_keys: Optional[Sequence[str]] = None,
    actor_net_kwargs=None,
    qvalue_net_kwargs=None,
    value_net_kwargs=None,
    observation_key=None,
    annealing_frames: int = 1000000,
    noisy: bool = False,
    ou_exploration: bool = False,
    ou_sigma: float = 0.2,
    ou_theta: float = 0.15,
    distributional: bool = False,
    atoms: int = 51,
    gSDE: bool = False,
    tanh_loc: bool = False,
    default_policy_scale: float = 1.0,
    distribution: str = "tanh_normal",
    actor_cells: int = 256,
    qvalue_cells: int = 256,
    scale_lb: float = 0.1,
    value_cells: int = 256,
    activation: str = "tanh",
    model_device: str = "",
) -> nn.ModuleList:

    tanh_loc = tanh_loc
    default_policy_scale = default_policy_scale
    gSDE = gSDE

    proof_environment.reset()
    action_spec = proof_environment.action_spec

    if actor_net_kwargs is None:
        actor_net_kwargs = {}
    if value_net_kwargs is None:
        value_net_kwargs = {}
    if qvalue_net_kwargs is None:
        qvalue_net_kwargs = {}

    if in_keys is None:
        in_keys = ["observation_vector"]

    actor_net_kwargs_default = {
        "num_cells": [actor_cells, actor_cells],
        "out_features": (2 - gSDE) * action_spec.shape[-1],
        "activation_class": ACTIVATIONS[activation],
    }
    actor_net_kwargs_default.update(actor_net_kwargs)
    actor_net = MLP(**actor_net_kwargs_default)

    qvalue_net_kwargs_default = {
        "num_cells": [qvalue_cells, qvalue_cells],
        "out_features": 1,
        "activation_class": ACTIVATIONS[activation],
    }
    qvalue_net_kwargs_default.update(qvalue_net_kwargs)
    qvalue_net = MLP(
        **qvalue_net_kwargs_default,
    )

    value_net_kwargs_default = {
        "num_cells": [value_cells, value_cells],
        "out_features": 1,
        "activation_class": ACTIVATIONS[activation],
    }
    value_net_kwargs_default.update(value_net_kwargs)
    value_net = MLP(
        **value_net_kwargs_default,
    )

    dist_class = TanhNormal
    dist_kwargs = {
        "min": action_spec.space.minimum,
        "max": action_spec.space.maximum,
        "tanh_loc": tanh_loc,
    }

    if not gSDE:
        actor_net = NormalParamWrapper(
            actor_net,
            scale_mapping=f"biased_softplus_{default_policy_scale}",
            scale_lb=scale_lb,
        )
        in_keys_actor = in_keys
        actor_module = SafeModule(
            actor_net,
            in_keys=in_keys_actor,
            out_keys=[
                "loc",
                "scale",
            ],
        )

    else:
        gSDE_state_key = in_keys[0]
        actor_module = SafeModule(
            actor_net,
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

    actor = ProbabilisticActor(
        spec=action_spec,
        dist_in_keys=["loc", "scale"],
        module=actor_module,
        distribution_class=dist_class,
        distribution_kwargs=dist_kwargs,
        default_interaction_mode="random",
        return_log_prob=False,
    )

    qvalue = ValueOperator(
        in_keys=["action"] + in_keys,
        module=qvalue_net,
    )
    value = ValueOperator(
        in_keys=in_keys,
        module=value_net,
    )
    model = nn.ModuleDict({
        "policy": actor, # For the collector
        "actor_network": actor,
        "qvalue_network": qvalue,
        "value_network": value,
    }).to(device)

    # init nets
    with torch.no_grad(), set_exploration_mode("random"):
        td = proof_environment.reset()
        td = td.to(device)
        for key, net in model.items():
            net(td)
    del td

    return model
