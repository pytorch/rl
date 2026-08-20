# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import sys

from torchrl.modules.tensordict_module.common import DistributionalDQNnet

from . import dreamer_v3
from .act import ACTModel
from .batchrenorm import BatchRenorm1d
from .cross_group_critic import CrossCriticGroupSpec, CrossGroupCritic

from .decision_transformer import DecisionTransformer
from .dreamer_v3 import (
    DreamerV3MLP,
    RSSMPosteriorV3,
    RSSMPriorV3,
    RSSMRolloutV3,
    SymExpTwoHot,
)
from .exploration import (
    ConsistentDropout,
    ConsistentDropoutModule,
    NoisyLazyLinear,
    NoisyLinear,
    reset_noise,
)
from .gp import GPWorldModel
from .llm import GPT2RewardModel
from .model_based import (
    DreamerActor,
    ObsDecoder,
    ObsEncoder,
    RSSMPosterior,
    RSSMPrior,
    RSSMRollout,
)
from .models import (
    Conv2dNet,
    Conv3dNet,
    ConvNet,
    DdpgCnnActor,
    DdpgCnnQNet,
    DdpgMlpActor,
    DdpgMlpQNet,
    DTActor,
    DuelingCnnDQNet,
    DuelingMlpDQNet,
    MLP,
    OnlineDTActor,
)
from .multiagent import (
    MultiAgentConvNet,
    MultiAgentMLP,
    MultiAgentNetBase,
    QMixer,
    VDNMixer,
)
from .rbf_controller import RBFController
from .utils import Squeeze2dLayer, SqueezeLayer


# Preserve the module path shipped in TorchRL 0.13 without keeping a duplicate
# compatibility file. New code should import from ``dreamer_v3``.
sys.modules[f"{__name__}.model_based_v3"] = dreamer_v3

__all__ = [
    "ACTModel",
    "BatchRenorm1d",
    "CrossCriticGroupSpec",
    "CrossGroupCritic",
    "ConsistentDropout",
    "ConsistentDropoutModule",
    "Conv2dNet",
    "Conv3dNet",
    "ConvNet",
    "DdpgCnnActor",
    "DdpgCnnQNet",
    "DdpgMlpActor",
    "DdpgMlpQNet",
    "DecisionTransformer",
    "DistributionalDQNnet",
    "DreamerActor",
    "DreamerV3MLP",
    "DTActor",
    "DuelingCnnDQNet",
    "DuelingMlpDQNet",
    "GPT2RewardModel",
    "GPWorldModel",
    "MLP",
    "MultiAgentConvNet",
    "MultiAgentMLP",
    "MultiAgentNetBase",
    "NoisyLazyLinear",
    "NoisyLinear",
    "ObsDecoder",
    "ObsEncoder",
    "OnlineDTActor",
    "QMixer",
    "RBFController",
    "RSSMPosterior",
    "RSSMPosteriorV3",
    "RSSMPrior",
    "RSSMPriorV3",
    "RSSMRollout",
    "RSSMRolloutV3",
    "Squeeze2dLayer",
    "SqueezeLayer",
    "SymExpTwoHot",
    "VDNMixer",
    "reset_noise",
]
