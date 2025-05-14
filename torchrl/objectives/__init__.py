# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from torchrl.objectives.a2c import A2CLoss
from torchrl.objectives.common import LossModule
from torchrl.objectives.cql import CQLLoss, DiscreteCQLLoss
from torchrl.objectives.crossq import CrossQLoss
from torchrl.objectives.ddpg import DDPGLoss
from torchrl.objectives.decision_transformer import DTLoss, OnlineDTLoss
from torchrl.objectives.dqn import DistributionalDQNLoss, DQNLoss
from torchrl.objectives.dreamer import (
    DreamerActorLoss,
    DreamerModelLoss,
    DreamerValueLoss,
)
from torchrl.objectives.gail import GAILLoss
from torchrl.objectives.iql import DiscreteIQLLoss, IQLLoss
from torchrl.objectives.multiagent import QMixerLoss
from torchrl.objectives.ppo import ClipPPOLoss, KLPENPPOLoss, PPOLoss
from torchrl.objectives.redq import REDQLoss
from torchrl.objectives.reinforce import ReinforceLoss
from torchrl.objectives.sac import DiscreteSACLoss, SACLoss
from torchrl.objectives.td3 import TD3Loss
from torchrl.objectives.td3_bc import TD3BCLoss
from torchrl.objectives.utils import (
    default_value_kwargs,
    distance_loss,
    group_optimizers,
    HardUpdate,
    hold_out_net,
    hold_out_params,
    next_state_value,
    SoftUpdate,
    TargetNetUpdater,
    ValueEstimators,
)

__all__ = [
    "A2CLoss",
    "CQLLoss",
    "ClipPPOLoss",
    "CrossQLoss",
    "DDPGLoss",
    "DQNLoss",
    "DTLoss",
    "DiscreteCQLLoss",
    "DiscreteIQLLoss",
    "DiscreteSACLoss",
    "DistributionalDQNLoss",
    "DreamerActorLoss",
    "DreamerModelLoss",
    "DreamerValueLoss",
    "GAILLoss",
    "HardUpdate",
    "IQLLoss",
    "KLPENPPOLoss",
    "LossModule",
    "OnlineDTLoss",
    "PPOLoss",
    "QMixerLoss",
    "REDQLoss",
    "ReinforceLoss",
    "SACLoss",
    "SoftUpdate",
    "TD3BCLoss",
    "TD3Loss",
    "TargetNetUpdater",
    "ValueEstimators",
    "add_random_module",
    "default_value_kwargs",
    "distance_loss",
    "group_optimizers",
    "hold_out_net",
    "hold_out_params",
    "next_state_value",
]
