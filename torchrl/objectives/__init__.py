# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from .a2c import A2CLoss
from .common import LossModule
from .cql import CQLLoss, DiscreteCQLLoss
from .ddpg import DDPGLoss
from .decision_transformer import DTLoss, OnlineDTLoss
from .dqn import DistributionalDQNLoss, DQNLoss
from .dreamer import DreamerActorLoss, DreamerModelLoss, DreamerValueLoss
from .iql import IQLLoss
from .multiagent import QMixerLoss
from .ppo import ClipPPOLoss, KLPENPPOLoss, PPOLoss
from .redq import REDQLoss
from .reinforce import ReinforceLoss
from .sac import DiscreteSACLoss, SACLoss
from .td3 import TD3Loss
from .utils import (
    default_value_kwargs,
    distance_loss,
    HardUpdate,
    hold_out_net,
    hold_out_params,
    next_state_value,
    SoftUpdate,
    ValueEstimators,
)

# from .value import bellman_max, c_val, dv_val, vtrace, GAE, TDLambdaEstimate, TDEstimate
