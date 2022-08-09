from math import sqrt

import torch
import torch.distributions as d
import torch.nn as nn
import torch.nn.functional as F

from torchrl.modules.tensordict_module import TensorDictModule, TensorDictSequence
from ..model_based import ModelBasedEnv


class DreamerEnv(ModelBasedEnv):
    def __init__(
        self,
        obs_depth=32,
        rssm_hidden=200,
        rnn_hidden_dim=200,
        state_dim=20,
        device="cpu",
        dtype=None,
        batch_size=None,
    ):
        super().__init__(
            world_model=DreamerModel(
                obs_depth=obs_depth,
                rssm_hidden=rssm_hidden,
                rnn_hidden_dim=rnn_hidden_dim,
                state_dim=state_dim,
            ),
            reward_model=TensorDictModule(
                RewardModel(),
                in_keys=[
                    "posterior_states",
                    "beliefs",
                ],
                out_keys=["predicted_reward"],
            ),
            device=device,
            dtype=dtype,
            batch_size=batch_size,
        )
