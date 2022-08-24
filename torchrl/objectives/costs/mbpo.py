# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn

from torchrl.data import TensorDict
from .common import LossModule


class MBPOModelLoss(LossModule):
    """
    Loss function for the MBPO algorithm.
    """

    def __init__(
        self,
        single_world_model,
        cfg,
        observation_key="observation_vector",
        lambda_obs=None,
        lambda_reward=None,
    ):
        """
        Args:
            gamma (float): discount factor
            lam (float): lambda parameter for GAE
        """
        super().__init__()

        self.convert_to_functional(
            single_world_model,
            "world_model",
            expand_dim=cfg.num_world_models_ensemble,
        )
        self.cfg = cfg
        self.observation_key = observation_key
        self.lambda_obs = lambda_obs
        self.lambda_reward = lambda_reward

    def forward(self, tensordict):
        tensordict_clone = tensordict.select(*self.world_model.in_keys)
        tensordict_expand = self.world_model(
            tensordict_clone,
            tensordict_out=TensorDict(
                {}, [self.cfg.num_world_models_ensemble, *tensordict_clone.shape]
            ),
            params=self.world_model_params,
            buffers=self.world_model_buffers,
            vmap=True,
        )
        loss_obs = self.model_loss(
            tensordict_expand[f"next_{self.observation_key}_loc"],
            tensordict_expand[f"next_{self.observation_key}_scale"],
            tensordict[f"next_{self.observation_key}"],
        )
        loss_reward = self.model_loss(
            tensordict_expand[f"reward_loc"],
            tensordict_expand[f"reward_scale"],
            tensordict[f"reward"],
        )
        if self.lambda_obs is None or self.lambda_reward is None:
            N = tensordict[f"next_{self.observation_key}"].shape[-1]
            loss_model = (N - 1) / N * loss_obs + 1 / N * loss_reward
        else:
            loss_model = self.lambda_obs * loss_obs + self.lambda_reward * loss_reward
        return TensorDict({"loss_world_model": loss_model}, [])

    def model_loss(self, mean, sigma, target):
        log_likelihood = torch.pow(mean - target, 2) / (
            2 * torch.pow(sigma, 2)
        ) + torch.log(sigma)
        return torch.sum(torch.mean(log_likelihood, dim=(1, 2)))
