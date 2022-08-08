# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import torch
import torch.nn as nn
import torch.distributions as d
from torch.distributions import kl_divergence
from torchrl.data import TensorDict


class DreamerModelLoss(nn.Module):
    """
    Dreamer Model Loss
    Computes the loss of the dreamer model given a tensordict that contains, prior and posterior means and stds,
    the observation, the reconstruction observation, the reward and the predicted reward.

    Args:
        reco_loss (nn.Module): loss function for the reconstruction of the observation
        reward_loss (nn.Module): loss function for the prediction of the reward

    """
    def __init__(
        self,
        reco_loss: nn.Module = nn.MSELoss(reduction="none"),
        reward_loss: nn.Module = nn.MSELoss(),
    ):
        super().__init__()
        self.reco_loss = reco_loss
        self.reward_loss = reward_loss

    def forward(self, tensordict:TensorDict)-> torch.Tensor:
        flat_prior = d.Normal(
            tensordict.get("prior_means"), tensordict.get("prior_stds")
        )
        flat_posterior = d.Normal(
            tensordict.get("posterior_means"), tensordict.get("posterior_stds")
        )
        kl_loss = kl_divergence(flat_posterior, flat_prior).mean()
        reco_loss = (
            0.5
            * self.reco_loss(
                tensordict.get("observation"),
                tensordict.get("reco_observation"),
            )
            .mean(dim=[0, 1])
            .sum()
        )
        reward_loss = 0.5 * self.reward_loss(
            tensordict.get("reward"), tensordict.get("predicted_reward")
        )
        return kl_loss + reco_loss + reward_loss
