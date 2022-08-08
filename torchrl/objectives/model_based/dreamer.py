# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import torch
import torch.distributions as d
import torch.nn as nn
from torch.distributions import kl_divergence

from torchrl.data import TensorDict
from torchrl.objectives.utils import LogLikelihood


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
        lambda_kl: float = 1.0,
        lambda_reco: float = 1.0,
        lambda_reward: float = 1.0,
        reco_loss: nn.Module = LogLikelihood(reduction="none"),
        reward_loss: nn.Module = LogLikelihood(),
        free_nats: int = 3,
    ):
        super().__init__()
        self.reco_loss = reco_loss
        self.reward_loss = reward_loss
        self.lambda_kl = lambda_kl
        self.lambda_reco = lambda_reco
        self.lambda_reward = lambda_reward
        self.free_nats = free_nats

    def forward(self, tensordict: TensorDict) -> torch.Tensor:
        kl_loss = self.kl_loss(
            tensordict["prior_means"],
            tensordict["prior_stds"],
            tensordict["posterior_means"],
            tensordict["posterior_stds"],
        )
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
        return (
            self.lambda_kl * kl_loss
            + self.lambda_reco * reco_loss
            + self.lambda_reward * reward_loss
        )

    def kl_loss(self, prior_mean, prior_std, posterior_mean, posterior_std):
        flat_prior = d.Normal(prior_mean, prior_std)
        flat_posterior = d.Normal(posterior_mean, posterior_std)
        kl = kl_divergence(flat_posterior, flat_prior)
        kl = kl.clamp(min=self.free_nats).mean()
        return kl
