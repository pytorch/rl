# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from turtle import forward
import torch
import torch.distributions as d
import torch.nn as nn
from torch.distributions import kl_divergence

from torchrl.data import TensorDict
from torchrl.objectives.costs.common import LossModule
from torchrl.objectives.costs.utils import LogLikelihood
from torchrl.objectives.returns.functional import vec_td_lambda_return_estimate


class DreamerModelLoss(LossModule):
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
        loss = (
            self.lambda_kl * kl_loss
            + self.lambda_reco * reco_loss
            + self.lambda_reward * reward_loss
        )
        return TensorDict(
            {
                "loss": loss,
                "loss_kl": kl_loss,
                "loss_reco": reco_loss,
                "loss_reward": reward_loss,
            },
            [],
        )

    def kl_loss(self, prior_mean, prior_std, posterior_mean, posterior_std):
        flat_prior = d.Normal(prior_mean, prior_std)
        flat_posterior = d.Normal(posterior_mean, posterior_std)
        kl = kl_divergence(flat_posterior, flat_prior)
        kl = kl.clamp(min=self.free_nats).mean()
        return kl



class DreamerBehaviourLoss(LossModule):
    def __init__(self, value_loss: nn.Module = nn.MSELoss(), gamma=0.99, lmbda=0.95):
        super().__init__()
        self.value_loss = value_loss
        self.gamma = gamma
        self.lmbda = lmbda
    def forward(self, tensordict: TensorDict) -> torch.Tensor:
        reward = tensordict.get("predicted_reward")
        value = tensordict.get("predicted_value")
        lambda_target = self.lambda_target(reward, value)
        value_loss = 0.5*self.value_loss(value, lambda_target)
        actor_loss = -lambda_target.mean()
        return TensorDict(
            {
                "loss_value": value_loss,
                "loss_actor": actor_loss,
                "loss_lambda_target": lambda_target,
            }
        )
    def lambda_target(self, reward, value):
        done = torch.zeros(reward.batch_size).bool().to(reward.device)
        return vec_td_lambda_return_estimate((self.gamma, self.lmbda, value, reward, done))
