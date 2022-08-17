# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from turtle import forward

import torch
import torch.nn as nn

from torchrl.data import TensorDict
from torchrl.objectives.costs.common import LossModule
from torchrl.objectives.costs.utils import hold_out_net
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
        world_model: nn.Module,
        cfg: "DictConfig",
        lambda_kl: float = 1.0,
        lambda_reco: float = 1.0,
        lambda_reward: float = 1.0,
        reco_loss: nn.Module = nn.MSELoss(reduction="none"),
        reward_loss: nn.Module = nn.MSELoss(),
        free_nats: int = 3,
    ):
        super().__init__()
        self.world_model = world_model
        self.cfg = cfg
        self.reco_loss = reco_loss
        self.reward_loss = reward_loss
        self.lambda_kl = lambda_kl
        self.lambda_reco = lambda_reco
        self.lambda_reward = lambda_reward
        self.free_nats = free_nats

    def forward(self, tensordict: TensorDict) -> torch.Tensor:
        tensordict.batch_size = [tensordict.shape[0]]
        tensordict["prior_state"] = torch.zeros(
            (tensordict.batch_size[0], self.cfg.state_dim)
        )
        tensordict["belief"] = torch.zeros(
            (tensordict.batch_size[0], self.cfg.rssm_hidden_dim)
        )
        tensordict = self.world_model(tensordict)
        # compute model loss
        kl_loss = self.kl_loss(
            tensordict["prior_means"],
            tensordict["prior_stds"],
            tensordict["posterior_means"],
            tensordict["posterior_stds"],
        )
        reco_loss = (
            0.5
            * self.reco_loss(
                tensordict.get("pixels"),
                tensordict.get("reco_pixels"),
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
        return (
            TensorDict(
                {
                    "loss_world_model": loss,
                    "loss_kl": kl_loss,
                    "loss_reco": reco_loss,
                    "loss_reward": reward_loss,
                },
                [],
            ),
            tensordict,
        )

    def kl_loss(self, prior_mean, prior_std, posterior_mean, posterior_std):
        kl = torch.log(prior_std/posterior_std) + (posterior_std**2 + (prior_mean - posterior_mean)**2)/(2*prior_std**2) - 0.5
        kl = kl.mean().clamp(min=self.free_nats)
        return kl


class DreamerActorLoss(LossModule):
    def __init__(
        self,
        actor_model,
        value_model,
        model_based_env,
        cfg,
        gamma=0.99,
        lmbda=0.95,
    ):
        super().__init__()
        self.actor_model = actor_model
        self.value_model = value_model
        self.model_based_env = model_based_env
        self.cfg = cfg
        self.gamma = gamma
        self.lmbda = lmbda

    def forward(self, tensordict) -> torch.Tensor:
        with torch.no_grad():
            tensordict = tensordict.select("posterior_states", "next_belief")

            tensordict.batch_size = [
                tensordict.shape[0],
                tensordict.get("next_belief").shape[1],
            ]
            tensordict.rename_key("posterior_states", "prior_state")
            tensordict.rename_key("next_belief", "belief")
            tensordict = tensordict.view(-1).detach()
        with hold_out_net(self.model_based_env):
            tensordict = self.model_based_env.rollout(
                max_steps=self.cfg.imagination_horizon,
                policy=self.actor_model,
                auto_reset=False,
                tensordict=tensordict,
            )
            with hold_out_net(self.value_model):
                tensordict = self.value_model(tensordict)
        tensordict["lambda_target"] = self.lambda_target(
            tensordict.get("reward"), tensordict.get("predicted_value")
        )

        actor_loss = -tensordict.get("lambda_target").mean()
        return (
            TensorDict(
                {
                    "loss_actor": actor_loss,
                },
                batch_size=[],
            ),
            tensordict,
        )

    def lambda_target(self, reward, value):
        done = torch.zeros(reward.shape).bool().to(reward.device)
        return vec_td_lambda_return_estimate(
            self.gamma, self.lmbda, value, reward, done
        )
class DreamerValueLoss(LossModule):
    def __init__(
        self,
        value_model,
        value_loss: nn.Module = nn.MSELoss(),
    ):
        super().__init__()
        self.value_model = value_model
        self.value_loss = value_loss

    def forward(self, tensordict) -> torch.Tensor:
        with torch.no_grad():
            value_td = tensordict.clone().detach()
        value_td = self.value_model(value_td)

        value_loss = 0.5 * self.value_loss(
            value_td.get("predicted_value"), tensordict.get("lambda_target").detach()
        )

        return (
            TensorDict(
                {
                    "loss_value": value_loss,
                },
                batch_size=[],
            ),
            value_td.detach(),
        )