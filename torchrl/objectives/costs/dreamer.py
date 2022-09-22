# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from typing import Optional

import torch

from torchrl.data import TensorDict
from torchrl.envs.utils import set_exploration_mode
from torchrl.modules import TensorDictModule
from torchrl.objectives.costs.common import LossModule
from torchrl.objectives.costs.utils import hold_out_net, distance_loss
from torchrl.objectives.returns.functional import vec_td_lambda_return_estimate
from torchrl.envs.utils import step_tensordict


class DreamerModelLoss(LossModule):
    """
    Dreamer Model Loss
    Computes the loss of the dreamer model given a tensordict that contains, prior and posterior means and stds,
    the observation, the reconstruction observation, the reward and the predicted reward.

    Args:
        TODO

    """

    def __init__(
        self,
        world_model: TensorDictModule,
        cfg: "DictConfig",
        lambda_kl: float = 1.0,
        lambda_reco: float = 1.0,
        lambda_reward: float = 1.0,
        reco_loss: Optional[str] = None,
        reward_loss: Optional[str] = None,
        free_nats: int = 3,
    ):
        super().__init__()
        self.world_model = world_model
        self.cfg = cfg
        self.reco_loss = reco_loss if reco_loss is not None else "l2"
        self.reward_loss = reward_loss if reward_loss is not None else "l2"
        self.lambda_kl = lambda_kl
        self.lambda_reco = lambda_reco
        self.lambda_reward = lambda_reward
        self.free_nats = free_nats

    def forward(self, tensordict: TensorDict) -> torch.Tensor:
        tensordict = tensordict.clone(recurse=False)

        # prepare tensordict: remove time in batch dimensions
        tensordict.batch_size = tensordict.batch_size[:1]
        # take the first tensor for prev_posterior_state and prev_belief
        tensordict["prev_posterior_state"] = tensordict["prev_posterior_state"][:, 0]
        tensordict["prev_belief"] = tensordict["prev_belief"][:, 0]

        tensordict = self.world_model(tensordict)
        # compute model loss
        kl_loss = self.kl_loss(
            tensordict.get("prior_means"),
            tensordict.get("prior_stds"),
            tensordict.get("posterior_means"),
            tensordict.get("posterior_stds"),
        )
        reco_loss = (
            distance_loss(
                tensordict.get("pixels"),
                tensordict.get("reco_pixels"),
                self.reco_loss,
            )
            .sum((-1, -2, -3))
            .mean()
        )
        reward_loss = distance_loss(
            tensordict.get("reward"),
            tensordict.get("predicted_reward"),
            self.reward_loss,
        ).mean()
        loss = (
            self.lambda_kl * kl_loss
            + self.lambda_reco * reco_loss
            + self.lambda_reward * reward_loss
        )
        return (
            TensorDict(
                {
                    "loss_world_model": loss,
                    "kl_model_loss": kl_loss,
                    "reco_model_loss": reco_loss,
                    "reward_model_loss": reward_loss,
                },
                [],
            ),
            tensordict.detach(),
        )

    def kl_loss(self, prior_mean, prior_std, posterior_mean, posterior_std):
        kl = (
            torch.log(prior_std / posterior_std)
            + (posterior_std ** 2 + (prior_mean - posterior_mean) ** 2)
            / (2 * prior_std ** 2)
            - 0.5
        )
        kl = kl.sum(-1).clamp_min(self.free_nats).mean()
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
            tensordict = tensordict.select("posterior_state", "belief", "predicted_reward")

            tensordict.batch_size = [
                tensordict.shape[0],
                tensordict.get("belief").shape[1],
            ]
            tensordict.rename_key("posterior_state", "prior_state")
            tensordict.rename_key("predicted_reward", "reward")
            tensordict = tensordict.view(-1).detach()
        with hold_out_net(self.model_based_env), set_exploration_mode("random"):
            tensordict = self.model_based_env.rollout(
                max_steps=self.cfg.imagination_horizon,
                policy=self.actor_model,
                auto_reset=False,
                tensordict=tensordict,
            )
            tensordict = step_tensordict(tensordict, keep_other=True,
                    exclude_reward=False,
                    exclude_action=False,)
            with hold_out_net(self.value_model):
                tensordict = self.value_model(tensordict)

        lambda_target = self.lambda_target(
            tensordict.get("reward"), tensordict.get("predicted_value")
        )
        tensordict = tensordict[:, :-1]
        tensordict.set("lambda_target", lambda_target)

        discount = self.gamma * torch.ones_like(lambda_target, device=tensordict.device)
        discount[:, 0] = 1
        discount = discount.cumprod(dim=1).detach()
        actor_loss = -(lambda_target * discount).mean()
        return (
            TensorDict(
                {
                    "loss_actor": actor_loss,
                },
                batch_size=[],
            ),
            tensordict.detach(),
        )

    def lambda_target(self, reward, value):
        done = torch.zeros(reward.shape, dtype=torch.bool, device=reward.device)
        return vec_td_lambda_return_estimate(
            self.gamma, self.lmbda, value[:, 1:], reward[:, :-1], done[:, :-1]
        )


class DreamerValueLoss(LossModule):
    def __init__(
        self,
        value_model,
        value_loss: Optional[str] = None,
        gamma=0.99,
    ):
        super().__init__()
        self.value_model = value_model
        self.value_loss = value_loss if value_loss is not None else "l2"
        self.gamma = gamma

    def forward(self, tensordict) -> torch.Tensor:
        tensordict = self.value_model(tensordict)
        discount = self.gamma * torch.ones_like(
            tensordict.get("lambda_target"), device=tensordict.device
        )
        discount[:, 0] = 1
        discount = discount.cumprod(dim=1).detach()
        value_loss = (
            discount
            * distance_loss(
                tensordict.get("predicted_value"),
                tensordict.get("lambda_target"),
                self.value_loss,
            )
        ).sum((-1,-2)).mean()

        return (
            TensorDict(
                {
                    "loss_value": value_loss,
                },
                batch_size=[],
            ),
            tensordict.detach(),
        )
