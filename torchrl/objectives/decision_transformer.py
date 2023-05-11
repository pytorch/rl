# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math

import numpy as np

import torch
from tensordict.tensordict import TensorDict, TensorDictBase
from torch import distributions as d
from torchrl.modules import ProbabilisticActor

from .common import LossModule


class OnlineDTLoss(LossModule):
    r"""TorchRL implementation of the Online Decision Transformer loss.

    Presented in "Online Decision Transformer" https://arxiv.org/abs/2202.05607
    Args:
        actor_network (ProbabilisticActor): stochastic actor
        alpha_init:
        samples_mc_entropy:

    """

    def __init__(
        self,
        actor_network: ProbabilisticActor,
        alpha_init: float = 1.0,
        samples_mc_entropy: int = 1,
    ) -> None:
        super().__init__()

        # Actor Network
        self.convert_to_functional(
            actor_network,
            "actor_network",
            create_target_params=False,
            funs_to_decorate=["forward", "get_dist"],
        )
        try:
            device = next(self.parameters()).device
        except AttributeError:
            device = torch.device("cpu")
        self.register_buffer("alpha_init", torch.tensor(alpha_init, device=device))
        self.register_parameter(
            "log_alpha",
            torch.nn.Parameter(torch.tensor(math.log(alpha_init), device=device)),
        )

        target_entropy = -float(np.prod(actor_network.spec["action"].shape))
        self.register_buffer(
            "target_entropy", torch.tensor(target_entropy, device=device)
        )
        self.samples_mc_entropy = samples_mc_entropy

    @property
    def device(self) -> torch.device:
        for p in self.parameters():
            return p.device
        raise RuntimeError(
            "At least one of the networks of OnlineDTLoss must have trainable "
            "parameters."
        )

    def get_entropy_bonus(self, dist: d.Distribution) -> torch.Tensor:
        x = dist.rsample((self.samples_mc_entropy,))
        log_p = dist.log_prob(x)
        # log_p: (batch_size, context_len,
        return -log_p.mean(axis=0)

    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        """Compute the loss for the Online Decision Transformer."""
        # extract action targets
        target_actions = torch.clone(tensordict["action"].detach()).to(self.device)

        action_dist = self.actor_network.get_dist(
            tensordict.to(self.device), params=self.actor_network_params
        )

        loss_log_likelihood = action_dist.log_prob(target_actions).mean()
        entropy = self.get_entropy_bonus(action_dist).mean()
        loss = -(loss_log_likelihood + self.log_alpha.exp().detach() * entropy)

        loss_alpha = self.log_alpha.exp() * (entropy - self.target_entropy).detach()

        out = {
            "loss": loss,
            "loss_log_likelihood": -loss_log_likelihood,
            "entropy": entropy,
            "loss_alpha": loss_alpha,
            "alpha": self.log_alpha.exp(),
        }
        return TensorDict(out, [])
