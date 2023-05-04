# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math

import numpy as np

import torch
from tensordict.tensordict import TensorDict, TensorDictBase

from torchrl.modules import ProbabilisticActor

from .common import LossModule


class OnlineDTLoss(LossModule):
    r"""TorchRL implementation of the Online Decision Transformer loss.

    Presented in "Online Decision Transformer" https://arxiv.org/abs/2202.05607
    Args:
        actor_network (ProbabilisticActor): stochastic actor
        qvalue_network (SafeModule): Q(s, a) parametric model
        value_network (SafeModule, optional): V(s) parametric model. If not
            provided, the second version of SAC is assumed.
        qvalue_network_bis (ProbabilisticTDModule, optional): if required, the
            Q-value can be computed twice independently using two separate
            networks. The minimum predicted value will then be used for
            inference.
        gamma (number, optional): discount for return computation
            Default is 0.99
        priority_key (str, optional): tensordict key where to write the
            priority (for prioritized replay buffer usage). Default is
            `"td_error"`.
        loss_function (str, optional): loss function to be used with
            the value function loss. Default is `"smooth_l1"`.
        temperature (float, optional):  Inverse temperature (beta).
            For smaller hyperparameter values, the objective behaves similarly to
            behavioral cloning, while for larger values, it attempts to recover the
            maximum of the Q-function.
        expectile (float, optional): expectile :math:`\tau`. A larger value of :math:`\tau` is crucial
            for antmaze tasks that require dynamical programming ("stichting").

    """

    def __init__(
        self,
        actor_network: ProbabilisticActor,
        alpha_init: float = 1.0,
        min_alpha: float = 0.1,
        max_alpha: float = 10.0,
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
        self.register_buffer(
            "min_log_alpha", torch.tensor(min_alpha, device=device).log()
        )
        self.register_buffer(
            "max_log_alpha", torch.tensor(max_alpha, device=device).log()
        )

        self.register_parameter(
            "log_alpha",
            torch.nn.Parameter(torch.tensor(math.log(alpha_init), device=device)),
        )

        target_entropy = -float(np.prod(actor_network.spec["action"].shape))
        self.register_buffer(
            "target_entropy", torch.tensor(target_entropy, device=device)
        )

    @property
    def device(self) -> torch.device:
        for p in self.parameters():
            return p.device
        raise RuntimeError(
            "At least one of the networks of SACLoss must have trainable " "parameters."
        )

    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        """Compute the loss for the Online Decision Transformer.

            # a_hat is a SquashedNormal Distribution
            log_likelihood = a_hat_dist.log_likelihood(a)[attention_mask > 0].mean()

            entropy = a_hat_dist.entropy().mean()
            loss = -(log_likelihood + entropy_reg * entropy)

            return (
                loss,
                -log_likelihood,
                entropy,
            )
        dist.log_prob(x).sum(axis=2)
        """
        shape = None
        if tensordict.ndimension() > 1:
            shape = tensordict.shape
            tensordict_reshape = tensordict.reshape(-1)
        else:
            tensordict_reshape = tensordict

        # device = self.device
        # td_device = tensordict_reshape.to(device)

        out_td = self.actor_network(tensordict)

        target_actions = tensordict["action"]

        # log_prob = out_td["log_prob"]
        action_dist = out_td["distribution"]
        loss_log_likelihood = action_dist.log_prob(target_actions).sum(axis=2)
        entropy = action_dist.entropy().mean()
        loss = -(loss_log_likelihood + self.target_entropy.detach() * entropy)

        loss_alpha = self.log_alpha.exp() * (entropy - self.target_entropy).detach()
        if shape:
            tensordict.update(tensordict_reshape.view(shape))
        out = {
            "loss": loss.mean(),
            "loss_log_likelihood": loss_log_likelihood.mean(),
            "entropy": entropy.mean(),
            "loss_alpha": loss_alpha.mean(),
            "alpha": self._alpha,
        }
        return TensorDict(out, [])

    @property
    def _alpha(self):
        self.log_alpha.data.clamp_(self.min_log_alpha, self.max_log_alpha)
        with torch.no_grad():
            alpha = self.log_alpha.exp()
        return alpha
