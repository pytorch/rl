# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Callable, Optional, Tuple

import torch
from tensordict.tensordict import TensorDict, TensorDictBase
from torch import distributions as d

from torchrl.modules import TensorDictModule
from torchrl.modules.tensordict_module import ProbabilisticTensorDictModule
from torchrl.objectives.common import LossModule
from torchrl.objectives.utils import distance_loss


class A2CLoss(LossModule):
    """TorchRL implementation of the A2C loss.

    A2C (Advantage Actor Critic) is a model-free, online RL algorithm that uses parallel rollouts of n steps to
    update the policy, relying on the REINFORCE estimator to compute the gradient. It also adds an entropy term to the
    objective function to improve exploration.

    For more details regarding A2C, refer to: "Asynchronous Methods for Deep Reinforcment Learning",
    https://arxiv.org/abs/1602.01783v2

    Args:
        actor (ProbabilisticTensorDictModule): policy operator.
        critic (ValueOperator): value operator.
        advantage_key (str): the input tensordict key where the advantage is expected to be written.
            default: "advantage"
        advantage_diff_key (str): the input tensordict key where advantage_diff is expected to be written.
            default: "value_error"
        entropy_coef (float): the weight of the entropy loss.
        critic_coef (float): the weight of the critic loss.
        gamma (scalar): a discount factor for return computation.
        loss_function_type (str): loss function for the value discrepancy. Can be one of "l1", "l2" or "smooth_l1".
        advantage_module (nn.Module): TensorDictModule used to compute tha advantage function.
    """

    def __init__(
        self,
        actor: ProbabilisticTensorDictModule,
        critic: TensorDictModule,
        advantage_key: str = "advantage",
        advantage_diff_key: str = "value_error",
        entropy_bonus: bool = True,
        samples_mc_entropy: int = 1,
        entropy_coef: float = 0.01,
        critic_coef: float = 1.0,
        gamma: float = 0.99,
        loss_critic_type: str = "smooth_l1",
        advantage_module: Optional[Callable[[TensorDictBase], TensorDictBase]] = None,
    ):
        super().__init__()
        self.convert_to_functional(actor, "actor")
        self.convert_to_functional(critic, "critic", compare_against=self.actor_params)
        self.advantage_key = advantage_key
        self.advantage_diff_key = advantage_diff_key
        self.samples_mc_entropy = samples_mc_entropy
        self.entropy_bonus = entropy_bonus and entropy_coef
        self.register_buffer(
            "entropy_coef", torch.tensor(entropy_coef, device=self.device)
        )
        self.register_buffer(
            "critic_coef", torch.tensor(critic_coef, device=self.device)
        )
        self.register_buffer("gamma", torch.tensor(gamma, device=self.device))
        self.loss_critic_type = loss_critic_type
        self.advantage_module = advantage_module
        if advantage_module:
            self.advantage_module = advantage_module.to(self.device)

    def reset(self) -> None:
        pass

    def get_entropy_bonus(self, dist: d.Distribution) -> torch.Tensor:
        try:
            entropy = dist.entropy()
        except NotImplementedError:
            x = dist.rsample((self.samples_mc_entropy,))
            entropy = -dist.log_prob(x)
        return entropy.unsqueeze(-1)

    def _log_probs(
        self, tensordict: TensorDictBase
    ) -> Tuple[torch.Tensor, d.Distribution]:
        # current log_prob of actions
        action = tensordict.get("action")
        if action.requires_grad:
            raise RuntimeError("tensordict stored action require grad.")
        tensordict_clone = tensordict.select(*self.actor.in_keys).clone()

        dist, *_ = self.actor.get_dist(
            tensordict_clone, params=self.actor_params, buffers=self.actor_buffers
        )
        log_prob = dist.log_prob(action)
        log_prob = log_prob.unsqueeze(-1)
        return log_prob, dist

    def loss_critic(self, tensordict: TensorDictBase) -> torch.Tensor:
        if self.advantage_diff_key in tensordict.keys():
            advantage_diff = tensordict.get(self.advantage_diff_key)
            if not advantage_diff.requires_grad:
                raise RuntimeError(
                    "value_target retrieved from tensordict does not require grad."
                )
            loss_value = distance_loss(
                advantage_diff,
                torch.zeros_like(advantage_diff),
                loss_function=self.loss_critic_type,
            )
        else:
            advantage = tensordict.get(self.advantage_key)
            tensordict_select = tensordict.select(*self.critic.in_keys)
            value = self.critic(
                tensordict_select,
                params=self.critic_params,
                buffers=self.critic_buffers,
            ).get("state_value")
            value_target = advantage + value.detach()
            loss_value = distance_loss(
                value, value_target, loss_function=self.loss_critic_type
            )
        return self.critic_coef * loss_value

    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        if self.advantage_module is not None:
            tensordict = self.advantage_module(
                tensordict,
            )
        tensordict = tensordict.clone()
        advantage = tensordict.get(self.advantage_key)
        log_probs, dist = self._log_probs(tensordict)
        loss = -(log_probs * advantage)
        td_out = TensorDict({"loss_objective": loss.mean()}, [])
        if self.entropy_bonus:
            entropy = self.get_entropy_bonus(dist)
            td_out.set("entropy", entropy.mean().detach())  # for logging
            td_out.set("loss_entropy", -self.entropy_coef * entropy.mean())
        if self.critic_coef:
            loss_critic = self.loss_critic(tensordict).mean()
            td_out.set("loss_critic", loss_critic.mean())
        return td_out
