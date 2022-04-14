# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Union, Optional, List

import torch

# for value, log_policy, reward, entropy in list(zip(values, log_policies, rewards, entropies))[::-1]:
#     gae = gae * opt.gamma * opt.tau
#     gae = gae + reward + opt.gamma * next_value.detach() - value.detach()
#     next_value = value
#     actor_loss = actor_loss + log_policy * gae
#     R = R * opt.gamma + reward
#     critic_loss = critic_loss + (R - value) ** 2 / 2
#     entropy_loss = entropy_loss + entropy
from torch import Tensor

from torchrl.envs.utils import step_tensordict
from ...data.tensordict.tensordict import _TensorDict
from ...modules import TDModule
from .functional import generalized_advantage_estimate


class GAE:
    """
    A class wrapper around the generalized advantage estimate functional.
    Refer to "HIGH-DIMENSIONAL CONTINUOUS CONTROL USING GENERALIZED ADVANTAGE ESTIMATION"
    https://arxiv.org/pdf/1506.02438.pdf for more context.

    Args:
        gamma (scalar): exponential mean discount.
        lamda (scalar): trajectory discount.
        value_network (TDModule): value operator used to retrieve the value estimates.
        average_rewards (bool): if True, rewards will be standardized before the GAE is computed.
        gradient_mode (bool): if True, gradients are propagated throught the computation of the value function.
            Default is `False`.
    """

    def __init__(
        self,
        gamma: Union[float, torch.Tensor],
        lamda: float,
        value_network: TDModule,
        average_rewards: bool = False,
        gradient_mode: bool = False,
    ):
        self.gamma = gamma
        self.lamda = lamda
        self.value_network = value_network
        self.average_rewards = average_rewards
        self.gradient_mode = gradient_mode

    def __call__(
        self,
        tensordict: _TensorDict,
        params: Optional[List[Tensor]] = None,
        buffers: Optional[List[Tensor]] = None,
        target_params: Optional[List[Tensor]] = None,
        target_buffers: Optional[List[Tensor]] = None,
    ) -> _TensorDict:
        """Computes the GAE given the data in tensordict.

        Args:
            tensordict (_TensorDict): A TensorDict containing the data (observation, action, reward, done state)
                necessary to compute the value estimates and the GAE.

        Returns:
            An updated TensorDict with an "advantage" and a "value_target" keys

        """
        with torch.set_grad_enabled(self.gradient_mode):
            if tensordict.batch_dims < 1:
                raise RuntimeError(
                    "Expected input tensordict to have at least one dimensions, got"
                    f"tensordict.batch_size = {tensordict.batch_size}"
                )
            reward = tensordict.get("reward")
            if self.average_rewards:
                reward = reward - reward.mean()
                reward = reward / reward.std().clamp_min(1e-4)
                tensordict.set_(
                    "reward", reward
                )  # we must update the rewards if they are used later in the code

            gamma, lamda = self.gamma, self.lamda
            kwargs = {}
            if params is not None:
                kwargs["params"] = params
            if buffers is not None:
                kwargs["buffers"] = buffers
            self.value_network(tensordict, **kwargs)
            value = tensordict.get("state_value")

        with torch.set_grad_enabled(False):
            step_td = step_tensordict(tensordict)
            if target_params is not None:
                kwargs["params"] = target_params
            if target_buffers is not None:
                kwargs["buffers"] = target_buffers
            self.value_network(step_td, **kwargs)
            next_value = step_td.get("state_value")

        done = tensordict.get("done")
        with torch.set_grad_enabled(self.gradient_mode):
            adv, value_target = generalized_advantage_estimate(
                gamma, lamda, value, next_value, reward, done
            )
            tensordict.set("advantage", adv.detach())
            tensordict.set("advantage_diff", adv)
        return tensordict
