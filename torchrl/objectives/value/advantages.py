# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional, Union

import torch
from tensordict.tensordict import TensorDictBase
from torch import nn, Tensor

from torchrl.envs.utils import step_mdp
from torchrl.modules import SafeModule
from torchrl.objectives.value.functional import (
    td_lambda_advantage_estimate,
    vec_generalized_advantage_estimate,
    vec_td_lambda_advantage_estimate,
)

from ..utils import hold_out_net
from .functional import td_advantage_estimate


class TDEstimate(nn.Module):
    """Temporal Difference estimate of advantage function.

    Args:
        gamma (scalar): exponential mean discount.
        value_network (SafeModule): value operator used to retrieve the value estimates.
        average_rewards (bool, optional): if True, rewards will be standardized
            before the TD is computed.
        gradient_mode (bool, optional): if True, gradients are propagated throught
            the computation of the value function. Default is :obj:`False`.
        value_key (str, optional): key pointing to the state value. Default is
            `"state_value"`.
    """

    def __init__(
        self,
        gamma: Union[float, torch.Tensor],
        value_network: SafeModule,
        average_rewards: bool = False,
        gradient_mode: bool = False,
        value_key: str = "state_value",
    ):
        super().__init__()
        self.register_buffer("gamma", torch.tensor(gamma))
        self.value_network = value_network

        self.average_rewards = average_rewards
        self.gradient_mode = gradient_mode
        self.value_key = value_key

    @property
    def is_functional(self):
        return (
            "_is_stateless" in self.value_network.__dict__
            and self.value_network.__dict__["_is_stateless"]
        )

    def forward(
        self,
        tensordict: TensorDictBase,
        *unused_args,
        params: Optional[List[Tensor]] = None,
        target_params: Optional[List[Tensor]] = None,
    ) -> TensorDictBase:
        """Computes the GAE given the data in tensordict.

        Args:
            tensordict (TensorDictBase): A TensorDict containing the data (observation, action, reward, done state)
                necessary to compute the value estimates and the GAE.

        Returns:
            An updated TensorDict with an "advantage" and a "value_error" keys

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
                tensordict.set(
                    "reward", reward
                )  # we must update the rewards if they are used later in the code

            gamma = self.gamma
            kwargs = {}
            if self.is_functional and params is None:
                raise RuntimeError(
                    "Expected params to be passed to advantage module but got none."
                )
            if params is not None:
                kwargs["params"] = params
            self.value_network(tensordict, **kwargs)
            value = tensordict.get(self.value_key)

        with hold_out_net(self.value_network):
            # we may still need to pass gradient, but we don't want to assign grads to
            # value net params
            step_td = step_mdp(tensordict)
            if target_params is not None:
                # we assume that target parameters are not differentiable
                kwargs["params"] = target_params
            elif "params" in kwargs:
                kwargs["params"] = [param.detach() for param in kwargs["params"]]
            self.value_network(step_td, **kwargs)
            next_value = step_td.get(self.value_key)

        done = tensordict.get("done")
        with torch.set_grad_enabled(self.gradient_mode):
            adv = td_advantage_estimate(gamma, value, next_value, reward, done)
            tensordict.set("advantage", adv.detach())
            if self.gradient_mode:
                tensordict.set("value_error", adv)
        return tensordict


class TDLambdaEstimate(nn.Module):
    """TD-Lambda estimate of advantage function.

    Args:
        gamma (scalar): exponential mean discount.
        lmbda (scalar): trajectory discount.
        value_network (SafeModule): value operator used to retrieve the value estimates.
        average_rewards (bool, optional): if True, rewards will be standardized
            before the TD is computed.
        gradient_mode (bool, optional): if True, gradients are propagated throught
            the computation of the value function. Default is `False`.
        value_key (str, optional): key pointing to the state value. Default is
            `"state_value"`.
        vectorized (bool, optional): whether to use the vectorized version of the
            lambda return. Default is `True`.
    """

    def __init__(
        self,
        gamma: Union[float, torch.Tensor],
        lmbda: Union[float, torch.Tensor],
        value_network: SafeModule,
        average_rewards: bool = False,
        gradient_mode: bool = False,
        value_key: str = "state_value",
        vectorized: bool = True,
    ):
        super().__init__()
        self.register_buffer("gamma", torch.tensor(gamma))
        self.register_buffer("lmbda", torch.tensor(lmbda))
        self.value_network = value_network
        self.vectorized = vectorized

        self.average_rewards = average_rewards
        self.gradient_mode = gradient_mode
        self.value_key = value_key

    @property
    def is_functional(self):
        return (
            "_is_stateless" in self.value_network.__dict__
            and self.value_network.__dict__["_is_stateless"]
        )

    def forward(
        self,
        tensordict: TensorDictBase,
        *unused_args,
        params: Optional[List[Tensor]] = None,
        target_params: Optional[List[Tensor]] = None,
    ) -> TensorDictBase:
        """Computes the GAE given the data in tensordict.

        Args:
            tensordict (TensorDictBase): A TensorDict containing the data (observation, action, reward, done state)
                necessary to compute the value estimates and the GAE.

        Returns:
            An updated TensorDict with an "advantage" and a "value_error" keys

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
                tensordict.set(
                    "reward", reward
                )  # we must update the rewards if they are used later in the code

            gamma = self.gamma
            lmbda = self.lmbda

            kwargs = {}
            if self.is_functional and params is None:
                raise RuntimeError(
                    "Expected params to be passed to advantage module but got none."
                )
            if params is not None:
                kwargs["params"] = params
            self.value_network(tensordict, **kwargs)
            value = tensordict.get(self.value_key)

        with hold_out_net(self.value_network):
            # we may still need to pass gradient, but we don't want to assign grads to
            # value net params
            step_td = step_mdp(tensordict)
            if target_params is not None:
                # we assume that target parameters are not differentiable
                kwargs["params"] = target_params
            elif "params" in kwargs:
                kwargs["params"] = [param.detach() for param in kwargs["params"]]
            self.value_network(step_td, **kwargs)
            next_value = step_td.get(self.value_key)

        done = tensordict.get("done")
        with torch.set_grad_enabled(self.gradient_mode):
            if self.vectorized:
                adv = vec_td_lambda_advantage_estimate(
                    gamma, lmbda, value, next_value, reward, done
                )
            else:
                adv = td_lambda_advantage_estimate(
                    gamma, lmbda, value, next_value, reward, done
                )

            tensordict.set("advantage", adv.detach())
            if self.gradient_mode:
                tensordict.set("value_error", adv)
        return tensordict


class GAE(nn.Module):
    """A class wrapper around the generalized advantage estimate functional.

    Refer to "HIGH-DIMENSIONAL CONTINUOUS CONTROL USING GENERALIZED ADVANTAGE ESTIMATION"
    https://arxiv.org/pdf/1506.02438.pdf for more context.

    Args:
        gamma (scalar): exponential mean discount.
        lmbda (scalar): trajectory discount.
        value_network (SafeModule): value operator used to retrieve the value estimates.
        average_rewards (bool): if True, rewards will be standardized before the GAE is computed.
        gradient_mode (bool): if True, gradients are propagated throught the computation of the value function.
            Default is `False`.

    """

    def __init__(
        self,
        gamma: Union[float, torch.Tensor],
        lmbda: float,
        value_network: SafeModule,
        average_rewards: bool = False,
        gradient_mode: bool = False,
    ):
        super().__init__()
        self.register_buffer("gamma", torch.tensor(gamma))
        self.register_buffer("lmbda", torch.tensor(lmbda))
        self.value_network = value_network

        self.average_rewards = average_rewards
        self.gradient_mode = gradient_mode

    @property
    def is_functional(self):
        return (
            "_is_stateless" in self.value_network.__dict__
            and self.value_network.__dict__["_is_stateless"]
        )

    def forward(
        self,
        tensordict: TensorDictBase,
        *unused_args,
        params: Optional[List[Tensor]] = None,
        target_params: Optional[List[Tensor]] = None,
    ) -> TensorDictBase:
        """Computes the GAE given the data in tensordict.

        Args:
            tensordict (TensorDictBase): A TensorDict containing the data (observation, action, reward, done state)
                necessary to compute the value estimates and the GAE.

        Returns:
            An updated TensorDict with an "advantage" and a "value_error" keys

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
                tensordict.set(
                    "reward", reward
                )  # we must update the rewards if they are used later in the code

            gamma, lmbda = self.gamma, self.lmbda
            kwargs = {}
            if self.is_functional and params is None:
                raise RuntimeError(
                    "Expected params to be passed to advantage module but got none."
                )
            if params is not None:
                kwargs["params"] = params
            self.value_network(tensordict, **kwargs)
            value = tensordict.get("state_value")

        with hold_out_net(self.value_network):
            # we may still need to pass gradient, but we don't want to assign grads to
            # value net params
            step_td = step_mdp(tensordict)
            if target_params is not None:
                # we assume that target parameters are not differentiable
                kwargs["params"] = target_params
            elif "params" in kwargs:
                kwargs["params"] = [param.detach() for param in kwargs["params"]]
            self.value_network(step_td, **kwargs)
            next_value = step_td.get("state_value")
            done = tensordict.get("done")
            adv, value_target = vec_generalized_advantage_estimate(
                gamma, lmbda, value, next_value, reward, done
            )

        tensordict.set("advantage", adv.detach())
        if self.gradient_mode:
            tensordict.set("value_error", value_target - value)

        return tensordict
