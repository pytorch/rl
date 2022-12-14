# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from functools import wraps
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


def _self_set_grad_enabled(fun):
    @wraps(fun)
    def new_fun(self, *args, **kwargs):
        with torch.set_grad_enabled(self.differentiable):
            return fun(self, *args, **kwargs)

    return new_fun


class TDEstimate(nn.Module):
    """Temporal Difference estimate of advantage function.

    Args:
        gamma (scalar): exponential mean discount.
        value_network (SafeModule): value operator used to retrieve the value estimates.
        average_rewards (bool, optional): if True, rewards will be standardized
            before the TD is computed.
        differentiable (bool, optional): if True, gradients are propagated throught
            the computation of the value function. Default is :obj:`False`.
        value_key (str, optional): key pointing to the state value. Default is
            `"state_value"`.
    """

    def __init__(
        self,
        gamma: Union[float, torch.Tensor],
        value_network: SafeModule,
        average_rewards: bool = False,
        differentiable: bool = False,
        value_key: str = "state_value",
    ):
        super().__init__()
        try:
            device = next(value_network.parameters()).device
        except StopIteration:
            device = torch.device("cpu")
        self.register_buffer("gamma", torch.tensor(gamma, device=device))
        self.value_network = value_network

        self.average_rewards = average_rewards
        self.differentiable = differentiable
        self.value_key = value_key

    @property
    def is_functional(self):
        return (
            "_is_stateless" in self.value_network.__dict__
            and self.value_network.__dict__["_is_stateless"]
        )

    @_self_set_grad_enabled
    def forward(
        self,
        tensordict: TensorDictBase,
        *unused_args,
        params: Optional[List[Tensor]] = None,
        target_params: Optional[List[Tensor]] = None,
    ) -> TensorDictBase:
        """Computes the GAE given the data in tensordict.

        Args:
            tensordict (TensorDictBase): A TensorDict containing the data
                (an observation key, "action", "reward", "done" and "next" tensordict state
                as returned by the environment) necessary to compute the value estimates and the TDEstimate.

        Returns:
            An updated TensorDict with an "advantage" and a "value_error" keys

        """
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
        with hold_out_net(self.value_network):
            self.value_network(tensordict, **kwargs)
            value = tensordict.get(self.value_key)

        # we may still need to pass gradient, but we don't want to assign grads to
        # value net params
        step_td = step_mdp(tensordict)
        if target_params is not None:
            # we assume that target parameters are not differentiable
            kwargs["params"] = target_params
        elif "params" in kwargs:
            kwargs["params"] = kwargs["params"].detach()
        with hold_out_net(self.value_network):
            self.value_network(step_td, **kwargs)
            next_value = step_td.get(self.value_key)

        done = tensordict.get("done")
        adv = td_advantage_estimate(gamma, value, next_value, reward, done)
        tensordict.set("advantage", adv)
        tensordict.set("value_target", adv + value)
        return tensordict


class TDLambdaEstimate(nn.Module):
    """TD-Lambda estimate of advantage function.

    Args:
        gamma (scalar): exponential mean discount.
        lmbda (scalar): trajectory discount.
        value_network (SafeModule): value operator used to retrieve the value estimates.
        average_rewards (bool, optional): if True, rewards will be standardized
            before the TD is computed.
        differentiable (bool, optional): if True, gradients are propagated throught
            the computation of the value function. Default is :obj:`False`.
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
        differentiable: bool = False,
        value_key: str = "state_value",
        vectorized: bool = True,
    ):
        super().__init__()
        try:
            device = next(value_network.parameters()).device
        except StopIteration:
            device = torch.device("cpu")
        self.register_buffer("gamma", torch.tensor(gamma, device=device))
        self.register_buffer("lmbda", torch.tensor(lmbda, device=device))
        self.value_network = value_network
        self.vectorized = vectorized

        self.average_rewards = average_rewards
        self.differentiable = differentiable
        self.value_key = value_key

    @property
    def is_functional(self):
        return (
            "_is_stateless" in self.value_network.__dict__
            and self.value_network.__dict__["_is_stateless"]
        )

    @_self_set_grad_enabled
    def forward(
        self,
        tensordict: TensorDictBase,
        *unused_args,
        params: Optional[List[Tensor]] = None,
        target_params: Optional[List[Tensor]] = None,
    ) -> TensorDictBase:
        """Computes the GAE given the data in tensordict.

        Args:
            tensordict (TensorDictBase): A TensorDict containing the data
                (an observation key, "action", "reward", "done" and "next" tensordict state
                as returned by the environment) necessary to compute the value
                estimates and the TDLambdaEstimate.

        Returns:
            An updated TensorDict with an "advantage" and a "value_error" keys

        """
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
        with hold_out_net(self.value_network):
            self.value_network(tensordict, **kwargs)
            value = tensordict.get(self.value_key)

        step_td = step_mdp(tensordict)
        if target_params is not None:
            # we assume that target parameters are not differentiable
            kwargs["params"] = target_params
        elif "params" in kwargs:
            kwargs["params"] = kwargs["params"].detach()
        with hold_out_net(self.value_network):
            # we may still need to pass gradient, but we don't want to assign grads to
            # value net params
            self.value_network(step_td, **kwargs)
            next_value = step_td.get(self.value_key)

        done = tensordict.get("done")
        if self.vectorized:
            adv = vec_td_lambda_advantage_estimate(
                gamma, lmbda, value, next_value, reward, done
            )
        else:
            adv = td_lambda_advantage_estimate(
                gamma, lmbda, value, next_value, reward, done
            )

        tensordict.set("advantage", adv)
        tensordict.set("value_target", adv + value)
        return tensordict


class GAE(nn.Module):
    """A class wrapper around the generalized advantage estimate functional.

    Refer to "HIGH-DIMENSIONAL CONTINUOUS CONTROL USING GENERALIZED ADVANTAGE ESTIMATION"
    https://arxiv.org/pdf/1506.02438.pdf for more context.

    Args:
        gamma (scalar): exponential mean discount.
        lmbda (scalar): trajectory discount.
        value_network (SafeModule): value operator used to retrieve the value estimates.
        average_gae (bool): if True, the resulting GAE values will be standardized.
            Default is :obj:`False`.
        differentiable (bool, optional): if True, gradients are propagated throught
            the computation of the value function. Default is :obj:`False`.

    GAE will return an :obj:`"advantage"` entry containing the advange value. It will also
    return a :obj:`"value_target"` entry with the return value that is to be used
    to train the value network. Finally, if :obj:`gradient_mode` is :obj:`True`,
    an additional and differentiable :obj:`"value_error"` entry will be returned,
    which simple represents the difference between the return and the value network
    output (i.e. an additional distance loss should be applied to that signed value).

    """

    def __init__(
        self,
        gamma: Union[float, torch.Tensor],
        lmbda: float,
        value_network: SafeModule,
        average_gae: bool = False,
        differentiable: bool = False,
    ):
        super().__init__()
        try:
            device = next(value_network.parameters()).device
        except StopIteration:
            device = torch.device("cpu")
        self.register_buffer("gamma", torch.tensor(gamma, device=device))
        self.register_buffer("lmbda", torch.tensor(lmbda, device=device))
        self.value_network = value_network

        self.average_gae = average_gae
        self.differentiable = differentiable

    @property
    def is_functional(self):
        return (
            "_is_stateless" in self.value_network.__dict__
            and self.value_network.__dict__["_is_stateless"]
        )

    @_self_set_grad_enabled
    def forward(
        self,
        tensordict: TensorDictBase,
        *unused_args,
        params: Optional[List[Tensor]] = None,
        target_params: Optional[List[Tensor]] = None,
    ) -> TensorDictBase:
        """Computes the GAE given the data in tensordict.

        Args:
            tensordict (TensorDictBase): A TensorDict containing the data
                (an observation key, "action", "reward", "done" and "next" tensordict state
                as returned by the environment) necessary to compute the value estimates and the GAE.

        Returns:
            An updated TensorDict with an "advantage" and a "value_error" keys

        """
        if tensordict.batch_dims < 1:
            raise RuntimeError(
                "Expected input tensordict to have at least one dimensions, got"
                f"tensordict.batch_size = {tensordict.batch_size}"
            )
        reward = tensordict.get("reward")
        gamma, lmbda = self.gamma, self.lmbda
        kwargs = {}
        if self.is_functional and params is None:
            raise RuntimeError(
                "Expected params to be passed to advantage module but got none."
            )
        if params is not None:
            kwargs["params"] = params
        with hold_out_net(self.value_network):
            # we may still need to pass gradient, but we don't want to assign grads to
            # value net params
            self.value_network(tensordict, **kwargs)

        value = tensordict.get("state_value")

        step_td = step_mdp(tensordict)
        if target_params is not None:
            # we assume that target parameters are not differentiable
            kwargs["params"] = target_params
        elif "params" in kwargs:
            kwargs["params"] = kwargs["params"].detach()
        with hold_out_net(self.value_network):
            # we may still need to pass gradient, but we don't want to assign grads to
            # value net params
            self.value_network(step_td, **kwargs)
        next_value = step_td.get("state_value")
        done = tensordict.get("done")
        adv, value_target = vec_generalized_advantage_estimate(
            gamma, lmbda, value, next_value, reward, done
        )

        if self.average_gae:
            adv = adv - adv.mean()
            adv = adv / adv.std().clamp_min(1e-4)

        tensordict.set("advantage", adv)
        tensordict.set("value_target", value_target)

        return tensordict
