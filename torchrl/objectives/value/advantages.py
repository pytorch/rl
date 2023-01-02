# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from functools import wraps
from typing import List, Optional, Tuple, Union

import torch
from tensordict.nn import dispatch_kwargs
from tensordict.tensordict import TensorDictBase
from torch import nn, Tensor

from torchrl.envs.utils import step_mdp
from torchrl.modules import SafeModule

from torchrl.objectives.utils import hold_out_net
from torchrl.objectives.value.functional import (
    td_advantage_estimate,
    td_lambda_advantage_estimate,
    vec_generalized_advantage_estimate,
    vec_td_lambda_advantage_estimate,
)


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
        advantage_key (str or tuple of str, optional): the key of the advantage entry.
            Defaults to "advantage".
        value_target_key (str or tuple of str, optional): the key of the advantage entry.
            Defaults to "value_target".
        value_key (str or tuple of str, optional): the value key to read from the input tensordict.
            Defaults to "state_value".

    """

    def __init__(
        self,
        gamma: Union[float, torch.Tensor],
        value_network: SafeModule,
        average_rewards: bool = False,
        differentiable: bool = False,
        advantage_key: Union[str, Tuple] = "advantage",
        value_target_key: Union[str, Tuple] = "value_target",
        value_key: Union[str, Tuple] = "state_value",
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
        if value_key not in value_network.out_keys:
            raise KeyError(
                f"value key '{value_key}' not found in value network out_keys."
            )

        self.advantage_key = advantage_key
        self.value_target_key = value_target_key

        self.in_keys = (
            value_network.in_keys
            + ["reward", "done"]
            + [("next", in_key) for in_key in value_network.in_keys]
        )
        self.out_keys = [self.advantage_key, self.value_target_key]

    @property
    def is_functional(self):
        return (
            "_is_stateless" in self.value_network.__dict__
            and self.value_network.__dict__["_is_stateless"]
        )

    @_self_set_grad_enabled
    @dispatch_kwargs
    def forward(
        self,
        tensordict: TensorDictBase,
        params: Optional[TensorDictBase] = None,
        target_params: Optional[TensorDictBase] = None,
    ) -> TensorDictBase:
        """Computes the TDEstimate given the data in tensordict.

        If a functional module is provided, a nested TensorDict containing the parameters
        (and if relevant the target parameters) can be passed to the module.

        Args:
            tensordict (TensorDictBase): A TensorDict containing the data
                (an observation key, "action", "reward", "done" and "next" tensordict state
                as returned by the environment) necessary to compute the value estimates and the TDEstimate.
                The data passed to this module should be structured as :obj:`[*B, T, F]` where :obj:`B` are
                the batch size, :obj:`T` the time dimension and :obj:`F` the feature dimension(s).
            params (TensorDictBase, optional): A nested TensorDict containing the params
                to be passed to the functional value network module.
            target_params (TensorDictBase, optional): A nested TensorDict containing the
                target params to be passed to the functional value network module.

        Returns:
            An updated TensorDict with an advantage and a value_error keys as defined in the constructor.

        Examples:
            >>> from tensordict import TensorDict
            >>> value_net = SafeModule(
            ...     nn.Linear(3, 1), in_keys=["obs"], out_keys=["state_value"]
            ... )
            >>> module = TDEstimate(
            ...     gamma=0.98,
            ...     value_network=value_net,
            ...     differentiable=False,
            ... )
            >>> obs, next_obs = torch.randn(2, 1, 10, 3)
            >>> reward = torch.randn(1, 10, 1)
            >>> done = torch.zeros(1, 10, 1, dtype=torch.bool)
            >>> tensordict = TensorDict({"obs": obs, "next": {"obs": next_obs}, "done": done, "reward": reward}, [1, 10])
            >>> _ = module(tensordict)
            >>> assert "advantage" in tensordict.keys()

        The module supports non-tensordict (i.e. unpacked tensordict) inputs too:

        Examples:
            >>> value_net = SafeModule(
            ...     nn.Linear(3, 1), in_keys=["obs"], out_keys=["state_value"]
            ... )
            >>> module = TDEstimate(
            ...     gamma=0.98,
            ...     value_network=value_net,
            ...     differentiable=False,
            ... )
            >>> obs, next_obs = torch.randn(2, 1, 10, 3)
            >>> reward = torch.randn(1, 10, 1)
            >>> done = torch.zeros(1, 10, 1, dtype=torch.bool)
            >>> advantage, value_target = module(obs=obs, reward=reward, done=done, next_obs=next_obs)

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
            kwargs["params"] = params.detach()
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
        vectorized (bool, optional): whether to use the vectorized version of the
            lambda return. Default is `True`.
        advantage_key (str or tuple of str, optional): the key of the advantage entry.
            Defaults to "advantage".
        value_target_key (str or tuple of str, optional): the key of the advantage entry.
            Defaults to "value_target".
        value_key (str or tuple of str, optional): the value key to read from the input tensordict.
            Defaults to "state_value".

    """

    def __init__(
        self,
        gamma: Union[float, torch.Tensor],
        lmbda: Union[float, torch.Tensor],
        value_network: SafeModule,
        average_rewards: bool = False,
        differentiable: bool = False,
        vectorized: bool = True,
        advantage_key: Union[str, Tuple] = "advantage",
        value_target_key: Union[str, Tuple] = "value_target",
        value_key: Union[str, Tuple] = "state_value",
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
        if value_key not in value_network.out_keys:
            raise KeyError(
                f"value key '{value_key}' not found in value network out_keys."
            )

        self.advantage_key = advantage_key
        self.value_target_key = value_target_key

        self.in_keys = (
            value_network.in_keys
            + ["reward", "done"]
            + [("next", in_key) for in_key in value_network.in_keys]
        )
        self.out_keys = [self.advantage_key, self.value_target_key]

    @property
    def is_functional(self):
        return (
            "_is_stateless" in self.value_network.__dict__
            and self.value_network.__dict__["_is_stateless"]
        )

    @_self_set_grad_enabled
    @dispatch_kwargs
    def forward(
        self,
        tensordict: TensorDictBase,
        params: Optional[List[Tensor]] = None,
        target_params: Optional[List[Tensor]] = None,
    ) -> TensorDictBase:
        """Computes the TDLambdaEstimate given the data in tensordict.

        If a functional module is provided, a nested TensorDict containing the parameters
        (and if relevant the target parameters) can be passed to the module.

        Args:
            tensordict (TensorDictBase): A TensorDict containing the data
                (an observation key, "action", "reward", "done" and "next" tensordict state
                as returned by the environment) necessary to compute the value estimates and the TDLambdaEstimate.
                The data passed to this module should be structured as :obj:`[*B, T, F]` where :obj:`B` are
                the batch size, :obj:`T` the time dimension and :obj:`F` the feature dimension(s).
            params (TensorDictBase, optional): A nested TensorDict containing the params
                to be passed to the functional value network module.
            target_params (TensorDictBase, optional): A nested TensorDict containing the
                target params to be passed to the functional value network module.

        Returns:
            An updated TensorDict with an advantage and a value_error keys as defined in the constructor.

        Examples:
            >>> from tensordict import TensorDict
            >>> value_net = SafeModule(
            ...     nn.Linear(3, 1), in_keys=["obs"], out_keys=["state_value"]
            ... )
            >>> module = TDLambdaEstimate(
            ...     gamma=0.98,
            ...     lmbda=0.94,
            ...     value_network=value_net,
            ...     differentiable=False,
            ... )
            >>> obs, next_obs = torch.randn(2, 1, 10, 3)
            >>> reward = torch.randn(1, 10, 1)
            >>> done = torch.zeros(1, 10, 1, dtype=torch.bool)
            >>> tensordict = TensorDict({"obs": obs, "next": {"obs": next_obs}, "done": done, "reward": reward}, [1, 10])
            >>> _ = module(tensordict)
            >>> assert "advantage" in tensordict.keys()

        The module supports non-tensordict (i.e. unpacked tensordict) inputs too:

        Examples:
            >>> value_net = SafeModule(
            ...     nn.Linear(3, 1), in_keys=["obs"], out_keys=["state_value"]
            ... )
            >>> module = TDLambdaEstimate(
            ...     gamma=0.98,
            ...     lmbda=0.94,
            ...     value_network=value_net,
            ...     differentiable=False,
            ... )
            >>> obs, next_obs = torch.randn(2, 1, 10, 3)
            >>> reward = torch.randn(1, 10, 1)
            >>> done = torch.zeros(1, 10, 1, dtype=torch.bool)
            >>> advantage, value_target = module(obs=obs, reward=reward, done=done, next_obs=next_obs)

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

        tensordict.set(self.advantage_key, adv)
        tensordict.set(self.value_target_key, adv + value)
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
        advantage_key (str or tuple of str, optional): the key of the advantage entry.
            Defaults to "advantage".
        value_target_key (str or tuple of str, optional): the key of the advantage entry.
            Defaults to "value_target".
        value_key (str or tuple of str, optional): the value key to read from the input tensordict.
            Defaults to "state_value".

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
        advantage_key: Union[str, Tuple] = "advantage",
        value_target_key: Union[str, Tuple] = "value_target",
        value_key: Union[str, Tuple] = "state_value",
    ):
        super().__init__()
        try:
            device = next(value_network.parameters()).device
        except StopIteration:
            device = torch.device("cpu")
        self.register_buffer("gamma", torch.tensor(gamma, device=device))
        self.register_buffer("lmbda", torch.tensor(lmbda, device=device))
        self.value_network = value_network
        self.value_key = value_key
        if value_key not in value_network.out_keys:
            raise KeyError(
                f"value key '{value_key}' not found in value network out_keys."
            )

        self.average_gae = average_gae
        self.differentiable = differentiable

        self.advantage_key = advantage_key
        self.value_target_key = value_target_key

        self.in_keys = (
            value_network.in_keys
            + ["reward", "done"]
            + [("next", in_key) for in_key in value_network.in_keys]
        )
        self.out_keys = [self.advantage_key, self.value_target_key]

    @property
    def is_functional(self):
        return (
            "_is_stateless" in self.value_network.__dict__
            and self.value_network.__dict__["_is_stateless"]
        )

    @_self_set_grad_enabled
    @dispatch_kwargs
    def forward(
        self,
        tensordict: TensorDictBase,
        *unused_args,
        params: Optional[List[Tensor]] = None,
        target_params: Optional[List[Tensor]] = None,
    ) -> TensorDictBase:
        """Computes the GAE given the data in tensordict.

        If a functional module is provided, a nested TensorDict containing the parameters
        (and if relevant the target parameters) can be passed to the module.

        Args:
            tensordict (TensorDictBase): A TensorDict containing the data
                (an observation key, "action", "reward", "done" and "next" tensordict state
                as returned by the environment) necessary to compute the value estimates and the GAE.
                The data passed to this module should be structured as :obj:`[*B, T, F]` where :obj:`B` are
                the batch size, :obj:`T` the time dimension and :obj:`F` the feature dimension(s).
            params (TensorDictBase, optional): A nested TensorDict containing the params
                to be passed to the functional value network module.
            target_params (TensorDictBase, optional): A nested TensorDict containing the
                target params to be passed to the functional value network module.

        Returns:
            An updated TensorDict with an advantage and a value_error keys as defined in the constructor.

        Examples:
            >>> from tensordict import TensorDict
            >>> value_net = SafeModule(
            ...     nn.Linear(3, 1), in_keys=["obs"], out_keys=["state_value"]
            ... )
            >>> module = GAE(
            ...     gamma=0.98,
            ...     lmbda=0.94,
            ...     value_network=value_net,
            ...     differentiable=False,
            ... )
            >>> obs, next_obs = torch.randn(2, 1, 10, 3)
            >>> reward = torch.randn(1, 10, 1)
            >>> done = torch.zeros(1, 10, 1, dtype=torch.bool)
            >>> tensordict = TensorDict({"obs": obs, "next": {"obs": next_obs}, "done": done, "reward": reward}, [1, 10])
            >>> _ = module(tensordict)
            >>> assert "advantage" in tensordict.keys()

        The module supports non-tensordict (i.e. unpacked tensordict) inputs too:

        Examples:
            >>> value_net = SafeModule(
            ...     nn.Linear(3, 1), in_keys=["obs"], out_keys=["state_value"]
            ... )
            >>> module = GAE(
            ...     gamma=0.98,
            ...     lmbda=0.94,
            ...     value_network=value_net,
            ...     differentiable=False,
            ... )
            >>> obs, next_obs = torch.randn(2, 1, 10, 3)
            >>> reward = torch.randn(1, 10, 1)
            >>> done = torch.zeros(1, 10, 1, dtype=torch.bool)
            >>> advantage, value_target = module(obs=obs, reward=reward, done=done, next_obs=next_obs)

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
        adv, value_target = vec_generalized_advantage_estimate(
            gamma, lmbda, value, next_value, reward, done
        )

        if self.average_gae:
            adv = adv - adv.mean()
            adv = adv / adv.std().clamp_min(1e-4)

        tensordict.set(self.advantage_key, adv)
        tensordict.set(self.value_target_key, value_target)

        return tensordict
