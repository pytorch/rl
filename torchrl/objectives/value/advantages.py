# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import abc
import functools
import warnings
from dataclasses import asdict, dataclass
from functools import wraps
from typing import Callable, List, Optional, Union

import torch
from tensordict.nn import (
    dispatch,
    is_functional,
    set_skip_existing,
    TensorDictModule,
    TensorDictModuleBase,
)
from tensordict.tensordict import TensorDictBase
from tensordict.utils import NestedKey
from torch import nn, Tensor

from torchrl._utils import RL_WARNINGS
from torchrl.envs.utils import step_mdp

from torchrl.objectives.utils import hold_out_net
from torchrl.objectives.value.functional import (
    generalized_advantage_estimate,
    td0_return_estimate,
    td_lambda_return_estimate,
    vec_generalized_advantage_estimate,
    vec_td1_return_estimate,
    vec_td_lambda_return_estimate,
)

try:
    from torch import vmap
except ImportError as err:
    try:
        from functorch import vmap
    except ImportError:
        raise ImportError(
            "vmap couldn't be found. Make sure you have torch>1.13 installed."
        ) from err


def _self_set_grad_enabled(fun):
    @wraps(fun)
    def new_fun(self, *args, **kwargs):
        with torch.set_grad_enabled(self.differentiable):
            return fun(self, *args, **kwargs)

    return new_fun


def _self_set_skip_existing(fun):
    @functools.wraps(fun)
    def new_func(self, *args, **kwargs):
        if self.skip_existing is not None:
            with set_skip_existing(self.skip_existing):
                return fun(self, *args, **kwargs)
        return fun(self, *args, **kwargs)

    return new_func


def _call_value_nets(
    value_net: TensorDictModuleBase,
    data: TensorDictBase,
    params: TensorDictBase,
    next_params: TensorDictBase,
    single_call: bool,
    value_key: NestedKey,
    detach_next: bool,
):
    in_keys = value_net.in_keys
    if single_call:
        for i, name in enumerate(data.names):
            if name == "time":
                ndim = i + 1
                break
        else:
            ndim = None
        if ndim is not None:
            # get data at t and last of t+1
            idx0 = (slice(None),) * (ndim - 1) + (slice(-1, None),)
            idx = (slice(None),) * (ndim - 1) + (slice(None, -1),)
            idx_ = (slice(None),) * (ndim - 1) + (slice(1, None),)
            data_in = torch.cat(
                [
                    data.select(*in_keys, value_key, strict=False),
                    data.get("next").select(*in_keys, value_key, strict=False)[idx0],
                ],
                ndim - 1,
            )
        else:
            if RL_WARNINGS:
                warnings.warn(
                    "Got a tensordict without a time-marked dimension, assuming time is along the last dimension. "
                    "This warning can be turned off by setting the environment variable RL_WARNINGS to False."
                )
            ndim = data.ndim
            idx = (slice(None),) * (ndim - 1) + (slice(None, data.shape[ndim - 1]),)
            idx_ = (slice(None),) * (ndim - 1) + (slice(data.shape[ndim - 1], None),)
            data_in = torch.cat(
                [
                    data.select(*in_keys, value_key, strict=False),
                    data.get("next").select(*in_keys, value_key, strict=False),
                ],
                ndim - 1,
            )

        # next_params should be None or be identical to params
        if next_params is not None and next_params is not params:
            raise ValueError(
                "the value at t and t+1 cannot be retrieved in a single call without recurring to vmap when both params and next params are passed."
            )
        if params is not None:
            value_est = value_net(data_in, params).get(value_key)
        else:
            value_est = value_net(data_in).get(value_key)
        value, value_ = value_est[idx], value_est[idx_]
    else:
        data_in = torch.stack(
            [
                data.select(*in_keys, value_key, strict=False),
                data.get("next").select(*in_keys, value_key, strict=False),
            ],
            0,
        )
        if (params is not None) ^ (next_params is not None):
            raise ValueError(
                "params and next_params must be either both provided or not."
            )
        elif params is not None:
            params_stack = torch.stack([params, next_params], 0)
            data_out = vmap(value_net, (0, 0))(data_in, params_stack)
        else:
            data_out = vmap(value_net, (0,))(data_in)
        value_est = data_out.get(value_key)
        value, value_ = value_est[0], value_est[1]
    data.set(value_key, value)
    data.set(("next", value_key), value_)
    if detach_next:
        value_ = value_.detach()
    return value, value_


class ValueEstimatorBase(TensorDictModuleBase):
    """An abstract parent class for value function modules.

    Its :meth:`ValueFunctionBase.forward` method will compute the value (given
    by the value network) and the value estimate (given by the value estimator)
    as well as the advantage and write these values in the output tensordict.

    If only the value estimate is needed, the :meth:`ValueFunctionBase.value_estimate`
    should be used instead.

    """

    @dataclass
    class _AcceptedKeys:
        """Maintains default values for all configurable tensordict keys.

        This class defines which tensordict keys can be set using '.set_keys(key_name=key_value)' and their
        default values.

        Attributes:
            advantage (NestedKey): The input tensordict key where the advantage is written to.
                Will be used for the underlying value estimator. Defaults to ``"advantage"``.
            value_target (NestedKey): The input tensordict key where the target state value is written to.
                Will be used for the underlying value estimator Defaults to ``"value_target"``.
            value (NestedKey): The input tensordict key where the state value is expected.
                Will be used for the underlying value estimator. Defaults to ``"state_value"``.
            reward (NestedKey): The input tensordict key where the reward is written to.
                Defaults to ``"reward"``.
            done (NestedKey): The key in the input TensorDict that indicates
                whether a trajectory is done.  Defaults to ``"done"``.
            terminated (NestedKey): The key in the input TensorDict that indicates
                whether a trajectory is terminated.  Defaults to ``"terminated"``.
            steps_to_next_obs_key (NestedKey): The key in the input tensordict
                that indicates the number of steps to the next observation.
                Defaults to ``"steps_to_next_obs"``.
        """

        advantage: NestedKey = "advantage"
        value_target: NestedKey = "value_target"
        value: NestedKey = "state_value"
        reward: NestedKey = "reward"
        done: NestedKey = "done"
        terminated: NestedKey = "terminated"
        steps_to_next_obs: NestedKey = "steps_to_next_obs"

    default_keys = _AcceptedKeys()
    value_network: Union[TensorDictModule, Callable]

    @property
    def advantage_key(self):
        return self.tensor_keys.advantage

    @property
    def value_key(self):
        return self.tensor_keys.value

    @property
    def value_target_key(self):
        return self.tensor_keys.value_target

    @property
    def reward_key(self):
        return self.tensor_keys.reward

    @property
    def done_key(self):
        return self.tensor_keys.done

    @property
    def terminated_key(self):
        return self.tensor_keys.terminated

    @property
    def steps_to_next_obs_key(self):
        return self.tensor_keys.steps_to_next_obs

    @abc.abstractmethod
    def forward(
        self,
        tensordict: TensorDictBase,
        params: Optional[TensorDictBase] = None,
        target_params: Optional[TensorDictBase] = None,
    ) -> TensorDictBase:
        """Computes the advantage estimate given the data in tensordict.

        If a functional module is provided, a nested TensorDict containing the parameters
        (and if relevant the target parameters) can be passed to the module.

        Args:
            tensordict (TensorDictBase): A TensorDict containing the data
                (an observation key, ``"action"``, ``("next", "reward")``,
                ``("next", "done")``, ``("next", "terminated")``,
                and ``"next"`` tensordict state as returned by the environment)
                necessary to compute the value estimates and the TDEstimate.
                The data passed to this module should be structured as
                :obj:`[*B, T, *F]` where :obj:`B` are
                the batch size, :obj:`T` the time dimension and :obj:`F` the
                feature dimension(s). The tensordict must have shape ``[*B, T]``.
            params (TensorDictBase, optional): A nested TensorDict containing the params
                to be passed to the functional value network module.
            target_params (TensorDictBase, optional): A nested TensorDict containing the
                target params to be passed to the functional value network module.

        Returns:
            An updated TensorDict with an advantage and a value_error keys as defined in the constructor.
        """
        raise NotImplementedError

    def __init__(
        self,
        *,
        value_network: TensorDictModule,
        shifted: bool = False,
        differentiable: bool = False,
        skip_existing: Optional[bool] = None,
        advantage_key: NestedKey = None,
        value_target_key: NestedKey = None,
        value_key: NestedKey = None,
    ):
        super().__init__()
        self._tensor_keys = None
        self.differentiable = differentiable
        self.skip_existing = skip_existing
        self.value_network = value_network
        self.dep_keys = {}
        self.shifted = shifted

        if advantage_key is not None:
            warnings.warn(
                "Setting 'advantage_key' via ctor is deprecated, use .set_keys(advantage_key='some_key') instead.",
                category=DeprecationWarning,
            )
            self.dep_keys["advantage"] = advantage_key
        if value_target_key is not None:
            warnings.warn(
                "Setting 'value_target_key' via ctor is deprecated, use .set_keys(value_target_key='some_key') instead.",
                category=DeprecationWarning,
            )
            self.dep_keys["value_target"] = value_target_key
        if value_key is not None:
            warnings.warn(
                "Setting 'value_key' via ctor is deprecated, use .set_keys(value_key='some_key') instead.",
                category=DeprecationWarning,
            )
            self.dep_keys["value"] = value_key

    @property
    def tensor_keys(self) -> _AcceptedKeys:
        if self._tensor_keys is None:
            self.set_keys()
        return self._tensor_keys

    @tensor_keys.setter
    def tensor_keys(self, value):
        if not isinstance(value, type(self._AcceptedKeys)):
            raise ValueError("value must be an instance of _AcceptedKeys")
        self._keys = value

    @property
    def in_keys(self):
        try:
            in_keys = (
                self.value_network.in_keys
                + [
                    ("next", self.tensor_keys.reward),
                    ("next", self.tensor_keys.done),
                    ("next", self.tensor_keys.terminated),
                ]
                + [("next", in_key) for in_key in self.value_network.in_keys]
            )
        except AttributeError:
            # value network does not have an `in_keys` attribute
            in_keys = []
            pass
        return in_keys

    @property
    def out_keys(self):
        return [
            self.tensor_keys.advantage,
            self.tensor_keys.value_target,
        ]

    def set_keys(self, **kwargs) -> None:
        """Set tensordict key names."""
        for key, value in kwargs.items():
            if not isinstance(value, (str, tuple)):
                raise ValueError(
                    f"key name must be of type NestedKey (Union[str, Tuple[str]]) but got {type(value)}"
                )
            if value is None:
                raise ValueError("tensordict keys cannot be None")
            if key not in self._AcceptedKeys.__dict__:
                raise KeyError(
                    f"{key} it not an accepted tensordict key for advantages"
                )
            if (
                key == "value"
                and hasattr(self.value_network, "out_keys")
                and (value not in self.value_network.out_keys)
            ):
                raise KeyError(
                    f"value key '{value}' not found in value network out_keys {self.value_network.out_keys}"
                )
        if self._tensor_keys is None:
            conf = asdict(self.default_keys)
            conf.update(self.dep_keys)
        else:
            conf = asdict(self._tensor_keys)
        conf.update(kwargs)
        self._tensor_keys = self._AcceptedKeys(**conf)

    def value_estimate(
        self,
        tensordict,
        target_params: Optional[TensorDictBase] = None,
        next_value: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """Gets a value estimate, usually used as a target value for the value network.

        If the state value key is present under ``tensordict.get(("next", self.tensor_keys.value))``
        then this value will be used without recurring to the value network.

        Args:
            tensordict (TensorDictBase): the tensordict containing the data to
                read.
            target_params (TensorDictBase, optional): A nested TensorDict containing the
                target params to be passed to the functional value network module.
            next_value (torch.Tensor, optional): the value of the next state
                or state-action pair. Exclusive with ``target_params``.
            **kwargs: the keyword arguments to be passed to the value network.

        Returns: a tensor corresponding to the state value.

        """
        raise NotImplementedError

    @property
    def is_functional(self):
        if isinstance(self.value_network, nn.Module):
            return is_functional(self.value_network)
        elif self.value_network is None:
            return None
        else:
            raise RuntimeError("Cannot determine if value network is functional.")

    @property
    def is_stateless(self):
        if not self.is_functional:
            return False
        return self.value_network._is_stateless

    def _next_value(self, tensordict, target_params, kwargs):
        step_td = step_mdp(tensordict, keep_other=False)
        if self.value_network is not None:
            if target_params is not None:
                kwargs["params"] = target_params
            with hold_out_net(self.value_network):
                self.value_network(step_td, **kwargs)
        next_value = step_td.get(self.tensor_keys.value)
        return next_value


class TD0Estimator(ValueEstimatorBase):
    """Temporal Difference (TD(0)) estimate of advantage function.

    AKA bootstrapped temporal difference or 1-step return.

    Keyword Args:
        gamma (scalar): exponential mean discount.
        value_network (TensorDictModule): value operator used to retrieve
            the value estimates.
        shifted (bool, optional): if ``True``, the value and next value are
            estimated with a single call to the value network. This is faster
            but is only valid whenever (1) the ``"next"`` value is shifted by
            only one time step (which is not the case with multi-step value
            estimation, for instance) and (2) when the parameters used at time
            ``t`` and ``t+1`` are identical (which is not the case when target
            parameters are to be used). Defaults to ``False``.
        average_rewards (bool, optional): if ``True``, rewards will be standardized
            before the TD is computed.
        differentiable (bool, optional): if ``True``, gradients are propagated through
            the computation of the value function. Default is ``False``.

            .. note::
              The proper way to make the function call non-differentiable is to
              decorate it in a `torch.no_grad()` context manager/decorator or
              pass detached parameters for functional modules.

        skip_existing (bool, optional): if ``True``, the value network will skip
            modules which outputs are already present in the tensordict.
            Defaults to ``None``, ie. the value of :func:`tensordict.nn.skip_existing()`
            is not affected.
        advantage_key (str or tuple of str, optional): [Deprecated] the key of
            the advantage entry.  Defaults to ``"advantage"``.
        value_target_key (str or tuple of str, optional): [Deprecated] the key
            of the advantage entry.  Defaults to ``"value_target"``.
        value_key (str or tuple of str, optional): [Deprecated] the value key to
            read from the input tensordict.  Defaults to ``"state_value"``.

    """

    def __init__(
        self,
        *,
        gamma: Union[float, torch.Tensor],
        value_network: TensorDictModule,
        shifted: bool = False,
        average_rewards: bool = False,
        differentiable: bool = False,
        advantage_key: NestedKey = None,
        value_target_key: NestedKey = None,
        value_key: NestedKey = None,
        skip_existing: Optional[bool] = None,
    ):
        super().__init__(
            value_network=value_network,
            differentiable=differentiable,
            shifted=shifted,
            advantage_key=advantage_key,
            value_target_key=value_target_key,
            value_key=value_key,
            skip_existing=skip_existing,
        )
        try:
            device = next(value_network.parameters()).device
        except (AttributeError, StopIteration):
            device = torch.device("cpu")
        self.register_buffer("gamma", torch.tensor(gamma, device=device))
        self.average_rewards = average_rewards

    @_self_set_skip_existing
    @_self_set_grad_enabled
    @dispatch
    def forward(
        self,
        tensordict: TensorDictBase,
        params: Optional[TensorDictBase] = None,
        target_params: Optional[TensorDictBase] = None,
    ) -> TensorDictBase:
        """Computes the TD(0) advantage given the data in tensordict.

        If a functional module is provided, a nested TensorDict containing the parameters
        (and if relevant the target parameters) can be passed to the module.

        Args:
            tensordict (TensorDictBase): A TensorDict containing the data
                (an observation key, ``"action"``, ``("next", "reward")``,
                ``("next", "done")``, ``("next", "terminated")``, and ``"next"``
                tensordict state as returned by the environment) necessary to
                compute the value estimates and the TDEstimate.
                The data passed to this module should be structured as
                :obj:`[*B, T, *F]` where :obj:`B` are
                the batch size, :obj:`T` the time dimension and :obj:`F` the
                feature dimension(s). The tensordict must have shape ``[*B, T]``.
            params (TensorDictBase, optional): A nested TensorDict containing the params
                to be passed to the functional value network module.
            target_params (TensorDictBase, optional): A nested TensorDict containing the
                target params to be passed to the functional value network module.

        Returns:
            An updated TensorDict with an advantage and a value_error keys as defined in the constructor.

        Examples:
            >>> from tensordict import TensorDict
            >>> value_net = TensorDictModule(
            ...     nn.Linear(3, 1), in_keys=["obs"], out_keys=["state_value"]
            ... )
            >>> module = TDEstimate(
            ...     gamma=0.98,
            ...     value_network=value_net,
            ... )
            >>> obs, next_obs = torch.randn(2, 1, 10, 3)
            >>> reward = torch.randn(1, 10, 1)
            >>> done = torch.zeros(1, 10, 1, dtype=torch.bool)
            >>> terminated = torch.zeros(1, 10, 1, dtype=torch.bool)
            >>> tensordict = TensorDict({"obs": obs, "next": {"obs": next_obs, "done": done, "terminated": terminated, "reward": reward}}, [1, 10])
            >>> _ = module(tensordict)
            >>> assert "advantage" in tensordict.keys()

        The module supports non-tensordict (i.e. unpacked tensordict) inputs too:

        Examples:
            >>> value_net = TensorDictModule(
            ...     nn.Linear(3, 1), in_keys=["obs"], out_keys=["state_value"]
            ... )
            >>> module = TDEstimate(
            ...     gamma=0.98,
            ...     value_network=value_net,
            ... )
            >>> obs, next_obs = torch.randn(2, 1, 10, 3)
            >>> reward = torch.randn(1, 10, 1)
            >>> done = torch.zeros(1, 10, 1, dtype=torch.bool)
            >>> terminated = torch.zeros(1, 10, 1, dtype=torch.bool)
            >>> advantage, value_target = module(obs=obs, next_reward=reward, next_done=done, next_obs=next_obs, next_terminated=terminated)

        """
        if tensordict.batch_dims < 1:
            raise RuntimeError(
                "Expected input tensordict to have at least one dimensions, got"
                f"tensordict.batch_size = {tensordict.batch_size}"
            )

        if self.is_stateless and params is None:
            raise RuntimeError(
                "Expected params to be passed to advantage module but got none."
            )
        if self.value_network is not None:
            if params is not None:
                params = params.detach()
                if target_params is None:
                    target_params = params.clone(False)
            with hold_out_net(self.value_network):
                # we may still need to pass gradient, but we don't want to assign grads to
                # value net params
                value, next_value = _call_value_nets(
                    value_net=self.value_network,
                    data=tensordict,
                    params=params,
                    next_params=target_params,
                    single_call=self.shifted,
                    value_key=self.tensor_keys.value,
                    detach_next=True,
                )
        else:
            value = tensordict.get(self.tensor_keys.value)
            next_value = tensordict.get(("next", self.tensor_keys.value))

        value_target = self.value_estimate(tensordict, next_value=next_value)
        tensordict.set(self.tensor_keys.advantage, value_target - value)
        tensordict.set(self.tensor_keys.value_target, value_target)
        return tensordict

    def value_estimate(
        self,
        tensordict,
        target_params: Optional[TensorDictBase] = None,
        next_value: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        reward = tensordict.get(("next", self.tensor_keys.reward))
        device = reward.device
        gamma = self.gamma.to(device)
        steps_to_next_obs = tensordict.get(self.tensor_keys.steps_to_next_obs, None)
        if steps_to_next_obs is not None:
            gamma = gamma ** steps_to_next_obs.view_as(reward)

        if self.average_rewards:
            reward = reward - reward.mean()
            reward = reward / reward.std().clamp_min(1e-4)
            tensordict.set(
                ("next", self.tensor_keys.reward), reward
            )  # we must update the rewards if they are used later in the code
        if next_value is None:
            next_value = self._next_value(tensordict, target_params, kwargs=kwargs)

        done = tensordict.get(("next", self.tensor_keys.done))
        terminated = tensordict.get(("next", self.tensor_keys.terminated), default=done)
        value_target = td0_return_estimate(
            gamma=gamma,
            next_state_value=next_value,
            reward=reward,
            done=done,
            terminated=terminated,
        )
        return value_target


class TD1Estimator(ValueEstimatorBase):
    r""":math:`\infty`-Temporal Difference (TD(1)) estimate of advantage function.

    Keyword Args:
        gamma (scalar): exponential mean discount.
        value_network (TensorDictModule): value operator used to retrieve the value estimates.
        average_rewards (bool, optional): if ``True``, rewards will be standardized
            before the TD is computed.
        differentiable (bool, optional): if ``True``, gradients are propagated through
            the computation of the value function. Default is ``False``.

            .. note::
              The proper way to make the function call non-differentiable is to
              decorate it in a `torch.no_grad()` context manager/decorator or
              pass detached parameters for functional modules.

        skip_existing (bool, optional): if ``True``, the value network will skip
            modules which outputs are already present in the tensordict.
            Defaults to ``None``, ie. the value of :func:`tensordict.nn.skip_existing()`
            is not affected.
        advantage_key (str or tuple of str, optional): [Deprecated] the key of
            the advantage entry.  Defaults to ``"advantage"``.
        value_target_key (str or tuple of str, optional): [Deprecated] the key
            of the advantage entry.  Defaults to ``"value_target"``.
        value_key (str or tuple of str, optional): [Deprecated] the value key to
            read from the input tensordict.  Defaults to ``"state_value"``.
        shifted (bool, optional): if ``True``, the value and next value are
            estimated with a single call to the value network. This is faster
            but is only valid whenever (1) the ``"next"`` value is shifted by
            only one time step (which is not the case with multi-step value
            estimation, for instance) and (2) when the parameters used at time
            ``t`` and ``t+1`` are identical (which is not the case when target
            parameters are to be used). Defaults to ``False``.

    """

    def __init__(
        self,
        *,
        gamma: Union[float, torch.Tensor],
        value_network: TensorDictModule,
        average_rewards: bool = False,
        differentiable: bool = False,
        skip_existing: Optional[bool] = None,
        advantage_key: NestedKey = None,
        value_target_key: NestedKey = None,
        value_key: NestedKey = None,
        shifted: bool = False,
    ):
        super().__init__(
            value_network=value_network,
            differentiable=differentiable,
            advantage_key=advantage_key,
            value_target_key=value_target_key,
            value_key=value_key,
            shifted=shifted,
            skip_existing=skip_existing,
        )
        try:
            device = next(value_network.parameters()).device
        except (AttributeError, StopIteration):
            device = torch.device("cpu")
        self.register_buffer("gamma", torch.tensor(gamma, device=device))
        self.average_rewards = average_rewards

    @_self_set_skip_existing
    @_self_set_grad_enabled
    @dispatch
    def forward(
        self,
        tensordict: TensorDictBase,
        params: Optional[TensorDictBase] = None,
        target_params: Optional[TensorDictBase] = None,
    ) -> TensorDictBase:
        """Computes the TD(1) advantage given the data in tensordict.

        If a functional module is provided, a nested TensorDict containing the parameters
        (and if relevant the target parameters) can be passed to the module.

        Args:
            tensordict (TensorDictBase): A TensorDict containing the data
                (an observation key, ``"action"``, ``("next", "reward")``,
                ``("next", "done")``, ``("next", "terminated")``,
                and ``"next"`` tensordict state as returned by the environment)
                necessary to compute the value estimates and the TDEstimate.
                The data passed to this module should be structured as :obj:`[*B, T, *F]` where :obj:`B` are
                the batch size, :obj:`T` the time dimension and :obj:`F` the feature dimension(s).
                The tensordict must have shape ``[*B, T]``.
            params (TensorDictBase, optional): A nested TensorDict containing the params
                to be passed to the functional value network module.
            target_params (TensorDictBase, optional): A nested TensorDict containing the
                target params to be passed to the functional value network module.

        Returns:
            An updated TensorDict with an advantage and a value_error keys as defined in the constructor.

        Examples:
            >>> from tensordict import TensorDict
            >>> value_net = TensorDictModule(
            ...     nn.Linear(3, 1), in_keys=["obs"], out_keys=["state_value"]
            ... )
            >>> module = TDEstimate(
            ...     gamma=0.98,
            ...     value_network=value_net,
            ... )
            >>> obs, next_obs = torch.randn(2, 1, 10, 3)
            >>> reward = torch.randn(1, 10, 1)
            >>> done = torch.zeros(1, 10, 1, dtype=torch.bool)
            >>> terminated = torch.zeros(1, 10, 1, dtype=torch.bool)
            >>> tensordict = TensorDict({"obs": obs, "next": {"obs": next_obs, "done": done, "reward": reward, "terminated": terminated}}, [1, 10])
            >>> _ = module(tensordict)
            >>> assert "advantage" in tensordict.keys()

        The module supports non-tensordict (i.e. unpacked tensordict) inputs too:

        Examples:
            >>> value_net = TensorDictModule(
            ...     nn.Linear(3, 1), in_keys=["obs"], out_keys=["state_value"]
            ... )
            >>> module = TDEstimate(
            ...     gamma=0.98,
            ...     value_network=value_net,
            ... )
            >>> obs, next_obs = torch.randn(2, 1, 10, 3)
            >>> reward = torch.randn(1, 10, 1)
            >>> done = torch.zeros(1, 10, 1, dtype=torch.bool)
            >>> terminated = torch.zeros(1, 10, 1, dtype=torch.bool)
            >>> advantage, value_target = module(obs=obs, next_reward=reward, next_done=done, next_obs=next_obs, next_terminated=terminated)

        """
        if tensordict.batch_dims < 1:
            raise RuntimeError(
                "Expected input tensordict to have at least one dimensions, got"
                f"tensordict.batch_size = {tensordict.batch_size}"
            )

        if self.is_stateless and params is None:
            raise RuntimeError(
                "Expected params to be passed to advantage module but got none."
            )
        if self.value_network is not None:
            if params is not None:
                params = params.detach()
                if target_params is None:
                    target_params = params.clone(False)
            with hold_out_net(self.value_network):
                # we may still need to pass gradient, but we don't want to assign grads to
                # value net params
                value, next_value = _call_value_nets(
                    value_net=self.value_network,
                    data=tensordict,
                    params=params,
                    next_params=target_params,
                    single_call=self.shifted,
                    value_key=self.tensor_keys.value,
                    detach_next=True,
                )
        else:
            value = tensordict.get(self.tensor_keys.value)
            next_value = tensordict.get(("next", self.tensor_keys.value))

        value_target = self.value_estimate(tensordict, next_value=next_value)

        tensordict.set(self.tensor_keys.advantage, value_target - value)
        tensordict.set(self.tensor_keys.value_target, value_target)
        return tensordict

    def value_estimate(
        self,
        tensordict,
        target_params: Optional[TensorDictBase] = None,
        next_value: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        reward = tensordict.get(("next", self.tensor_keys.reward))
        device = reward.device
        gamma = self.gamma.to(device)
        steps_to_next_obs = tensordict.get(self.tensor_keys.steps_to_next_obs, None)
        if steps_to_next_obs is not None:
            gamma = gamma ** steps_to_next_obs.view_as(reward)

        if self.average_rewards:
            reward = reward - reward.mean()
            reward = reward / reward.std().clamp_min(1e-4)
            tensordict.set(
                ("next", self.tensor_keys.reward), reward
            )  # we must update the rewards if they are used later in the code
        if next_value is None:
            next_value = self._next_value(tensordict, target_params, kwargs=kwargs)

        done = tensordict.get(("next", self.tensor_keys.done))
        terminated = tensordict.get(("next", self.tensor_keys.terminated), default=done)
        value_target = vec_td1_return_estimate(
            gamma,
            next_value,
            reward,
            done=done,
            terminated=terminated,
            time_dim=tensordict.ndim - 1,
        )
        return value_target


class TDLambdaEstimator(ValueEstimatorBase):
    r"""TD(:math:`\lambda`) estimate of advantage function.

    Args:
        gamma (scalar): exponential mean discount.
        lmbda (scalar): trajectory discount.
        value_network (TensorDictModule): value operator used to retrieve the value estimates.
        average_rewards (bool, optional): if ``True``, rewards will be standardized
            before the TD is computed.
        differentiable (bool, optional): if ``True``, gradients are propagated through
            the computation of the value function. Default is ``False``.

            .. note::
              The proper way to make the function call non-differentiable is to
              decorate it in a `torch.no_grad()` context manager/decorator or
              pass detached parameters for functional modules.

        vectorized (bool, optional): whether to use the vectorized version of the
            lambda return. Default is `True`.
        skip_existing (bool, optional): if ``True``, the value network will skip
            modules which outputs are already present in the tensordict.
            Defaults to ``None``, ie. the value of :func:`tensordict.nn.skip_existing()`
            is not affected.
        advantage_key (str or tuple of str, optional): [Deprecated] the key of
            the advantage entry.  Defaults to ``"advantage"``.
        value_target_key (str or tuple of str, optional): [Deprecated] the key
            of the advantage entry.  Defaults to ``"value_target"``.
        value_key (str or tuple of str, optional): [Deprecated] the value key to
            read from the input tensordict.  Defaults to ``"state_value"``.
        shifted (bool, optional): if ``True``, the value and next value are
            estimated with a single call to the value network. This is faster
            but is only valid whenever (1) the ``"next"`` value is shifted by
            only one time step (which is not the case with multi-step value
            estimation, for instance) and (2) when the parameters used at time
            ``t`` and ``t+1`` are identical (which is not the case when target
            parameters are to be used). Defaults to ``False``.

    """

    def __init__(
        self,
        *,
        gamma: Union[float, torch.Tensor],
        lmbda: Union[float, torch.Tensor],
        value_network: TensorDictModule,
        average_rewards: bool = False,
        differentiable: bool = False,
        vectorized: bool = True,
        skip_existing: Optional[bool] = None,
        advantage_key: NestedKey = None,
        value_target_key: NestedKey = None,
        value_key: NestedKey = None,
        shifted: bool = False,
    ):
        super().__init__(
            value_network=value_network,
            differentiable=differentiable,
            advantage_key=advantage_key,
            value_target_key=value_target_key,
            value_key=value_key,
            skip_existing=skip_existing,
            shifted=shifted,
        )
        try:
            device = next(value_network.parameters()).device
        except (AttributeError, StopIteration):
            device = torch.device("cpu")
        self.register_buffer("gamma", torch.tensor(gamma, device=device))
        self.register_buffer("lmbda", torch.tensor(lmbda, device=device))
        self.average_rewards = average_rewards
        self.vectorized = vectorized

    @_self_set_skip_existing
    @_self_set_grad_enabled
    @dispatch
    def forward(
        self,
        tensordict: TensorDictBase,
        params: Optional[List[Tensor]] = None,
        target_params: Optional[List[Tensor]] = None,
    ) -> TensorDictBase:
        r"""Computes the TD(:math:`\lambda`) advantage given the data in tensordict.

        If a functional module is provided, a nested TensorDict containing the parameters
        (and if relevant the target parameters) can be passed to the module.

        Args:
            tensordict (TensorDictBase): A TensorDict containing the data
                (an observation key, ``"action"``, ``("next", "reward")``,
                ``("next", "done")``, ``("next", "terminated")``,
                and ``"next"`` tensordict state as returned by the environment)
                necessary to compute the value estimates and the TDLambdaEstimate.
                The data passed to this module should be structured as :obj:`[*B, T, *F]` where :obj:`B` are
                the batch size, :obj:`T` the time dimension and :obj:`F` the feature dimension(s).
                The tensordict must have shape ``[*B, T]``.
            params (TensorDictBase, optional): A nested TensorDict containing the params
                to be passed to the functional value network module.
            target_params (TensorDictBase, optional): A nested TensorDict containing the
                target params to be passed to the functional value network module.

        Returns:
            An updated TensorDict with an advantage and a value_error keys as defined in the constructor.

        Examples:
            >>> from tensordict import TensorDict
            >>> value_net = TensorDictModule(
            ...     nn.Linear(3, 1), in_keys=["obs"], out_keys=["state_value"]
            ... )
            >>> module = TDLambdaEstimator(
            ...     gamma=0.98,
            ...     lmbda=0.94,
            ...     value_network=value_net,
            ... )
            >>> obs, next_obs = torch.randn(2, 1, 10, 3)
            >>> reward = torch.randn(1, 10, 1)
            >>> done = torch.zeros(1, 10, 1, dtype=torch.bool)
            >>> terminated = torch.zeros(1, 10, 1, dtype=torch.bool)
            >>> tensordict = TensorDict({"obs": obs, "next": {"obs": next_obs, "done": done, "reward": reward, "terminated": terminated}}, [1, 10])
            >>> _ = module(tensordict)
            >>> assert "advantage" in tensordict.keys()

        The module supports non-tensordict (i.e. unpacked tensordict) inputs too:

        Examples:
            >>> value_net = TensorDictModule(
            ...     nn.Linear(3, 1), in_keys=["obs"], out_keys=["state_value"]
            ... )
            >>> module = TDLambdaEstimator(
            ...     gamma=0.98,
            ...     lmbda=0.94,
            ...     value_network=value_net,
            ... )
            >>> obs, next_obs = torch.randn(2, 1, 10, 3)
            >>> reward = torch.randn(1, 10, 1)
            >>> done = torch.zeros(1, 10, 1, dtype=torch.bool)
            >>> terminated = torch.zeros(1, 10, 1, dtype=torch.bool)
            >>> advantage, value_target = module(obs=obs, next_reward=reward, next_done=done, next_obs=next_obs, next_terminated=terminated)

        """
        if tensordict.batch_dims < 1:
            raise RuntimeError(
                "Expected input tensordict to have at least one dimensions, got"
                f"tensordict.batch_size = {tensordict.batch_size}"
            )
        if self.is_stateless and params is None:
            raise RuntimeError(
                "Expected params to be passed to advantage module but got none."
            )
        if self.value_network is not None:
            if params is not None:
                params = params.detach()
                if target_params is None:
                    target_params = params.clone(False)
            with hold_out_net(self.value_network):
                # we may still need to pass gradient, but we don't want to assign grads to
                # value net params
                value, next_value = _call_value_nets(
                    value_net=self.value_network,
                    data=tensordict,
                    params=params,
                    next_params=target_params,
                    single_call=self.shifted,
                    value_key=self.tensor_keys.value,
                    detach_next=True,
                )
        else:
            value = tensordict.get(self.tensor_keys.value)
            next_value = tensordict.get(("next", self.tensor_keys.value))
        value_target = self.value_estimate(tensordict, next_value=next_value)

        tensordict.set(self.tensor_keys.advantage, value_target - value)
        tensordict.set(self.tensor_keys.value_target, value_target)
        return tensordict

    def value_estimate(
        self,
        tensordict,
        target_params: Optional[TensorDictBase] = None,
        next_value: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        reward = tensordict.get(("next", self.tensor_keys.reward))
        device = reward.device
        gamma = self.gamma.to(device)
        steps_to_next_obs = tensordict.get(self.tensor_keys.steps_to_next_obs, None)
        if steps_to_next_obs is not None:
            gamma = gamma ** steps_to_next_obs.view_as(reward)

        lmbda = self.lmbda
        if self.average_rewards:
            reward = reward - reward.mean()
            reward = reward / reward.std().clamp_min(1e-4)
            tensordict.set(
                ("next", self.tensor_keys.steps_to_next_obs), reward
            )  # we must update the rewards if they are used later in the code

        if next_value is None:
            next_value = self._next_value(tensordict, target_params, kwargs=kwargs)

        done = tensordict.get(("next", self.tensor_keys.done))
        terminated = tensordict.get(("next", self.tensor_keys.terminated), default=done)
        if self.vectorized:
            val = vec_td_lambda_return_estimate(
                gamma,
                lmbda,
                next_value,
                reward,
                done=done,
                terminated=terminated,
                time_dim=tensordict.ndim - 1,
            )
        else:
            val = td_lambda_return_estimate(
                gamma,
                lmbda,
                next_value,
                reward,
                done=done,
                terminated=terminated,
                time_dim=tensordict.ndim - 1,
            )
        return val


class GAE(ValueEstimatorBase):
    """A class wrapper around the generalized advantage estimate functional.

    Refer to "HIGH-DIMENSIONAL CONTINUOUS CONTROL USING GENERALIZED ADVANTAGE ESTIMATION"
    https://arxiv.org/pdf/1506.02438.pdf for more context.

    Args:
        gamma (scalar): exponential mean discount.
        lmbda (scalar): trajectory discount.
        value_network (TensorDictModule): value operator used to retrieve the value estimates.
        average_gae (bool): if ``True``, the resulting GAE values will be standardized.
            Default is ``False``.
        differentiable (bool, optional): if ``True``, gradients are propagated through
            the computation of the value function. Default is ``False``.

            .. note::
              The proper way to make the function call non-differentiable is to
              decorate it in a `torch.no_grad()` context manager/decorator or
              pass detached parameters for functional modules.

        vectorized (bool, optional): whether to use the vectorized version of the
            lambda return. Default is `True`.
        skip_existing (bool, optional): if ``True``, the value network will skip
            modules which outputs are already present in the tensordict.
            Defaults to ``None``, ie. the value of :func:`tensordict.nn.skip_existing()`
            is not affected.
            Defaults to "state_value".
        advantage_key (str or tuple of str, optional): [Deprecated] the key of
            the advantage entry.  Defaults to ``"advantage"``.
        value_target_key (str or tuple of str, optional): [Deprecated] the key
            of the advantage entry.  Defaults to ``"value_target"``.
        value_key (str or tuple of str, optional): [Deprecated] the value key to
            read from the input tensordict.  Defaults to ``"state_value"``.
        shifted (bool, optional): if ``True``, the value and next value are
            estimated with a single call to the value network. This is faster
            but is only valid whenever (1) the ``"next"`` value is shifted by
            only one time step (which is not the case with multi-step value
            estimation, for instance) and (2) when the parameters used at time
            ``t`` and ``t+1`` are identical (which is not the case when target
            parameters are to be used). Defaults to ``False``.

    GAE will return an :obj:`"advantage"` entry containing the advange value. It will also
    return a :obj:`"value_target"` entry with the return value that is to be used
    to train the value network. Finally, if :obj:`gradient_mode` is ``True``,
    an additional and differentiable :obj:`"value_error"` entry will be returned,
    which simple represents the difference between the return and the value network
    output (i.e. an additional distance loss should be applied to that signed value).

    .. note::
      As other advantage functions do, if the ``value_key`` is already present
      in the input tensordict, the GAE module will ignore the calls to the value
      network (if any) and use the provided value instead.

    """

    def __init__(
        self,
        *,
        gamma: Union[float, torch.Tensor],
        lmbda: float,
        value_network: TensorDictModule,
        average_gae: bool = False,
        differentiable: bool = False,
        vectorized: bool = True,
        skip_existing: Optional[bool] = None,
        advantage_key: NestedKey = None,
        value_target_key: NestedKey = None,
        value_key: NestedKey = None,
        shifted: bool = False,
    ):
        super().__init__(
            shifted=shifted,
            value_network=value_network,
            differentiable=differentiable,
            advantage_key=advantage_key,
            value_target_key=value_target_key,
            value_key=value_key,
            skip_existing=skip_existing,
        )
        try:
            device = next(value_network.parameters()).device
        except (AttributeError, StopIteration):
            device = torch.device("cpu")
        self.register_buffer("gamma", torch.tensor(gamma, device=device))
        self.register_buffer("lmbda", torch.tensor(lmbda, device=device))
        self.average_gae = average_gae
        self.vectorized = vectorized

    @_self_set_skip_existing
    @_self_set_grad_enabled
    @dispatch
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
                (an observation key, ``"action"``, ``("next", "reward")``,
                ``("next", "done")``, ``("next", "terminated")``,
                and ``"next"`` tensordict state as returned by the environment)
                necessary to compute the value estimates and the GAE.
                The data passed to this module should be structured as :obj:`[*B, T, *F]` where :obj:`B` are
                the batch size, :obj:`T` the time dimension and :obj:`F` the feature dimension(s).
                The tensordict must have shape ``[*B, T]``.
            params (TensorDictBase, optional): A nested TensorDict containing the params
                to be passed to the functional value network module.
            target_params (TensorDictBase, optional): A nested TensorDict containing the
                target params to be passed to the functional value network module.

        Returns:
            An updated TensorDict with an advantage and a value_error keys as defined in the constructor.

        Examples:
            >>> from tensordict import TensorDict
            >>> value_net = TensorDictModule(
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
            >>> terminated = torch.zeros(1, 10, 1, dtype=torch.bool)
            >>> tensordict = TensorDict({"obs": obs, "next": {"obs": next_obs}, "done": done, "reward": reward, "terminated": terminated}, [1, 10])
            >>> _ = module(tensordict)
            >>> assert "advantage" in tensordict.keys()

        The module supports non-tensordict (i.e. unpacked tensordict) inputs too:

        Examples:
            >>> value_net = TensorDictModule(
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
            >>> terminated = torch.zeros(1, 10, 1, dtype=torch.bool)
            >>> advantage, value_target = module(obs=obs, next_reward=reward, next_done=done, next_obs=next_obs, next_terminated=terminated)

        """
        if tensordict.batch_dims < 1:
            raise RuntimeError(
                "Expected input tensordict to have at least one dimensions, got "
                f"tensordict.batch_size = {tensordict.batch_size}"
            )
        reward = tensordict.get(("next", self.tensor_keys.reward))
        device = reward.device
        gamma, lmbda = self.gamma.to(device), self.lmbda.to(device)
        steps_to_next_obs = tensordict.get(self.tensor_keys.steps_to_next_obs, None)
        if steps_to_next_obs is not None:
            gamma = gamma ** steps_to_next_obs.view_as(reward)

        if self.value_network is not None:
            if params is not None:
                params = params.detach()
                if target_params is None:
                    target_params = params.clone(False)
            with hold_out_net(self.value_network):
                # we may still need to pass gradient, but we don't want to assign grads to
                # value net params
                value, next_value = _call_value_nets(
                    value_net=self.value_network,
                    data=tensordict,
                    params=params,
                    next_params=target_params,
                    single_call=self.shifted,
                    value_key=self.tensor_keys.value,
                    detach_next=True,
                )
        else:
            value = tensordict.get(self.tensor_keys.value)
            next_value = tensordict.get(("next", self.tensor_keys.value))

        done = tensordict.get(("next", self.tensor_keys.done))
        terminated = tensordict.get(("next", self.tensor_keys.terminated), default=done)
        if self.vectorized:
            adv, value_target = vec_generalized_advantage_estimate(
                gamma,
                lmbda,
                value,
                next_value,
                reward,
                done=done,
                terminated=terminated,
                time_dim=tensordict.ndim - 1,
            )
        else:
            adv, value_target = generalized_advantage_estimate(
                gamma,
                lmbda,
                value,
                next_value,
                reward,
                done=done,
                terminated=terminated,
                time_dim=tensordict.ndim - 1,
            )

        if self.average_gae:
            loc = adv.mean()
            scale = adv.std().clamp_min(1e-4)
            adv = adv - loc
            adv = adv / scale

        tensordict.set(self.tensor_keys.advantage, adv)
        tensordict.set(self.tensor_keys.value_target, value_target)

        return tensordict

    def value_estimate(
        self,
        tensordict,
        params: Optional[TensorDictBase] = None,
        target_params: Optional[TensorDictBase] = None,
        **kwargs,
    ):
        if tensordict.batch_dims < 1:
            raise RuntimeError(
                "Expected input tensordict to have at least one dimensions, got"
                f"tensordict.batch_size = {tensordict.batch_size}"
            )
        reward = tensordict.get(("next", self.tensor_keys.reward))
        device = reward.device
        gamma, lmbda = self.gamma.to(device), self.lmbda.to(device)
        steps_to_next_obs = tensordict.get(self.tensor_keys.steps_to_next_obs, None)
        if steps_to_next_obs is not None:
            gamma = gamma ** steps_to_next_obs.view_as(reward)

        if self.is_stateless and params is None:
            raise RuntimeError(
                "Expected params to be passed to advantage module but got none."
            )
        if self.value_network is not None:
            if params is not None:
                params = params.detach()
                if target_params is None:
                    target_params = params.clone(False)
            with hold_out_net(self.value_network):
                # we may still need to pass gradient, but we don't want to assign grads to
                # value net params
                value, next_value = _call_value_nets(
                    value_net=self.value_network,
                    data=tensordict,
                    params=params,
                    next_params=target_params,
                    single_call=self.shifted,
                    value_key=self.tensor_keys.value,
                    detach_next=True,
                )
        else:
            value = tensordict.get(self.tensor_keys.value)
            next_value = tensordict.get(("next", self.tensor_keys.value))
        done = tensordict.get(("next", self.tensor_keys.done))
        terminated = tensordict.get(("next", self.tensor_keys.terminated), default=done)
        _, value_target = vec_generalized_advantage_estimate(
            gamma,
            lmbda,
            value,
            next_value,
            reward,
            done=done,
            terminated=terminated,
            time_dim=tensordict.ndim - 1,
        )
        return value_target


def _deprecate_class(cls, new_cls):
    @wraps(cls.__init__)
    def new_init(self, *args, **kwargs):
        warnings.warn(f"class {cls} is deprecated, please use {new_cls} instead.")
        cls.__init__(self, *args, **kwargs)

    cls.__init__ = new_init


TD0Estimate = type("TD0Estimate", TD0Estimator.__bases__, dict(TD0Estimator.__dict__))
_deprecate_class(TD0Estimate, TD0Estimator)
TD1Estimate = type("TD1Estimate", TD1Estimator.__bases__, dict(TD1Estimator.__dict__))
_deprecate_class(TD1Estimate, TD1Estimator)
TDLambdaEstimate = type(
    "TDLambdaEstimate", TDLambdaEstimator.__bases__, dict(TDLambdaEstimator.__dict__)
)
_deprecate_class(TDLambdaEstimate, TDLambdaEstimator)
