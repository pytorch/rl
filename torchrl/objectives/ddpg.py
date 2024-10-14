# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Tuple

import torch
from tensordict import TensorDict, TensorDictBase, TensorDictParams
from tensordict.nn import dispatch, TensorDictModule

from tensordict.utils import NestedKey, unravel_key
from torchrl.modules.tensordict_module.actors import ActorCriticWrapper
from torchrl.objectives.common import LossModule
from torchrl.objectives.utils import (
    _cache_values,
    _GAMMA_LMBDA_DEPREC_ERROR,
    _reduce,
    default_value_kwargs,
    distance_loss,
    ValueEstimators,
)
from torchrl.objectives.value import TD0Estimator, TD1Estimator, TDLambdaEstimator


class DDPGLoss(LossModule):
    """The DDPG Loss class.

    Args:
        actor_network (TensorDictModule): a policy operator.
        value_network (TensorDictModule): a Q value operator.
        loss_function (str): loss function for the value discrepancy. Can be one of "l1", "l2" or "smooth_l1".
        delay_actor (bool, optional): whether to separate the target actor networks from the actor networks used for
            data collection. Default is ``False``.
        delay_value (bool, optional): whether to separate the target value networks from the value networks used for
            data collection. Default is ``True``.
        separate_losses (bool, optional): if ``True``, shared parameters between
            policy and critic will only be trained on the policy loss.
            Defaults to ``False``, i.e., gradients are propagated to shared
            parameters for both policy and critic losses.
        reduction (str, optional): Specifies the reduction to apply to the output:
            ``"none"`` | ``"mean"`` | ``"sum"``. ``"none"``: no reduction will be applied,
            ``"mean"``: the sum of the output will be divided by the number of
            elements in the output, ``"sum"``: the output will be summed. Default: ``"mean"``.

    Examples:
        >>> import torch
        >>> from torch import nn
        >>> from torchrl.data import Bounded
        >>> from torchrl.modules.tensordict_module.actors import Actor, ValueOperator
        >>> from torchrl.objectives.ddpg import DDPGLoss
        >>> from tensordict import TensorDict
        >>> n_act, n_obs = 4, 3
        >>> spec = Bounded(-torch.ones(n_act), torch.ones(n_act), (n_act,))
        >>> actor = Actor(spec=spec, module=nn.Linear(n_obs, n_act))
        >>> class ValueClass(nn.Module):
        ...     def __init__(self):
        ...         super().__init__()
        ...         self.linear = nn.Linear(n_obs + n_act, 1)
        ...     def forward(self, obs, act):
        ...         return self.linear(torch.cat([obs, act], -1))
        >>> module = ValueClass()
        >>> value = ValueOperator(
        ...     module=module,
        ...     in_keys=["observation", "action"])
        >>> loss = DDPGLoss(actor, value)
        >>> batch = [2, ]
        >>> data = TensorDict({
        ...        "observation": torch.randn(*batch, n_obs),
        ...        "action": spec.rand(batch),
        ...        ("next", "done"): torch.zeros(*batch, 1, dtype=torch.bool),
        ...        ("next", "terminated"): torch.zeros(*batch, 1, dtype=torch.bool),
        ...        ("next", "reward"): torch.randn(*batch, 1),
        ...        ("next", "observation"): torch.randn(*batch, n_obs),
        ...    }, batch)
        >>> loss(data)
        TensorDict(
            fields={
                loss_actor: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False),
                loss_value: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False),
                pred_value: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False),
                pred_value_max: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False),
                target_value: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False),
                target_value_max: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False)},
            batch_size=torch.Size([]),
            device=None,
            is_shared=False)

    This class is compatible with non-tensordict based modules too and can be
    used without recurring to any tensordict-related primitive. In this case,
    the expected keyword arguments are:
    ``["next_reward", "next_done", "next_terminated"]`` + in_keys of the actor_network and value_network.
    The return value is a tuple of tensors in the following order:
    ``["loss_actor", "loss_value", "pred_value", "target_value", "pred_value_max", "target_value_max"]``

    Examples:
        >>> import torch
        >>> from torch import nn
        >>> from torchrl.data import Bounded
        >>> from torchrl.modules.tensordict_module.actors import Actor, ValueOperator
        >>> from torchrl.objectives.ddpg import DDPGLoss
        >>> _ = torch.manual_seed(42)
        >>> n_act, n_obs = 4, 3
        >>> spec = Bounded(-torch.ones(n_act), torch.ones(n_act), (n_act,))
        >>> actor = Actor(spec=spec, module=nn.Linear(n_obs, n_act))
        >>> class ValueClass(nn.Module):
        ...     def __init__(self):
        ...         super().__init__()
        ...         self.linear = nn.Linear(n_obs + n_act, 1)
        ...     def forward(self, obs, act):
        ...         return self.linear(torch.cat([obs, act], -1))
        >>> module = ValueClass()
        >>> value = ValueOperator(
        ...     module=module,
        ...     in_keys=["observation", "action"])
        >>> loss = DDPGLoss(actor, value)
        >>> loss_actor, loss_value, pred_value, target_value, pred_value_max, target_value_max = loss(
        ...     observation=torch.randn(n_obs),
        ...     action=spec.rand(),
        ...     next_done=torch.zeros(1, dtype=torch.bool),
        ...     next_terminated=torch.zeros(1, dtype=torch.bool),
        ...     next_observation=torch.randn(n_obs),
        ...     next_reward=torch.randn(1))
        >>> loss_actor.backward()

    The output keys can also be filtered using the :meth:`DDPGLoss.select_out_keys`
    method.

    Examples:
        >>> loss.select_out_keys('loss_actor', 'loss_value')
        >>> loss_actor, loss_value = loss(
        ...     observation=torch.randn(n_obs),
        ...     action=spec.rand(),
        ...     next_done=torch.zeros(1, dtype=torch.bool),
        ...     next_terminated=torch.zeros(1, dtype=torch.bool),
        ...     next_observation=torch.randn(n_obs),
        ...     next_reward=torch.randn(1))
        >>> loss_actor.backward()

    """

    @dataclass
    class _AcceptedKeys:
        """Maintains default values for all configurable tensordict keys.

        This class defines which tensordict keys can be set using '.set_keys(key_name=key_value)' and their
        default values.

        Attributes:
            state_action_value (NestedKey): The input tensordict key where the
                state action value is expected. Will be used for the underlying
                value estimator as value key. Defaults to ``"state_action_value"``.
            priority (NestedKey): The input tensordict key where the target
                priority is written to. Defaults to ``"td_error"``.
            reward (NestedKey): The input tensordict key where the reward is expected.
                Will be used for the underlying value estimator. Defaults to ``"reward"``.
            done (NestedKey): The key in the input TensorDict that indicates
                whether a trajectory is done. Will be used for the underlying value estimator.
                Defaults to ``"done"``.
            terminated (NestedKey): The key in the input TensorDict that indicates
                whether a trajectory is terminated. Will be used for the underlying value estimator.
                Defaults to ``"terminated"``.

        """

        state_action_value: NestedKey = "state_action_value"
        priority: NestedKey = "td_error"
        reward: NestedKey = "reward"
        done: NestedKey = "done"
        terminated: NestedKey = "terminated"

    default_keys = _AcceptedKeys()
    default_value_estimator: ValueEstimators = ValueEstimators.TD0
    out_keys = [
        "loss_actor",
        "loss_value",
        "pred_value",
        "target_value",
        "pred_value_max",
        "target_value_max",
    ]

    actor_network: TensorDictModule
    value_network: actor_network
    actor_network_params: TensorDictParams
    value_network_params: TensorDictParams
    target_actor_network_params: TensorDictParams
    target_value_network_params: TensorDictParams

    def __init__(
        self,
        actor_network: TensorDictModule,
        value_network: TensorDictModule,
        *,
        loss_function: str = "l2",
        delay_actor: bool = False,
        delay_value: bool = True,
        gamma: float = None,
        separate_losses: bool = False,
        reduction: str = None,
    ) -> None:
        self._in_keys = None
        if reduction is None:
            reduction = "mean"
        super().__init__()
        self.delay_actor = delay_actor
        self.delay_value = delay_value

        actor_critic = ActorCriticWrapper(actor_network, value_network)
        params = TensorDict.from_module(actor_critic)
        params_meta = params.apply(
            self._make_meta_params, device=torch.device("meta"), filter_empty=False
        )
        with params_meta.to_module(actor_critic):
            self.__dict__["actor_critic"] = deepcopy(actor_critic)

        self.convert_to_functional(
            actor_network,
            "actor_network",
            create_target_params=self.delay_actor,
        )
        if separate_losses:
            # we want to make sure there are no duplicates in the params: the
            # params of critic must be refs to actor if they're shared
            policy_params = list(actor_network.parameters())
        else:
            policy_params = None
        self.convert_to_functional(
            value_network,
            "value_network",
            create_target_params=self.delay_value,
            compare_against=policy_params,
        )
        self.actor_critic.module[0] = self.actor_network
        self.actor_critic.module[1] = self.value_network

        self.actor_in_keys = actor_network.in_keys
        self.value_exclusive_keys = set(self.value_network.in_keys) - (
            set(self.actor_in_keys) | set(self.actor_network.out_keys)
        )

        self.loss_function = loss_function
        self.reduction = reduction
        if gamma is not None:
            raise TypeError(_GAMMA_LMBDA_DEPREC_ERROR)

    def _forward_value_estimator_keys(self, **kwargs) -> None:
        if self._value_estimator is not None:
            self._value_estimator.set_keys(
                value=self._tensor_keys.state_action_value,
                reward=self._tensor_keys.reward,
                done=self._tensor_keys.done,
                terminated=self._tensor_keys.terminated,
            )
        self._set_in_keys()

    def _set_in_keys(self):
        in_keys = {
            unravel_key(("next", self.tensor_keys.reward)),
            unravel_key(("next", self.tensor_keys.done)),
            unravel_key(("next", self.tensor_keys.terminated)),
            *self.actor_in_keys,
            *[unravel_key(("next", key)) for key in self.actor_in_keys],
            *self.value_network.in_keys,
            *[unravel_key(("next", key)) for key in self.value_network.in_keys],
        }
        self._in_keys = sorted(in_keys, key=str)

    @property
    def in_keys(self):
        if self._in_keys is None:
            self._set_in_keys()
        return self._in_keys

    @in_keys.setter
    def in_keys(self, values):
        self._in_keys = values

    @dispatch
    def forward(self, tensordict: TensorDictBase) -> TensorDict:
        """Computes the DDPG losses given a tensordict sampled from the replay buffer.

        This function will also write a "td_error" key that can be used by prioritized replay buffers to assign
            a priority to items in the tensordict.

        Args:
            tensordict (TensorDictBase): a tensordict with keys ["done", "terminated", "reward"] and the in_keys of the actor
                and value networks.

        Returns:
            a tuple of 2 tensors containing the DDPG loss.

        """
        loss_value, metadata = self.loss_value(tensordict)
        loss_actor, metadata_actor = self.loss_actor(tensordict)
        metadata.update(metadata_actor)
        td_out = TensorDict(
            source={"loss_actor": loss_actor, "loss_value": loss_value, **metadata},
            batch_size=[],
        )
        return td_out

    def loss_actor(
        self,
        tensordict: TensorDictBase,
    ) -> [torch.Tensor, dict]:
        td_copy = tensordict.select(
            *self.actor_in_keys, *self.value_exclusive_keys, strict=False
        ).detach()
        with self.actor_network_params.to_module(self.actor_network):
            td_copy = self.actor_network(td_copy)
        with self._cached_detached_value_params.to_module(self.value_network):
            td_copy = self.value_network(td_copy)
        loss_actor = -td_copy.get(self.tensor_keys.state_action_value).squeeze(-1)
        metadata = {}
        loss_actor = _reduce(loss_actor, self.reduction)
        return loss_actor, metadata

    def loss_value(
        self,
        tensordict: TensorDictBase,
    ) -> Tuple[torch.Tensor, dict]:
        # value loss
        td_copy = tensordict.select(*self.value_network.in_keys, strict=False).detach()
        with self.value_network_params.to_module(self.value_network):
            self.value_network(td_copy)
        pred_val = td_copy.get(self.tensor_keys.state_action_value).squeeze(-1)

        target_value = self.value_estimator.value_estimate(
            tensordict, target_params=self._cached_target_params
        ).squeeze(-1)

        # td_error = pred_val - target_value
        loss_value = distance_loss(
            pred_val, target_value, loss_function=self.loss_function
        )

        td_error = (pred_val - target_value).pow(2)
        td_error = td_error.detach()
        if tensordict.device is not None:
            td_error = td_error.to(tensordict.device)
        tensordict.set(
            self.tensor_keys.priority,
            td_error,
            inplace=True,
        )
        with torch.no_grad():
            metadata = {
                "td_error": td_error,
                "pred_value": pred_val,
                "target_value": target_value,
                "target_value_max": target_value.max(),
                "pred_value_max": pred_val.max(),
            }
        loss_value = _reduce(loss_value, self.reduction)
        return loss_value, metadata

    def make_value_estimator(self, value_type: ValueEstimators = None, **hyperparams):
        if value_type is None:
            value_type = self.default_value_estimator
        self.value_type = value_type
        hp = dict(default_value_kwargs(value_type))
        if hasattr(self, "gamma"):
            hp["gamma"] = self.gamma
        hp.update(hyperparams)
        if value_type == ValueEstimators.TD1:
            self._value_estimator = TD1Estimator(value_network=self.actor_critic, **hp)
        elif value_type == ValueEstimators.TD0:
            self._value_estimator = TD0Estimator(value_network=self.actor_critic, **hp)
        elif value_type == ValueEstimators.GAE:
            raise NotImplementedError(
                f"Value type {value_type} it not implemented for loss {type(self)}."
            )
        elif value_type == ValueEstimators.TDLambda:
            self._value_estimator = TDLambdaEstimator(
                value_network=self.actor_critic, **hp
            )
        else:
            raise NotImplementedError(f"Unknown value type {value_type}")

        tensor_keys = {
            "value": self.tensor_keys.state_action_value,
            "reward": self.tensor_keys.reward,
            "done": self.tensor_keys.done,
            "terminated": self.tensor_keys.terminated,
        }
        self._value_estimator.set_keys(**tensor_keys)

    @property
    @_cache_values
    def _cached_target_params(self):
        target_params = TensorDict(
            {
                "module": {
                    "0": self.target_actor_network_params,
                    "1": self.target_value_network_params,
                }
            },
            batch_size=self.target_actor_network_params.batch_size,
            device=self.target_actor_network_params.device,
        )
        return target_params

    @property
    @_cache_values
    def _cached_detached_value_params(self):
        return self.value_network_params.detach()
