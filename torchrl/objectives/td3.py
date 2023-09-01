# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import warnings
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from tensordict.nn import dispatch, TensorDictModule

from tensordict.tensordict import TensorDict, TensorDictBase
from tensordict.utils import NestedKey
from torchrl.data import BoundedTensorSpec, CompositeSpec, TensorSpec

from torchrl.envs.utils import step_mdp
from torchrl.objectives.common import LossModule
from torchrl.objectives.utils import (
    _cache_values,
    _GAMMA_LMBDA_DEPREC_WARNING,
    default_value_kwargs,
    distance_loss,
    ValueEstimators,
)
from torchrl.objectives.value import TD0Estimator, TD1Estimator, TDLambdaEstimator

try:
    try:
        from torch import vmap
    except ImportError:
        from functorch import vmap

    FUNCTORCH_ERR = ""
    _has_functorch = True
except ImportError as err:
    FUNCTORCH_ERR = str(err)
    _has_functorch = False


class TD3Loss(LossModule):
    """TD3 Loss module.

    Args:
        actor_network (TensorDictModule): the actor to be trained
        qvalue_network (TensorDictModule): a single Q-value network that will
            be multiplicated as many times as needed.

    Keyword Args:
        bounds (tuple of float, optional): the bounds of the action space.
            Exclusive with action_spec. Either this or ``action_spec`` must
            be provided.
        action_spec (TensorSpec, optional): the action spec.
            Exclusive with bounds. Either this or ``bounds`` must be provided.
        num_qvalue_nets (int, optional): Number of Q-value networks to be
            trained. Default is ``10``.
        policy_noise (float, optional): Standard deviation for the target
            policy action noise. Default is ``0.2``.
        noise_clip (float, optional): Clipping range value for the sampled
            target policy action noise. Default is ``0.5``.
        priority_key (str, optional): Key where to write the priority value
            for prioritized replay buffers. Default is
            `"td_error"`.
        loss_function (str, optional): loss function to be used for the Q-value.
            Can be one of  ``"smooth_l1"``, ``"l2"``,
            ``"l1"``, Default is ``"smooth_l1"``.
        delay_actor (bool, optional): whether to separate the target actor
            networks from the actor networks used for
            data collection. Default is ``True``.
        delay_qvalue (bool, optional): Whether to separate the target Q value
            networks from the Q value networks used
            for data collection. Default is ``True``.
        spec (TensorSpec, optional): the action tensor spec. If not provided
            and the target entropy is ``"auto"``, it will be retrieved from
            the actor.
        separate_losses (bool, optional): if ``True``, shared parameters between
            policy and critic will only be trained on the policy loss.
            Defaults to ``False``, ie. gradients are propagated to shared
            parameters for both policy and critic losses.

    Examples:
        >>> import torch
        >>> from torch import nn
        >>> from torchrl.data import BoundedTensorSpec
        >>> from torchrl.modules.distributions.continuous import NormalParamWrapper, TanhNormal
        >>> from torchrl.modules.tensordict_module.actors import Actor, ProbabilisticActor, ValueOperator
        >>> from torchrl.modules.tensordict_module.common import SafeModule
        >>> from torchrl.objectives.td3 import TD3Loss
        >>> from tensordict.tensordict import TensorDict
        >>> n_act, n_obs = 4, 3
        >>> spec = BoundedTensorSpec(-torch.ones(n_act), torch.ones(n_act), (n_act,))
        >>> module = nn.Linear(n_obs, n_act)
        >>> actor = Actor(
        ...     module=module,
        ...     spec=spec)
        >>> class ValueClass(nn.Module):
        ...     def __init__(self):
        ...         super().__init__()
        ...         self.linear = nn.Linear(n_obs + n_act, 1)
        ...     def forward(self, obs, act):
        ...         return self.linear(torch.cat([obs, act], -1))
        >>> module = ValueClass()
        >>> qvalue = ValueOperator(
        ...     module=module,
        ...     in_keys=['observation', 'action'])
        >>> loss = TD3Loss(actor, qvalue, action_spec=actor.spec)
        >>> batch = [2, ]
        >>> action = spec.rand(batch)
        >>> data = TensorDict({
        ...      "observation": torch.randn(*batch, n_obs),
        ...      "action": action,
        ...      ("next", "done"): torch.zeros(*batch, 1, dtype=torch.bool),
        ...      ("next", "reward"): torch.randn(*batch, 1),
        ...      ("next", "observation"): torch.randn(*batch, n_obs),
        ...  }, batch)
        >>> loss(data)
        TensorDict(
            fields={
                loss_actor: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False),
                loss_qvalue: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False),
                next_state_value: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False),
                pred_value: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False),
                state_action_value_actor: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False),
                target_value: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False)},
            batch_size=torch.Size([]),
            device=None,
            is_shared=False)

    This class is compatible with non-tensordict based modules too and can be
    used without recurring to any tensordict-related primitive. In this case,
    the expected keyword arguments are:
    ``["action", "next_reward", "next_done"]`` + in_keys of the actor and qvalue network
    The return value is a tuple of tensors in the following order:
    ``["loss_actor", "loss_qvalue", "pred_value", "state_action_value_actor", "next_state_value", "target_value",]``.

    Examples:
        >>> import torch
        >>> from torch import nn
        >>> from torchrl.data import BoundedTensorSpec
        >>> from torchrl.modules.tensordict_module.actors import Actor, ValueOperator
        >>> from torchrl.objectives.td3 import TD3Loss
        >>> n_act, n_obs = 4, 3
        >>> spec = BoundedTensorSpec(-torch.ones(n_act), torch.ones(n_act), (n_act,))
        >>> module = nn.Linear(n_obs, n_act)
        >>> actor = Actor(
        ...     module=module,
        ...     spec=spec)
        >>> class ValueClass(nn.Module):
        ...     def __init__(self):
        ...         super().__init__()
        ...         self.linear = nn.Linear(n_obs + n_act, 1)
        ...     def forward(self, obs, act):
        ...         return self.linear(torch.cat([obs, act], -1))
        >>> module = ValueClass()
        >>> qvalue = ValueOperator(
        ...     module=module,
        ...     in_keys=['observation', 'action'])
        >>> loss = TD3Loss(actor, qvalue, action_spec=actor.spec)
        >>> _ = loss.select_out_keys("loss_actor", "loss_qvalue")
        >>> batch = [2, ]
        >>> action = spec.rand(batch)
        >>> loss_actor, loss_qvalue = loss(
        ...         observation=torch.randn(*batch, n_obs),
        ...         action=action,
        ...         next_done=torch.zeros(*batch, 1, dtype=torch.bool),
        ...         next_reward=torch.randn(*batch, 1),
        ...         next_observation=torch.randn(*batch, n_obs))
        >>> loss_actor.backward()

    """

    @dataclass
    class _AcceptedKeys:
        """Maintains default values for all configurable tensordict keys.

        This class defines which tensordict keys can be set using '.set_keys(key_name=key_value)' and their
        default values.

        Attributes:
            action (NestedKey): The input tensordict key where the action is expected.
                Defaults to ``"action"``.
            state_action_value (NestedKey): The input tensordict key where the state action value is expected.
                Will be used for the underlying value estimator. Defaults to ``"state_action_value"``.
            priority (NestedKey): The input tensordict key where the target priority is written to.
                Defaults to ``"td_error"``.
            reward (NestedKey): The input tensordict key where the reward is expected.
                Will be used for the underlying value estimator. Defaults to ``"reward"``.
            done (NestedKey): The key in the input TensorDict that indicates
                whether a trajectory is done. Will be used for the underlying value estimator.
                Defaults to ``"done"``.
        """

        action: NestedKey = "action"
        state_action_value: NestedKey = "state_action_value"
        priority: NestedKey = "td_error"
        reward: NestedKey = "reward"
        done: NestedKey = "done"

    default_keys = _AcceptedKeys()
    default_value_estimator = ValueEstimators.TD0
    out_keys = [
        "loss_actor",
        "loss_qvalue",
        "pred_value",
        "state_action_value_actor",
        "next_state_value",
        "target_value",
    ]

    def __init__(
        self,
        actor_network: TensorDictModule,
        qvalue_network: TensorDictModule,
        *,
        action_spec: TensorSpec = None,
        bounds: Optional[Tuple[float]] = None,
        num_qvalue_nets: int = 2,
        policy_noise: float = 0.2,
        noise_clip: float = 0.5,
        loss_function: str = "smooth_l1",
        delay_actor: bool = True,
        delay_qvalue: bool = True,
        gamma: float = None,
        priority_key: str = None,
        separate_losses: bool = False,
    ) -> None:
        if not _has_functorch:
            raise ImportError(
                f"Failed to import functorch with error message:\n{FUNCTORCH_ERR}"
            )

        super().__init__()
        self._in_keys = None
        self._set_deprecated_ctor_keys(priority=priority_key)

        self.delay_actor = delay_actor
        self.delay_qvalue = delay_qvalue

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
            qvalue_network,
            "qvalue_network",
            num_qvalue_nets,
            create_target_params=self.delay_qvalue,
            compare_against=policy_params,
        )

        for p in self.parameters():
            device = p.device
            break
        else:
            device = None
        self.num_qvalue_nets = num_qvalue_nets
        self.loss_function = loss_function
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        if not ((action_spec is not None) ^ (bounds is not None)):
            raise ValueError(
                "One of 'bounds' and 'action_spec' must be provided, "
                f"but not both. Got bounds={bounds} and action_spec={action_spec}."
            )
        elif action_spec is not None:
            if isinstance(action_spec, CompositeSpec):
                action_spec = action_spec[self.tensor_keys.action]
            if not isinstance(action_spec, BoundedTensorSpec):
                raise ValueError(
                    f"action_spec is not of type BoundedTensorSpec but {type(action_spec)}."
                )
            low = action_spec.space.minimum
            high = action_spec.space.maximum
        else:
            low, high = bounds
        if not isinstance(low, torch.Tensor):
            low = torch.tensor(low)
        if not isinstance(high, torch.Tensor):
            high = torch.tensor(high, device=low.device, dtype=low.dtype)
        if (low > high).any():
            raise ValueError("Got a low bound higher than a high bound.")
        if device is not None:
            low = low.to(device)
            high = high.to(device)
        self.register_buffer("max_action", high)
        self.register_buffer("min_action", low)
        if gamma is not None:
            warnings.warn(_GAMMA_LMBDA_DEPREC_WARNING, category=DeprecationWarning)
            self.gamma = gamma
        self._vmap_qvalue_network00 = vmap(self.qvalue_network)
        self._vmap_actor_network00 = vmap(self.actor_network)

    def _forward_value_estimator_keys(self, **kwargs) -> None:
        if self._value_estimator is not None:
            self._value_estimator.set_keys(
                value=self._tensor_keys.state_action_value,
                reward=self.tensor_keys.reward,
                done=self.tensor_keys.done,
            )
        self._set_in_keys()

    def _set_in_keys(self):
        keys = [
            self.tensor_keys.action,
            ("next", self.tensor_keys.reward),
            ("next", self.tensor_keys.done),
            *self.actor_network.in_keys,
            *[("next", key) for key in self.actor_network.in_keys],
            *self.qvalue_network.in_keys,
        ]
        self._in_keys = list(set(keys))

    @property
    def in_keys(self):
        if self._in_keys is None:
            self._set_in_keys()
        return self._in_keys

    @in_keys.setter
    def in_keys(self, values):
        self._in_keys = values

    @property
    @_cache_values
    def _cached_detach_qvalue_network_params(self):
        return self.qvalue_network_params.detach()

    @property
    @_cache_values
    def _cached_stack_actor_params(self):
        return torch.stack(
            [self.actor_network_params, self.target_actor_network_params], 0
        )

    @dispatch
    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        obs_keys = self.actor_network.in_keys
        tensordict_save = tensordict
        tensordict = tensordict.clone(False)
        act = tensordict.get(self.tensor_keys.action)
        action_shape = act.shape
        action_device = act.device
        # computing early for reprod
        noise = torch.normal(
            mean=torch.zeros(action_shape),
            std=torch.full(action_shape, self.policy_noise),
        ).to(action_device)
        noise = noise.clamp(-self.noise_clip, self.noise_clip)

        tensordict_actor_grad = tensordict.select(
            *obs_keys
        )  # to avoid overwriting keys
        next_td_actor = step_mdp(tensordict).select(
            *self.actor_network.in_keys
        )  # next_observation ->
        tensordict_actor = torch.stack([tensordict_actor_grad, next_td_actor], 0)
        # DO NOT call contiguous bc we'll update the tds later
        actor_output_td = self._vmap_actor_network00(
            tensordict_actor,
            self._cached_stack_actor_params,
        )
        # add noise to target policy
        actor_output_td1 = actor_output_td[1]
        next_action = (actor_output_td1.get(self.tensor_keys.action) + noise).clamp(
            self.min_action, self.max_action
        )
        actor_output_td1.set(self.tensor_keys.action, next_action)
        tensordict_actor.set(
            self.tensor_keys.action,
            actor_output_td.get(self.tensor_keys.action),
        )

        # repeat tensordict_actor to match the qvalue size
        _actor_loss_td = (
            tensordict_actor[0]
            .select(*self.qvalue_network.in_keys)
            .expand(self.num_qvalue_nets, *tensordict_actor[0].batch_size)
        )  # for actor loss
        _qval_td = tensordict.select(*self.qvalue_network.in_keys).expand(
            self.num_qvalue_nets,
            *tensordict.select(*self.qvalue_network.in_keys).batch_size,
        )  # for qvalue loss
        _next_val_td = (
            tensordict_actor[1]
            .select(*self.qvalue_network.in_keys)
            .expand(self.num_qvalue_nets, *tensordict_actor[1].batch_size)
        )  # for next value estimation
        tensordict_qval = torch.cat(
            [
                _actor_loss_td,
                _next_val_td,
                _qval_td,
            ],
            0,
        )

        # cat params
        qvalue_params = torch.cat(
            [
                self._cached_detach_qvalue_network_params,
                self.target_qvalue_network_params,
                self.qvalue_network_params,
            ],
            0,
        )
        tensordict_qval = self._vmap_qvalue_network00(
            tensordict_qval,
            qvalue_params,
        )

        state_action_value = tensordict_qval.get(
            self.tensor_keys.state_action_value
        ).squeeze(-1)
        (
            state_action_value_actor,
            next_state_action_value_qvalue,
            state_action_value_qvalue,
        ) = state_action_value.split(
            [self.num_qvalue_nets, self.num_qvalue_nets, self.num_qvalue_nets],
            dim=0,
        )

        loss_actor = -(state_action_value_actor.min(0)[0]).mean()

        next_state_value = next_state_action_value_qvalue.min(0)[0]
        tensordict.set(
            ("next", self.tensor_keys.state_action_value),
            next_state_value.unsqueeze(-1),
        )
        target_value = self.value_estimator.value_estimate(tensordict).squeeze(-1)
        pred_val = state_action_value_qvalue
        td_error = (pred_val - target_value).pow(2)
        loss_qval = (
            distance_loss(
                pred_val,
                target_value.expand_as(pred_val),
                loss_function=self.loss_function,
            )
            .mean(-1)
            .sum()
            * 0.5
        )

        tensordict_save.set(self.tensor_keys.priority, td_error.detach().max(0)[0])

        if not loss_qval.shape == loss_actor.shape:
            raise RuntimeError(
                f"QVal and actor loss have different shape: {loss_qval.shape} and {loss_actor.shape}"
            )
        td_out = TensorDict(
            source={
                "loss_actor": loss_actor.mean(),
                "loss_qvalue": loss_qval.mean(),
                "pred_value": pred_val.mean().detach(),
                "state_action_value_actor": state_action_value_actor.mean().detach(),
                "next_state_value": next_state_value.mean().detach(),
                "target_value": target_value.mean().detach(),
            },
            batch_size=[],
        )

        return td_out

    def make_value_estimator(self, value_type: ValueEstimators = None, **hyperparams):
        if value_type is None:
            value_type = self.default_value_estimator
        self.value_type = value_type
        hp = dict(default_value_kwargs(value_type))
        if hasattr(self, "gamma"):
            hp["gamma"] = self.gamma
        hp.update(hyperparams)
        # we do not need a value network bc the next state value is already passed
        if value_type == ValueEstimators.TD1:
            self._value_estimator = TD1Estimator(value_network=None, **hp)
        elif value_type == ValueEstimators.TD0:
            self._value_estimator = TD0Estimator(value_network=None, **hp)
        elif value_type == ValueEstimators.GAE:
            raise NotImplementedError(
                f"Value type {value_type} it not implemented for loss {type(self)}."
            )
        elif value_type == ValueEstimators.TDLambda:
            self._value_estimator = TDLambdaEstimator(value_network=None, **hp)
        else:
            raise NotImplementedError(f"Unknown value type {value_type}")

        tensor_keys = {
            "value": self.tensor_keys.state_action_value,
            "reward": self.tensor_keys.reward,
            "done": self.tensor_keys.done,
        }
        self._value_estimator.set_keys(**tensor_keys)
