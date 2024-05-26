# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from functools import wraps
from numbers import Number
from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch
from tensordict import TensorDict, TensorDictBase, TensorDictParams

from tensordict.nn import dispatch, TensorDictModule
from tensordict.utils import NestedKey
from torch import Tensor
from torchrl.data.tensor_specs import CompositeSpec, TensorSpec
from torchrl.data.utils import _find_action_space
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.modules import ProbabilisticActor
from torchrl.modules.tensordict_module.actors import ActorCriticWrapper
from torchrl.objectives.common import LossModule

from torchrl.objectives.utils import (
    _cache_values,
    _GAMMA_LMBDA_DEPREC_ERROR,
    _reduce,
    _vmap_func,
    default_value_kwargs,
    distance_loss,
    ValueEstimators,
)
from torchrl.objectives.value import TD0Estimator, TD1Estimator, TDLambdaEstimator


def _delezify(func):
    @wraps(func)
    def new_func(self, *args, **kwargs):
        self.target_entropy
        return func(self, *args, **kwargs)

    return new_func


class SACLoss(LossModule):
    """TorchRL implementation of the SAC loss.

    Presented in "Soft Actor-Critic: Off-Policy Maximum Entropy Deep
    Reinforcement Learning with a Stochastic Actor" https://arxiv.org/abs/1801.01290
    and "Soft Actor-Critic Algorithms and Applications" https://arxiv.org/abs/1812.05905

    Args:
        actor_network (ProbabilisticActor): stochastic actor
        qvalue_network (TensorDictModule): Q(s, a) parametric model.
            This module typically outputs a ``"state_action_value"`` entry.
        value_network (TensorDictModule, optional): V(s) parametric model.
            This module typically outputs a ``"state_value"`` entry.

            .. note::
              If not provided, the second version of SAC is assumed, where
              only the Q-Value network is needed.

        num_qvalue_nets (integer, optional): number of Q-Value networks used.
            Defaults to ``2``.
        loss_function (str, optional): loss function to be used with
            the value function loss. Default is `"smooth_l1"`.
        alpha_init (float, optional): initial entropy multiplier.
            Default is 1.0.
        min_alpha (float, optional): min value of alpha.
            Default is None (no minimum value).
        max_alpha (float, optional): max value of alpha.
            Default is None (no maximum value).
        action_spec (TensorSpec, optional): the action tensor spec. If not provided
            and the target entropy is ``"auto"``, it will be retrieved from
            the actor.
        fixed_alpha (bool, optional): if ``True``, alpha will be fixed to its
            initial value. Otherwise, alpha will be optimized to
            match the 'target_entropy' value.
            Default is ``False``.
        target_entropy (float or str, optional): Target entropy for the
            stochastic policy. Default is "auto", where target entropy is
            computed as :obj:`-prod(n_actions)`.
        delay_actor (bool, optional): Whether to separate the target actor
            networks from the actor networks used for data collection.
            Default is ``False``.
        delay_qvalue (bool, optional): Whether to separate the target Q value
            networks from the Q value networks used for data collection.
            Default is ``True``.
        delay_value (bool, optional): Whether to separate the target value
            networks from the value networks used for data collection.
            Default is ``True``.
        priority_key (str, optional): [Deprecated, use .set_keys(priority_key=priority_key) instead]
            Tensordict key where to write the
            priority (for prioritized replay buffer usage). Defaults to ``"td_error"``.
        separate_losses (bool, optional): if ``True``, shared parameters between
            policy and critic will only be trained on the policy loss.
            Defaults to ``False``, ie. gradients are propagated to shared
            parameters for both policy and critic losses.
        reduction (str, optional): Specifies the reduction to apply to the output:
            ``"none"`` | ``"mean"`` | ``"sum"``. ``"none"``: no reduction will be applied,
            ``"mean"``: the sum of the output will be divided by the number of
            elements in the output, ``"sum"``: the output will be summed. Default: ``"mean"``.

    Examples:
        >>> import torch
        >>> from torch import nn
        >>> from torchrl.data import BoundedTensorSpec
        >>> from torchrl.modules.distributions.continuous import NormalParamWrapper, TanhNormal
        >>> from torchrl.modules.tensordict_module.actors import ProbabilisticActor, ValueOperator
        >>> from torchrl.modules.tensordict_module.common import SafeModule
        >>> from torchrl.objectives.sac import SACLoss
        >>> from tensordict import TensorDict
        >>> n_act, n_obs = 4, 3
        >>> spec = BoundedTensorSpec(-torch.ones(n_act), torch.ones(n_act), (n_act,))
        >>> net = NormalParamWrapper(nn.Linear(n_obs, 2 * n_act))
        >>> module = SafeModule(net, in_keys=["observation"], out_keys=["loc", "scale"])
        >>> actor = ProbabilisticActor(
        ...     module=module,
        ...     in_keys=["loc", "scale"],
        ...     spec=spec,
        ...     distribution_class=TanhNormal)
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
        >>> module = nn.Linear(n_obs, 1)
        >>> value = ValueOperator(
        ...     module=module,
        ...     in_keys=["observation"])
        >>> loss = SACLoss(actor, qvalue, value)
        >>> batch = [2, ]
        >>> action = spec.rand(batch)
        >>> data = TensorDict({
        ...         "observation": torch.randn(*batch, n_obs),
        ...         "action": action,
        ...         ("next", "done"): torch.zeros(*batch, 1, dtype=torch.bool),
        ...         ("next", "terminated"): torch.zeros(*batch, 1, dtype=torch.bool),
        ...         ("next", "reward"): torch.randn(*batch, 1),
        ...         ("next", "observation"): torch.randn(*batch, n_obs),
        ...     }, batch)
        >>> loss(data)
        TensorDict(
            fields={
                alpha: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False),
                entropy: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False),
                loss_actor: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False),
                loss_alpha: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False),
                loss_qvalue: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False),
                loss_value: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False)},
            batch_size=torch.Size([]),
            device=None,
            is_shared=False)

    This class is compatible with non-tensordict based modules too and can be
    used without recurring to any tensordict-related primitive. In this case,
    the expected keyword arguments are:
    ``["action", "next_reward", "next_done", "next_terminated"]`` + in_keys of the actor, value, and qvalue network.
    The return value is a tuple of tensors in the following order:
    ``["loss_actor", "loss_qvalue", "loss_alpha", "alpha", "entropy"]`` + ``"loss_value"`` if version one is used.

    Examples:
        >>> import torch
        >>> from torch import nn
        >>> from torchrl.data import BoundedTensorSpec
        >>> from torchrl.modules.distributions.continuous import NormalParamWrapper, TanhNormal
        >>> from torchrl.modules.tensordict_module.actors import ProbabilisticActor, ValueOperator
        >>> from torchrl.modules.tensordict_module.common import SafeModule
        >>> from torchrl.objectives.sac import SACLoss
        >>> _ = torch.manual_seed(42)
        >>> n_act, n_obs = 4, 3
        >>> spec = BoundedTensorSpec(-torch.ones(n_act), torch.ones(n_act), (n_act,))
        >>> net = NormalParamWrapper(nn.Linear(n_obs, 2 * n_act))
        >>> module = SafeModule(net, in_keys=["observation"], out_keys=["loc", "scale"])
        >>> actor = ProbabilisticActor(
        ...     module=module,
        ...     in_keys=["loc", "scale"],
        ...     spec=spec,
        ...     distribution_class=TanhNormal)
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
        >>> module = nn.Linear(n_obs, 1)
        >>> value = ValueOperator(
        ...     module=module,
        ...     in_keys=["observation"])
        >>> loss = SACLoss(actor, qvalue, value)
        >>> batch = [2, ]
        >>> action = spec.rand(batch)
        >>> loss_actor, loss_qvalue, _, _, _, _ = loss(
        ...     observation=torch.randn(*batch, n_obs),
        ...     action=action,
        ...     next_done=torch.zeros(*batch, 1, dtype=torch.bool),
        ...     next_terminated=torch.zeros(*batch, 1, dtype=torch.bool),
        ...     next_observation=torch.zeros(*batch, n_obs),
        ...     next_reward=torch.randn(*batch, 1))
        >>> loss_actor.backward()

    The output keys can also be filtered using the :meth:`SACLoss.select_out_keys`
    method.

    Examples:
        >>> _ = loss.select_out_keys('loss_actor', 'loss_qvalue')
        >>> loss_actor, loss_qvalue = loss(
        ...     observation=torch.randn(*batch, n_obs),
        ...     action=action,
        ...     next_done=torch.zeros(*batch, 1, dtype=torch.bool),
        ...     next_terminated=torch.zeros(*batch, 1, dtype=torch.bool),
        ...     next_observation=torch.zeros(*batch, n_obs),
        ...     next_reward=torch.randn(*batch, 1))
        >>> loss_actor.backward()
    """

    @dataclass
    class _AcceptedKeys:
        """Maintains default values for all configurable tensordict keys.

        This class defines which tensordict keys can be set using '.set_keys(key_name=key_value)' and their
        default values.

        Attributes:
            action (NestedKey): The input tensordict key where the action is expected.
                Defaults to ``"advantage"``.
            value (NestedKey): The input tensordict key where the state value is expected.
                Will be used for the underlying value estimator. Defaults to ``"state_value"``.
            state_action_value (NestedKey): The input tensordict key where the
                state action value is expected.  Defaults to ``"state_action_value"``.
            log_prob (NestedKey): The input tensordict key where the log probability is expected.
                Defaults to ``"_log_prob"``.
            priority (NestedKey): The input tensordict key where the target priority is written to.
                Defaults to ``"td_error"``.
            reward (NestedKey): The input tensordict key where the reward is expected.
                Will be used for the underlying value estimator. Defaults to ``"reward"``.
            done (NestedKey): The key in the input TensorDict that indicates
                whether a trajectory is done. Will be used for the underlying value estimator.
                Defaults to ``"done"``.
            terminated (NestedKey): The key in the input TensorDict that indicates
                whether a trajectory is terminated. Will be used for the underlying value estimator.
                Defaults to ``"terminated"``.
        """

        action: NestedKey = "action"
        value: NestedKey = "state_value"
        state_action_value: NestedKey = "state_action_value"
        log_prob: NestedKey = "_log_prob"
        priority: NestedKey = "td_error"
        reward: NestedKey = "reward"
        done: NestedKey = "done"
        terminated: NestedKey = "terminated"

    default_keys = _AcceptedKeys()
    default_value_estimator = ValueEstimators.TD0

    actor_network: TensorDictModule
    qvalue_network: TensorDictModule
    value_network: TensorDictModule | None
    actor_network_params: TensorDictParams
    qvalue_network_params: TensorDictParams
    value_network_params: TensorDictParams | None
    target_actor_network_params: TensorDictParams
    target_qvalue_network_params: TensorDictParams
    target_value_network_params: TensorDictParams | None

    def __init__(
        self,
        actor_network: ProbabilisticActor,
        qvalue_network: TensorDictModule,
        value_network: Optional[TensorDictModule] = None,
        *,
        num_qvalue_nets: int = 2,
        loss_function: str = "smooth_l1",
        alpha_init: float = 1.0,
        min_alpha: float = None,
        max_alpha: float = None,
        action_spec=None,
        fixed_alpha: bool = False,
        target_entropy: Union[str, float] = "auto",
        delay_actor: bool = False,
        delay_qvalue: bool = True,
        delay_value: bool = True,
        gamma: float = None,
        priority_key: str = None,
        separate_losses: bool = False,
        reduction: str = None,
    ) -> None:
        self._in_keys = None
        self._out_keys = None
        if reduction is None:
            reduction = "mean"
        super().__init__()
        self._set_deprecated_ctor_keys(priority_key=priority_key)

        # Actor
        self.delay_actor = delay_actor
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
            q_value_policy_params = None
        # Value
        if value_network is not None:
            self._version = 1
            self.delay_value = delay_value
            self.convert_to_functional(
                value_network,
                "value_network",
                create_target_params=self.delay_value,
                compare_against=policy_params,
            )
        else:
            self._version = 2

        # Q value
        self.delay_qvalue = delay_qvalue
        self.num_qvalue_nets = num_qvalue_nets
        if self._version == 1:
            if separate_losses:
                value_params = list(value_network.parameters())
                q_value_policy_params = policy_params + value_params
            else:
                q_value_policy_params = policy_params
        else:
            q_value_policy_params = policy_params
        self.convert_to_functional(
            qvalue_network,
            "qvalue_network",
            num_qvalue_nets,
            create_target_params=self.delay_qvalue,
            compare_against=q_value_policy_params,
        )

        self.loss_function = loss_function
        try:
            device = next(self.parameters()).device
        except AttributeError:
            device = torch.device("cpu")
        self.register_buffer("alpha_init", torch.tensor(alpha_init, device=device))
        if bool(min_alpha) ^ bool(max_alpha):
            min_alpha = min_alpha if min_alpha else 0.0
            if max_alpha == 0:
                raise ValueError("max_alpha must be either None or greater than 0.")
            max_alpha = max_alpha if max_alpha else 1e9
        if min_alpha:
            self.register_buffer(
                "min_log_alpha", torch.tensor(min_alpha, device=device).log()
            )
        else:
            self.min_log_alpha = None
        if max_alpha:
            self.register_buffer(
                "max_log_alpha", torch.tensor(max_alpha, device=device).log()
            )
        else:
            self.max_log_alpha = None
        self.fixed_alpha = fixed_alpha
        if fixed_alpha:
            self.register_buffer(
                "log_alpha", torch.tensor(math.log(alpha_init), device=device)
            )
        else:
            self.register_parameter(
                "log_alpha",
                torch.nn.Parameter(torch.tensor(math.log(alpha_init), device=device)),
            )

        self._target_entropy = target_entropy
        self._action_spec = action_spec
        if self._version == 1:
            self.__dict__["actor_critic"] = ActorCriticWrapper(
                self.actor_network, self.value_network
            )
        if gamma is not None:
            raise TypeError(_GAMMA_LMBDA_DEPREC_ERROR)
        self._vmap_qnetworkN0 = _vmap_func(
            self.qvalue_network, (None, 0), randomness=self.vmap_randomness
        )
        if self._version == 1:
            self._vmap_qnetwork00 = _vmap_func(
                qvalue_network, randomness=self.vmap_randomness
            )
        self.reduction = reduction

    @property
    def target_entropy_buffer(self):
        return self.target_entropy

    @property
    def target_entropy(self):
        target_entropy = self._buffers.get("_target_entropy", None)
        if target_entropy is not None:
            return target_entropy
        target_entropy = self._target_entropy
        action_spec = self._action_spec
        actor_network = self.actor_network
        device = next(self.parameters()).device
        if target_entropy == "auto":
            action_spec = (
                action_spec
                if action_spec is not None
                else getattr(actor_network, "spec", None)
            )
            if action_spec is None:
                raise RuntimeError(
                    "Cannot infer the dimensionality of the action. Consider providing "
                    "the target entropy explicitely or provide the spec of the "
                    "action tensor in the actor network."
                )
            if not isinstance(action_spec, CompositeSpec):
                action_spec = CompositeSpec({self.tensor_keys.action: action_spec})
            if (
                isinstance(self.tensor_keys.action, tuple)
                and len(self.tensor_keys.action) > 1
            ):
                action_container_shape = action_spec[self.tensor_keys.action[:-1]].shape
            else:
                action_container_shape = action_spec.shape
            target_entropy = -float(
                action_spec[self.tensor_keys.action]
                .shape[len(action_container_shape) :]
                .numel()
            )
        delattr(self, "_target_entropy")
        self.register_buffer(
            "_target_entropy", torch.tensor(target_entropy, device=device)
        )
        return self._target_entropy

    state_dict = _delezify(LossModule.state_dict)
    load_state_dict = _delezify(LossModule.load_state_dict)

    def _forward_value_estimator_keys(self, **kwargs) -> None:
        if self._value_estimator is not None:
            self._value_estimator.set_keys(
                value=self.tensor_keys.value,
                reward=self.tensor_keys.reward,
                done=self.tensor_keys.done,
                terminated=self.tensor_keys.terminated,
            )
        self._set_in_keys()

    def make_value_estimator(self, value_type: ValueEstimators = None, **hyperparams):
        if value_type is None:
            value_type = self.default_value_estimator
        self.value_type = value_type
        if self._version == 1:
            value_net = self.actor_critic
        elif self._version == 2:
            # we will take care of computing the next value inside this module
            value_net = None
        else:
            # unreachable
            raise NotImplementedError

        hp = dict(default_value_kwargs(value_type))
        hp.update(hyperparams)
        if value_type is ValueEstimators.TD1:
            self._value_estimator = TD1Estimator(
                **hp,
                value_network=value_net,
            )
        elif value_type is ValueEstimators.TD0:
            self._value_estimator = TD0Estimator(
                **hp,
                value_network=value_net,
            )
        elif value_type is ValueEstimators.GAE:
            raise NotImplementedError(
                f"Value type {value_type} it not implemented for loss {type(self)}."
            )
        elif value_type is ValueEstimators.TDLambda:
            self._value_estimator = TDLambdaEstimator(
                **hp,
                value_network=value_net,
            )
        else:
            raise NotImplementedError(f"Unknown value type {value_type}")

        tensor_keys = {
            "value_target": "value_target",
            "value": self.tensor_keys.value,
            "reward": self.tensor_keys.reward,
            "done": self.tensor_keys.done,
            "terminated": self.tensor_keys.terminated,
        }
        self._value_estimator.set_keys(**tensor_keys)

    @property
    def device(self) -> torch.device:
        for p in self.parameters():
            return p.device
        raise RuntimeError(
            "At least one of the networks of SACLoss must have trainable " "parameters."
        )

    def _set_in_keys(self):
        keys = [
            self.tensor_keys.action,
            ("next", self.tensor_keys.reward),
            ("next", self.tensor_keys.done),
            ("next", self.tensor_keys.terminated),
            *self.actor_network.in_keys,
            *[("next", key) for key in self.actor_network.in_keys],
            *self.qvalue_network.in_keys,
        ]
        if self._version == 1:
            keys.extend(self.value_network.in_keys)
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
    def out_keys(self):
        if self._out_keys is None:
            keys = ["loss_actor", "loss_qvalue", "loss_alpha", "alpha", "entropy"]
            if self._version == 1:
                keys.append("loss_value")
            self._out_keys = keys
        return self._out_keys

    @out_keys.setter
    def out_keys(self, values):
        self._out_keys = values

    @dispatch
    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        shape = None
        if tensordict.ndimension() > 1:
            shape = tensordict.shape
            tensordict_reshape = tensordict.reshape(-1)
        else:
            tensordict_reshape = tensordict

        if self._version == 1:
            loss_qvalue, value_metadata = self._qvalue_v1_loss(tensordict_reshape)
            loss_value, _ = self._value_loss(tensordict_reshape)
        else:
            loss_qvalue, value_metadata = self._qvalue_v2_loss(tensordict_reshape)
            loss_value = None
        loss_actor, metadata_actor = self._actor_loss(tensordict_reshape)
        loss_alpha = self._alpha_loss(log_prob=metadata_actor["log_prob"])
        tensordict_reshape.set(self.tensor_keys.priority, value_metadata["td_error"])
        if (loss_actor.shape != loss_qvalue.shape) or (
            loss_value is not None and loss_actor.shape != loss_value.shape
        ):
            raise RuntimeError(
                f"Losses shape mismatch: {loss_actor.shape}, {loss_qvalue.shape} and {loss_value.shape}"
            )
        if shape:
            tensordict.update(tensordict_reshape.view(shape))
        entropy = -metadata_actor["log_prob"]
        out = {
            "loss_actor": loss_actor,
            "loss_qvalue": loss_qvalue,
            "loss_alpha": loss_alpha,
            "alpha": self._alpha,
            "entropy": entropy.detach().mean(),
        }
        if self._version == 1:
            out["loss_value"] = loss_value
        td_out = TensorDict(out, [])
        td_out = td_out.named_apply(
            lambda name, value: _reduce(value, reduction=self.reduction)
            if name.startswith("loss_")
            else value,
            batch_size=[],
        )
        return td_out

    @property
    @_cache_values
    def _cached_detached_qvalue_params(self):
        return self.qvalue_network_params.detach()

    def _actor_loss(
        self, tensordict: TensorDictBase
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        with set_exploration_type(
            ExplorationType.RANDOM
        ), self.actor_network_params.to_module(self.actor_network):
            dist = self.actor_network.get_dist(tensordict)
            a_reparm = dist.rsample()
        log_prob = dist.log_prob(a_reparm)

        td_q = tensordict.select(*self.qvalue_network.in_keys, strict=False)
        td_q.set(self.tensor_keys.action, a_reparm)
        td_q = self._vmap_qnetworkN0(
            td_q,
            self._cached_detached_qvalue_params,  # should we clone?
        )
        min_q_logprob = (
            td_q.get(self.tensor_keys.state_action_value).min(0)[0].squeeze(-1)
        )

        if log_prob.shape != min_q_logprob.shape:
            raise RuntimeError(
                f"Losses shape mismatch: {log_prob.shape} and {min_q_logprob.shape}"
            )

        return self._alpha * log_prob - min_q_logprob, {"log_prob": log_prob.detach()}

    @property
    @_cache_values
    def _cached_target_params_actor_value(self):
        return TensorDict(
            {
                "module": {
                    "0": self.target_actor_network_params,
                    "1": self.target_value_network_params,
                }
            },
            torch.Size([]),
            _run_checks=False,
        )

    def _qvalue_v1_loss(
        self, tensordict: TensorDictBase
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        target_params = self._cached_target_params_actor_value
        with set_exploration_type(ExplorationType.MODE):
            target_value = self.value_estimator.value_estimate(
                tensordict, target_params=target_params
            ).squeeze(-1)

        # Q-nets must be trained independently: as such, we split the data in 2
        # if required and train each q-net on one half of the data.
        shape = tensordict.shape
        if shape[0] % self.num_qvalue_nets != 0:
            raise RuntimeError(
                f"Batch size={tensordict.shape} is incompatible "
                f"with num_qvqlue_nets={self.num_qvalue_nets}."
            )
        tensordict_chunks = tensordict.reshape(
            self.num_qvalue_nets, -1, *tensordict.shape[1:]
        )
        target_chunks = target_value.reshape(
            self.num_qvalue_nets, -1, *target_value.shape[1:]
        )

        # if vmap=True, it is assumed that the input tensordict must be cast to the param shape
        tensordict_chunks = self._vmap_qnetwork00(
            tensordict_chunks, self.qvalue_network_params
        )
        pred_val = tensordict_chunks.get(self.tensor_keys.state_action_value)
        pred_val = pred_val.squeeze(-1)
        loss_value = distance_loss(
            pred_val, target_chunks, loss_function=self.loss_function
        ).view(*shape)
        metadata = {"td_error": (pred_val - target_chunks).pow(2).flatten(0, 1)}

        return loss_value, metadata

    def _compute_target_v2(self, tensordict) -> Tensor:
        r"""Value network for SAC v2.

        SAC v2 is based on a value estimate of the form:

        .. math::

          V = Q(s,a) - \alpha * \log p(a | s)

        This class computes this value given the actor and qvalue network

        """
        tensordict = tensordict.clone(False)
        # get actions and log-probs
        with torch.no_grad():
            with set_exploration_type(
                ExplorationType.RANDOM
            ), self.actor_network_params.to_module(self.actor_network):
                next_tensordict = tensordict.get("next").clone(False)
                next_dist = self.actor_network.get_dist(next_tensordict)
                next_action = next_dist.rsample()
                next_tensordict.set(self.tensor_keys.action, next_action)
                next_sample_log_prob = next_dist.log_prob(next_action)

            # get q-values
            next_tensordict_expand = self._vmap_qnetworkN0(
                next_tensordict, self.target_qvalue_network_params
            )
            state_action_value = next_tensordict_expand.get(
                self.tensor_keys.state_action_value
            )
            if (
                state_action_value.shape[-len(next_sample_log_prob.shape) :]
                != next_sample_log_prob.shape
            ):
                next_sample_log_prob = next_sample_log_prob.unsqueeze(-1)
            next_state_value = state_action_value - self._alpha * next_sample_log_prob
            next_state_value = next_state_value.min(0)[0]
            tensordict.set(
                ("next", self.value_estimator.tensor_keys.value), next_state_value
            )
            target_value = self.value_estimator.value_estimate(tensordict).squeeze(-1)
            return target_value

    def _qvalue_v2_loss(
        self, tensordict: TensorDictBase
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        # we pass the alpha value to the tensordict. Since it's a scalar, we must erase the batch-size first.
        target_value = self._compute_target_v2(tensordict)

        tensordict_expand = self._vmap_qnetworkN0(
            tensordict.select(*self.qvalue_network.in_keys, strict=False),
            self.qvalue_network_params,
        )
        pred_val = tensordict_expand.get(self.tensor_keys.state_action_value).squeeze(
            -1
        )
        td_error = abs(pred_val - target_value)
        loss_qval = distance_loss(
            pred_val,
            target_value.expand_as(pred_val),
            loss_function=self.loss_function,
        ).sum(0)
        metadata = {"td_error": td_error.detach().max(0)[0]}
        return loss_qval, metadata

    def _value_loss(
        self, tensordict: TensorDictBase
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        # value loss
        td_copy = tensordict.select(*self.value_network.in_keys, strict=False).detach()
        with self.value_network_params.to_module(self.value_network):
            self.value_network(td_copy)
        pred_val = td_copy.get(self.tensor_keys.value).squeeze(-1)
        with self.target_actor_network_params.to_module(self.actor_network):
            action_dist = self.actor_network.get_dist(td_copy)  # resample an action
        action = action_dist.rsample()

        td_copy.set(self.tensor_keys.action, action, inplace=False)

        td_copy = self._vmap_qnetworkN0(
            td_copy,
            self.target_qvalue_network_params,
        )

        min_qval = (
            td_copy.get(self.tensor_keys.state_action_value).squeeze(-1).min(0)[0]
        )

        log_p = action_dist.log_prob(action)
        if log_p.shape != min_qval.shape:
            raise RuntimeError(
                f"Losses shape mismatch: {min_qval.shape} and {log_p.shape}"
            )
        target_val = min_qval - self._alpha * log_p

        loss_value = distance_loss(
            pred_val, target_val, loss_function=self.loss_function
        )
        return loss_value, {}

    def _alpha_loss(self, log_prob: Tensor) -> Tensor:
        if self.target_entropy is not None:
            # we can compute this loss even if log_alpha is not a parameter
            alpha_loss = -self.log_alpha * (log_prob + self.target_entropy)
        else:
            # placeholder
            alpha_loss = torch.zeros_like(log_prob)
        return alpha_loss

    @property
    def _alpha(self):
        if self.min_log_alpha is not None:
            self.log_alpha.data.clamp_(self.min_log_alpha, self.max_log_alpha)
        with torch.no_grad():
            alpha = self.log_alpha.exp()
        return alpha


class DiscreteSACLoss(LossModule):
    """Discrete SAC Loss module.

    Args:
        actor_network (ProbabilisticActor): the actor to be trained
        qvalue_network (TensorDictModule): a single Q-value network that will be multiplicated as many times as needed.
        action_space (str or TensorSpec): Action space. Must be one of
            ``"one-hot"``, ``"mult_one_hot"``, ``"binary"`` or ``"categorical"``,
            or an instance of the corresponding specs (:class:`torchrl.data.OneHotDiscreteTensorSpec`,
            :class:`torchrl.data.MultiOneHotDiscreteTensorSpec`,
            :class:`torchrl.data.BinaryDiscreteTensorSpec` or :class:`torchrl.data.DiscreteTensorSpec`).
        num_actions (int, optional): number of actions in the action space.
            To be provided if target_entropy is set to "auto".
        num_qvalue_nets (int, optional): Number of Q-value networks to be trained. Default is 2.
        loss_function (str, optional): loss function to be used for the Q-value. Can be one of `"smooth_l1"`, "l2",
            "l1", Default is "smooth_l1".
        alpha_init (float, optional): initial entropy multiplier.
            Default is 1.0.
        min_alpha (float, optional): min value of alpha.
            Default is None (no minimum value).
        max_alpha (float, optional): max value of alpha.
            Default is None (no maximum value).
        fixed_alpha (bool, optional): whether alpha should be trained to match a target entropy. Default is ``False``.
        target_entropy_weight (float, optional): weight for the target entropy term.
        target_entropy (Union[str, Number], optional): Target entropy for the stochastic policy. Default is "auto".
        delay_qvalue (bool, optional): Whether to separate the target Q value networks from the Q value networks used
            for data collection. Default is ``False``.
        priority_key (str, optional): [Deprecated, use .set_keys(priority_key=priority_key) instead]
            Key where to write the priority value for prioritized replay buffers.
            Default is `"td_error"`.
        separate_losses (bool, optional): if ``True``, shared parameters between
            policy and critic will only be trained on the policy loss.
            Defaults to ``False``, ie. gradients are propagated to shared
            parameters for both policy and critic losses.
        reduction (str, optional): Specifies the reduction to apply to the output:
            ``"none"`` | ``"mean"`` | ``"sum"``. ``"none"``: no reduction will be applied,
            ``"mean"``: the sum of the output will be divided by the number of
            elements in the output, ``"sum"``: the output will be summed. Default: ``"mean"``.

    Examples:
    >>> import torch
    >>> from torch import nn
    >>> from torchrl.data.tensor_specs import OneHotDiscreteTensorSpec
    >>> from torchrl.modules.distributions.continuous import NormalParamWrapper
    >>> from torchrl.modules.distributions.discrete import OneHotCategorical
    >>> from torchrl.modules.tensordict_module.actors import ProbabilisticActor, ValueOperator
    >>> from torchrl.modules.tensordict_module.common import SafeModule
    >>> from torchrl.objectives.sac import DiscreteSACLoss
    >>> from tensordict import TensorDict
    >>> from tensordict.nn import TensorDictModule
    >>> n_act, n_obs = 4, 3
    >>> spec = OneHotDiscreteTensorSpec(n_act)
    >>> module = TensorDictModule(nn.Linear(n_obs, n_act), in_keys=["observation"], out_keys=["logits"])
    >>> actor = ProbabilisticActor(
    ...     module=module,
    ...     in_keys=["logits"],
    ...     out_keys=["action"],
    ...     spec=spec,
    ...     distribution_class=OneHotCategorical)
    >>> qvalue = TensorDictModule(
    ...     nn.Linear(n_obs, n_act),
    ...     in_keys=["observation"],
    ...     out_keys=["action_value"],
    ... )
    >>> loss = DiscreteSACLoss(actor, qvalue, action_space=spec, num_actions=spec.space.n)
    >>> batch = [2,]
    >>> action = spec.rand(batch)
    >>> data = TensorDict({
    ...     "observation": torch.randn(*batch, n_obs),
    ...     "action": action,
    ...     ("next", "done"): torch.zeros(*batch, 1, dtype=torch.bool),
    ...     ("next", "terminated"): torch.zeros(*batch, 1, dtype=torch.bool),
    ...     ("next", "reward"): torch.randn(*batch, 1),
    ...     ("next", "observation"): torch.randn(*batch, n_obs),
    ...     }, batch)
    >>> loss(data)
    TensorDict(
    fields={
        alpha: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False),
        entropy: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False),
        loss_actor: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False),
        loss_alpha: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False),
        loss_qvalue: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False)},
    batch_size=torch.Size([]),
    device=None,
    is_shared=False)


    This class is compatible with non-tensordict based modules too and can be
    used without recurring to any tensordict-related primitive. In this case,
    the expected keyword arguments are:
    ``["action", "next_reward", "next_done", "next_terminated"]`` + in_keys of the actor and qvalue network.
    The return value is a tuple of tensors in the following order:
    ``["loss_actor", "loss_qvalue", "loss_alpha",
       "alpha", "entropy"]``
    The output keys can also be filtered using :meth:`DiscreteSACLoss.select_out_keys` method.

    Examples:
        >>> import torch
        >>> from torch import nn
        >>> from torchrl.data.tensor_specs import OneHotDiscreteTensorSpec
        >>> from torchrl.modules.distributions.continuous import NormalParamWrapper
        >>> from torchrl.modules.distributions.discrete import OneHotCategorical
        >>> from torchrl.modules.tensordict_module.actors import ProbabilisticActor, ValueOperator
        >>> from torchrl.modules.tensordict_module.common import SafeModule
        >>> from torchrl.objectives.sac import DiscreteSACLoss
        >>> n_act, n_obs = 4, 3
        >>> spec = OneHotDiscreteTensorSpec(n_act)
        >>> net = NormalParamWrapper(nn.Linear(n_obs, 2 * n_act))
        >>> module = SafeModule(net, in_keys=["observation"], out_keys=["logits"])
        >>> actor = ProbabilisticActor(
        ...     module=module,
        ...     in_keys=["logits"],
        ...     out_keys=["action"],
        ...     spec=spec,
        ...     distribution_class=OneHotCategorical)
        >>> class ValueClass(nn.Module):
        ...     def __init__(self):
        ...         super().__init__()
        ...         self.linear = nn.Linear(n_obs, n_act)
        ...     def forward(self, obs):
        ...         return self.linear(obs)
        >>> module = ValueClass()
        >>> qvalue = ValueOperator(
        ...     module=module,
        ...     in_keys=['observation'])
        >>> loss = DiscreteSACLoss(actor, qvalue, num_actions=actor.spec["action"].space.n)
        >>> batch = [2, ]
        >>> action = spec.rand(batch)
        >>> # filter output keys to "loss_actor", and "loss_qvalue"
        >>> _ = loss.select_out_keys("loss_actor", "loss_qvalue")
        >>> loss_actor, loss_qvalue = loss(
        ...     observation=torch.randn(*batch, n_obs),
        ...     action=action,
        ...     next_done=torch.zeros(*batch, 1, dtype=torch.bool),
        ...     next_terminated=torch.zeros(*batch, 1, dtype=torch.bool),
        ...     next_observation=torch.zeros(*batch, n_obs),
        ...     next_reward=torch.randn(*batch, 1))
        >>> loss_actor.backward()
    """

    @dataclass
    class _AcceptedKeys:
        """Maintains default values for all configurable tensordict keys.

        This class defines which tensordict keys can be set using '.set_keys(key_name=key_value)' and their
        default values

        Attributes:
            action (NestedKey): The input tensordict key where the action is expected.
                Defaults to ``"action"``.
            value (NestedKey): The input tensordict key where the state value is expected.
                Will be used for the underlying value estimator. Defaults to ``"state_value"``.
            priority (NestedKey): The input tensordict key where the target priority is written to.
                Defaults to ``"td_error"``.
            reward (NestedKey): The input tensordict key where the reward is expected.
                Will be used for the underlying value estimator. Defaults to ``"reward"``.
            done (NestedKey): The key in the input TensorDict that indicates
                whether a trajectory is done. Will be used for the underlying value estimator.
                Defaults to ``"done"``.
            terminated (NestedKey): The key in the input TensorDict that indicates
                whether a trajectory is terminated. Will be used for the underlying value estimator.
                Defaults to ``"terminated"``.
        """

        action: NestedKey = "action"
        value: NestedKey = "state_value"
        action_value: NestedKey = "action_value"
        priority: NestedKey = "td_error"
        reward: NestedKey = "reward"
        done: NestedKey = "done"
        terminated: NestedKey = "terminated"
        log_prob: NestedKey = "log_prob"

    default_keys = _AcceptedKeys()
    default_value_estimator = ValueEstimators.TD0
    delay_actor: bool = False
    out_keys = [
        "loss_actor",
        "loss_qvalue",
        "loss_alpha",
        "alpha",
        "entropy",
    ]

    actor_network: TensorDictModule
    qvalue_network: TensorDictModule
    value_network: TensorDictModule | None
    actor_network_params: TensorDictParams
    qvalue_network_params: TensorDictParams
    value_network_params: TensorDictParams | None
    target_actor_network_params: TensorDictParams
    target_qvalue_network_params: TensorDictParams
    target_value_network_params: TensorDictParams | None

    def __init__(
        self,
        actor_network: ProbabilisticActor,
        qvalue_network: TensorDictModule,
        *,
        action_space: Union[str, TensorSpec] = None,
        num_actions: Optional[int] = None,
        num_qvalue_nets: int = 2,
        loss_function: str = "smooth_l1",
        alpha_init: float = 1.0,
        min_alpha: float = None,
        max_alpha: float = None,
        fixed_alpha: bool = False,
        target_entropy_weight: float = 0.98,
        target_entropy: Union[str, Number] = "auto",
        delay_qvalue: bool = True,
        priority_key: str = None,
        separate_losses: bool = False,
        reduction: str = None,
    ):
        if reduction is None:
            reduction = "mean"
        self._in_keys = None
        super().__init__()
        self._set_deprecated_ctor_keys(priority_key=priority_key)

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
        self.delay_qvalue = delay_qvalue
        self.convert_to_functional(
            qvalue_network,
            "qvalue_network",
            num_qvalue_nets,
            create_target_params=self.delay_qvalue,
            compare_against=policy_params,
        )
        self.num_qvalue_nets = num_qvalue_nets
        self.loss_function = loss_function

        try:
            device = next(self.parameters()).device
        except AttributeError:
            device = torch.device("cpu")

        self.register_buffer("alpha_init", torch.tensor(alpha_init, device=device))
        if bool(min_alpha) ^ bool(max_alpha):
            min_alpha = min_alpha if min_alpha else 0.0
            if max_alpha == 0:
                raise ValueError("max_alpha must be either None or greater than 0.")
            max_alpha = max_alpha if max_alpha else 1e9
        if min_alpha:
            self.register_buffer(
                "min_log_alpha", torch.tensor(min_alpha, device=device).log()
            )
        else:
            self.min_log_alpha = None
        if max_alpha:
            self.register_buffer(
                "max_log_alpha", torch.tensor(max_alpha, device=device).log()
            )
        else:
            self.max_log_alpha = None
        self.fixed_alpha = fixed_alpha
        if fixed_alpha:
            self.register_buffer(
                "log_alpha", torch.tensor(math.log(alpha_init), device=device)
            )
        else:
            self.register_parameter(
                "log_alpha",
                torch.nn.Parameter(torch.tensor(math.log(alpha_init), device=device)),
            )

        if action_space is None:
            warnings.warn(
                "action_space was not specified. DiscreteSACLoss will default to 'one-hot'."
                "This behaviour will be deprecated soon and a space will have to be passed. "
                "Check the DiscreteSACLoss documentation to see how to pass the action space. "
            )
            action_space = "one-hot"
        self.action_space = _find_action_space(action_space)
        if target_entropy == "auto":
            if num_actions is None:
                raise ValueError(
                    "num_actions needs to be provided if target_entropy == 'auto'"
                )
            target_entropy = -float(np.log(1.0 / num_actions) * target_entropy_weight)
        self.register_buffer(
            "target_entropy", torch.tensor(target_entropy, device=device)
        )
        self._vmap_qnetworkN0 = _vmap_func(
            self.qvalue_network, (None, 0), randomness=self.vmap_randomness
        )
        self.reduction = reduction

    def _forward_value_estimator_keys(self, **kwargs) -> None:
        if self._value_estimator is not None:
            self._value_estimator.set_keys(
                value=self._tensor_keys.value,
                reward=self.tensor_keys.reward,
                done=self.tensor_keys.done,
                terminated=self.tensor_keys.terminated,
            )
        self._set_in_keys()

    def _set_in_keys(self):
        keys = [
            self.tensor_keys.action,
            ("next", self.tensor_keys.reward),
            ("next", self.tensor_keys.done),
            ("next", self.tensor_keys.terminated),
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

    @dispatch
    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        shape = None
        if tensordict.ndimension() > 1:
            shape = tensordict.shape
            tensordict_reshape = tensordict.reshape(-1)
        else:
            tensordict_reshape = tensordict

        loss_value, metadata_value = self._value_loss(tensordict_reshape)
        loss_actor, metadata_actor = self._actor_loss(tensordict_reshape)
        loss_alpha = self._alpha_loss(
            log_prob=metadata_actor["log_prob"],
        )

        tensordict_reshape.set(self.tensor_keys.priority, metadata_value["td_error"])
        if loss_actor.shape != loss_value.shape:
            raise RuntimeError(
                f"Losses shape mismatch: {loss_actor.shape}, and {loss_value.shape}"
            )
        if shape:
            tensordict.update(tensordict_reshape.view(shape))
        entropy = -metadata_actor["log_prob"]
        out = {
            "loss_actor": loss_actor,
            "loss_qvalue": loss_value,
            "loss_alpha": loss_alpha,
            "alpha": self._alpha,
            "entropy": entropy.detach().mean(),
        }
        td_out = TensorDict(out, [])
        td_out = td_out.named_apply(
            lambda name, value: _reduce(value, reduction=self.reduction)
            if name.startswith("loss_")
            else value,
            batch_size=[],
        )
        return td_out

    def _compute_target(self, tensordict) -> Tensor:
        r"""Value network for SAC v2.

        SAC v2 is based on a value estimate of the form:

        .. math::

          V = Q(s,a) - \alpha * \log p(a | s)

        This class computes this value given the actor and qvalue network

        """
        tensordict = tensordict.clone(False)
        # get actions and log-probs
        with torch.no_grad():
            next_tensordict = tensordict.get("next").clone(False)

            # get probs and log probs for actions computed from "next"
            with self.actor_network_params.to_module(self.actor_network):
                next_dist = self.actor_network.get_dist(next_tensordict)
            next_prob = next_dist.probs
            next_log_prob = torch.log(torch.where(next_prob == 0, 1e-8, next_prob))

            # get q-values for all actions
            next_tensordict_expand = self._vmap_qnetworkN0(
                next_tensordict, self.target_qvalue_network_params
            )
            next_action_value = next_tensordict_expand.get(
                self.tensor_keys.action_value
            )

            # like in continuous SAC, we take the minimum of the value ensemble and subtract the entropy term
            next_state_value = next_action_value.min(0)[0] - self._alpha * next_log_prob
            # unlike in continuous SAC, we can compute the exact expectation over all discrete actions
            next_state_value = (next_prob * next_state_value).sum(-1).unsqueeze(-1)

            tensordict.set(
                ("next", self.value_estimator.tensor_keys.value), next_state_value
            )
            target_value = self.value_estimator.value_estimate(tensordict).squeeze(-1)
            return target_value

    def _value_loss(
        self, tensordict: TensorDictBase
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        target_value = self._compute_target(tensordict)
        tensordict_expand = self._vmap_qnetworkN0(
            tensordict.select(*self.qvalue_network.in_keys, strict=False),
            self.qvalue_network_params,
        )

        action_value = tensordict_expand.get(self.tensor_keys.action_value)
        action = tensordict.get(self.tensor_keys.action)
        action = action.expand((action_value.shape[0], *action.shape))  # Add vmap dim

        # TODO this block comes from the dqn loss, we need to swap all these with a proper
        #  helper function which selects the value given the action for all discrete spaces
        if self.action_space == "categorical":
            if action.shape != action_value.shape:
                # unsqueeze the action if it lacks on trailing singleton dim
                action = action.unsqueeze(-1)
            chosen_action_value = torch.gather(action_value, -1, index=action).squeeze(
                -1
            )
        else:
            action = action.to(torch.float)
            chosen_action_value = (action_value * action).sum(-1)

        td_error = torch.abs(chosen_action_value - target_value)
        loss_qval = distance_loss(
            chosen_action_value,
            target_value.expand_as(chosen_action_value),
            loss_function=self.loss_function,
        ).sum(0)

        metadata = {
            "td_error": td_error.detach().max(0)[0],
        }
        return loss_qval, metadata

    def _actor_loss(
        self, tensordict: TensorDictBase
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        # get probs and log probs for actions
        with self.actor_network_params.to_module(self.actor_network):
            dist = self.actor_network.get_dist(tensordict.clone(False))
        prob = dist.probs
        log_prob = dist.logits

        td_q = tensordict.select(*self.qvalue_network.in_keys, strict=False)

        td_q = self._vmap_qnetworkN0(
            td_q, self._cached_detached_qvalue_params  # should we clone?
        )
        min_q = td_q.get(self.tensor_keys.action_value).min(0)[0]

        if log_prob.shape != min_q.shape:
            raise RuntimeError(
                f"Losses shape mismatch: {log_prob.shape} and {min_q.shape}"
            )

        # like in continuous SAC, we take the entropy term and subtract the minimum of the value ensemble
        loss = self._alpha * log_prob - min_q
        # unlike in continuous SAC, we can compute the exact expectation over all discrete actions
        loss = (prob * loss).sum(-1)

        return loss, {"log_prob": (log_prob * prob).sum(-1).detach()}

    def _alpha_loss(self, log_prob: Tensor) -> Tensor:
        if self.target_entropy is not None:
            # we can compute this loss even if log_alpha is not a parameter
            alpha_loss = -self.log_alpha * (log_prob + self.target_entropy)
        else:
            # placeholder
            alpha_loss = torch.zeros_like(log_prob)
        return alpha_loss

    @property
    def _alpha(self):
        if self.min_log_alpha is not None:
            self.log_alpha.data = self.log_alpha.data.clamp(
                self.min_log_alpha, self.max_log_alpha
            )
        with torch.no_grad():
            alpha = self.log_alpha.exp()
        return alpha

    @property
    @_cache_values
    def _cached_detached_qvalue_params(self):
        return self.qvalue_network_params.detach()

    def make_value_estimator(self, value_type: ValueEstimators = None, **hyperparams):
        if value_type is None:
            value_type = self.default_value_estimator
        self.value_type = value_type
        hp = dict(default_value_kwargs(value_type))
        hp.update(hyperparams)
        if hasattr(self, "gamma"):
            hp["gamma"] = self.gamma
        if value_type is ValueEstimators.TD1:
            self._value_estimator = TD1Estimator(
                **hp,
                value_network=None,
            )
        elif value_type is ValueEstimators.TD0:
            self._value_estimator = TD0Estimator(
                **hp,
                value_network=None,
            )
        elif value_type is ValueEstimators.GAE:
            raise NotImplementedError(
                f"Value type {value_type} it not implemented for loss {type(self)}."
            )
        elif value_type is ValueEstimators.TDLambda:
            self._value_estimator = TDLambdaEstimator(
                **hp,
                value_network=None,
            )
        else:
            raise NotImplementedError(f"Unknown value type {value_type}")

        tensor_keys = {
            "value": self.tensor_keys.value,
            "value_target": "value_target",
            "reward": self.tensor_keys.reward,
            "done": self.tensor_keys.done,
            "terminated": self.tensor_keys.terminated,
        }
        self._value_estimator.set_keys(**tensor_keys)
