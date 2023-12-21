# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import math
import warnings
from copy import deepcopy
from dataclasses import dataclass

from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from tensordict.nn import dispatch, TensorDictModule
from tensordict.tensordict import TensorDict, TensorDictBase
from tensordict.utils import NestedKey, unravel_key
from torch import Tensor

from torchrl.data.tensor_specs import CompositeSpec
from torchrl.data.utils import _find_action_space
from torchrl.envs.utils import ExplorationType, set_exploration_type

from torchrl.modules import ProbabilisticActor, QValueActor
from torchrl.modules.tensordict_module.common import ensure_tensordict_compatible
from torchrl.objectives.common import LossModule
from torchrl.objectives.utils import (
    _cache_values,
    _GAMMA_LMBDA_DEPREC_WARNING,
    _vmap_func,
    default_value_kwargs,
    distance_loss,
    ValueEstimators,
)

from torchrl.objectives.value import TD0Estimator, TD1Estimator, TDLambdaEstimator


class CQLLoss(LossModule):
    """TorchRL implementation of the continuous CQL loss.

    Presented in "Conservative Q-Learning for Offline Reinforcement Learning" https://arxiv.org/abs/2006.04779

    Args:
        actor_network (ProbabilisticActor): stochastic actor
        qvalue_network (TensorDictModule): Q(s, a) parametric model.
            This module typically outputs a ``"state_action_value"`` entry.
    Keyword args:
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
        gamma (float, optional): Discount factor. Default is ``None``.
        temperature (float, optional): CQL temperature. Default is 1.0.
        min_q_weight (float, optional): Minimum Q weight. Default is 1.0.
        max_q_backup (bool, optional): Whether to use the max-min Q backup.
            Default is ``False``.
        deterministic_backup (bool, optional): Whether to use the deterministic. Default is ``True``.
        num_random (int, optional): Number of random actions to sample for the CQL loss.
            Default is 10.
        with_lagrange (bool, optional): Whether to use the Lagrange multiplier.
            Default is ``False``.
        lagrange_thresh (float, optional): Lagrange threshold. Default is 0.0.

    Examples:
        >>> import torch
        >>> from torch import nn
        >>> from torchrl.data import BoundedTensorSpec
        >>> from torchrl.modules.distributions.continuous import NormalParamWrapper, TanhNormal
        >>> from torchrl.modules.tensordict_module.actors import ProbabilisticActor, ValueOperator
        >>> from torchrl.modules.tensordict_module.common import SafeModule
        >>> from torchrl.objectives.cql import CQLLoss
        >>> from tensordict.tensordict import TensorDict
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
        >>> loss = CQLLoss(actor, qvalue)
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
                loss_alpha_prime: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False),
                loss_qvalue: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False)},
            batch_size=torch.Size([]),
            device=None,
            is_shared=False)

    This class is compatible with non-tensordict based modules too and can be
    used without recurring to any tensordict-related primitive. In this case,
    the expected keyword arguments are:
    ``["action", "next_reward", "next_done", "next_terminated"]`` + in_keys of the actor, value, and qvalue network.
    The return value is a tuple of tensors in the following order:
    ``["loss_actor", "loss_qvalue", "loss_alpha", "loss_alpha_prime", "alpha", "entropy"]``.

    Examples:
        >>> import torch
        >>> from torch import nn
        >>> from torchrl.data import BoundedTensorSpec
        >>> from torchrl.modules.distributions.continuous import NormalParamWrapper, TanhNormal
        >>> from torchrl.modules.tensordict_module.actors import ProbabilisticActor, ValueOperator
        >>> from torchrl.modules.tensordict_module.common import SafeModule
        >>> from torchrl.objectives.cql import CQLLoss
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
        >>> loss = CQLLoss(actor, qvalue, value)
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

    The output keys can also be filtered using the :meth:`CQLLoss.select_out_keys`
    method.

    Examples:
        >>> loss.select_out_keys('loss_actor', 'loss_qvalue')
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
            pred_q1 (NestedKey): The input tensordict key where the predicted Q1 values are expected.
                Defaults to ``"pred_q1"``.
            pred_q2 (NestedKey): The input tensordict key where the predicted Q2 values are expected.
                Defaults to ``"pred_q2"``.
            priority (NestedKey): The input tensordict key where the target priority is written to.
                Defaults to ``"td_error"``.
            cql_q1_loss (NestedKey): The input tensordict key where the CQL Q1 loss is expected.
                Defaults to ``"cql_q1_loss"``.
            cql_q2_loss (NestedKey): The input tensordict key where the CQL Q2 loss is expected.
                Defaults to ``"cql_q2_loss"``.
            reward (NestedKey): The input tensordict key where the reward is expected.
                Defaults to ``"reward"``.
            done (NestedKey): The input tensordict key where the done flag is expected.
                Defaults to ``"done"``.
            terminated (NestedKey): The input tensordict key where the terminated flag is expected.
                Defaults to ``"terminated"``.
        """

        action: NestedKey = "action"
        value: NestedKey = "state_value"
        state_action_value: NestedKey = "state_action_value"
        log_prob: NestedKey = "_log_prob"
        pred_q1: NestedKey = "pred_q1"
        pred_q2: NestedKey = "pred_q2"
        priority: NestedKey = "td_error"
        cql_q1_loss: NestedKey = "cql_q1_loss"
        cql_q2_loss: NestedKey = "cql_q2_loss"
        priority: NestedKey = "td_error"
        reward: NestedKey = "reward"
        done: NestedKey = "done"
        terminated: NestedKey = "terminated"

    default_keys = _AcceptedKeys()
    default_value_estimator = ValueEstimators.TD0

    def __init__(
        self,
        actor_network: ProbabilisticActor,
        qvalue_network: TensorDictModule,
        *,
        loss_function: str = "smooth_l1",
        alpha_init: float = 1.0,
        min_alpha: float = None,
        max_alpha: float = None,
        action_spec=None,
        fixed_alpha: bool = False,
        target_entropy: Union[str, float] = "auto",
        delay_actor: bool = False,
        delay_qvalue: bool = True,
        gamma: float = None,
        temperature: float = 1.0,
        min_q_weight: float = 1.0,
        max_q_backup: bool = False,
        deterministic_backup: bool = True,
        num_random: int = 10,
        with_lagrange: bool = False,
        lagrange_thresh: float = 0.0,
    ) -> None:
        self._out_keys = None
        super().__init__()

        # Actor
        self.delay_actor = delay_actor
        self.convert_to_functional(
            actor_network,
            "actor_network",
            create_target_params=self.delay_actor,
        )

        # Q value
        self.delay_qvalue = delay_qvalue
        self.num_qvalue_nets = 2

        self.convert_to_functional(
            qvalue_network,
            "qvalue_network",
            self.num_qvalue_nets,
            create_target_params=self.delay_qvalue,
            compare_against=list(actor_network.parameters()),
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
        self.target_entropy_buffer = None

        if gamma is not None:
            warnings.warn(_GAMMA_LMBDA_DEPREC_WARNING, category=DeprecationWarning)
            self.gamma = gamma

        self.temperature = temperature
        self.min_q_weight = min_q_weight
        self.max_q_backup = max_q_backup
        self.deterministic_backup = deterministic_backup
        self.num_random = num_random
        self.with_lagrange = with_lagrange

        if self.with_lagrange:
            self.target_action_gap = lagrange_thresh
            self.register_parameter(
                "log_alpha_prime",
                torch.nn.Parameter(torch.tensor(math.log(1.0), device=device)),
            )

        self._vmap_qvalue_networkN0 = _vmap_func(self.qvalue_network, (None, 0))
        self._vmap_qvalue_network00 = _vmap_func(self.qvalue_network)

    @property
    def target_entropy(self):
        target_entropy = self.target_entropy_buffer
        if target_entropy is None:
            delattr(self, "target_entropy_buffer")
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
                    action_container_shape = action_spec[
                        self.tensor_keys.action[:-1]
                    ].shape
                else:
                    action_container_shape = action_spec.shape
                target_entropy = -float(
                    action_spec[self.tensor_keys.action]
                    .shape[len(action_container_shape) :]
                    .numel()
                )
            self.register_buffer(
                "target_entropy_buffer", torch.tensor(target_entropy, device=device)
            )
            return self.target_entropy_buffer
        return target_entropy

    def _forward_value_estimator_keys(self, **kwargs) -> None:
        if self._value_estimator is not None:
            self._value_estimator.set_keys(
                value=self._tensor_keys.value,
                reward=self.tensor_keys.reward,
                done=self.tensor_keys.done,
                terminated=self.tensor_keys.terminated,
            )

    def make_value_estimator(self, value_type: ValueEstimators = None, **hyperparams):
        if value_type is None:
            value_type = self.default_value_estimator
        self.value_type = value_type

        # we will take care of computing the next value inside this module
        value_net = None

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
    def in_keys(self):
        keys = [
            self.tensor_keys.action,
            ("next", self.tensor_keys.reward),
            ("next", self.tensor_keys.done),
            ("next", self.tensor_keys.terminated),
            *self.actor_network.in_keys,
            *[("next", key) for key in self.actor_network.in_keys],
            *self.qvalue_network.in_keys,
        ]

        return list(set(keys))

    @property
    def out_keys(self):
        if self._out_keys is None:
            keys = [
                "loss_actor",
                "loss_actor_bc",
                "loss_qvalue",
                "loss_cql",
                "loss_alpha",
                "loss_alpha_prime",
                "alpha",
                "entropy",
            ]
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

        td_device = tensordict_reshape.to(tensordict.device)

        q_loss, metadata = self.q_loss(td_device)
        cql_loss, cql_metadata = self.cql_loss(td_device)
        if self.with_lagrange:
            alpha_prime_loss, alpha_prime_metadata = self.alpha_prime_loss(td_device)
            metadata.update(alpha_prime_metadata)
        loss_actor_bc, bc_metadata = self.actor_bc_loss(td_device)
        loss_actor, actor_metadata = self.actor_loss(td_device)
        loss_alpha, alpha_metadata = self.alpha_loss(td_device)
        metadata.update(bc_metadata)
        metadata.update(cql_metadata)
        metadata.update(actor_metadata)
        metadata.update(alpha_metadata)
        tensordict_reshape.set(
            self.tensor_keys.priority, metadata.pop("td_error").detach().max(0).values
        )
        if shape:
            tensordict.update(tensordict_reshape.view(shape))
        out = {
            "loss_actor": loss_actor.mean(),
            "loss_actor_bc": loss_actor_bc.mean(),
            "loss_qvalue": q_loss.mean(),
            "loss_cql": cql_loss.mean(),
            "loss_alpha": loss_alpha.mean(),
            "alpha": self._alpha,
            "entropy": -td_device.get(self.tensor_keys.log_prob).mean().detach(),
        }
        if self.with_lagrange:
            out["loss_alpha_prime"] = alpha_prime_loss.mean()
        return TensorDict(out, [])

    @property
    @_cache_values
    def _cached_detach_qvalue_params(self):
        return self.qvalue_network_params.detach()

    def actor_bc_loss(self, tensordict: TensorDictBase) -> Tensor:
        with set_exploration_type(
            ExplorationType.RANDOM
        ), self.actor_network_params.to_module(self.actor_network):
            dist = self.actor_network.get_dist(
                tensordict,
            )
            a_reparm = dist.rsample()
        log_prob = dist.log_prob(a_reparm)
        bc_log_prob = dist.log_prob(tensordict.get(self.tensor_keys.action))

        bc_actor_loss = self._alpha * log_prob - bc_log_prob
        metadata = {"bc_log_prob": bc_log_prob.mean().detach()}
        return bc_actor_loss, metadata

    def actor_loss(self, tensordict: TensorDictBase) -> Tensor:
        with set_exploration_type(
            ExplorationType.RANDOM
        ), self.actor_network_params.to_module(self.actor_network):
            dist = self.actor_network.get_dist(
                tensordict,
            )
            a_reparm = dist.rsample()
        log_prob = dist.log_prob(a_reparm)

        td_q = tensordict.select(*self.qvalue_network.in_keys)
        td_q.set(self.tensor_keys.action, a_reparm)
        td_q = self._vmap_qvalue_networkN0(
            td_q,
            self._cached_detach_qvalue_params,
        )
        min_q_logprob = (
            td_q.get(self.tensor_keys.state_action_value).min(0)[0].squeeze(-1)
        )

        if log_prob.shape != min_q_logprob.shape:
            raise RuntimeError(
                f"Losses shape mismatch: {log_prob.shape} and {min_q_logprob.shape}"
            )

        # write log_prob in tensordict for alpha loss
        tensordict.set(self.tensor_keys.log_prob, log_prob.detach())
        actor_loss = self._alpha * log_prob - min_q_logprob

        return actor_loss, {}

    def _get_policy_actions(self, data, actor_params, num_actions=10):
        batch_size = data.batch_size
        batch_size = list(batch_size[:-1]) + [batch_size[-1] * num_actions]
        tensordict = data.select(*self.actor_network.in_keys).apply(
            lambda x: x.repeat_interleave(num_actions, dim=data.ndim - 1),
            batch_size=batch_size,
        )
        with torch.no_grad():
            with set_exploration_type(ExplorationType.RANDOM), actor_params.to_module(
                self.actor_network
            ):
                dist = self.actor_network.get_dist(tensordict)
                action = dist.rsample()
                tensordict.set(self.tensor_keys.action, action)
                sample_log_prob = dist.log_prob(action)
                # tensordict.del_("loc")
                # tensordict.del_("scale")

        return (
            tensordict.select(*self.actor_network.in_keys, self.tensor_keys.action),
            sample_log_prob,
        )

    def _get_value_v(self, tensordict, _alpha, actor_params, qval_params):
        tensordict = tensordict.clone(False)
        # get actions and log-probs
        with torch.no_grad():
            with set_exploration_type(ExplorationType.RANDOM), actor_params.to_module(
                self.actor_network
            ):
                next_tensordict = tensordict.get("next").clone(False)
                next_dist = self.actor_network.get_dist(next_tensordict)
                next_action = next_dist.rsample()
                next_tensordict.set(self.tensor_keys.action, next_action)
                next_sample_log_prob = next_dist.log_prob(next_action)

            # get q-values
            if not self.max_q_backup:
                next_tensordict_expand = self._vmap_qvalue_networkN0(
                    next_tensordict, qval_params
                )
                next_state_value = next_tensordict_expand.get(
                    self.tensor_keys.state_action_value
                ).min(0)[0]
                if (
                    next_state_value.shape[-len(next_sample_log_prob.shape) :]
                    != next_sample_log_prob.shape
                ):
                    next_sample_log_prob = next_sample_log_prob.unsqueeze(-1)
                if not self.deterministic_backup:
                    next_state_value = next_state_value - _alpha * next_sample_log_prob

            if self.max_q_backup:
                next_tensordict, _ = self._get_policy_actions(
                    tensordict.get("next"),
                    actor_params,
                    num_actions=self.num_random,
                )
                next_tensordict_expand = self._vmap_qvalue_networkN0(
                    next_tensordict, qval_params
                )

                state_action_value = next_tensordict_expand.get(
                    self.tensor_keys.state_action_value
                )
                # take max over actions
                state_action_value = state_action_value.reshape(
                    self.num_qvalue_nets, tensordict.shape[0], self.num_random, -1
                ).max(-2)[0]
                # take min over qvalue nets
                next_state_value = state_action_value.min(0)[0]

            tensordict.set(
                ("next", self.value_estimator.tensor_keys.value), next_state_value
            )
            target_value = self.value_estimator.value_estimate(tensordict).squeeze(-1)
            return target_value

    def q_loss(self, tensordict: TensorDictBase) -> Tensor:
        # we pass the alpha value to the tensordict. Since it's a scalar, we must erase the batch-size first.
        target_value = self._get_value_v(
            tensordict,
            self._alpha,
            self.actor_network_params,
            self.target_qvalue_network_params,
        )

        tensordict_pred_q = tensordict.select(*self.qvalue_network.in_keys)
        q_pred = self._vmap_qvalue_networkN0(
            tensordict_pred_q, self.qvalue_network_params
        ).get(self.tensor_keys.state_action_value)

        # write pred values in tensordict for cql loss
        tensordict.set(self.tensor_keys.pred_q1, q_pred[0])
        tensordict.set(self.tensor_keys.pred_q2, q_pred[1])

        q_pred = q_pred.squeeze(-1)
        loss_qval = distance_loss(
            q_pred,
            target_value.expand_as(q_pred),
            loss_function=self.loss_function,
        )
        td_error = (q_pred - target_value).pow(2)
        metadata = {"td_error": td_error.detach()}

        return loss_qval.sum(0).mean(), metadata

    def cql_loss(self, tensordict: TensorDictBase) -> Tensor:
        pred_q1 = tensordict.get(self.tensor_keys.pred_q1)
        pred_q2 = tensordict.get(self.tensor_keys.pred_q2)

        if pred_q1 is None:
            raise KeyError(
                f"Couldn't find the pred_q1 with key {self.tensor_keys.pred_q1} in the input tensordict. "
                "This could be caused by calling cql_loss method before q_loss method."
            )
        if pred_q2 is None:
            raise KeyError(
                f"Couldn't find the pred_q2 with key {self.tensor_keys.pred_q2} in the input tensordict. "
                "This could be caused by calling cql_loss method before q_loss method."
            )

        random_actions_tensor = (
            torch.FloatTensor(
                tensordict.shape[0] * self.num_random,
                tensordict[self.tensor_keys.action].shape[-1],
            )
            .uniform_(-1, 1)
            .to(tensordict.device)
        )
        curr_actions_td, curr_log_pis = self._get_policy_actions(
            tensordict,
            self.actor_network_params,
            num_actions=self.num_random,
        )
        new_curr_actions_td, new_log_pis = self._get_policy_actions(
            tensordict.get("next"),
            self.actor_network_params,
            num_actions=self.num_random,
        )

        # process all in one forward pass
        # stack qvalue params
        qvalue_params = torch.cat(
            [
                self.qvalue_network_params,
                self.qvalue_network_params,
                self.qvalue_network_params,
            ],
            0,
        )
        # select and stack input params
        # q value random action
        tensordict_q_random = tensordict.select(*self.actor_network.in_keys)

        batch_size = tensordict_q_random.batch_size
        batch_size = list(batch_size[:-1]) + [batch_size[-1] * self.num_random]
        tensordict_q_random = tensordict_q_random.select(
            *self.actor_network.in_keys
        ).apply(
            lambda x: x.repeat_interleave(
                self.num_random, dim=tensordict_q_random.ndim - 1
            ),
            batch_size=batch_size,
        )
        tensordict_q_random.set(self.tensor_keys.action, random_actions_tensor)
        cql_tensordict = torch.cat(
            [
                tensordict_q_random.expand(
                    self.num_qvalue_nets, *curr_actions_td.batch_size
                ),
                curr_actions_td.expand(
                    self.num_qvalue_nets, *curr_actions_td.batch_size
                ),
                new_curr_actions_td.expand(
                    self.num_qvalue_nets, *curr_actions_td.batch_size
                ),
            ],
            0,
        )
        cql_tensordict = cql_tensordict.contiguous()

        cql_tensordict_expand = self._vmap_qvalue_network00(
            cql_tensordict, qvalue_params
        )
        # get q values
        state_action_value = cql_tensordict_expand.get(
            self.tensor_keys.state_action_value
        )
        # split q values
        (q_random, q_curr, q_new,) = state_action_value.split(
            [
                self.num_qvalue_nets,
                self.num_qvalue_nets,
                self.num_qvalue_nets,
            ],
            dim=0,
        )

        # importance sammpled version
        random_density = np.log(
            0.5 ** curr_actions_td[self.tensor_keys.action].shape[-1]
        )
        cat_q1 = torch.cat(
            [
                q_random[0] - random_density,
                q_new[0] - new_log_pis.detach().unsqueeze(-1),
                q_curr[0] - curr_log_pis.detach().unsqueeze(-1),
            ],
            1,
        )
        cat_q2 = torch.cat(
            [
                q_random[1] - random_density,
                q_new[1] - new_log_pis.detach().unsqueeze(-1),
                q_curr[1] - curr_log_pis.detach().unsqueeze(-1),
            ],
            1,
        )

        min_qf1_loss = (
            torch.logsumexp(cat_q1 / self.temperature, dim=1)
            * self.min_q_weight
            * self.temperature
        )
        min_qf2_loss = (
            torch.logsumexp(cat_q2 / self.temperature, dim=1)
            * self.min_q_weight
            * self.temperature
        )

        # Subtract the log likelihood of data
        cql_q1_loss = min_qf1_loss - pred_q1 * self.min_q_weight
        cql_q2_loss = min_qf2_loss - pred_q2 * self.min_q_weight

        # write cql losses in tensordict for alpha prime loss
        tensordict.set(self.tensor_keys.cql_q1_loss, cql_q1_loss)
        tensordict.set(self.tensor_keys.cql_q2_loss, cql_q2_loss)

        return (cql_q1_loss + cql_q2_loss).mean(), {}

    def alpha_prime_loss(self, tensordict: TensorDictBase) -> Tensor:
        cql_q1_loss = tensordict.get(self.tensor_keys.cql_q1_loss)
        cql_q2_loss = tensordict.get(self.tensor_keys.cql_q2_loss)

        if cql_q1_loss is None:
            raise KeyError(
                f"Couldn't find the cql_q1_loss with key {self.tensor_keys.cql_q1_loss} in the input tensordict. "
                "This could be caused by calling alpha_prime_loss method before cql_loss method."
            )
        if cql_q2_loss is None:
            raise KeyError(
                f"Couldn't find the cql_q2_loss with key {self.tensor_keys.cql_q2_loss} in the input tensordict. "
                "This could be caused by calling alpha_prime_loss method before cql_loss method."
            )

        alpha_prime = torch.clamp_max(self.log_alpha_prime.exp(), max=1000000.0)
        min_qf1_loss = alpha_prime * (cql_q1_loss.mean() - self.target_action_gap)
        min_qf2_loss = alpha_prime * (cql_q2_loss.mean() - self.target_action_gap)

        alpha_prime_loss = (-min_qf1_loss - min_qf2_loss) * 0.5
        return alpha_prime_loss, {}

    def alpha_loss(self, tensordict: TensorDictBase) -> Tensor:
        log_pi = tensordict.get(self.tensor_keys.log_prob)
        if self.target_entropy is not None:
            # we can compute this loss even if log_alpha is not a parameter
            alpha_loss = -self.log_alpha * (log_pi.detach() + self.target_entropy)
        else:
            # placeholder
            alpha_loss = torch.zeros_like(log_pi)
        return alpha_loss, {}

    @property
    def _alpha(self):
        if self.min_log_alpha is not None:
            self.log_alpha.data.clamp_(self.min_log_alpha, self.max_log_alpha)
        with torch.no_grad():
            alpha = self.log_alpha.exp()
        return alpha


class DiscreteCQLLoss(LossModule):
    """TorchRL implementation of the discrete CQL loss.

    This class implements the discrete conservative Q-learning (CQL) loss function, as presented in the paper
    "Conservative Q-Learning for Offline Reinforcement Learning" (https://arxiv.org/abs/2006.04779).

    Args:
        value_network (Union[QValueActor, nn.Module]): The Q-value network used to estimate state-action values.
    Keyword Args:
        loss_function (Optional[str]): The distance function used to calculate the distance between the predicted
            Q-values and the target Q-values. Defaults to ``l2``.
        delay_value (bool): Whether to separate the target Q value
            networks from the Q value networks used for data collection.
            Default is ``True``.
        gamma (float, optional): Discount factor. Default is ``None``.
        action_space: The action space of the environment. If None, it is inferred from the value network.
            Defaults to None.


    Examples:
        >>> from torchrl.modules import MLP
        >>> from torchrl.data import OneHotDiscreteTensorSpec
        >>> n_obs, n_act = 4, 3
        >>> value_net = MLP(in_features=n_obs, out_features=n_act)
        >>> spec = OneHotDiscreteTensorSpec(n_act)
        >>> actor = QValueActor(value_net, in_keys=["observation"], action_space=spec)
        >>> loss = DiscreteCQLLoss(actor, action_space=spec)
        >>> batch = [10,]
        >>> data = TensorDict({
        ...     "observation": torch.randn(*batch, n_obs),
        ...     "action": spec.rand(batch),
        ...     ("next", "observation"): torch.randn(*batch, n_obs),
        ...     ("next", "done"): torch.zeros(*batch, 1, dtype=torch.bool),
        ...     ("next", "terminated"): torch.zeros(*batch, 1, dtype=torch.bool),
        ...     ("next", "reward"): torch.randn(*batch, 1)
        ... }, batch)
        >>> loss(data)
        TensorDict(
            fields={
                loss: Tensor(shape=torch.Size([]), device=cuda:0, dtype=torch.float32, is_shared=True),
                loss_cql: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False)},
            batch_size=torch.Size([]),
            device=None,
            is_shared=False)

    This class is compatible with non-tensordict based modules too and can be
    used without recurring to any tensordict-related primitive. In this case,
    the expected keyword arguments are:
    ``["observation", "next_observation", "action", "next_reward", "next_done", "next_terminated"]``,
    and a single loss value is returned.

    Examples:
        >>> from torchrl.objectives import DiscreteCQLLoss
        >>> from torchrl.data import OneHotDiscreteTensorSpec
        >>> from torch import nn
        >>> import torch
        >>> n_obs = 3
        >>> n_action = 4
        >>> action_spec = OneHotDiscreteTensorSpec(n_action)
        >>> value_network = nn.Linear(n_obs, n_action) # a simple value model
        >>> dcql_loss = DiscreteCQLLoss(value_network, action_space=action_spec)
        >>> # define data
        >>> observation = torch.randn(n_obs)
        >>> next_observation = torch.randn(n_obs)
        >>> action = action_spec.rand()
        >>> next_reward = torch.randn(1)
        >>> next_done = torch.zeros(1, dtype=torch.bool)
        >>> next_terminated = torch.zeros(1, dtype=torch.bool)
        >>> loss_val = dcql_loss(
        ...     observation=observation,
        ...     next_observation=next_observation,
        ...     next_reward=next_reward,
        ...     next_done=next_done,
        ...     next_terminated=next_terminated,
        ...     action=action)
    """

    @dataclass
    class _AcceptedKeys:
        """Maintains default values for all configurable tensordict keys.

        This class defines which tensordict keys can be set using '.set_keys(key_name=key_value)' and their
        default values.

        Attributes:
            value_target (NestedKey): The input tensordict key where the target state value is expected.
                Will be used for the underlying value estimator Defaults to ``"value_target"``.
            value (NestedKey): The input tensordict key where the chosen action value is expected.
                Will be used for the underlying value estimator. Defaults to ``"chosen_action_value"``.
            action_value (NestedKey): The input tensordict key where the action value is expected.
                Defaults to ``"action_value"``.
            action (NestedKey): The input tensordict key where the action is expected.
                Defaults to ``"action"``.
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
            pred_val (NestedKey): The key where the predicted value will be written
                in the input tensordict. This value is subsequently used by cql_loss.
                Defaults to ``"pred_val"``.

        """

        value_target: NestedKey = "value_target"
        value: NestedKey = "chosen_action_value"
        action_value: NestedKey = "action_value"
        action: NestedKey = "action"
        priority: NestedKey = "td_error"
        reward: NestedKey = "reward"
        done: NestedKey = "done"
        terminated: NestedKey = "terminated"
        pred_val: NestedKey = "pred_val"

    default_keys = _AcceptedKeys()
    default_value_estimator = ValueEstimators.TD0
    out_keys = [
        "loss_qvalue",
        "loss_cql",
    ]

    def __init__(
        self,
        value_network: Union[QValueActor, nn.Module],
        *,
        loss_function: Optional[str] = "l2",
        delay_value: bool = True,
        gamma: float = None,
        action_space=None,
    ) -> None:
        super().__init__()
        self._in_keys = None
        self.delay_value = delay_value
        value_network = ensure_tensordict_compatible(
            module=value_network,
            wrapper_type=QValueActor,
            action_space=action_space,
        )

        self.convert_to_functional(
            value_network,
            "value_network",
            create_target_params=self.delay_value,
        )

        self.value_network_in_keys = value_network.in_keys

        self.loss_function = loss_function
        if action_space is None:
            # infer from value net
            try:
                action_space = value_network.spec
            except AttributeError:
                # let's try with action_space then
                try:
                    action_space = value_network.action_space
                except AttributeError:
                    raise ValueError(self.ACTION_SPEC_ERROR)
        if action_space is None:
            warnings.warn(
                "action_space was not specified. DiscreteCQLLoss will default to 'one-hot'. "
                "This behaviour will be deprecated soon and a space will have to be passed. "
                "Check the DiscreteCQLLoss documentation to see how to pass the action space."
            )
            action_space = "one-hot"
        self.action_space = _find_action_space(action_space)

        if gamma is not None:
            warnings.warn(_GAMMA_LMBDA_DEPREC_WARNING, category=DeprecationWarning)
            self.gamma = gamma

    def _forward_value_estimator_keys(self, **kwargs) -> None:
        if self._value_estimator is not None:
            self._value_estimator.set_keys(
                value_target=self.tensor_keys.value_target,
                value=self._tensor_keys.value,
                reward=self._tensor_keys.reward,
                done=self._tensor_keys.done,
                terminated=self._tensor_keys.terminated,
            )
        self._set_in_keys()

    def _set_in_keys(self):
        in_keys = {
            self.tensor_keys.action,
            unravel_key(("next", self.tensor_keys.reward)),
            unravel_key(("next", self.tensor_keys.done)),
            unravel_key(("next", self.tensor_keys.terminated)),
            *self.value_network.in_keys,
            *[unravel_key(("next", key)) for key in self.value_network.in_keys],
        }
        self._in_keys = sorted(in_keys, key=str)

    def make_value_estimator(self, value_type: ValueEstimators = None, **hyperparams):
        if value_type is None:
            value_type = self.default_value_estimator
        self.value_type = value_type

        # we will take care of computing the next value inside this module
        value_net = deepcopy(self.value_network)
        self.value_network_params.to_module(value_net, return_swap=False)

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
    def in_keys(self):
        if self._in_keys is None:
            self._set_in_keys()
        return self._in_keys

    @in_keys.setter
    def in_keys(self, values):
        self._in_keys = values

    @dispatch
    def value_loss(
        self,
        tensordict: TensorDictBase,
    ) -> Tuple[torch.Tensor, dict]:
        td_copy = tensordict.clone(False)
        with self.value_network_params.to_module(self.value_network):
            self.value_network(td_copy)

        action = tensordict.get(self.tensor_keys.action)
        pred_val = td_copy.get(self.tensor_keys.action_value)

        if self.action_space == "categorical":
            if action.shape != pred_val.shape:
                # unsqueeze the action if it lacks on trailing singleton dim
                action = action.unsqueeze(-1)
            pred_val_index = torch.gather(pred_val, -1, index=action).squeeze(-1)
        else:
            action = action.to(torch.float)
            pred_val_index = (pred_val * action).sum(-1)

        # calculate target value
        with torch.no_grad():
            target_value = self.value_estimator.value_estimate(
                td_copy, params=self._cached_detached_target_value_params
            ).squeeze(-1)

        with torch.no_grad():
            td_error = (pred_val_index - target_value).pow(2)
            td_error = td_error.unsqueeze(-1)
        if tensordict.device is not None:
            td_error = td_error.to(tensordict.device)

        tensordict.set(
            self.tensor_keys.priority,
            td_error,
            inplace=True,
        )
        tensordict.set(
            self.tensor_keys.pred_val,
            pred_val,
            inplace=True,
        )
        loss = (
            0.5 * distance_loss(pred_val_index, target_value, self.loss_function).mean()
        )

        metadata = {
            "td_error": td_error.mean(0).detach(),
            "pred_value": pred_val.mean().detach(),
            "target_value": target_value.mean().detach(),
        }

        return loss, metadata

    @dispatch
    def forward(self, tensordict: TensorDictBase) -> TensorDict:
        """Computes the (DQN) CQL loss given a tensordict sampled from the replay buffer.

        This function will also write a "td_error" key that can be used by prioritized replay buffers to assign
            a priority to items in the tensordict.

        Args:
            tensordict (TensorDictBase): a tensordict with keys ["action"] and the in_keys of
                the value network (observations, "done", "terminated", "reward" in a "next" tensordict).

        Returns:
            a tensor containing the CQL loss.

        """
        loss_qval, metadata = self.value_loss(tensordict)
        loss_cql, _ = self.cql_loss(tensordict)
        source = {
            "loss_qvalue": loss_qval,
            "loss_cql": loss_cql,
        }
        source.update(metadata)
        td_out = TensorDict(
            source=source,
            batch_size=[],
        )

        return td_out

    @property
    @_cache_values
    def _cached_detached_target_value_params(self):
        return self.target_value_network_params.detach()

    def cql_loss(self, tensordict):
        qvalues = tensordict.get(self.tensor_keys.pred_val, default=None)
        if qvalues is None:
            raise KeyError(
                "Couldn't find the predicted qvalue with key {self.tensor_keys.pred_val} in the input tensordict. "
                "This could be caused by calling cql_loss method before value_loss."
            )

        current_action = tensordict.get(self.tensor_keys.action)

        logsumexp = torch.logsumexp(qvalues, dim=-1, keepdim=True)
        if self.action_space == "categorical":
            if current_action.shape != qvalues.shape:
                # unsqueeze the action if it lacks on trailing singleton dim
                current_action = current_action.unsqueeze(-1)
            q_a = qvalues.gather(-1, current_action)
        else:
            q_a = (qvalues * current_action).sum(dim=-1, keepdim=True)

        return (logsumexp - q_a).mean(), {}
