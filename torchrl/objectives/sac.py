# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import math
import warnings
from dataclasses import dataclass
from numbers import Number
from typing import Optional, Tuple, Union

import numpy as np
import torch
from tensordict.nn import dispatch, make_functional, TensorDictModule
from tensordict.tensordict import TensorDict, TensorDictBase
from tensordict.utils import NestedKey
from torch import Tensor

from torchrl.data import CompositeSpec
from torchrl.envs.utils import ExplorationType, set_exploration_type, step_mdp

from torchrl.modules import ProbabilisticActor
from torchrl.modules.tensordict_module.actors import ActorCriticWrapper
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

    _has_functorch = True
    err = ""
except ImportError as err:
    _has_functorch = False
    FUNCTORCH_ERROR = err


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

    Examples:
        >>> import torch
        >>> from torch import nn
        >>> from torchrl.data import BoundedTensorSpec
        >>> from torchrl.modules.distributions.continuous import NormalParamWrapper, TanhNormal
        >>> from torchrl.modules.tensordict_module.actors import ProbabilisticActor, ValueOperator
        >>> from torchrl.modules.tensordict_module.common import SafeModule
        >>> from torchrl.objectives.sac import SACLoss
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
    ``["action", "next_reward", "next_done"]`` + in_keys of the actor, value, and qvalue network.
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
        """

        action: NestedKey = "action"
        value: NestedKey = "state_value"
        state_action_value: NestedKey = "state_action_value"
        log_prob: NestedKey = "_log_prob"
        priority: NestedKey = "td_error"
        reward: NestedKey = "reward"
        done: NestedKey = "done"

    default_keys = _AcceptedKeys()
    default_value_estimator = ValueEstimators.TD0

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
    ) -> None:
        self._in_keys = None
        self._out_keys = None
        if not _has_functorch:
            raise ImportError("Failed to import functorch.") from FUNCTORCH_ERROR
        super().__init__()
        self._set_deprecated_ctor_keys(priority_key=priority_key)

        # Actor
        self.delay_actor = delay_actor
        self.convert_to_functional(
            actor_network,
            "actor_network",
            create_target_params=self.delay_actor,
            funs_to_decorate=["forward", "get_dist"],
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
        self.target_entropy_buffer = None
        if self._version == 1:
            self.actor_critic = ActorCriticWrapper(
                self.actor_network, self.value_network
            )
            make_functional(self.actor_critic)
        if gamma is not None:
            warnings.warn(_GAMMA_LMBDA_DEPREC_WARNING, category=DeprecationWarning)
            self.gamma = gamma
        self._vmap_qnetworkN0 = vmap(self.qvalue_network, (None, 0))
        if self._version == 1:
            self._vmap_qnetwork00 = vmap(qvalue_network)

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
                target_entropy = -float(
                    np.prod(action_spec[self.tensor_keys.action].shape)
                )
            self.register_buffer(
                "target_entropy_buffer", torch.tensor(target_entropy, device=device)
            )
            return self.target_entropy_buffer
        return target_entropy

    def _forward_value_estimator_keys(self, **kwargs) -> None:
        if self._value_estimator is not None:
            self._value_estimator.set_keys(
                value=self.tensor_keys.value,
                reward=self.tensor_keys.reward,
                done=self.tensor_keys.done,
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

        device = self.device
        td_device = tensordict_reshape.to(device)

        if self._version == 1:
            loss_qvalue, priority = self._loss_qvalue_v1(td_device)
            loss_value = self._loss_value(td_device)
        else:
            loss_qvalue, priority = self._loss_qvalue_v2(td_device)
            loss_value = None
        loss_actor = self._loss_actor(td_device)
        loss_alpha = self._loss_alpha(td_device)
        tensordict_reshape.set(self.tensor_keys.priority, priority)
        if (loss_actor.shape != loss_qvalue.shape) or (
            loss_value is not None and loss_actor.shape != loss_value.shape
        ):
            raise RuntimeError(
                f"Losses shape mismatch: {loss_actor.shape}, {loss_qvalue.shape} and {loss_value.shape}"
            )
        if shape:
            tensordict.update(tensordict_reshape.view(shape))
        out = {
            "loss_actor": loss_actor.mean(),
            "loss_qvalue": loss_qvalue.mean(),
            "loss_alpha": loss_alpha.mean(),
            "alpha": self._alpha,
            "entropy": -td_device.get(self.tensor_keys.log_prob).mean().detach(),
        }
        if self._version == 1:
            out["loss_value"] = loss_value.mean()
        return TensorDict(out, [])

    @property
    @_cache_values
    def _cached_detached_qvalue_params(self):
        return self.qvalue_network_params.detach()

    def _loss_actor(self, tensordict: TensorDictBase) -> Tensor:
        with set_exploration_type(ExplorationType.RANDOM):
            dist = self.actor_network.get_dist(
                tensordict,
                params=self.actor_network_params,
            )
            a_reparm = dist.rsample()
        log_prob = dist.log_prob(a_reparm)

        td_q = tensordict.select(*self.qvalue_network.in_keys)
        td_q.set(self.tensor_keys.action, a_reparm)
        td_q = self._vmap_qnetworkN0(
            td_q, self._cached_detached_qvalue_params  # should we clone?
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
        return self._alpha * log_prob - min_q_logprob

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

    def _loss_qvalue_v1(self, tensordict: TensorDictBase) -> Tuple[Tensor, Tensor]:
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
        tensordict_chunks = torch.stack(
            tensordict.chunk(self.num_qvalue_nets, dim=0), 0
        )
        target_chunks = torch.stack(target_value.chunk(self.num_qvalue_nets, dim=0), 0)

        # if vmap=True, it is assumed that the input tensordict must be cast to the param shape
        tensordict_chunks = self._vmap_qnetwork00(
            tensordict_chunks, self.qvalue_network_params
        )
        pred_val = tensordict_chunks.get(self.tensor_keys.state_action_value).squeeze(
            -1
        )
        loss_value = distance_loss(
            pred_val, target_chunks, loss_function=self.loss_function
        ).view(*shape)
        priority_value = torch.cat((pred_val - target_chunks).pow(2).unbind(0), 0)

        return loss_value, priority_value

    def _get_value_v2(self, tensordict, _alpha, actor_params, qval_params):
        r"""Value network for SAC v2.

        SAC v2 is based on a value estimate of the form:

        .. math::

          V = Q(s,a) - \alpha * \log p(a | s)

        This class computes this value given the actor and qvalue network

        """
        tensordict = tensordict.clone(False)
        # get actions and log-probs
        with torch.no_grad():
            with set_exploration_type(ExplorationType.RANDOM):
                next_tensordict = tensordict.get("next").clone(False)
                next_dist = self.actor_network.get_dist(
                    next_tensordict, params=actor_params
                )
                next_action = next_dist.rsample()
                next_tensordict.set(self.tensor_keys.action, next_action)
                next_sample_log_prob = next_dist.log_prob(next_action)

            # get q-values
            next_tensordict_expand = self._vmap_qnetworkN0(next_tensordict, qval_params)
            state_action_value = next_tensordict_expand.get(
                self.tensor_keys.state_action_value
            )
            if (
                state_action_value.shape[-len(next_sample_log_prob.shape) :]
                != next_sample_log_prob.shape
            ):
                next_sample_log_prob = next_sample_log_prob.unsqueeze(-1)
            next_state_value = state_action_value - _alpha * next_sample_log_prob
            next_state_value = next_state_value.min(0)[0]
            tensordict.set(
                ("next", self.value_estimator.tensor_keys.value), next_state_value
            )
            target_value = self.value_estimator.value_estimate(tensordict).squeeze(-1)
            return target_value

    def _loss_qvalue_v2(self, tensordict: TensorDictBase) -> Tuple[Tensor, Tensor]:
        # we pass the alpha value to the tensordict. Since it's a scalar, we must erase the batch-size first.
        target_value = self._get_value_v2(
            tensordict,
            self._alpha,
            self.actor_network_params,
            self.target_qvalue_network_params,
        )

        tensordict_expand = self._vmap_qnetworkN0(
            tensordict.select(*self.qvalue_network.in_keys),
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
        ).mean(0)
        return loss_qval, td_error.detach().max(0)[0]

    def _loss_value(self, tensordict: TensorDictBase) -> Tensor:
        # value loss
        td_copy = tensordict.select(*self.value_network.in_keys).detach()
        self.value_network(
            td_copy,
            params=self.value_network_params,
        )
        pred_val = td_copy.get(self.tensor_keys.value).squeeze(-1)

        action_dist = self.actor_network.get_dist(
            td_copy,
            params=self.target_actor_network_params,
        )  # resample an action
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
        return loss_value

    def _loss_alpha(self, tensordict: TensorDictBase) -> Tensor:
        log_pi = tensordict.get(self.tensor_keys.log_prob)
        if self.target_entropy is not None:
            # we can compute this loss even if log_alpha is not a parameter
            alpha_loss = -self.log_alpha * (log_pi.detach() + self.target_entropy)
        else:
            # placeholder
            alpha_loss = torch.zeros_like(log_pi)
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
        num_actions (int): number of actions in the action space.
        num_qvalue_nets (int, optional): Number of Q-value networks to be trained. Default is 10.
        loss_function (str, optional): loss function to be used for the Q-value. Can be one of  `"smooth_l1"`, "l2",
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

    Examples:
    >>> import torch
    >>> from torch import nn
    >>> from torchrl.data.tensor_specs import OneHotDiscreteTensorSpec
    >>> from torchrl.modules.distributions.continuous import NormalParamWrapper
    >>> from torchrl.modules.distributions.discrete import OneHotCategorical
    >>> from torchrl.modules.tensordict_module.actors import ProbabilisticActor, ValueOperator
    >>> from torchrl.modules.tensordict_module.common import SafeModule
    >>> from torchrl.objectives.sac import DiscreteSACLoss
    >>> from tensordict.tensordict import TensorDict
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
    >>> data = TensorDict({
    ...         "observation": torch.randn(*batch, n_obs),
    ...         "action": action,
    ...         ("next", "done"): torch.zeros(*batch, 1, dtype=torch.bool),
    ...         ("next", "reward"): torch.randn(*batch, 1),
    ...         ("next", "observation"): torch.randn(*batch, n_obs),
    ...     }, batch)
    >>> loss(data)
    TensorDict(
        fields={
            action_log_prob_actor: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False),
            alpha: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False),
            entropy: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False),
            loss_actor: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False),
            loss_alpha: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False),
            loss_qvalue: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False),
            next.state_value: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False),
            state_action_value_actor: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False),
            target_value: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False)},
        batch_size=torch.Size([]),
        device=None,
        is_shared=False)

    This class is compatible with non-tensordict based modules too and can be
    used without recurring to any tensordict-related primitive. In this case,
    the expected keyword arguments are:
    ``["action", "next_reward", "next_done"]`` + in_keys of the actor and qvalue network.
    The return value is a tuple of tensors in the following order:
    ``["loss_actor", "loss_qvalue", "loss_alpha",
       "alpha", "entropy", "state_action_value_actor",
       "action_log_prob_actor", "next.state_value", "target_value"]``
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
        """

        action: NestedKey = "action"
        value: NestedKey = "state_value"
        priority: NestedKey = "td_error"
        reward: NestedKey = "reward"
        done: NestedKey = "done"

    default_keys = _AcceptedKeys()
    default_value_estimator = ValueEstimators.TD0
    delay_actor: bool = False
    out_keys = [
        "loss_actor",
        "loss_qvalue",
        "loss_alpha",
        "alpha",
        "entropy",
        "state_action_value_actor",
        "action_log_prob_actor",
        "next.state_value",
        "target_value",
    ]

    def __init__(
        self,
        actor_network: ProbabilisticActor,
        qvalue_network: TensorDictModule,
        num_actions: int,  # replace with spec?
        *,
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
    ):
        self._in_keys = None
        if not _has_functorch:
            raise ImportError("Failed to import functorch.") from FUNCTORCH_ERROR
        super().__init__()
        self._set_deprecated_ctor_keys(priority_key=priority_key)

        self.convert_to_functional(
            actor_network,
            "actor_network",
            create_target_params=self.delay_actor,
            funs_to_decorate=["forward", "get_dist_params"],
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

        if target_entropy == "auto":
            target_entropy = -float(np.log(1.0 / num_actions) * target_entropy_weight)
        self.register_buffer(
            "target_entropy", torch.tensor(target_entropy, device=device)
        )

        self._vmap_getdist = vmap(self.actor_network.get_dist_params)
        self._vmap_qnetwork = vmap(self.qvalue_network)

    @property
    def alpha(self):
        if self.min_log_alpha is not None:
            self.log_alpha.data.clamp_(self.min_log_alpha, self.max_log_alpha)
        with torch.no_grad():
            alpha = self.log_alpha.exp()
        return alpha

    def _forward_value_estimator_keys(self, **kwargs) -> None:
        if self._value_estimator is not None:
            self._value_estimator.set_keys(
                value=self._tensor_keys.value,
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

    @dispatch
    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        obs_keys = self.actor_network.in_keys
        tensordict_select = tensordict.clone(False).select(
            "next", *obs_keys, self.tensor_keys.action
        )

        actor_params = torch.stack(
            [self.actor_network_params, self.target_actor_network_params], 0
        )

        tensordict_actor_grad = tensordict_select.select(
            *obs_keys
        )  # to avoid overwriting keys
        next_td_actor = step_mdp(tensordict_select).select(
            *self.actor_network.in_keys
        )  # next_observation ->
        tensordict_actor = torch.stack([tensordict_actor_grad, next_td_actor], 0)
        tensordict_actor = tensordict_actor.contiguous()

        with set_exploration_type(ExplorationType.RANDOM):
            # vmap doesn't support sampling, so we take it out from the vmap
            td_params = self._vmap_getdist(
                tensordict_actor,
                actor_params,
            )
            if isinstance(self.actor_network, ProbabilisticActor):
                tensordict_actor_dist = self.actor_network.build_dist_from_params(
                    td_params
                )
            else:
                tensordict_actor_dist = self.actor_network.build_dist_from_params(
                    td_params
                )
            probs = tensordict_actor_dist.probs
            z = (probs == 0.0).float() * 1e-8
            logp_pi = torch.log(probs + z)
            logp_pi_pol = torch.sum(probs * logp_pi, dim=-1, keepdim=True)

        # repeat tensordict_actor to match the qvalue size
        _actor_loss_td = (
            tensordict_actor[0]
            .select(*self.qvalue_network.in_keys)
            .expand(self.num_qvalue_nets, *tensordict_actor[0].batch_size)
        )  # for actor loss
        _qval_td = tensordict_select.select(*self.qvalue_network.in_keys).expand(
            self.num_qvalue_nets,
            *tensordict_select.select(*self.qvalue_network.in_keys).batch_size,
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
        q_params_detach = self.qvalue_network_params.detach()
        qvalue_params = torch.cat(
            [
                q_params_detach,
                self.target_qvalue_network_params,
                self.qvalue_network_params,
            ],
            0,
        )
        tensordict_qval = self._vmap_qnetwork(
            tensordict_qval,
            qvalue_params,
        )

        state_action_value = tensordict_qval.get(self.tensor_keys.value).squeeze(-1)
        (
            state_action_value_actor,
            next_state_action_value_qvalue,
            state_action_value_qvalue,
        ) = state_action_value.split(
            [self.num_qvalue_nets, self.num_qvalue_nets, self.num_qvalue_nets],
            dim=0,
        )

        loss_actor = -(
            (state_action_value_actor.min(0)[0] * probs[0]).sum(-1, keepdim=True)
            - self.alpha * logp_pi_pol[0]
        ).mean()

        pred_next_val = (
            probs[1]
            * (next_state_action_value_qvalue.min(0)[0] - self.alpha * logp_pi[1])
        ).sum(dim=-1, keepdim=True)

        tensordict_select.set(
            ("next", self.value_estimator.tensor_keys.value), pred_next_val
        )
        target_value = self.value_estimator.value_estimate(tensordict_select).squeeze(
            -1
        )

        actions = torch.argmax(tensordict_select.get(self.tensor_keys.action), dim=-1)

        pred_val_1 = (
            state_action_value_qvalue[0].gather(-1, actions.unsqueeze(-1)).unsqueeze(0)
        )
        pred_val_2 = (
            state_action_value_qvalue[1].gather(-1, actions.unsqueeze(-1)).unsqueeze(0)
        )
        pred_val = torch.cat([pred_val_1, pred_val_2], dim=0).squeeze()
        td_error = (pred_val - target_value.expand_as(pred_val)).pow(2)
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

        tensordict.set(self.tensor_keys.priority, td_error.detach().max(0)[0])

        loss_alpha = self._loss_alpha(logp_pi_pol)
        if not loss_qval.shape == loss_actor.shape:
            raise RuntimeError(
                f"QVal and actor loss have different shape: {loss_qval.shape} and {loss_actor.shape}"
            )
        td_out = TensorDict(
            {
                "loss_actor": loss_actor.mean(),
                "loss_qvalue": loss_qval.mean(),
                "loss_alpha": loss_alpha.mean(),
                "alpha": self.alpha.detach(),
                "entropy": -logp_pi.mean().detach(),
                "state_action_value_actor": state_action_value_actor.mean().detach(),
                "action_log_prob_actor": logp_pi.mean().detach(),
                "next.state_value": pred_next_val.mean().detach(),
                "target_value": target_value.mean().detach(),
            },
            [],
        )

        return td_out

    def _loss_alpha(self, log_pi: Tensor) -> Tensor:
        if torch.is_grad_enabled() and not log_pi.requires_grad:
            raise RuntimeError(
                "expected log_pi to require gradient for the alpha loss)"
            )
        if self.target_entropy is not None:
            # we can compute this loss even if log_alpha is not a parameter
            alpha_loss = -self.log_alpha * (log_pi.detach() + self.target_entropy)
        else:
            # placeholder
            alpha_loss = torch.zeros_like(log_pi)
        return alpha_loss

    def make_value_estimator(self, value_type: ValueEstimators = None, **hyperparams):
        if value_type is None:
            value_type = self.default_value_estimator
        self.value_type = value_type
        value_net = None
        hp = dict(default_value_kwargs(value_type))
        hp.update(hyperparams)
        if hasattr(self, "gamma"):
            hp["gamma"] = self.gamma
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
            "value": self.tensor_keys.value,
            "value_target": "value_target",
            "reward": self.tensor_keys.reward,
            "done": self.tensor_keys.done,
        }
        self._value_estimator.set_keys(**tensor_keys)
