# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Number
from typing import List, Union

import torch
from tensordict import TensorDict, TensorDictBase, TensorDictParams

from tensordict.nn import dispatch, TensorDictModule
from tensordict.utils import NestedKey
from torch import Tensor

from torchrl.data.tensor_specs import Composite
from torchrl.envs.utils import ExplorationType, set_exploration_type, step_mdp
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


class REDQLoss(LossModule):
    """REDQ Loss module.

    REDQ (RANDOMIZED ENSEMBLED DOUBLE Q-LEARNING: LEARNING FAST WITHOUT A MODEL
    https://openreview.net/pdf?id=AY8zfZm0tDd) generalizes the idea of using an ensemble of Q-value functions to
    train a SAC-like algorithm.

    Args:
        actor_network (TensorDictModule): the actor to be trained
        qvalue_network (TensorDictModule): a single Q-value network or a list of Q-value networks.
            If a single instance of `qvalue_network` is provided, it will be duplicated ``num_qvalue_nets``
            times. If a list of modules is passed, their
            parameters will be stacked unless they share the same identity (in which case
            the original parameter will be expanded).

            .. warning:: When a list of parameters if passed, it will __not__ be compared against the policy parameters
              and all the parameters will be considered as untied.

    Keyword Args:
        num_qvalue_nets (int, optional): Number of Q-value networks to be trained.
            Default is ``10``.
        sub_sample_len (int, optional): number of Q-value networks to be
            subsampled to evaluate the next state value
            Default is ``2``.
        loss_function (str, optional): loss function to be used for the Q-value.
            Can be one of  ``"smooth_l1"``, ``"l2"``,
            ``"l1"``, Default is ``"smooth_l1"``.
        alpha_init (:obj:`float`, optional): initial entropy multiplier.
            Default is ``1.0``.
        min_alpha (:obj:`float`, optional): min value of alpha.
            Default is ``0.1``.
        max_alpha (:obj:`float`, optional): max value of alpha.
            Default is ``10.0``.
        action_spec (TensorSpec, optional): the action tensor spec. If not provided
            and the target entropy is ``"auto"``, it will be retrieved from
            the actor.
        fixed_alpha (bool, optional): whether alpha should be trained to match
            a target entropy. Default is ``False``.
        target_entropy (Union[str, Number], optional): Target entropy for the
            stochastic policy. Default is "auto".
        delay_qvalue (bool, optional): Whether to separate the target Q value
            networks from the Q value networks used
            for data collection. Default is ``False``.
        gSDE (bool, optional): Knowing if gSDE is used is necessary to create
            random noise variables.
            Default is ``False``.
        priority_key (str, optional): [Deprecated, use .set_keys() instead] Key where to write the priority value
            for prioritized replay buffers. Default is
            ``"td_error"``.
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
        >>> from torchrl.modules.distributions import NormalParamExtractor, TanhNormal
        >>> from torchrl.modules.tensordict_module.actors import ProbabilisticActor, ValueOperator
        >>> from torchrl.modules.tensordict_module.common import SafeModule
        >>> from torchrl.objectives.redq import REDQLoss
        >>> from tensordict import TensorDict
        >>> n_act, n_obs = 4, 3
        >>> spec = Bounded(-torch.ones(n_act), torch.ones(n_act), (n_act,))
        >>> net = nn.Sequential(nn.Linear(n_obs, 2 * n_act), NormalParamExtractor())
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
        >>> loss = REDQLoss(actor, qvalue)
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
    ``["action", "next_reward", "next_done", "next_terminated"]`` + in_keys of the actor and qvalue network
    The return value is a tuple of tensors in the following order:
    ``["loss_actor", "loss_qvalue", "loss_alpha", "alpha", "entropy", "state_action_value_actor", "action_log_prob_actor", "next.state_value", "target_value",]``.

    Examples:
        >>> import torch
        >>> from torch import nn
        >>> from torchrl.data import Bounded
        >>> from torchrl.modules.distributions import NormalParamExtractor, TanhNormal
        >>> from torchrl.modules.tensordict_module.actors import ProbabilisticActor, ValueOperator
        >>> from torchrl.modules.tensordict_module.common import SafeModule
        >>> from torchrl.objectives.redq import REDQLoss
        >>> n_act, n_obs = 4, 3
        >>> spec = Bounded(-torch.ones(n_act), torch.ones(n_act), (n_act,))
        >>> net = nn.Sequential(nn.Linear(n_obs, 2 * n_act), NormalParamExtractor())
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
        >>> loss = REDQLoss(actor, qvalue)
        >>> batch = [2, ]
        >>> action = spec.rand(batch)
        >>> # filter output keys to "loss_actor", and "loss_qvalue"
        >>> _ = loss.select_out_keys("loss_actor", "loss_qvalue")
        >>> loss_actor, loss_qvalue = loss(
        ...         observation=torch.randn(*batch, n_obs),
        ...         action=action,
        ...         next_done=torch.zeros(*batch, 1, dtype=torch.bool),
        ...         next_terminated=torch.zeros(*batch, 1, dtype=torch.bool),
        ...         next_reward=torch.randn(*batch, 1),
        ...         next_observation=torch.randn(*batch, n_obs))
        >>> loss_actor.backward()

    """

    @dataclass
    class _AcceptedKeys:
        """Maintains default values for all configurable tensordict keys.

        This class defines which tensordict keys can be set using '.set_keys(key_name=key_value)' and their
        default values

        Attributes:
            value (NestedKey): The input tensordict key where the state value is expected.
                Will be used for the underlying value estimator. Defaults to ``"state_value"``.
            action (NestedKey): The input tensordict key where the action is expected. Defaults to ``"action"``.
            sample_log_prob (NestedKey): The input tensordict key where the
                sample log probability is expected. Defaults to ``"sample_log_prob"``.
            priority (NestedKey): The input tensordict key where the target
                priority is written to. Defaults to ``"td_error"``.
            state_action_value (NestedKey): The input tensordict key where the
                state action value is expected. Defaults to ``"state_action_value"``.
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
        sample_log_prob: NestedKey = "sample_log_prob"
        priority: NestedKey = "td_error"
        state_action_value: NestedKey = "state_action_value"
        reward: NestedKey = "reward"
        done: NestedKey = "done"
        terminated: NestedKey = "terminated"

    default_keys = _AcceptedKeys()
    delay_actor: bool = False
    default_value_estimator = ValueEstimators.TD0
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

    actor_network: TensorDictModule
    qvalue_network: TensorDictModule
    actor_network_params: TensorDictParams
    qvalue_network_params: TensorDictParams
    target_actor_network_params: TensorDictParams
    target_qvalue_network_params: TensorDictParams

    def __init__(
        self,
        actor_network: TensorDictModule,
        qvalue_network: TensorDictModule | List[TensorDictModule],
        *,
        num_qvalue_nets: int = 10,
        sub_sample_len: int = 2,
        loss_function: str = "smooth_l1",
        alpha_init: float = 1.0,
        min_alpha: float = 0.1,
        max_alpha: float = 10.0,
        action_spec=None,
        fixed_alpha: bool = False,
        target_entropy: Union[str, Number] = "auto",
        delay_qvalue: bool = True,
        gSDE: bool = False,
        gamma: float = None,
        priority_key: str = None,
        separate_losses: bool = False,
        reduction: str = None,
    ):
        if reduction is None:
            reduction = "mean"
        super().__init__()
        self._in_keys = None
        self._set_deprecated_ctor_keys(priority_key=priority_key)

        self.convert_to_functional(
            actor_network,
            "actor_network",
            create_target_params=self.delay_actor,
        )

        # let's make sure that actor_network has `return_log_prob` set to True
        self.actor_network.return_log_prob = True
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
        self.sub_sample_len = max(1, min(sub_sample_len, num_qvalue_nets - 1))
        self.loss_function = loss_function

        try:
            device = next(self.parameters()).device
        except AttributeError:
            device = torch.device("cpu")

        self.register_buffer("alpha_init", torch.tensor(alpha_init, device=device))
        self.register_buffer(
            "min_log_alpha", torch.tensor(min_alpha, device=device).log()
        )
        self.register_buffer(
            "max_log_alpha", torch.tensor(max_alpha, device=device).log()
        )
        self.fixed_alpha = fixed_alpha
        if fixed_alpha:
            self.register_buffer(
                "log_alpha", torch.tensor(math.log(alpha_init), device=device)
            )
        else:
            self.register_parameter(
                "log_alpha",
                torch.nn.Parameter(
                    torch.tensor(
                        math.log(alpha_init), device=device, requires_grad=True
                    )
                ),
            )

        self._target_entropy = target_entropy
        self._action_spec = action_spec
        self.target_entropy_buffer = None
        self.reduction = reduction
        self.gSDE = gSDE
        if gamma is not None:
            raise TypeError(_GAMMA_LMBDA_DEPREC_ERROR)
        self._make_vmap()

    def _make_vmap(self):
        self._vmap_qvalue_network00 = _vmap_func(
            self.qvalue_network, randomness=self.vmap_randomness
        )
        self._vmap_getdist = _vmap_func(
            self.actor_network, func="get_dist_params", randomness=self.vmap_randomness
        )

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
                if not isinstance(action_spec, Composite):
                    action_spec = Composite({self.tensor_keys.action: action_spec})
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
        self._set_in_keys()

    @property
    def alpha(self):
        with torch.no_grad():
            return self.log_alpha.clamp(self.min_log_alpha, self.max_log_alpha).exp()

    def _set_in_keys(self):
        keys = [
            self.tensor_keys.action,
            self.tensor_keys.sample_log_prob,
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

    @property
    @_cache_values
    def _cached_detach_qvalue_network_params(self):
        return self.qvalue_network_params.detach()

    def _qvalue_params_cat(self, selected_q_params):
        qvalue_params = torch.cat(
            [
                self._cached_detach_qvalue_network_params,
                selected_q_params,
                self.qvalue_network_params,
            ],
            0,
        )
        return qvalue_params

    @dispatch
    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        obs_keys = self.actor_network.in_keys
        tensordict_select = tensordict.select(
            "next", *obs_keys, self.tensor_keys.action, strict=False
        )
        # We need to copy bc select does not copy sub-tds
        tensordict_select = tensordict_select.copy()

        selected_models_idx = torch.randperm(self.num_qvalue_nets)[
            : self.sub_sample_len
        ].sort()[0]
        selected_q_params = self.target_qvalue_network_params[selected_models_idx]

        actor_params = torch.stack(
            [self.actor_network_params, self.target_actor_network_params], 0
        )

        tensordict_actor_grad = tensordict_select.select(
            *obs_keys, strict=False
        )  # to avoid overwriting keys
        next_td_actor = step_mdp(tensordict_select).select(
            *self.actor_network.in_keys, strict=False
        )  # next_observation ->
        tensordict_actor = torch.stack([tensordict_actor_grad, next_td_actor], 0)

        with set_exploration_type(ExplorationType.RANDOM):
            if self.gSDE:
                tensordict_actor.set(
                    "_eps_gSDE",
                    torch.zeros(tensordict_actor.shape, device=tensordict_actor.device),
                )
            # vmap doesn't support sampling, so we take it out from the vmap
            td_params = self._vmap_getdist(
                tensordict_actor,
                actor_params,
            )
            sample_key = self.tensor_keys.action
            sample_key_lp = self.tensor_keys.sample_log_prob
            tensordict_actor_dist = self.actor_network.build_dist_from_params(td_params)
            tensordict_actor.set(sample_key, tensordict_actor_dist.rsample())
            tensordict_actor.set(
                sample_key_lp,
                tensordict_actor_dist.log_prob(tensordict_actor.get(sample_key)),
            )

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
            .expand(self.sub_sample_len, *tensordict_actor[1].batch_size)
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
        tensordict_qval = self._vmap_qvalue_network00(
            tensordict_qval,
            self._qvalue_params_cat(selected_q_params),
        )

        state_action_value = tensordict_qval.get(
            self.tensor_keys.state_action_value
        ).squeeze(-1)
        (
            state_action_value_actor,
            next_state_action_value_qvalue,
            state_action_value_qvalue,
        ) = state_action_value.split(
            [self.num_qvalue_nets, self.sub_sample_len, self.num_qvalue_nets],
            dim=0,
        )
        sample_log_prob = tensordict_actor.get(
            self.tensor_keys.sample_log_prob
        ).squeeze(-1)
        (
            action_log_prob_actor,
            next_action_log_prob_qvalue,
        ) = sample_log_prob.unbind(0)

        loss_actor = -(state_action_value_actor - self.alpha * action_log_prob_actor)

        next_state_value = (
            next_state_action_value_qvalue - self.alpha * next_action_log_prob_qvalue
        )
        next_state_value = next_state_value.min(0)[0]

        tensordict_select.set(
            ("next", self.tensor_keys.value), next_state_value.unsqueeze(-1)
        )
        target_value = self.value_estimator.value_estimate(tensordict_select).squeeze(
            -1
        )

        pred_val = state_action_value_qvalue
        td_error = (pred_val - target_value).pow(2)
        loss_qval = distance_loss(
            pred_val,
            target_value.expand_as(pred_val),
            loss_function=self.loss_function,
        )

        tensordict.set(self.tensor_keys.priority, td_error.detach().max(0)[0])

        loss_alpha = self._loss_alpha(sample_log_prob)
        if not loss_qval.shape == loss_actor.shape:
            raise RuntimeError(
                f"QVal and actor loss have different shape: {loss_qval.shape} and {loss_actor.shape}"
            )
        td_out = TensorDict(
            {
                "loss_actor": loss_actor,
                "loss_qvalue": loss_qval,
                "loss_alpha": loss_alpha,
                "alpha": self.alpha.detach(),
                "entropy": -sample_log_prob.detach().mean(),
                "state_action_value_actor": state_action_value_actor.detach(),
                "action_log_prob_actor": action_log_prob_actor.detach(),
                "next.state_value": next_state_value.detach(),
                "target_value": target_value.detach(),
            },
            [],
        )
        td_out = td_out.named_apply(
            lambda name, value: _reduce(value, reduction=self.reduction)
            if name.startswith("loss_")
            else value,
            batch_size=[],
        )
        return td_out

    def _loss_alpha(self, log_pi: Tensor) -> Tensor:
        if torch.is_grad_enabled() and not log_pi.requires_grad:
            raise RuntimeError(
                "expected log_pi to require gradient for the alpha loss)"
            )
        if self.target_entropy is not None:
            # we can compute this loss even if log_alpha is not a parameter
            alpha_loss = -self._safe_log_alpha.exp() * (
                log_pi.detach() + self.target_entropy
            )
        else:
            # placeholder
            alpha_loss = torch.zeros_like(log_pi)
        return alpha_loss

    @property
    def _safe_log_alpha(self):
        log_alpha = self.log_alpha
        with torch.no_grad():
            log_alpha_clamp = log_alpha.clamp(self.min_log_alpha, self.max_log_alpha)
            log_alpha_det = log_alpha.detach()
        return log_alpha - log_alpha_det + log_alpha_clamp

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
            "value": self.tensor_keys.value,
            "reward": self.tensor_keys.reward,
            "done": self.tensor_keys.done,
            "terminated": self.tensor_keys.terminated,
        }
        self._value_estimator.set_keys(**tensor_keys)
