# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import math
from dataclasses import dataclass
from functools import wraps

import torch
from tensordict import TensorDict, TensorDictBase, TensorDictParams
from tensordict.nn import dispatch, TensorDictModule
from tensordict.utils import NestedKey
from torch import Tensor

from torchrl.data.tensor_specs import Composite
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.modules import ProbabilisticActor
from torchrl.objectives.common import LossModule
from torchrl.objectives.utils import (
    _cache_values,
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


class CrossQLoss(LossModule):
    """TorchRL implementation of the CrossQ loss.

    Presented in "CROSSQ: BATCH NORMALIZATION IN DEEP REINFORCEMENT LEARNING
    FOR GREATER SAMPLE EFFICIENCY AND SIMPLICITY" https://openreview.net/pdf?id=PczQtTsTIX

    This class has three loss functions that will be called sequentially by the `forward` method:
    :meth:`~.qvalue_loss`, :meth:`~.actor_loss` and :meth:`~.alpha_loss`. Alternatively, they can
    be called by the user that order.

    Args:
        actor_network (ProbabilisticActor): stochastic actor
        qvalue_network (TensorDictModule): Q(s, a) parametric model.
            This module typically outputs a ``"state_action_value"`` entry.
            If a single instance of `qvalue_network` is provided, it will be duplicated ``num_qvalue_nets``
            times. If a list of modules is passed, their
            parameters will be stacked unless they share the same identity (in which case
            the original parameter will be expanded).

            .. warning:: When a list of parameters if passed, it will __not__ be compared against the policy parameters
              and all the parameters will be considered as untied.

    Keyword Args:
        num_qvalue_nets (integer, optional): number of Q-Value networks used.
            Defaults to ``2``.
        loss_function (str, optional): loss function to be used with
            the value function loss. Default is `"smooth_l1"`.
        alpha_init (:obj:`float`, optional): initial entropy multiplier.
            Default is 1.0.
        min_alpha (:obj:`float`, optional): min value of alpha.
            Default is None (no minimum value).
        max_alpha (:obj:`float`, optional): max value of alpha.
            Default is None (no maximum value).
        action_spec (TensorSpec, optional): the action tensor spec. If not provided
            and the target entropy is ``"auto"``, it will be retrieved from
            the actor.
        fixed_alpha (bool, optional): if ``True``, alpha will be fixed to its
            initial value. Otherwise, alpha will be optimized to
            match the 'target_entropy' value.
            Default is ``False``.
        target_entropy (:obj:`float` or str, optional): Target entropy for the
            stochastic policy. Default is "auto", where target entropy is
            computed as :obj:`-prod(n_actions)`.
        priority_key (str, optional): [Deprecated, use .set_keys(priority_key=priority_key) instead]
            Tensordict key where to write the
            priority (for prioritized replay buffer usage). Defaults to ``"td_error"``.
        separate_losses (bool, optional): if ``True``, shared parameters between
            policy and critic will only be trained on the policy loss.
            Defaults to ``False``, i.e., gradients are propagated to shared
            parameters for both policy and critic losses.
        reduction (str, optional): Specifies the reduction to apply to the output:
            ``"none"`` | ``"mean"`` | ``"sum"``. ``"none"``: no reduction will be applied,
            ``"mean"``: the sum of the output will be divided by the number of
            elements in the output, ``"sum"``: the output will be summed. Default: ``"mean"``.
        deactivate_vmap (bool, optional): whether to deactivate vmap calls and replace them with a plain for loop.
            Defaults to ``False``.

    Examples:
        >>> import torch
        >>> from torch import nn
        >>> from torchrl.data import Bounded
        >>> from torchrl.modules.distributions import NormalParamExtractor, TanhNormal
        >>> from torchrl.modules.tensordict_module.actors import ProbabilisticActor, ValueOperator
        >>> from torchrl.modules.tensordict_module.common import SafeModule
        >>> from torchrl.objectives.crossq import CrossQLoss
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
        >>> loss = CrossQLoss(actor, qvalue)
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
                loss_qvalue: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False)},
            batch_size=torch.Size([]),
            device=None,
            is_shared=False)

    This class is compatible with non-tensordict based modules too and can be
    used without recurring to any tensordict-related primitive. In this case,
    the expected keyword arguments are:
    ``["action", "next_reward", "next_done", "next_terminated"]`` + in_keys of the actor and qvalue network.
    The return value is a tuple of tensors in the following order:
    ``["loss_actor", "loss_qvalue", "loss_alpha", "alpha", "entropy"]``

    Examples:
        >>> import torch
        >>> from torch import nn
        >>> from torchrl.data import Bounded
        >>> from torchrl.modules.distributions import NormalParamExtractor, TanhNormal
        >>> from torchrl.modules.tensordict_module.actors import ProbabilisticActor, ValueOperator
        >>> from torchrl.modules.tensordict_module.common import SafeModule
        >>> from torchrl.objectives import CrossQLoss
        >>> _ = torch.manual_seed(42)
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
        >>> loss = CrossQLoss(actor, qvalue)
        >>> batch = [2, ]
        >>> action = spec.rand(batch)
        >>> loss_actor, loss_qvalue, _, _, _ = loss(
        ...     observation=torch.randn(*batch, n_obs),
        ...     action=action,
        ...     next_done=torch.zeros(*batch, 1, dtype=torch.bool),
        ...     next_terminated=torch.zeros(*batch, 1, dtype=torch.bool),
        ...     next_observation=torch.zeros(*batch, n_obs),
        ...     next_reward=torch.randn(*batch, 1))
        >>> loss_actor.backward()

    The output keys can also be filtered using the :meth:`CrossQLoss.select_out_keys`
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
            state_action_value (NestedKey): The input tensordict key where the
                state action value is expected.  Defaults to ``"state_action_value"``.
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
            log_prob (NestedKey): The input tensordict key where the log probability is expected.
                Defaults to ``"_log_prob"``.
        """

        action: NestedKey = "action"
        state_action_value: NestedKey = "state_action_value"
        priority: NestedKey = "td_error"
        reward: NestedKey = "reward"
        done: NestedKey = "done"
        terminated: NestedKey = "terminated"
        log_prob: NestedKey = "_log_prob"

    tensor_keys: _AcceptedKeys
    default_keys = _AcceptedKeys
    default_value_estimator = ValueEstimators.TD0

    actor_network: ProbabilisticActor
    actor_network_params: TensorDictParams
    qvalue_network: TensorDictModule
    qvalue_network_params: TensorDictParams
    target_actor_network_params: TensorDictParams
    target_qvalue_network_params: TensorDictParams

    def __init__(
        self,
        actor_network: ProbabilisticActor,
        qvalue_network: TensorDictModule | list[TensorDictModule],
        *,
        num_qvalue_nets: int = 2,
        loss_function: str = "smooth_l1",
        alpha_init: float = 1.0,
        min_alpha: float | None = None,
        max_alpha: float | None = None,
        action_spec=None,
        fixed_alpha: bool = False,
        target_entropy: str | float = "auto",
        priority_key: str = None,
        separate_losses: bool = False,
        reduction: str = None,
        deactivate_vmap: bool = False,
    ) -> None:
        self._in_keys = None
        self._out_keys = None
        if reduction is None:
            reduction = "mean"
        super().__init__()
        self._set_deprecated_ctor_keys(priority_key=priority_key)

        self.deactivate_vmap = deactivate_vmap

        # Actor
        self.convert_to_functional(
            actor_network,
            "actor_network",
            create_target_params=False,
        )
        if separate_losses:
            # we want to make sure there are no duplicates in the params: the
            # params of critic must be refs to actor if they're shared
            policy_params = list(actor_network.parameters())
        else:
            policy_params = None
            q_value_policy_params = None

        # Q value
        self.num_qvalue_nets = num_qvalue_nets

        q_value_policy_params = policy_params
        self.convert_to_functional(
            qvalue_network,
            "qvalue_network",
            num_qvalue_nets,
            create_target_params=False,
            compare_against=q_value_policy_params,
        )

        self.loss_function = loss_function
        try:
            device = next(self.parameters()).device
        except AttributeError:
            device = getattr(torch, "get_default_device", lambda: torch.device("cpu"))()
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
        self._make_vmap()
        self.reduction = reduction
        # init target entropy
        self.maybe_init_target_entropy()

    def _make_vmap(self):
        self._vmap_qnetworkN0 = _vmap_func(
            self.qvalue_network,
            (None, 0),
            randomness=self.vmap_randomness,
            pseudo_vmap=self.deactivate_vmap,
        )

    @property
    def target_entropy_buffer(self):
        """The target entropy.

        This value can be controlled via the `target_entropy` kwarg in the constructor.
        """
        return self.target_entropy

    def maybe_init_target_entropy(self, fault_tolerant=True):
        """Initialize the target entropy.

        Args:
            fault_tolerant (bool, optional): if ``True``, returns None if the target entropy
                cannot be determined. Raises an exception otherwise. Defaults to ``True``.

        """
        if "_target_entropy" in self._buffers:
            return
        target_entropy = self._target_entropy
        if target_entropy == "auto":
            device = next(self.parameters()).device
            action_spec = self.get_action_spec()
            if action_spec is None:
                if fault_tolerant:
                    return
                raise RuntimeError(
                    "Cannot infer the dimensionality of the action. Consider providing "
                    "the target entropy explicitly or provide the spec of the "
                    "action tensor in the actor network."
                )
            if not isinstance(action_spec, Composite):
                action_spec = Composite({self.tensor_keys.action: action_spec})
            elif fault_tolerant and self.tensor_keys.action not in action_spec:
                return
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

    def get_action_spec(self):
        action_spec = self._action_spec
        actor_network = self.actor_network
        action_spec = (
            action_spec
            if action_spec is not None
            else getattr(actor_network, "spec", None)
        )
        return action_spec

    @property
    def target_entropy(self):
        target_entropy = self._buffers.get("_target_entropy")
        if target_entropy is not None:
            return target_entropy
        return self.maybe_init_target_entropy(fault_tolerant=False)

    def set_keys(self, **kwargs) -> None:
        out = super().set_keys(**kwargs)
        self.maybe_init_target_entropy()
        return out

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
            self._out_keys = keys
        return self._out_keys

    @out_keys.setter
    def out_keys(self, values):
        self._out_keys = values

    @dispatch
    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        """The forward method.

        Computes successively the :meth:`~.qvalue_loss`, :meth:`~.actor_loss` and :meth:`~.alpha_loss`, and returns
        a tensordict with these values along with the `"alpha"` value and the `"entropy"` value (detached).
        To see what keys are expected in the input tensordict and what keys are expected as output, check the
        class's `"in_keys"` and `"out_keys"` attributes.
        """
        loss_qvalue, value_metadata = self.qvalue_loss(tensordict)
        loss_actor, metadata_actor = self.actor_loss(tensordict)
        loss_alpha = self.alpha_loss(log_prob=metadata_actor["log_prob"])
        tensordict.set(self.tensor_keys.priority, value_metadata["td_error"])
        if loss_actor.shape != loss_qvalue.shape:
            raise RuntimeError(
                f"Losses shape mismatch: {loss_actor.shape} and {loss_qvalue.shape}"
            )
        entropy = -metadata_actor["log_prob"]
        out = {
            "loss_actor": loss_actor,
            "loss_qvalue": loss_qvalue,
            "loss_alpha": loss_alpha,
            "alpha": self._alpha,
            "entropy": entropy.detach().mean(),
            **metadata_actor,
            **value_metadata,
        }
        td_out = TensorDict(out)
        self._clear_weakrefs(
            tensordict,
            td_out,
            "actor_network_params",
            "qvalue_network_params",
            "target_actor_network_params",
            "target_qvalue_network_params",
        )
        return td_out

    @property
    @_cache_values
    def _cached_detached_qvalue_params(self):
        return self.qvalue_network_params.detach()

    def actor_loss(
        self, tensordict: TensorDictBase
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """Compute the actor loss.

        The actor loss should be computed after the :meth:`~.qvalue_loss` and before the `~.alpha_loss` which
        requires the `log_prob` field of the `metadata` returned by this method.

        Args:
            tensordict (TensorDictBase): the input data for the loss. Check the class's `in_keys` to see what fields
                are required for this to be computed.

        Returns: a differentiable tensor with the alpha loss along with a metadata dictionary containing the detached `"log_prob"` of the sampled action.
        """
        tensordict = tensordict.copy()
        with set_exploration_type(
            ExplorationType.RANDOM
        ), self.actor_network_params.to_module(self.actor_network):
            dist = self.actor_network.get_dist(tensordict)
            a_reparm = dist.rsample()
        log_prob = dist.log_prob(a_reparm)

        td_q = tensordict.select(*self.qvalue_network.in_keys, strict=False)
        self.qvalue_network.eval()
        td_q.set(self.tensor_keys.action, a_reparm)
        td_q = self._vmap_qnetworkN0(
            td_q,
            self._cached_detached_qvalue_params,
        )

        min_q = td_q.get(self.tensor_keys.state_action_value).min(0)[0].squeeze(-1)
        self.qvalue_network.train()

        if log_prob.shape != min_q.shape:
            raise RuntimeError(
                f"Losses shape mismatch: {log_prob.shape} and {min_q.shape}"
            )
        actor_loss = self._alpha * log_prob - min_q
        return _reduce(actor_loss, reduction=self.reduction), {
            "log_prob": log_prob.detach()
        }

    def qvalue_loss(
        self, tensordict: TensorDictBase
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """Compute the q-value loss.

        The q-value loss should be computed before the :meth:`~.actor_loss`.

        Args:
            tensordict (TensorDictBase): the input data for the loss. Check the class's `in_keys` to see what fields
                are required for this to be computed.

        Returns: a differentiable tensor with the qvalue loss along with a metadata dictionary containing
            the detached `"td_error"` to be used for prioritized sampling.
        """
        tensordict = tensordict.copy()
        # # compute next action
        with torch.no_grad():
            with set_exploration_type(
                ExplorationType.RANDOM
            ), self.actor_network_params.to_module(self.actor_network):
                next_tensordict = tensordict.get("next").clone(False)
                next_dist = self.actor_network.get_dist(next_tensordict)
                next_action = next_dist.sample()
                next_tensordict.set(self.tensor_keys.action, next_action)
                next_sample_log_prob = next_dist.log_prob(next_action)

        combined = torch.cat(
            [
                tensordict.select(*self.qvalue_network.in_keys, strict=False),
                next_tensordict.select(*self.qvalue_network.in_keys, strict=False),
            ]
        )
        pred_qs = self._vmap_qnetworkN0(combined, self.qvalue_network_params).get(
            self.tensor_keys.state_action_value
        )
        (current_state_action_value, next_state_action_value) = pred_qs.split(
            tensordict.batch_size[0], dim=1
        )

        # compute target value
        if (
            next_state_action_value.shape[-len(next_sample_log_prob.shape) :]
            != next_sample_log_prob.shape
        ):
            next_sample_log_prob = next_sample_log_prob.unsqueeze(-1)
        next_state_action_value = next_state_action_value.min(0)[0]
        next_state_action_value = (
            next_state_action_value - self._alpha * next_sample_log_prob
        ).detach()

        target_value = self.value_estimator.value_estimate(
            tensordict, next_value=next_state_action_value
        ).squeeze(-1)

        # get current q-values
        pred_val = current_state_action_value.squeeze(-1)

        # compute loss
        td_error = abs(pred_val - target_value)
        loss_qval = distance_loss(
            pred_val,
            target_value.expand_as(pred_val),
            loss_function=self.loss_function,
        ).sum(0)
        metadata = {"td_error": td_error.detach().max(0)[0]}
        return _reduce(loss_qval, reduction=self.reduction), metadata

    def alpha_loss(self, log_prob: Tensor) -> Tensor:
        """Compute the entropy loss.

        The entropy loss should be computed last.

        Args:
            log_prob (torch.Tensor): a log-probability as computed by the :meth:`~.actor_loss` and returned in the `metadata`.

        Returns: a differentiable tensor with the entropy loss.
        """
        if self.target_entropy is not None:
            # we can compute this loss even if log_alpha is not a parameter
            alpha_loss = -self.log_alpha * (log_prob + self.target_entropy)
        else:
            # placeholder
            alpha_loss = torch.zeros_like(log_prob)
        return _reduce(alpha_loss, reduction=self.reduction)

    @property
    def _alpha(self):
        if self.min_log_alpha is not None or self.max_log_alpha is not None:
            self.log_alpha.data.clamp_(self.min_log_alpha, self.max_log_alpha)
        with torch.no_grad():
            alpha = self.log_alpha.exp()
        return alpha
