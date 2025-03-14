# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import contextlib
from copy import deepcopy
from dataclasses import dataclass

import torch
from tensordict import (
    is_tensor_collection,
    TensorDict,
    TensorDictBase,
    TensorDictParams,
)
from tensordict.nn import (
    composite_lp_aggregate,
    CompositeDistribution,
    dispatch,
    ProbabilisticTensorDictSequential,
    set_composite_lp_aggregate,
    TensorDictModule,
)
from tensordict.utils import NestedKey
from torch import distributions as d

from torchrl.modules.distributions import HAS_ENTROPY
from torchrl.objectives.common import LossModule
from torchrl.objectives.utils import (
    _cache_values,
    _clip_value_loss,
    _GAMMA_LMBDA_DEPREC_ERROR,
    _get_default_device,
    _reduce,
    default_value_kwargs,
    distance_loss,
    ValueEstimators,
)
from torchrl.objectives.value import (
    GAE,
    TD0Estimator,
    TD1Estimator,
    TDLambdaEstimator,
    VTrace,
)


class A2CLoss(LossModule):
    """TorchRL implementation of the A2C loss.

    A2C (Advantage Actor Critic) is a model-free, online RL algorithm that uses parallel rollouts of n steps to
    update the policy, relying on the REINFORCE estimator to compute the gradient. It also adds an entropy term to the
    objective function to improve exploration.

    For more details regarding A2C, refer to: "Asynchronous Methods for Deep Reinforcment Learning",
    https://arxiv.org/abs/1602.01783v2

    Args:
        actor_network (ProbabilisticTensorDictSequential): policy operator.
        critic_network (ValueOperator): value operator.
        entropy_bonus (bool): if ``True``, an entropy bonus will be added to the
            loss to favour exploratory policies.
        samples_mc_entropy (int): if the distribution retrieved from the policy
            operator does not have a closed form
            formula for the entropy, a Monte-Carlo estimate will be used.
            ``samples_mc_entropy`` will control how many
            samples will be used to compute this estimate.
            Defaults to ``1``.
        entropy_coef (:obj:`float`): the weight of the entropy loss. Defaults to `0.01``.
        critic_coef (:obj:`float`): the weight of the critic loss. Defaults to ``1.0``. If ``None``, the critic
            loss won't be included and the in-keys will miss the critic inputs.
        loss_critic_type (str): loss function for the value discrepancy.
            Can be one of "l1", "l2" or "smooth_l1". Defaults to ``"smooth_l1"``.
        separate_losses (bool, optional): if ``True``, shared parameters between
            policy and critic will only be trained on the policy loss.
            Defaults to ``False``, i.e., gradients are propagated to shared
            parameters for both policy and critic losses.
        advantage_key (str): [Deprecated, use set_keys(advantage_key=advantage_key) instead]
            The input tensordict key where the advantage is expected to be written.  default: "advantage"
        value_target_key (str): [Deprecated, use set_keys() instead] the input
            tensordict key where the target state value is expected to be written. Defaults to ``"value_target"``.
        functional (bool, optional): whether modules should be functionalized.
            Functionalizing permits features like meta-RL, but makes it
            impossible to use distributed models (DDP, FSDP, ...) and comes
            with a little cost. Defaults to ``True``.
        reduction (str, optional): Specifies the reduction to apply to the output:
            ``"none"`` | ``"mean"`` | ``"sum"``. ``"none"``: no reduction will be applied,
            ``"mean"``: the sum of the output will be divided by the number of
            elements in the output, ``"sum"``: the output will be summed. Default: ``"mean"``.
        clip_value (:obj:`float`, optional): If provided, it will be used to compute a clipped version of the value
            prediction with respect to the input value estimate and use it to calculate the value loss.
            The purpose of clipping is to limit the impact of extreme value predictions, helping stabilize training
            and preventing large updates. However, it will have no impact if the value estimate was done by the current
            version of the value estimator. Defaults to ``None``.

    .. note:
      The advantage (typically GAE) can be computed by the loss function or
      in the training loop. The latter option is usually preferred, but this is
      up to the user to choose which option is to be preferred.
      If the advantage key (``"advantage`` by default) is not present in the
      input tensordict, the advantage will be computed by the :meth:`~.forward`
      method.
      A custom advantage module can be built using :meth:`~.make_value_estimator`.
      The default is :class:`~torchrl.objectives.value.GAE` with hyperparameters
      dictated by :func:`~torchrl.objectives.utils.default_value_kwargs`.

    Examples:
        >>> import torch
        >>> from torch import nn
        >>> from torchrl.data import Bounded
        >>> from torchrl.modules.distributions import NormalParamExtractor, TanhNormal
        >>> from torchrl.modules.tensordict_module.actors import ProbabilisticActor, ValueOperator
        >>> from torchrl.modules.tensordict_module.common import SafeModule
        >>> from torchrl.objectives.a2c import A2CLoss
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
        >>> module = nn.Linear(n_obs, 1)
        >>> value = ValueOperator(
        ...     module=module,
        ...     in_keys=["observation"])
        >>> loss = A2CLoss(actor, value, loss_critic_type="l2")
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
                entropy: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False),
                loss_critic: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False),
                loss_entropy: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False),
                loss_objective: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False)},
            batch_size=torch.Size([]),
            device=None,
            is_shared=False)

    This class is compatible with non-tensordict based modules too and can be
    used without recurring to any tensordict-related primitive. In this case,
    the expected keyword arguments are:
    ``["action", "next_reward", "next_done", "next_terminated"]`` + in_keys of the actor and critic.
    The return value is a tuple of tensors in the following order:
    ``["loss_objective"]`` + ``["loss_critic"]`` if critic_coef is not None + ``["entropy", "loss_entropy"]`` if entropy_bonus is True and critic_coef is not None

    Examples:
        >>> import torch
        >>> from torch import nn
        >>> from torchrl.data import Bounded
        >>> from torchrl.modules.distributions import NormalParamExtractor, TanhNormal
        >>> from torchrl.modules.tensordict_module.actors import ProbabilisticActor, ValueOperator
        >>> from torchrl.modules.tensordict_module.common import SafeModule
        >>> from torchrl.objectives.a2c import A2CLoss
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
        >>> module = nn.Linear(n_obs, 1)
        >>> value = ValueOperator(
        ...     module=module,
        ...     in_keys=["observation"])
        >>> loss = A2CLoss(actor, value, loss_critic_type="l2")
        >>> batch = [2, ]
        >>> loss_obj, loss_critic, entropy, loss_entropy = loss(
        ...     observation = torch.randn(*batch, n_obs),
        ...     action = spec.rand(batch),
        ...     next_done = torch.zeros(*batch, 1, dtype=torch.bool),
        ...     next_terminated = torch.zeros(*batch, 1, dtype=torch.bool),
        ...     next_reward = torch.randn(*batch, 1),
        ...     next_observation = torch.randn(*batch, n_obs))
        >>> loss_obj.backward()

    The output keys can also be filtered using the :meth:`SACLoss.select_out_keys`
    method.

    Examples:
        >>> loss.select_out_keys('loss_objective', 'loss_critic')
        >>> loss_obj, loss_critic = loss(
        ...     observation = torch.randn(*batch, n_obs),
        ...     action = spec.rand(batch),
        ...     next_done = torch.zeros(*batch, 1, dtype=torch.bool),
        ...     next_terminated = torch.zeros(*batch, 1, dtype=torch.bool),
        ...     next_reward = torch.randn(*batch, 1),
        ...     next_observation = torch.randn(*batch, n_obs))
        >>> loss_obj.backward()

    .. note::
      There is an exception regarding compatibility with non-tensordict-based modules.
      If the actor network is probabilistic and uses a :class:`~tensordict.nn.distributions.CompositeDistribution`,
      this class must be used with tensordicts and cannot function as a tensordict-independent module.
      This is because composite action spaces inherently rely on the structured representation of data provided by
      tensordicts to handle their actions.
    """

    @dataclass
    class _AcceptedKeys:
        """Maintains default values for all configurable tensordict keys.

        This class defines which tensordict keys can be set using '.set_keys(key_name=key_value)' and their
        default values.

        Attributes:
            advantage (NestedKey): The input tensordict key where the advantage is expected.
                Will be used for the underlying value estimator. Defaults to ``"advantage"``.
            value_target (NestedKey): The input tensordict key where the target state value is expected.
                Will be used for the underlying value estimator Defaults to ``"value_target"``.
            value (NestedKey): The input tensordict key where the state value is expected.
                Will be used for the underlying value estimator. Defaults to ``"state_value"``.
            action (NestedKey): The input tensordict key where the action is expected.
                Defaults to ``"action"``.
            reward (NestedKey): The input tensordict key where the reward is expected.
                Will be used for the underlying value estimator. Defaults to ``"reward"``.
            done (NestedKey): The key in the input TensorDict that indicates
                whether a trajectory is done. Will be used for the underlying value estimator.
                Defaults to ``"done"``.
            terminated (NestedKey): The key in the input TensorDict that indicates
                whether a trajectory is terminated. Will be used for the underlying value estimator.
                Defaults to ``"terminated"``.
        """

        advantage: NestedKey = "advantage"
        value_target: NestedKey = "value_target"
        value: NestedKey = "state_value"
        action: NestedKey = "action"
        reward: NestedKey = "reward"
        done: NestedKey = "done"
        terminated: NestedKey = "terminated"
        sample_log_prob: NestedKey | None = None

        def __post_init__(self):
            if self.sample_log_prob is None:
                if composite_lp_aggregate(nowarn=True):
                    self.sample_log_prob = "sample_log_prob"
                else:
                    self.sample_log_prob = "action_log_prob"

    default_keys = _AcceptedKeys
    tensor_keys: _AcceptedKeys
    default_value_estimator: ValueEstimators = ValueEstimators.GAE

    actor_network: TensorDictModule
    critic_network: TensorDictModule
    actor_network_params: TensorDictParams | None
    critic_network_params: TensorDictParams | None
    target_actor_network_params: TensorDictParams | None
    target_critic_network_params: TensorDictParams | None

    def __init__(
        self,
        actor_network: ProbabilisticTensorDictSequential = None,
        critic_network: TensorDictModule = None,
        *,
        entropy_bonus: bool = True,
        samples_mc_entropy: int = 1,
        entropy_coef: float = 0.01,
        critic_coef: float = 1.0,
        loss_critic_type: str = "smooth_l1",
        gamma: float = None,
        separate_losses: bool = False,
        advantage_key: str = None,
        value_target_key: str = None,
        functional: bool = True,
        actor: ProbabilisticTensorDictSequential = None,
        critic: ProbabilisticTensorDictSequential = None,
        reduction: str = None,
        clip_value: float | None = None,
    ):
        if actor is not None:
            actor_network = actor
            del actor
        if critic is not None:
            critic_network = critic
            del critic
        if actor_network is None or critic_network is None:
            raise TypeError(
                "Missing positional arguments actor_network or critic_network."
            )
        if reduction is None:
            reduction = "mean"

        self._functional = functional
        self._out_keys = None
        super().__init__()
        self._set_deprecated_ctor_keys(
            advantage=advantage_key, value_target=value_target_key
        )

        if functional:
            self.convert_to_functional(
                actor_network,
                "actor_network",
            )
        else:
            self.actor_network = actor_network

        if separate_losses:
            # we want to make sure there are no duplicates in the params: the
            # params of critic must be refs to actor if they're shared
            policy_params = list(actor_network.parameters())
        else:
            policy_params = None
        if functional:
            self.convert_to_functional(
                critic_network, "critic_network", compare_against=policy_params
            )
        else:
            self.critic_network = critic_network
            self.target_critic_network_params = None

        self.samples_mc_entropy = samples_mc_entropy
        self.entropy_bonus = entropy_bonus and entropy_coef
        self.reduction = reduction

        device = _get_default_device(self)

        self.register_buffer(
            "entropy_coef", torch.as_tensor(entropy_coef, device=device)
        )
        if critic_coef is not None:
            self.register_buffer(
                "critic_coef", torch.as_tensor(critic_coef, device=device)
            )
        else:
            self.critic_coef = None

        if gamma is not None:
            raise TypeError(_GAMMA_LMBDA_DEPREC_ERROR)
        self.loss_critic_type = loss_critic_type

        if clip_value is not None:
            if isinstance(clip_value, float):
                clip_value = torch.tensor(clip_value)
            elif isinstance(clip_value, torch.Tensor):
                if clip_value.numel() != 1:
                    raise ValueError(
                        f"clip_value must be a float or a scalar tensor, got {clip_value}."
                    )
            else:
                raise ValueError(
                    f"clip_value must be a float or a scalar tensor, got {clip_value}."
                )
            self.register_buffer(
                "clip_value", torch.as_tensor(clip_value, device=device)
            )
        else:
            self.clip_value = None

        log_prob_keys = self.actor_network.log_prob_keys
        action_keys = self.actor_network.dist_sample_keys
        if len(log_prob_keys) > 1:
            self.set_keys(sample_log_prob=log_prob_keys, action=action_keys)
        else:
            self.set_keys(sample_log_prob=log_prob_keys[0], action=action_keys[0])

    @property
    def functional(self):
        return self._functional

    @property
    def in_keys(self):
        keys = [
            self.tensor_keys.action,
            ("next", self.tensor_keys.reward),
            ("next", self.tensor_keys.done),
            ("next", self.tensor_keys.terminated),
            *self.actor_network.in_keys,
            *[("next", key) for key in self.actor_network.in_keys],
        ]
        if self.critic_coef is not None:
            keys.extend(self.critic_network.in_keys)
        return list(set(keys))

    @property
    def out_keys(self):
        if self._out_keys is None:
            outs = ["loss_objective"]
            if self.critic_coef is not None:
                outs.append("loss_critic")
            if self.entropy_bonus:
                outs.append("entropy")
                outs.append("loss_entropy")
            self._out_keys = outs
        return self._out_keys

    @out_keys.setter
    def out_keys(self, value):
        self._out_keys = value

    def _forward_value_estimator_keys(self, **kwargs) -> None:
        if self._value_estimator is not None:
            self._value_estimator.set_keys(
                advantage=self.tensor_keys.advantage,
                value_target=self.tensor_keys.value_target,
                value=self.tensor_keys.value,
                reward=self.tensor_keys.reward,
                done=self.tensor_keys.done,
                terminated=self.tensor_keys.terminated,
            )

    def reset(self) -> None:
        pass

    @set_composite_lp_aggregate(False)
    def get_entropy_bonus(self, dist: d.Distribution) -> torch.Tensor:
        if HAS_ENTROPY.get(type(dist), False):
            entropy = dist.entropy()
        else:
            x = dist.rsample((self.samples_mc_entropy,))
            log_prob = dist.log_prob(x)
            if is_tensor_collection(log_prob):
                log_prob = sum(log_prob.sum(dim="feature").values(True, True))
            entropy = -log_prob.mean(0)
        return entropy.unsqueeze(-1)

    @set_composite_lp_aggregate(False)
    def _log_probs(
        self, tensordict: TensorDictBase
    ) -> tuple[torch.Tensor, d.Distribution]:
        # current log_prob of actions
        tensordict_clone = tensordict.select(
            *self.actor_network.in_keys, strict=False
        ).copy()
        with self.actor_network_params.to_module(
            self.actor_network
        ) if self.functional else contextlib.nullcontext():
            dist = self.actor_network.get_dist(tensordict_clone)
        if isinstance(dist, CompositeDistribution):
            action_keys = self.tensor_keys.action
            action = tensordict.select(
                *((action_keys,) if isinstance(action_keys, NestedKey) else action_keys)
            )
        else:
            action = tensordict.get(self.tensor_keys.action)

        if action.requires_grad:
            raise RuntimeError(
                f"tensordict stored {self.tensor_keys.action} requires grad."
            )
        log_prob = dist.log_prob(action)
        if not isinstance(action, torch.Tensor):
            log_prob = sum(
                dist.log_prob(tensordict).sum(dim="feature").values(True, True)
            )
        log_prob = log_prob.unsqueeze(-1)
        return log_prob, dist

    def loss_critic(self, tensordict: TensorDictBase) -> tuple[torch.Tensor, float]:
        """Returns the loss value of the critic, multiplied by ``critic_coef`` if it is not ``None``.

        Returns the loss and the clip-fraction.

        """
        if self.clip_value:
            old_state_value = tensordict.get(
                self.tensor_keys.value, None
            )  # TODO: None soon to be removed
            if old_state_value is None:
                raise KeyError(
                    f"clip_value is set to {self.clip_value}, but "
                    f"the key {self.tensor_keys.value} was not found in the input tensordict. "
                    f"Make sure that the value_key passed to A2C exists in the input tensordict."
                )
            old_state_value = old_state_value.clone()

        # TODO: if the advantage is gathered by forward, this introduces an
        #  overhead that we could easily reduce.
        target_return = tensordict.get(
            self.tensor_keys.value_target, None
        )  # TODO: None soon to be removed
        if target_return is None:
            raise KeyError(
                f"the key {self.tensor_keys.value_target} was not found in the input tensordict. "
                f"Make sure you provided the right key and the value_target (i.e. the target "
                f"return) has been retrieved accordingly. Advantage classes such as GAE, "
                f"TDLambdaEstimate and TDEstimate all return a 'value_target' entry that "
                f"can be used for the value loss."
            )
        tensordict_select = tensordict.select(
            *self.critic_network.in_keys, strict=False
        )
        with self.critic_network_params.to_module(
            self.critic_network
        ) if self.functional else contextlib.nullcontext():
            state_value = self.critic_network(
                tensordict_select,
            ).get(self.tensor_keys.value)
        loss_value = distance_loss(
            target_return,
            state_value,
            loss_function=self.loss_critic_type,
        )
        clip_fraction = None
        if self.clip_value:
            loss_value, clip_fraction = _clip_value_loss(
                old_state_value,
                state_value,
                self.clip_value,
                target_return,
                loss_value,
                self.loss_critic_type,
            )
        self._clear_weakrefs(
            tensordict,
            "actor_network_params",
            "critic_network_params",
            "target_actor_network_params",
            "target_critic_network_params",
        )
        if self.critic_coef is not None:
            return self.critic_coef * loss_value, clip_fraction
        return loss_value, clip_fraction

    @property
    @_cache_values
    def _cached_detach_critic_network_params(self):
        if not self.functional:
            return None
        return self.critic_network_params.detach()

    @dispatch()
    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        tensordict = tensordict.clone(False)
        advantage = tensordict.get(self.tensor_keys.advantage, None)
        if advantage is None:
            self.value_estimator(
                tensordict,
                params=self._cached_detach_critic_network_params,
                target_params=self.target_critic_network_params,
            )
            advantage = tensordict.get(self.tensor_keys.advantage)
        log_probs, dist = self._log_probs(tensordict)
        loss = -(log_probs * advantage)
        td_out = TensorDict({"loss_objective": loss}, batch_size=[])
        if self.entropy_bonus:
            entropy = self.get_entropy_bonus(dist)
            td_out.set("entropy", entropy.detach().mean())  # for logging
            td_out.set("loss_entropy", -self.entropy_coef * entropy)
        if self.critic_coef is not None:
            loss_critic, value_clip_fraction = self.loss_critic(tensordict)
            td_out.set("loss_critic", loss_critic)
            if value_clip_fraction is not None:
                td_out.set("value_clip_fraction", value_clip_fraction)
        td_out = td_out.named_apply(
            lambda name, value: _reduce(value, reduction=self.reduction).squeeze(-1)
            if name.startswith("loss_")
            else value,
        )
        self._clear_weakrefs(
            tensordict,
            td_out,
            "actor_network_params",
            "critic_network_params",
            "target_actor_network_params",
            "target_critic_network_params",
        )
        return td_out

    def make_value_estimator(self, value_type: ValueEstimators = None, **hyperparams):
        if value_type is None:
            value_type = self.default_value_estimator
        self.value_type = value_type
        hp = dict(default_value_kwargs(value_type))
        hp.update(hyperparams)

        device = _get_default_device(self)
        hp["device"] = device

        if hasattr(self, "gamma"):
            hp["gamma"] = self.gamma
        if value_type == ValueEstimators.TD1:
            self._value_estimator = TD1Estimator(
                value_network=self.critic_network, **hp
            )
        elif value_type == ValueEstimators.TD0:
            self._value_estimator = TD0Estimator(
                value_network=self.critic_network, **hp
            )
        elif value_type == ValueEstimators.GAE:
            self._value_estimator = GAE(value_network=self.critic_network, **hp)
        elif value_type == ValueEstimators.TDLambda:
            self._value_estimator = TDLambdaEstimator(
                value_network=self.critic_network, **hp
            )
        elif value_type == ValueEstimators.VTrace:
            # VTrace currently does not support functional call on the actor
            if self.functional:
                actor_with_params = deepcopy(self.actor_network)
                self.actor_network_params.to_module(actor_with_params)
            else:
                actor_with_params = self.actor_network
            self._value_estimator = VTrace(
                value_network=self.critic_network, actor_network=actor_with_params, **hp
            )
        else:
            raise NotImplementedError(f"Unknown value type {value_type}")

        tensor_keys = {
            "advantage": self.tensor_keys.advantage,
            "value": self.tensor_keys.value,
            "value_target": self.tensor_keys.value_target,
            "reward": self.tensor_keys.reward,
            "done": self.tensor_keys.done,
            "terminated": self.tensor_keys.terminated,
            "sample_log_prob": self.tensor_keys.sample_log_prob,
        }
        self._value_estimator.set_keys(**tensor_keys)
