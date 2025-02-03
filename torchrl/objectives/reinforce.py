# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import contextlib
from copy import deepcopy
from dataclasses import dataclass

import torch
from tensordict import TensorDict, TensorDictBase, TensorDictParams

from tensordict.nn import (
    composite_lp_aggregate,
    dispatch,
    ProbabilisticTensorDictSequential,
    TensorDictModule,
)
from tensordict.utils import NestedKey
from torchrl.objectives.common import LossModule

from torchrl.objectives.utils import (
    _clip_value_loss,
    _GAMMA_LMBDA_DEPREC_ERROR,
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


class ReinforceLoss(LossModule):
    """Reinforce loss module.

    Presented in "Simple statistical gradient-following sota-implementations for connectionist reinforcement learning", Williams, 1992
    https://doi.org/10.1007/BF00992696


    Args:
        actor_network (ProbabilisticTensorDictSequential): policy operator.
        critic_network (ValueOperator): value operator.

    Keyword Args:
        delay_value (bool, optional): if ``True``, a target network is needed
            for the critic. Defaults to ``False``. Incompatible with ``functional=False``.
        loss_critic_type (str): loss function for the value discrepancy.
            Can be one of "l1", "l2" or "smooth_l1". Defaults to ``"smooth_l1"``.
        advantage_key (str): [Deprecated, use .set_keys(advantage_key=advantage_key) instead]
            The input tensordict key where the advantage is expected to be written.
            Defaults to ``"advantage"``.
        value_target_key (str): [Deprecated, use .set_keys(value_target_key=value_target_key) instead]
            The input tensordict key where the target state
            value is expected to be written. Defaults to ``"value_target"``.
        separate_losses (bool, optional): if ``True``, shared parameters between
            policy and critic will only be trained on the policy loss.
            Defaults to ``False``, i.e., gradients are propagated to shared
            parameters for both policy and critic losses.
        functional (bool, optional): whether modules should be functionalized.
            Functionalizing permits features like meta-RL, but makes it
            impossible to use distributed models (DDP, FSDP, ...) and comes
            with a little cost. Defaults to ``True``.
        reduction (str, optional): Specifies the reduction to apply to the output:
            ``"none"`` | ``"mean"`` | ``"sum"``. ``"none"``: no reduction will be applied,
            ``"mean"``: the sum of the output will be divided by the number of
            elements in the output, ``"sum"``: the output will be summed. Default: ``"mean"``.
        clip_value (:obj:`float`, optional): If provided, it will be used to compute a clipped version of the value
            prediction with respect to the input tensordict value estimate and use it to calculate the value loss.
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

        >>> reinforce_loss = ReinforceLoss(actor, critic)
        >>> advantage = GAE(critic)
        >>> data = next(datacollector)
        >>> losses = reinforce_loss(data)
        >>> # equivalent
        >>> advantage(data)
        >>> losses = reinforce_loss(data)

      A custom advantage module can be built using :meth:`~.make_value_estimator`.
      The default is :class:`~torchrl.objectives.value.GAE` with hyperparameters
      dictated by :func:`~torchrl.objectives.utils.default_value_kwargs`.

        >>> reinforce_loss = ReinforceLoss(actor, critic)
        >>> reinforce_loss.make_value_estimator(ValueEstimators.TDLambda)
        >>> data = next(datacollector)
        >>> losses = reinforce_loss(data)

    Examples:
        >>> import torch
        >>> from torch import nn
        >>> from torchrl.data.tensor_specs import Unbounded
        >>> from torchrl.modules.distributions import NormalParamExtractor, TanhNormal
        >>> from torchrl.modules.tensordict_module.actors import ProbabilisticActor, ValueOperator
        >>> from torchrl.modules.tensordict_module.common import SafeModule
        >>> from torchrl.objectives.reinforce import ReinforceLoss
        >>> from tensordict import TensorDict
        >>> n_obs, n_act = 3, 5
        >>> value_net = ValueOperator(nn.Linear(n_obs, 1), in_keys=["observation"])
        >>> net = nn.Sequential(nn.Linear(n_obs, 2 * n_act), NormalParamExtractor())
        >>> module = SafeModule(net, in_keys=["observation"], out_keys=["loc", "scale"])
        >>> actor_net = ProbabilisticActor(
        ...     module,
        ...     distribution_class=TanhNormal,
        ...     return_log_prob=True,
        ...     in_keys=["loc", "scale"],
        ...     spec=Unbounded(n_act),)
        >>> loss = ReinforceLoss(actor_net, value_net)
        >>> batch = 2
        >>> data = TensorDict({
        ...     "observation": torch.randn(batch, n_obs),
        ...     "next": {
        ...         "observation": torch.randn(batch, n_obs),
        ...         "reward": torch.randn(batch, 1),
        ...         "done": torch.zeros(batch, 1, dtype=torch.bool),
        ...         "terminated": torch.zeros(batch, 1, dtype=torch.bool),
        ...     },
        ...     "action": torch.randn(batch, n_act),
        ... }, [batch])
        >>> loss(data)
        TensorDict(
            fields={
                loss_actor: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False),
                loss_value: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False)},
            batch_size=torch.Size([]),
            device=None,
            is_shared=False)

    This class is compatible with non-tensordict based modules too and can be
    used without recurring to any tensordict-related primitive. In this case,
    the expected keyword arguments are:
    ``["action", "next_reward", "next_done", "next_terminated"]`` + in_keys of the actor and critic network
    The return value is a tuple of tensors in the following order: ``["loss_actor", "loss_value"]``.

    Examples:
        >>> import torch
        >>> from torch import nn
        >>> from torchrl.data.tensor_specs import Unbounded
        >>> from torchrl.modules.distributions import NormalParamExtractor, TanhNormal
        >>> from torchrl.modules.tensordict_module.actors import ProbabilisticActor, ValueOperator
        >>> from torchrl.modules.tensordict_module.common import SafeModule
        >>> from torchrl.objectives.reinforce import ReinforceLoss
        >>> n_obs, n_act = 3, 5
        >>> value_net = ValueOperator(nn.Linear(n_obs, 1), in_keys=["observation"])
        >>> net = nn.Sequential(nn.Linear(n_obs, 2 * n_act), NormalParamExtractor())
        >>> module = SafeModule(net, in_keys=["observation"], out_keys=["loc", "scale"])
        >>> actor_net = ProbabilisticActor(
        ...     module,
        ...     distribution_class=TanhNormal,
        ...     return_log_prob=True,
        ...     in_keys=["loc", "scale"],
        ...     spec=Unbounded(n_act),)
        >>> loss = ReinforceLoss(actor_net, value_net)
        >>> batch = 2
        >>> loss_actor, loss_value = loss(
        ...     observation=torch.randn(batch, n_obs),
        ...     next_observation=torch.randn(batch, n_obs),
        ...     next_reward=torch.randn(batch, 1),
        ...     next_done=torch.zeros(batch, 1, dtype=torch.bool),
        ...     next_terminated=torch.zeros(batch, 1, dtype=torch.bool),
        ...     action=torch.randn(batch, n_act),)
        >>> loss_actor.backward()

    """

    @dataclass
    class _AcceptedKeys:
        """Maintains default values for all configurable tensordict keys.

        This class defines which tensordict keys can be set using '.set_keys(key_name=key_value)' and their
        default values.

        Attributes:
            advantage (NestedKey): he input tensordict key where the advantage is expected.
                Will be used for the underlying value estimator. Defaults to ``"advantage"``.
            value_target (NestedKey): The input tensordict key where the target state value is expected.
                Will be used for the underlying value estimator Defaults to ``"value_target"``.
            value (NestedKey): The input tensordict key where the state value is expected.
                Will be used for the underlying value estimator. Defaults to ``"state_value"``.
            sample_log_prob (NestedKey): The input tensordict key where the sample log probability is expected.
                Defaults to ``"sample_log_prob"`` when :func:`~tensordict.nn.composite_lp_aggregate` returns `True`,
                `"action_log_prob"`  otherwise.
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
        sample_log_prob: NestedKey | None = None
        action: NestedKey = "action"
        reward: NestedKey = "reward"
        done: NestedKey = "done"
        terminated: NestedKey = "terminated"

        def __post_init__(self):
            if self.sample_log_prob is None:
                if composite_lp_aggregate(nowarn=True):
                    self.sample_log_prob = "sample_log_prob"
                else:
                    self.sample_log_prob = "action_log_prob"

    tensor_keys: _AcceptedKeys
    default_keys = _AcceptedKeys
    default_value_estimator = ValueEstimators.GAE
    out_keys = ["loss_actor", "loss_value"]

    actor_network: TensorDictModule
    critic_network: TensorDictModule
    actor_network_params: TensorDictParams | None
    critic_network_params: TensorDictParams | None
    target_actor_network_params: TensorDictParams | None
    target_critic_network_params: TensorDictParams | None

    @classmethod
    def __new__(cls, *args, **kwargs):
        cls._tensor_keys = cls._AcceptedKeys()
        return super().__new__(cls)

    def __init__(
        self,
        actor_network: ProbabilisticTensorDictSequential,
        critic_network: TensorDictModule | None = None,
        *,
        delay_value: bool = False,
        loss_critic_type: str = "smooth_l1",
        gamma: float = None,
        advantage_key: str = None,
        value_target_key: str = None,
        separate_losses: bool = False,
        functional: bool = True,
        actor: ProbabilisticTensorDictSequential = None,
        critic: ProbabilisticTensorDictSequential = None,
        reduction: str = None,
        clip_value: float | None = None,
    ) -> None:
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
        if not functional and delay_value:
            raise RuntimeError(
                "delay_value and ~functional are incompatible, as delayed value currently relies on functional calls."
            )
        if reduction is None:
            reduction = "mean"

        self._functional = functional

        super().__init__()
        self.in_keys = None
        self._set_deprecated_ctor_keys(
            advantage=advantage_key, value_target=value_target_key
        )

        self.delay_value = delay_value
        self.loss_critic_type = loss_critic_type
        self.reduction = reduction

        # Actor
        if self.functional:
            self.convert_to_functional(
                actor_network,
                "actor_network",
                create_target_params=False,
            )
        else:
            self.actor_network = actor_network

        if separate_losses:
            # we want to make sure there are no duplicates in the params: the
            # params of critic must be refs to actor if they're shared
            policy_params = list(actor_network.parameters())
        else:
            policy_params = None
        # Value
        if critic_network is not None:
            if self.functional:
                self.convert_to_functional(
                    critic_network,
                    "critic_network",
                    create_target_params=self.delay_value,
                    compare_against=policy_params,
                )
            else:
                self.critic_network = critic_network
                self.target_critic_network_params = None

        if gamma is not None:
            raise TypeError(_GAMMA_LMBDA_DEPREC_ERROR)

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
        self.register_buffer("clip_value", clip_value)

    @property
    def functional(self):
        return self._functional

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
        self._set_in_keys()

    def _set_in_keys(self):
        keys = [
            self.tensor_keys.action,
            ("next", self.tensor_keys.reward),
            ("next", self.tensor_keys.done),
            ("next", self.tensor_keys.terminated),
            *self.actor_network.in_keys,
            *[("next", key) for key in self.actor_network.in_keys],
            *self.critic_network.in_keys,
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
        advantage = tensordict.get(self.tensor_keys.advantage, None)
        if advantage is None:
            self.value_estimator(
                tensordict,
                params=self.critic_network_params.detach() if self.functional else None,
                target_params=self.target_critic_network_params
                if self.functional
                else None,
            )
            advantage = tensordict.get(self.tensor_keys.advantage)

        # compute log-prob
        with self.actor_network_params.to_module(
            self.actor_network
        ) if self.functional else contextlib.nullcontext():
            tensordict = self.actor_network(tensordict)

        log_prob = tensordict.get(self.tensor_keys.sample_log_prob)
        if log_prob.shape == advantage.shape[:-1]:
            log_prob = log_prob.unsqueeze(-1)
        loss_actor = -log_prob * advantage.detach()
        td_out = TensorDict({"loss_actor": loss_actor}, batch_size=[])

        loss_value, value_clip_fraction = self.loss_critic(tensordict)
        td_out.set("loss_value", loss_value)
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

    def loss_critic(self, tensordict: TensorDictBase) -> torch.Tensor:

        if self.clip_value:
            old_state_value = tensordict.get(
                self.tensor_keys.value, None
            )  # TODO: None soon to be removed
            if old_state_value is None:
                raise KeyError(
                    f"clip_value is set to {self.clip_value}, but "
                    f"the key {self.tensor_keys.value} was not found in the input tensordict. "
                    f"Make sure that the value_key passed to Reinforce exists in the input tensordict."
                )
            old_state_value = old_state_value.clone()

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
            state_value = self.critic_network(tensordict_select).get(
                self.tensor_keys.value
            )
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
                self.clip_value.to(state_value.device),
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

        return loss_value, clip_fraction

    def make_value_estimator(self, value_type: ValueEstimators = None, **hyperparams):
        if value_type is None:
            value_type = self.default_value_estimator
        self.value_type = value_type
        hp = dict(default_value_kwargs(value_type))
        if hasattr(self, "gamma"):
            hp["gamma"] = self.gamma
        hp.update(hyperparams)
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
