# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import math
import warnings
from dataclasses import dataclass
from typing import Tuple

import torch
from tensordict.nn import ProbabilisticTensorDictSequential, TensorDictModule
from tensordict.tensordict import TensorDict, TensorDictBase
from tensordict.utils import NestedKey
from torch import distributions as d

from torchrl.objectives.utils import (
    _GAMMA_LMBDA_DEPREC_WARNING,
    default_value_kwargs,
    distance_loss,
    ValueEstimators,
)

from .common import LossModule
from .value import GAE, TD0Estimator, TD1Estimator, TDLambdaEstimator


class PPOLoss(LossModule):
    """A parent PPO loss class.

    PPO (Proximal Policy Optimisation) is a model-free, online RL algorithm
    that makes use of a recorded (batch of)
    trajectories to perform several optimization steps, while actively
    preventing the updated policy to deviate too
    much from its original parameter configuration.

    PPO loss can be found in different flavours, depending on the way the
    constrained optimisation is implemented: ClipPPOLoss and KLPENPPOLoss.
    Unlike its subclasses, this class does not implement any regularisation
    and should therefore be used cautiously.

    For more details regarding PPO, refer to: "Proximal Policy Optimization Algorithms",
    https://arxiv.org/abs/1707.06347

    Args:
        actor (ProbabilisticTensorDictSequential): policy operator.
        critic (ValueOperator): value operator.

    Keyword Args:
        entropy_bonus (bool, optional): if ``True``, an entropy bonus will be added to the
            loss to favour exploratory policies.
        samples_mc_entropy (int, optional): if the distribution retrieved from the policy
            operator does not have a closed form
            formula for the entropy, a Monte-Carlo estimate will be used.
            ``samples_mc_entropy`` will control how many
            samples will be used to compute this estimate.
            Defaults to ``1``.
        entropy_coef (scalar, optional): entropy multiplier when computing the total loss.
            Defaults to ``0.01``.
        critic_coef (scalar, optional): critic loss multiplier when computing the total
            loss. Defaults to ``1.0``.
        loss_critic_type (str, optional): loss function for the value discrepancy.
            Can be one of "l1", "l2" or "smooth_l1". Defaults to ``"smooth_l1"``.
        normalize_advantage (bool, optional): if ``True``, the advantage will be normalized
            before being used. Defaults to ``False``.
        separate_losses (bool, optional): if ``True``, shared parameters between
            policy and critic will only be trained on the policy loss.
            Defaults to ``False``, ie. gradients are propagated to shared
            parameters for both policy and critic losses.
        advantage_key (str, optional): [Deprecated, use set_keys(advantage_key=advantage_key) instead]
            The input tensordict key where the advantage is
            expected to be written. Defaults to ``"advantage"``.
        value_target_key (str, optional): [Deprecated, use set_keys(value_target_key=value_target_key) instead]
            The input tensordict key where the target state
            value is expected to be written. Defaults to ``"value_target"``.
        value_key (str, optional): [Deprecated, use set_keys(value_key) instead]
            The input tensordict key where the state
            value is expected to be written. Defaults to ``"state_value"``.

    .. note::
      The advantage (typically GAE) can be computed by the loss function or
      in the training loop. The latter option is usually preferred, but this is
      up to the user to choose which option is to be preferred.
      If the advantage key (``"advantage`` by default) is not present in the
      input tensordict, the advantage will be computed by the :meth:`~.forward`
      method.

        >>> ppo_loss = PPOLoss(actor, critic)
        >>> advantage = GAE(critic)
        >>> data = next(datacollector)
        >>> losses = ppo_loss(data)
        >>> # equivalent
        >>> advantage(data)
        >>> losses = ppo_loss(data)

      A custom advantage module can be built using :meth:`~.make_value_estimator`.
      The default is :class:`~torchrl.objectives.value.GAE` with hyperparameters
      dictated by :func:`~torchrl.objectives.utils.default_value_kwargs`.

        >>> ppo_loss = PPOLoss(actor, critic)
        >>> ppo_loss.make_value_estimator(ValueEstimators.TDLambda)
        >>> data = next(datacollector)
        >>> losses = ppo_loss(data)

    .. note::
      If the actor and the value function share parameters, one can avoid
      calling the common module multiple times by passing only the head of the
      value network to the PPO loss module:

        >>> common = SomeModule(in_keys=["observation"], out_keys=["hidden"])
        >>> actor_head = SomeActor(in_keys=["hidden"])
        >>> value_head = SomeValue(in_keys=["hidden"])
        >>> # first option, with 2 calls on the common module
        >>> model = ActorCriticOperator(common, actor_head, value_head)
        >>> loss_module = PPOLoss(model.get_policy_operator(), model.get_value_operator())
        >>> # second option, with a single call to the common module
        >>> loss_module = PPOLoss(ProbabilisticTensorDictSequential(model, actor_head), value_head)

      This will work regardless of whether separate_losses is activated or not.

    """

    @dataclass
    class _AcceptedKeys:
        """Maintains default values for all configurable tensordict keys.

        This class defines which tensordict keys can be set using '.set_keys(key_name=key_value)' and their
        default values

        Attributes:
            advantage (NestedKey): The input tensordict key where the advantage is expected.
                Will be used for the underlying value estimator. Defaults to ``"advantage"``.
            value_target (NestedKey): The input tensordict key where the target state value is expected.
                Will be used for the underlying value estimator Defaults to ``"value_target"``.
            value (NestedKey): The input tensordict key where the state value is expected.
                Will be used for the underlying value estimator. Defaults to ``"state_value"``.
            sample_log_prob (NestedKey): The input tensordict key where the
               sample log probability is expected.  Defaults to ``"sample_log_prob"``.
            action (NestedKey): The input tensordict key where the action is expected.
                Defaults to ``"action"``.
        """

        advantage: NestedKey = "advantage"
        value_target: NestedKey = "value_target"
        value: NestedKey = "state_value"
        sample_log_prob: NestedKey = "sample_log_prob"
        action: NestedKey = "action"

    default_keys = _AcceptedKeys()
    default_value_estimator = ValueEstimators.GAE

    def __init__(
        self,
        actor: ProbabilisticTensorDictSequential,
        critic: TensorDictModule,
        *,
        entropy_bonus: bool = True,
        samples_mc_entropy: int = 1,
        entropy_coef: float = 0.01,
        critic_coef: float = 1.0,
        loss_critic_type: str = "smooth_l1",
        normalize_advantage: bool = False,
        gamma: float = None,
        separate_losses: bool = False,
        advantage_key: str = None,
        value_target_key: str = None,
        value_key: str = None,
    ):
        super().__init__()
        self._set_deprecated_ctor_keys(
            advantage=advantage_key,
            value_target=value_target_key,
            value=value_key,
        )

        self.convert_to_functional(
            actor, "actor", funs_to_decorate=["forward", "get_dist"]
        )
        if separate_losses:
            # we want to make sure there are no duplicates in the params: the
            # params of critic must be refs to actor if they're shared
            policy_params = list(actor.parameters())
        else:
            policy_params = None
        self.convert_to_functional(critic, "critic", compare_against=policy_params)
        self.samples_mc_entropy = samples_mc_entropy
        self.entropy_bonus = entropy_bonus
        self.separate_losses = separate_losses
        self.register_buffer(
            "entropy_coef", torch.tensor(entropy_coef, device=self.device)
        )
        self.register_buffer(
            "critic_coef", torch.tensor(critic_coef, device=self.device)
        )
        self.loss_critic_type = loss_critic_type
        self.normalize_advantage = normalize_advantage
        if gamma is not None:
            warnings.warn(_GAMMA_LMBDA_DEPREC_WARNING, category=DeprecationWarning)
            self.gamma = gamma

    def _forward_value_estimator_keys(self, **kwargs) -> None:
        if self._value_estimator is not None:
            self._value_estimator.set_keys(
                advantage=self._tensor_keys.advantage,
                value_target=self._tensor_keys.value_target,
                value=self._tensor_keys.value,
            )

    def reset(self) -> None:
        pass

    def get_entropy_bonus(self, dist: d.Distribution) -> torch.Tensor:
        try:
            entropy = dist.entropy()
        except NotImplementedError:
            x = dist.rsample((self.samples_mc_entropy,))
            entropy = -dist.log_prob(x)
        return entropy.unsqueeze(-1)

    def _log_weight(
        self, tensordict: TensorDictBase
    ) -> Tuple[torch.Tensor, d.Distribution]:
        # current log_prob of actions
        action = tensordict.get(self.tensor_keys.action)
        if action.requires_grad:
            raise RuntimeError(
                f"tensordict stored {self.tensor_keys.action} requires grad."
            )

        dist = self.actor.get_dist(tensordict, params=self.actor_params)
        log_prob = dist.log_prob(action)

        prev_log_prob = tensordict.get(self.tensor_keys.sample_log_prob)
        if prev_log_prob.requires_grad:
            raise RuntimeError("tensordict prev_log_prob requires grad.")

        log_weight = (log_prob - prev_log_prob).unsqueeze(-1)
        return log_weight, dist

    def loss_critic(self, tensordict: TensorDictBase) -> torch.Tensor:
        # TODO: if the advantage is gathered by forward, this introduces an
        # overhead that we could easily reduce.
        if self.separate_losses:
            tensordict = tensordict.detach()
        try:
            target_return = tensordict.get(self.tensor_keys.value_target)
        except KeyError:
            raise KeyError(
                f"the key {self.tensor_keys.value_target} was not found in the input tensordict. "
                f"Make sure you provided the right key and the value_target (i.e. the target "
                f"return) has been retrieved accordingly. Advantage classes such as GAE, "
                f"TDLambdaEstimate and TDEstimate all return a 'value_target' entry that "
                f"can be used for the value loss."
            )

        state_value_td = self.critic(
            tensordict,
            params=self.critic_params,
        )

        try:
            state_value = state_value_td.get(self.tensor_keys.value)
        except KeyError:
            raise KeyError(
                f"the key {self.tensor_keys.value} was not found in the input tensordict. "
                f"Make sure that the value_key passed to PPO is accurate."
            )

        loss_value = distance_loss(
            target_return,
            state_value,
            loss_function=self.loss_critic_type,
        )
        return self.critic_coef * loss_value

    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        tensordict = tensordict.clone(False)
        advantage = tensordict.get(self.tensor_keys.advantage, None)
        if advantage is None:
            self.value_estimator(
                tensordict,
                params=self.critic_params.detach(),
                target_params=self.target_critic_params,
            )
            advantage = tensordict.get(self.tensor_keys.advantage)
        if self.normalize_advantage and advantage.numel() > 1:
            loc = advantage.mean().item()
            scale = advantage.std().clamp_min(1e-6).item()
            advantage = (advantage - loc) / scale

        log_weight, dist = self._log_weight(tensordict)
        neg_loss = (log_weight.exp() * advantage).mean()
        td_out = TensorDict({"loss_objective": -neg_loss.mean()}, [])
        if self.entropy_bonus:
            entropy = self.get_entropy_bonus(dist)
            td_out.set("entropy", entropy.mean().detach())  # for logging
            td_out.set("loss_entropy", -self.entropy_coef * entropy.mean())
        if self.critic_coef:
            loss_critic = self.loss_critic(tensordict).mean()
            td_out.set("loss_critic", loss_critic.mean())
        return td_out

    def make_value_estimator(self, value_type: ValueEstimators = None, **hyperparams):
        if value_type is None:
            value_type = self.default_value_estimator
        self.value_type = value_type
        hp = dict(default_value_kwargs(value_type))
        if hasattr(self, "gamma"):
            hp["gamma"] = self.gamma
        hp.update(hyperparams)
        if value_type == ValueEstimators.TD1:
            self._value_estimator = TD1Estimator(value_network=self.critic, **hp)
        elif value_type == ValueEstimators.TD0:
            self._value_estimator = TD0Estimator(value_network=self.critic, **hp)
        elif value_type == ValueEstimators.GAE:
            self._value_estimator = GAE(value_network=self.critic, **hp)
        elif value_type == ValueEstimators.TDLambda:
            self._value_estimator = TDLambdaEstimator(value_network=self.critic, **hp)
        else:
            raise NotImplementedError(f"Unknown value type {value_type}")

        tensor_keys = {
            "advantage": self.tensor_keys.advantage,
            "value": self.tensor_keys.value,
            "value_target": self.tensor_keys.value_target,
        }
        self._value_estimator.set_keys(**tensor_keys)


class ClipPPOLoss(PPOLoss):
    """Clipped PPO loss.

    The clipped importance weighted loss is computed as follows:
        loss = -min( weight * advantage, min(max(weight, 1-eps), 1+eps) * advantage)

    Args:
        actor (ProbabilisticTensorDictSequential): policy operator.
        critic (ValueOperator): value operator.

    Keyword Args:
        clip_epsilon (scalar, optional): weight clipping threshold in the clipped PPO loss equation.
            default: 0.2
        entropy_bonus (bool, optional): if ``True``, an entropy bonus will be added to the
            loss to favour exploratory policies.
        samples_mc_entropy (int, optional): if the distribution retrieved from the policy
            operator does not have a closed form
            formula for the entropy, a Monte-Carlo estimate will be used.
            ``samples_mc_entropy`` will control how many
            samples will be used to compute this estimate.
            Defaults to ``1``.
        entropy_coef (scalar, optional): entropy multiplier when computing the total loss.
            Defaults to ``0.01``.
        critic_coef (scalar, optional): critic loss multiplier when computing the total
            loss. Defaults to ``1.0``.
        loss_critic_type (str, optional): loss function for the value discrepancy.
            Can be one of "l1", "l2" or "smooth_l1". Defaults to ``"smooth_l1"``.
        normalize_advantage (bool, optional): if ``True``, the advantage will be normalized
            before being used. Defaults to ``False``.
        separate_losses (bool, optional): if ``True``, shared parameters between
            policy and critic will only be trained on the policy loss.
            Defaults to ``False``, ie. gradients are propagated to shared
            parameters for both policy and critic losses.
        advantage_key (str, optional): [Deprecated, use set_keys(advantage_key=advantage_key) instead]
            The input tensordict key where the advantage is
            expected to be written. Defaults to ``"advantage"``.
        value_target_key (str, optional): [Deprecated, use set_keys(value_target_key=value_target_key) instead]
            The input tensordict key where the target state
            value is expected to be written. Defaults to ``"value_target"``.
        value_key (str, optional): [Deprecated, use set_keys(value_key) instead]
            The input tensordict key where the state
            value is expected to be written. Defaults to ``"state_value"``.

    .. note:
      The advantage (typically GAE) can be computed by the loss function or
      in the training loop. The latter option is usually preferred, but this is
      up to the user to choose which option is to be preferred.
      If the advantage key (``"advantage`` by default) is not present in the
      input tensordict, the advantage will be computed by the :meth:`~.forward`
      method.

        >>> ppo_loss = ClipPPOLoss(actor, critic)
        >>> advantage = GAE(critic)
        >>> data = next(datacollector)
        >>> losses = ppo_loss(data)
        >>> # equivalent
        >>> advantage(data)
        >>> losses = ppo_loss(data)

      A custom advantage module can be built using :meth:`~.make_value_estimator`.
      The default is :class:`~torchrl.objectives.value.GAE` with hyperparameters
      dictated by :func:`~torchrl.objectives.utils.default_value_kwargs`.

        >>> ppo_loss = ClipPPOLoss(actor, critic)
        >>> ppo_loss.make_value_estimator(ValueEstimators.TDLambda)
        >>> data = next(datacollector)
        >>> losses = ppo_loss(data)

    .. note::
      If the actor and the value function share parameters, one can avoid
      calling the common module multiple times by passing only the head of the
      value network to the PPO loss module:

        >>> common = SomeModule(in_keys=["observation"], out_keys=["hidden"])
        >>> actor_head = SomeActor(in_keys=["hidden"])
        >>> value_head = SomeValue(in_keys=["hidden"])
        >>> # first option, with 2 calls on the common module
        >>> model = ActorCriticOperator(common, actor_head, value_head)
        >>> loss_module = PPOLoss(model.get_policy_operator(), model.get_value_operator())
        >>> # second option, with a single call to the common module
        >>> loss_module = PPOLoss(ProbabilisticTensorDictSequential(model, actor_head), value_head)

      This will work regardless of whether separate_losses is activated or not.

    """

    def __init__(
        self,
        actor: ProbabilisticTensorDictSequential,
        critic: TensorDictModule,
        *,
        clip_epsilon: float = 0.2,
        entropy_bonus: bool = True,
        samples_mc_entropy: int = 1,
        entropy_coef: float = 0.01,
        critic_coef: float = 1.0,
        loss_critic_type: str = "smooth_l1",
        normalize_advantage: bool = True,
        gamma: float = None,
        separate_losses: bool = False,
        **kwargs,
    ):
        super(ClipPPOLoss, self).__init__(
            actor,
            critic,
            entropy_bonus=entropy_bonus,
            samples_mc_entropy=samples_mc_entropy,
            entropy_coef=entropy_coef,
            critic_coef=critic_coef,
            loss_critic_type=loss_critic_type,
            normalize_advantage=normalize_advantage,
            gamma=gamma,
            separate_losses=separate_losses,
            **kwargs,
        )
        self.register_buffer("clip_epsilon", torch.tensor(clip_epsilon))

    @property
    def _clip_bounds(self):
        return (
            math.log1p(-self.clip_epsilon),
            math.log1p(self.clip_epsilon),
        )

    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        tensordict = tensordict.clone(False)
        advantage = tensordict.get(self.tensor_keys.advantage, None)
        if advantage is None:
            self.value_estimator(
                tensordict,
                params=self.critic_params.detach(),
                target_params=self.target_critic_params,
            )
            advantage = tensordict.get(self.tensor_keys.advantage)
        if self.normalize_advantage and advantage.numel() > 1:
            loc = advantage.mean().item()
            scale = advantage.std().clamp_min(1e-6).item()
            advantage = (advantage - loc) / scale

        log_weight, dist = self._log_weight(tensordict)
        # ESS for logging
        with torch.no_grad():
            # In theory, ESS should be computed on particles sampled from the same source. Here we sample according
            # to different, unrelated trajectories, which is not standard. Still it can give a idea of the dispersion
            # of the weights.
            lw = log_weight.squeeze()
            ess = (2 * lw.logsumexp(0) - (2 * lw).logsumexp(0)).exp()
            batch = log_weight.shape[0]

        if not advantage.shape == log_weight.shape:
            raise RuntimeError(
                f"advantage.shape and log_weight.shape do not match (got {advantage.shape} "
                f"and {log_weight.shape})"
            )
        gain1 = log_weight.exp() * advantage

        log_weight_clip = log_weight.clamp(*self._clip_bounds)
        gain2 = log_weight_clip.exp() * advantage

        gain = torch.stack([gain1, gain2], -1).min(dim=-1)[0]
        td_out = TensorDict({"loss_objective": -gain.mean()}, [])

        if self.entropy_bonus:
            entropy = self.get_entropy_bonus(dist)
            td_out.set("entropy", entropy.mean().detach())  # for logging
            td_out.set("loss_entropy", -self.entropy_coef * entropy.mean())
        if self.critic_coef:
            loss_critic = self.loss_critic(tensordict)
            td_out.set("loss_critic", loss_critic.mean())
        td_out.set("ESS", ess.mean() / batch)
        return td_out


class KLPENPPOLoss(PPOLoss):
    """KL Penalty PPO loss.

    The KL penalty loss has the following formula:
        loss = loss - beta * KL(old_policy, new_policy)
    The "beta" parameter is adapted on-the-fly to match a target KL divergence between the new and old policy, thus
    favouring a certain level of distancing between the two while still preventing them to be too much apart.

    Args:
        actor (ProbabilisticTensorDictSequential): policy operator.
        critic (ValueOperator): value operator.

    Keyword Args:
        dtarg (scalar, optional): target KL divergence. Defaults to ``0.01``.
        samples_mc_kl (int, optional): number of samples used to compute the KL divergence
            if no analytical formula can be found. Defaults to ``1``.
        beta (scalar, optional): initial KL divergence multiplier.
            Defaults to ``1.0``.
        decrement (scalar, optional): how much beta should be decremented if KL < dtarg. Valid range: decrement <= 1.0
            default: ``0.5``.
        increment (scalar, optional): how much beta should be incremented if KL > dtarg. Valid range: increment >= 1.0
            default: ``2.0``.
        entropy_bonus (bool, optional): if ``True``, an entropy bonus will be added to the
            loss to favour exploratory policies. Defaults to ``True``.
        samples_mc_entropy (int, optional): if the distribution retrieved from the policy
            operator does not have a closed form
            formula for the entropy, a Monte-Carlo estimate will be used.
            ``samples_mc_entropy`` will control how many
            samples will be used to compute this estimate.
            Defaults to ``1``.
        entropy_coef (scalar, optional): entropy multiplier when computing the total loss.
            Defaults to ``0.01``.
        critic_coef (scalar, optional): critic loss multiplier when computing the total
            loss. Defaults to ``1.0``.
        loss_critic_type (str, optional): loss function for the value discrepancy.
            Can be one of "l1", "l2" or "smooth_l1". Defaults to ``"smooth_l1"``.
        normalize_advantage (bool, optional): if ``True``, the advantage will be normalized
            before being used. Defaults to ``False``.
        separate_losses (bool, optional): if ``True``, shared parameters between
            policy and critic will only be trained on the policy loss.
            Defaults to ``False``, ie. gradients are propagated to shared
            parameters for both policy and critic losses.
        advantage_key (str, optional): [Deprecated, use set_keys(advantage_key=advantage_key) instead]
            The input tensordict key where the advantage is
            expected to be written. Defaults to ``"advantage"``.
        value_target_key (str, optional): [Deprecated, use set_keys(value_target_key=value_target_key) instead]
            The input tensordict key where the target state
            value is expected to be written. Defaults to ``"value_target"``.
        value_key (str, optional): [Deprecated, use set_keys(value_key) instead]
            The input tensordict key where the state
            value is expected to be written. Defaults to ``"state_value"``.


    .. note:
      The advantage (typically GAE) can be computed by the loss function or
      in the training loop. The latter option is usually preferred, but this is
      up to the user to choose which option is to be preferred.
      If the advantage key (``"advantage`` by default) is not present in the
      input tensordict, the advantage will be computed by the :meth:`~.forward`
      method.

        >>> ppo_loss = KLPENPPOLoss(actor, critic)
        >>> advantage = GAE(critic)
        >>> data = next(datacollector)
        >>> losses = ppo_loss(data)
        >>> # equivalent
        >>> advantage(data)
        >>> losses = ppo_loss(data)

      A custom advantage module can be built using :meth:`~.make_value_estimator`.
      The default is :class:`~torchrl.objectives.value.GAE` with hyperparameters
      dictated by :func:`~torchrl.objectives.utils.default_value_kwargs`.

        >>> ppo_loss = KLPENPPOLoss(actor, critic)
        >>> ppo_loss.make_value_estimator(ValueEstimators.TDLambda)
        >>> data = next(datacollector)
        >>> losses = ppo_loss(data)

    .. note::
      If the actor and the value function share parameters, one can avoid
      calling the common module multiple times by passing only the head of the
      value network to the PPO loss module:

        >>> common = SomeModule(in_keys=["observation"], out_keys=["hidden"])
        >>> actor_head = SomeActor(in_keys=["hidden"])
        >>> value_head = SomeValue(in_keys=["hidden"])
        >>> # first option, with 2 calls on the common module
        >>> model = ActorCriticOperator(common, actor_head, value_head)
        >>> loss_module = PPOLoss(model.get_policy_operator(), model.get_value_operator())
        >>> # second option, with a single call to the common module
        >>> loss_module = PPOLoss(ProbabilisticTensorDictSequential(model, actor_head), value_head)

      This will work regardless of whether separate_losses is activated or not.

    """

    def __init__(
        self,
        actor: ProbabilisticTensorDictSequential,
        critic: TensorDictModule,
        *,
        dtarg: float = 0.01,
        beta: float = 1.0,
        increment: float = 2,
        decrement: float = 0.5,
        samples_mc_kl: int = 1,
        entropy_bonus: bool = True,
        samples_mc_entropy: int = 1,
        entropy_coef: float = 0.01,
        critic_coef: float = 1.0,
        loss_critic_type: str = "smooth_l1",
        normalize_advantage: bool = True,
        gamma: float = None,
        separate_losses: bool = False,
        **kwargs,
    ):
        super(KLPENPPOLoss, self).__init__(
            actor,
            critic,
            entropy_bonus=entropy_bonus,
            samples_mc_entropy=samples_mc_entropy,
            entropy_coef=entropy_coef,
            critic_coef=critic_coef,
            loss_critic_type=loss_critic_type,
            normalize_advantage=normalize_advantage,
            gamma=gamma,
            separate_losses=separate_losses,
            **kwargs,
        )

        self.dtarg = dtarg
        self._beta_init = beta
        self.register_buffer("beta", torch.tensor(beta))

        if increment < 1.0:
            raise ValueError(
                f"increment should be >= 1.0 in KLPENPPOLoss, got {increment:4.4f}"
            )
        self.increment = increment
        if decrement > 1.0:
            raise ValueError(
                f"decrement should be <= 1.0 in KLPENPPOLoss, got {decrement:4.4f}"
            )
        self.decrement = decrement
        self.samples_mc_kl = samples_mc_kl

    def forward(self, tensordict: TensorDictBase) -> TensorDict:
        tensordict = tensordict.clone(False)
        advantage = tensordict.get(self.tensor_keys.advantage, None)
        if advantage is None:
            self.value_estimator(
                tensordict,
                params=self.critic_params.detach(),
                target_params=self.target_critic_params,
            )
            advantage = tensordict.get(self.tensor_keys.advantage)
        if self.normalize_advantage and advantage.numel() > 1:
            loc = advantage.mean().item()
            scale = advantage.std().clamp_min(1e-6).item()
            advantage = (advantage - loc) / scale
        log_weight, dist = self._log_weight(tensordict)
        neg_loss = log_weight.exp() * advantage

        previous_dist = self.actor.build_dist_from_params(tensordict)
        current_dist = self.actor.get_dist(tensordict, params=self.actor_params)
        try:
            kl = torch.distributions.kl.kl_divergence(previous_dist, current_dist)
        except NotImplementedError:
            x = previous_dist.sample((self.samples_mc_kl,))
            kl = (previous_dist.log_prob(x) - current_dist.log_prob(x)).mean(0)
        kl = kl.unsqueeze(-1)
        neg_loss = neg_loss - self.beta * kl
        if kl.mean() > self.dtarg * 1.5:
            self.beta.data *= self.increment
        elif kl.mean() < self.dtarg / 1.5:
            self.beta.data *= self.decrement
        td_out = TensorDict(
            {
                "loss_objective": -neg_loss.mean(),
                "kl": kl.detach().mean(),
            },
            [],
        )

        if self.entropy_bonus:
            entropy = self.get_entropy_bonus(dist)
            td_out.set("entropy", entropy.mean().detach())  # for logging
            td_out.set("loss_entropy", -self.entropy_coef * entropy.mean())

        if self.critic_coef:
            loss_critic = self.loss_critic(tensordict)
            td_out.set("loss_critic", loss_critic.mean())

        return td_out

    def reset(self) -> None:
        self.beta = self._beta_init
