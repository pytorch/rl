import math
from numbers import Number
from typing import Optional, Tuple, Callable

import torch
from torch import distributions as d

from torchrl.data.tensordict.tensordict import _TensorDict, TensorDict
from torchrl.envs.utils import step_tensor_dict
from torchrl.modules import Actor, ProbabilisticOperator

__all__ = ["PPOLoss", "ClipPPOLoss", "KLPENPPOLoss"]

from .common import _LossModule

from torchrl.objectives.costs.utils import distance_loss


class PPOLoss(_LossModule):
    """
    A parent PPO loss class.

    PPO (Proximal Policy Optimisation) is a model-free, online RL algorithm that makes use of a recorded (batch of)
    trajectories to perform several optimization steps, while actively preventing the updated policy to deviate too
    much from its original parameter configuration.

    PPO loss can be found in different flavours, depending on the way the constrained optimisation is implemented:
        ClipPPOLoss and KLPENPPOLoss.
    Unlike its subclasses, this class does not implement any regularisation and should therefore be used cautiously.

    For more details regarding PPO, refer to: "Proximal Policy Optimization Algorithms",
    https://arxiv.org/abs/1707.06347

    Args:
        actor (Actor): policy operator.
        critic (ProbabilisticOperator): value operator.
        advantage_key (str): the input tensordict key where the advantage is expected to be written.
            default: "advantage"
        entropy_bonus (bool): if True, an entropy bonus will be added to the loss to favour exploratory policies.
        samples_mc_entropy (int): if the distribution retrieved from the policy operator does not have a closed form
            formula for the entropy, a Monte-Carlo estimate will be used. samples_mc_entropy will control how many
            samples will be used to compute this estimate.
            default: 1
        entropy_factor (scalar): entropy multiplier when computing the total loss.
            default: 0.01
        critic_factor (scalar): critic loss multiplier when computing the total loss.
            default: 1.0
        gamma (scalar): a discount factor for return computation.
        loss_function (str): loss function for the value discrepancy. Can be one of "l1", "l2" or "smooth_l1".

    """

    def __init__(
        self,
        actor: Actor,
        critic: ProbabilisticOperator,
        advantage_key: str = "advantage",
        entropy_bonus: bool = True,
        samples_mc_entropy: int = 1,
        entropy_factor: Number = 0.01,
        critic_factor: Number = 1.0,
        gamma: Number = 0.99,
        loss_critic_type: str = "smooth_l1",
        advantage_module: Optional[Callable[[_TensorDict], _TensorDict]] = None,
    ):
        super().__init__()
        self.actor = actor
        self.critic = critic
        self.advantage_key = advantage_key
        self.samples_mc_entropy = samples_mc_entropy
        self.entropy_bonus = entropy_bonus and entropy_factor
        self.entropy_factor = entropy_factor
        self.critic_factor = critic_factor
        self.gamma = gamma
        self.loss_critic_type = loss_critic_type
        self.advantage_module = advantage_module

    def reset(self) -> None:
        pass

    def get_entropy_bonus(self, dist: Optional[d.Distribution] = None) -> torch.Tensor:
        try:
            entropy = dist.entropy()
        except:
            x = dist.rsample((self.samples_mc_entropy,))
            entropy = -dist.log_prob(x)
        return entropy.unsqueeze(-1)

    def _log_weight(
        self, tensor_dict: _TensorDict
    ) -> Tuple[torch.Tensor, d.Distribution]:
        # current log_prob of actions
        action = tensor_dict.get("action")
        if action.requires_grad:
            raise RuntimeError("tensor_dict stored action requires grad.")
        tensor_dict_clone = tensor_dict.select(*self.actor.in_keys).clone()

        dist, *_ = self.actor.get_dist(tensor_dict_clone)
        log_prob = dist.log_prob(action)
        log_prob = log_prob.unsqueeze(-1)

        prev_log_prob = tensor_dict.get("action_log_prob")
        if prev_log_prob.requires_grad:
            raise RuntimeError("tensor_dict prev_log_prob requires grad.")

        log_weight = log_prob - prev_log_prob
        return log_weight, dist

    def loss_critic(self, tensor_dict: _TensorDict) -> torch.Tensor:

        if "value_target" in tensor_dict.keys():
            value_target = tensor_dict.get("value_target")
            if value_target.requires_grad:
                raise RuntimeError(
                    "value_target retrieved from tensor_dict requires grad."
                )

        else:
            with torch.no_grad():
                reward = tensor_dict.get("reward")
                next_td = step_tensor_dict(tensor_dict)
                next_value = self.critic(next_td).get("state_value")
                value_target = reward + next_value * self.gamma
        tensor_dict_select = tensor_dict.select(*self.critic.in_keys).clone()
        value = self.critic(tensor_dict_select).get("state_value")
        loss_value = distance_loss(
            value, value_target, loss_function=self.loss_critic_type
        )
        return self.critic_factor * loss_value

    def __call__(self, tensor_dict: _TensorDict) -> _TensorDict:
        if self.advantage_module is not None:
            tensor_dict = self.advantage_module(tensor_dict)
        tensor_dict = tensor_dict.clone()
        advantage = tensor_dict.get(self.advantage_key)
        log_weight, dist = self._log_weight(tensor_dict)
        neg_loss = (log_weight.exp() * advantage).mean()
        td_out = TensorDict({"loss_objective": -neg_loss.mean()}, [])
        if self.entropy_bonus:
            entropy = self.get_entropy_bonus(dist)
            td_out.set("entropy", entropy.mean().detach())  # for logging
            td_out.set("loss_entropy", -self.entropy_factor * entropy.mean())
        if self.critic_factor:
            loss_critic = self.loss_critic(tensor_dict).mean()
            td_out.set("loss_critic", loss_critic.mean())
        return td_out


class ClipPPOLoss(PPOLoss):
    """
    Clipped PPO loss.

    The clipped importance weighted loss is computed as follows:
        loss = -min( weight * advantage, min(max(weight, 1-eps), 1+eps) * advantage)

    Args:
        actor (Actor): policy operator.
        critic (ProbabilisticOperator): value operator.
        advantage_key (str): the input tensordict key where the advantage is expected to be written.
            default: "advantage"
        clip_epsilon (scalar): weight clipping threshold in the clipped PPO loss equation.
            default: 0.2
        entropy_bonus (bool): if True, an entropy bonus will be added to the loss to favour exploratory policies.
        samples_mc_entropy (int): if the distribution retrieved from the policy operator does not have a closed form
            formula for the entropy, a Monte-Carlo estimate will be used. samples_mc_entropy will control how many
            samples will be used to compute this estimate.
            default: 1
        entropy_factor (scalar): entropy multiplier when computing the total loss.
            default: 0.01
        critic_factor (scalar): critic loss multiplier when computing the total loss.
            default: 1.0
        gamma (scalar): a discount factor for return computation.
        loss_function (str): loss function for the value discrepancy. Can be one of "l1", "l2" or "smooth_l1".

    """

    def __init__(
        self,
        actor: Actor,
        critic: ProbabilisticOperator,
        advantage_key: str = "advantage",
        clip_epsilon: Number = 0.2,
        entropy_bonus: bool = True,
        samples_mc_entropy: int = 1,
        entropy_factor: Number = 0.01,
        critic_factor: Number = 1.0,
        gamma: Number = 0.99,
        loss_critic_type: str = "l2",
        **kwargs,
    ):
        super(ClipPPOLoss, self).__init__(
            actor,
            critic,
            advantage_key,
            entropy_bonus=entropy_bonus,
            samples_mc_entropy=samples_mc_entropy,
            entropy_factor=entropy_factor,
            critic_factor=critic_factor,
            gamma=gamma,
            loss_critic_type=loss_critic_type,
            **kwargs,
        )
        self.clip_epsilon = clip_epsilon
        self._clip_bounds = (
            math.log1p(-self.clip_epsilon),
            math.log1p(self.clip_epsilon),
        )

    def __call__(self, tensor_dict: _TensorDict) -> _TensorDict:
        if self.advantage_module is not None:
            tensor_dict = self.advantage_module(tensor_dict)
        tensor_dict = tensor_dict.clone()
        for key, value in tensor_dict.items():
            if value.requires_grad:
                raise RuntimeError(
                    f"The key {key} returns a value that requires a gradient, consider detaching."
                )
        advantage = tensor_dict.get(self.advantage_key)
        log_weight, dist = self._log_weight(tensor_dict)
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
        log_weight_clip = torch.empty_like(log_weight)
        # log_weight_clip.data.clamp_(*self._clip_bounds)
        idx_pos = advantage >= 0
        log_weight_clip[idx_pos] = log_weight[idx_pos].clamp_max(self._clip_bounds[1])
        log_weight_clip[~idx_pos] = log_weight[~idx_pos].clamp_min(self._clip_bounds[0])

        gain2 = log_weight_clip.exp() * advantage
        gain = torch.stack([gain1, gain2], -1).min(dim=-1)[0]
        td_out = TensorDict({"loss_objective": -gain.mean()}, [])

        if self.entropy_bonus:
            entropy = self.get_entropy_bonus(dist)
            td_out.set("entropy", entropy.mean().detach())  # for logging
            td_out.set("loss_entropy", -self.entropy_factor * entropy.mean())
        if self.critic_factor:
            loss_critic = self.loss_critic(tensor_dict)
            td_out.set("loss_critic", loss_critic.mean())
        td_out.set("ESS", ess.mean()/batch)
        return td_out


class KLPENPPOLoss(PPOLoss):
    """
    KL Penalty PPO loss.

    The KL penalty loss has the following formula:
        loss = loss - beta * KL(old_policy, new_policy)
    The "beta" parameter is adapted on-the-fly to match a target KL divergence between the new and old policy, thus
    favouring a certain level of distancing between the two while still preventing them to be too much apart.

    Args:
        actor (Actor): policy operator.
        critic (ProbabilisticOperator): value operator.
        advantage_key (str): the input tensordict key where the advantage is expected to be written.
            default: "advantage"
        dtarg (scalar): target KL divergence.
        beta (scalar): initial KL divergence multiplier.
            default: 1.0
        increment (scalar): how much beta should be incremented if KL > dtarg. Valid range: increment >= 1.0
            default: 2.0
        decrement (scalar): how much beta should be decremented if KL < dtarg. Valid range: decrement <= 1.0
            default: 0.5
        entropy_bonus (bool): if True, an entropy bonus will be added to the loss to favour exploratory policies.
        samples_mc_entropy (int): if the distribution retrieved from the policy operator does not have a closed form
            formula for the entropy, a Monte-Carlo estimate will be used. samples_mc_entropy will control how many
            samples will be used to compute this estimate.
            default: 1
        entropy_factor (scalar): entropy multiplier when computing the total loss.
            default: 0.01
        critic_factor (scalar): critic loss multiplier when computing the total loss.
            default: 1.0
        gamma (scalar): a discount factor for return computation.
        loss_critic_type (str): loss function for the value discrepancy. Can be one of "l1", "l2" or "smooth_l1".

    """

    def __init__(
        self,
        actor: Actor,
        critic: ProbabilisticOperator,
        advantage_key="advantage",
        dtarg: Number = 0.01,
        beta: Number = 1.0,
        increment: Number = 2,
        decrement: Number = 0.5,
        samples_mc_kl: int = 1,
        entropy_bonus: bool = True,
        samples_mc_entropy: int = 1,
        entropy_factor: Number = 0.01,
        critic_factor: Number = 1.0,
        gamma: Number = 0.99,
        loss_critic_type: str = "l2",
        **kwargs,
    ):
        super(KLPENPPOLoss, self).__init__(
            actor,
            critic,
            advantage_key,
            entropy_bonus=entropy_bonus,
            samples_mc_entropy=samples_mc_entropy,
            entropy_factor=entropy_factor,
            critic_factor=critic_factor,
            gamma=gamma,
            loss_critic_type=loss_critic_type,
            **kwargs,
        )

        self.dtarg = dtarg
        self._beta_init = beta
        self.beta = beta

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

    def __call__(self, tensor_dict: _TensorDict) -> TensorDict:
        if self.advantage_module is not None:
            tensor_dict = self.advantage_module(tensor_dict)
        tensor_dict = tensor_dict.clone()
        advantage = tensor_dict.get(self.advantage_key)
        log_weight, dist = self._log_weight(tensor_dict)
        neg_loss = log_weight.exp() * advantage

        tensor_dict_clone = tensor_dict.select(*self.actor.in_keys).clone()
        params = []
        out_key = self.actor.out_keys[0]
        i = 0
        while True:
            key = f"{out_key}_dist_param_{i}"
            if key in tensor_dict.keys():
                params.append(tensor_dict.get(key))
                i += 1
            else:
                break

        if i == 0:
            raise Exception(
                "No parameter was found for the policy distribution. Consider building the policy with save_dist_params=True"
            )
        previous_dist, *_ = self.actor.build_dist_from_params(params)
        current_dist, *_ = self.actor.get_dist(tensor_dict_clone)
        try:
            kl = torch.distributions.kl.kl_divergence(previous_dist, current_dist)
        except NotImplementedError:
            x = previous_dist.sample((self.samples_mc_kl,))
            kl = (previous_dist.log_prob(x) - current_dist.log_prob(x)).mean(0)
        kl = kl.unsqueeze(-1)
        neg_loss = neg_loss - self.beta * kl
        if kl.mean() > self.dtarg * 1.5:
            self.beta *= self.increment
        elif kl.mean() < self.dtarg / 1.5:
            self.beta *= self.decrement
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
            td_out.set("loss_entropy", -self.entropy_factor * entropy.mean())

        if self.critic_factor:
            loss_critic = self.loss_critic(tensor_dict)
            td_out.set("loss_critic", loss_critic.mean())

        return td_out

    def reset(self) -> None:
        self.beta = self._beta_init
