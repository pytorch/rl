import math

import torch

from torchrl.envs.utils import step_tensor_dict
from torchrl.modules import Actor, ProbabilisticOperator

__all__ = ["PPOLoss", "ClipPPOLoss", "KLPENPPOLoss"]

from torchrl.objectives.costs.utils import distance_loss


class PPOLoss:
    def __init__(
            self,
            actor: Actor,
            critic: ProbabilisticOperator,
            advantage_key="advantage",
            entropy_bonus=True,
            samples_mc_entropy=1,
            entropy_factor=0.01,
            critic_factor=1.0,
            gamma=0.99,
            critic_loss_type="l2"
    ):
        self.actor = actor
        self.critic = critic
        self.advantage_key = advantage_key
        self.samples_mc_entropy = samples_mc_entropy
        self.entropy_bonus = entropy_bonus
        self.entropy_factor = entropy_factor
        self.critic_factor = critic_factor
        self.gamma = gamma
        self.critic_loss_type = critic_loss_type

    def reset(self):
        pass
    def get_entropy_bonus(self, dist=None):
        try:
            entropy = dist.entropy()
        except:
            x = dist.rsample((self.samples_mc_entropy,))
            entropy = - dist.log_prob(x)
        return self.entropy_factor * entropy.unsqueeze(-1)

    def _log_weight(self, tensor_dict):
        # current log_prob of actions
        action = tensor_dict.get("action")
        assert not action.requires_grad
        tensor_dict_clone = tensor_dict.select(*self.actor.in_keys).clone()

        dist, *_ = self.actor.get_dist(tensor_dict_clone)
        log_prob = dist.log_prob(action)
        log_prob = log_prob.unsqueeze(-1)

        prev_log_prob = tensor_dict.get("action_log_prob")
        assert not prev_log_prob.requires_grad
        log_weight = log_prob - prev_log_prob
        return log_weight, dist

    def critic_loss(self, tensor_dict):

        if "value_target" in tensor_dict.keys():
            value_target = tensor_dict.get("value_target")
        else:
            with torch.no_grad():
                reward = tensor_dict.get("reward")
                next_td = step_tensor_dict(tensor_dict)
                next_value = self.critic(next_td).get("state_value")
                value_target = reward + next_value * self.gamma
        tensor_dict_select = tensor_dict.select(*self.critic.in_keys).clone()
        value = self.critic(tensor_dict_select).get("state_value")
        assert not value_target.requires_grad
        value_loss = distance_loss(value, value_target, loss_type=self.critic_loss_type)
        return self.critic_factor * value_loss

    def __call__(self, tensor_dict):
        tensor_dict = tensor_dict.clone()
        advantage = tensor_dict.get(self.advantage_key)
        log_weight, dist = self._log_weight(tensor_dict)
        neg_loss = (log_weight.exp() * advantage).mean()
        if self.entropy_bonus:
            neg_loss = neg_loss + self.get_entropy_bonus(dist).mean()
        if self.critic_factor:
            neg_loss = neg_loss - self.critic_loss(tensor_dict).mean()
        loss = -neg_loss
        return loss


class ClipPPOLoss(PPOLoss):
    def __init__(
            self,
            actor, critic, advantage_key="advantage", clip_epsilon=0.2, entropy_bonus=True,
            samples_mc_entropy=1, entropy_factor=0.01, critic_factor=1.0, gamma=0.99, critic_loss_type="l2"
    ):
        super(ClipPPOLoss, self).__init__(actor, critic, advantage_key, entropy_bonus=entropy_bonus,
                                          samples_mc_entropy=samples_mc_entropy, entropy_factor=entropy_factor,
                                          critic_factor=critic_factor, gamma=gamma, critic_loss_type=critic_loss_type)
        self.clip_epsilon = clip_epsilon

    def __call__(self, tensor_dict):
        neg_loss = 0.0
        tensor_dict = tensor_dict.clone()
        for k, it in tensor_dict.items():
            assert not it.requires_grad
        advantage = tensor_dict.get(self.advantage_key)
        log_weight, dist = self._log_weight(tensor_dict)
        # print(tensordict.get("action").min(), tensordict.get("action").max())
        neg_loss1 = log_weight.exp() * advantage
        log_weight_clip = log_weight.clone()
        log_weight_clip = log_weight_clip.clamp_(math.log1p(-self.clip_epsilon), math.log1p(+self.clip_epsilon))
        neg_loss2 = log_weight_clip.exp() * advantage
        # neg_loss = log_weight * advantage
        neg_loss = neg_loss + torch.stack([neg_loss1, neg_loss2], -1).min(dim=-1)[0]
        if self.entropy_bonus:
            entropy_gain = self.get_entropy_bonus(dist)
            neg_loss = neg_loss + entropy_gain
        if self.critic_factor:
            critic_loss = self.critic_loss(tensor_dict)
            neg_loss = neg_loss - critic_loss
        return -neg_loss.mean()


class KLPENPPOLoss(PPOLoss):
    def __init__(self,
                 actor,
                 critic,
                 advantage_key="advantage",
                 dtarg=0.01,
                 beta=1.0,
                 increment=2,
                 decrement=0.5,
                 samples_mc_kl=1,
                 entropy_bonus=True,
                 samples_mc_entropy=1,
                 entropy_factor=0.01,
                 critic_factor=1.0,
                 gamma=0.99,
                 critic_loss_type="l2"
                 ):
        super(KLPENPPOLoss, self).__init__(actor, critic, advantage_key, entropy_bonus=entropy_bonus,
                                           samples_mc_entropy=samples_mc_entropy, entropy_factor=entropy_factor,
                                           critic_factor=critic_factor, gamma=gamma, critic_loss_type=critic_loss_type)

        self.dtarg = dtarg
        self._beta_init = beta
        self.beta = beta

        self.increment = increment
        self.decrement = decrement
        self.samples_mc_kl = samples_mc_kl

    def __call__(self, tensor_dict):
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
                "No parameter was found for the policy distribution. Consider building the policy with save_dist_params=True")
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

        if self.entropy_bonus:
            neg_loss = neg_loss + self.get_entropy_bonus(dist=current_dist)
        if self.critic_factor:
            neg_loss = neg_loss - self.critic_loss(tensor_dict)
        return -neg_loss.mean()

    def reset(self):
        self.beta = self._beta_init