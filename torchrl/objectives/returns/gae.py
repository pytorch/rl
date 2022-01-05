from numbers import Number
from typing import Union

import torch

# for value, log_policy, reward, entropy in list(zip(values, log_policies, rewards, entropies))[::-1]:
#     gae = gae * opt.gamma * opt.tau
#     gae = gae + reward + opt.gamma * next_value.detach() - value.detach()
#     next_value = value
#     actor_loss = actor_loss + log_policy * gae
#     R = R * opt.gamma + reward
#     critic_loss = critic_loss + (R - value) ** 2 / 2
#     entropy_loss = entropy_loss + entropy
from torchrl.envs.utils import step_tensor_dict
from .functional import generalized_advantage_estimate
# from https://github.com/H-Huang/rpc-rl-experiments/blob/6621f0aadb347d1c4e24bcf46517ac36907401ff/a3c/process.py#L14
# TODO: create function / object that vectorises that
# actor_loss = 0
# critic_loss = 0
# entropy_loss = 0
# next_value = R
from ...data.tensordict.tensordict import _TensorDict
from ...modules import ProbabilisticOperator


def gae(values: torch.Tensor, log_prob_actions: torch.Tensor, rewards: torch.Tensor, entropies: torch.Tensor,
        gamma: Union[Number, torch.Tensor], tau: Number) -> torch.Tensor:
    # https://arxiv.org/pdf/1506.02438.pdf
    gaes = []
    for value, log_policy, reward, entropy in list(
            zip(values, log_prob_actions, rewards, entropies)
    )[::-1]:
        if next_value is None:
            next_value = torch.zeros_like(value)
        gae = gae * gamma * tau
        gae = gae + reward + gamma * next_value.detach() - value.detach()
        next_value = value
        gaes.append(gae)
    return torch.stack(gae)


class GAE:
    def __init__(self, gamma: Union[Number, torch.Tensor], lamda: Number, critic: ProbabilisticOperator,
                 average_rewards: bool = True):
        self.gamma = gamma
        self.lamda = lamda
        self.critic = critic
        self.average_rewards = average_rewards

    def __call__(self, tensor_dict: _TensorDict) -> _TensorDict:
        if tensor_dict.batch_dims < 2:
            raise RuntimeError("Expected input tensordict to have at least two dimensions, got"
                               f"tensor_dict.batch_size = {tensor_dict.batch_size}")

        gamma, lamda = self.gamma, self.lamda
        self.critic(tensor_dict)
        value = tensor_dict.get("state_value")
        step_td = step_tensor_dict(tensor_dict)
        self.critic(step_td)
        next_value = step_td.get("state_value")
        reward = tensor_dict.get("reward")
        if self.average_rewards:
            reward = reward - reward.mean()
            reward = reward / reward.std().clamp_min(1e-4)
        done = tensor_dict.get("done")

        adv, value_target = generalized_advantage_estimate(
            gamma, lamda, value, next_value, reward, done)
        tensor_dict.set("advantage", adv)
        tensor_dict.set("value_target", value_target)
        return tensor_dict
