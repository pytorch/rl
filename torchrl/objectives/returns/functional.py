from numbers import Number

import torch


def generalized_advantage_estimate(
        gamma: Number, lamda: Number, value_old_state: torch.Tensor, value_new_state: torch.Tensor,
        reward: torch.Tensor, done: torch.Tensor
):
    """
    Get generalized advantage estimate of a trajectory
    Refer to "HIGH-DIMENSIONAL CONTINUOUS CONTROL USING GENERALIZED ADVANTAGE ESTIMATION"
    https://arxiv.org/pdf/1506.02438.pdf for more context.

    Args:
        gamma (scalar): exponential mean discount.
        lamda (scalar): trajectory discount.
        value_old_state (Tensor): value function result with old_state input.
            must be a [Batch x TimeSteps x 1] or [Batch x TimeSteps] tensor
        value_new_state (Tensor): value function result with new_state input.
            must be a [Batch x TimeSteps x 1] or [Batch x TimeSteps] tensor
        reward (Tensor): agent reward of taking actions in the environment.
            must be a [Batch x TimeSteps x 1] or [Batch x TimeSteps] tensor
        done (Tensor): boolean flag for end of episode.
    """
    not_done = 1 - done.to(value_new_state.dtype)
    batch_size, time_steps = not_done.shape[:2]
    device = value_old_state.device
    advantage = torch.zeros(batch_size, time_steps + 1, 1, device=device)

    for t in reversed(range(time_steps)):
        delta = reward[:, t] + (gamma * value_new_state[:, t] * not_done[:, t]) - value_old_state[:, t]
        advantage[:, t] = delta + (gamma * lamda * advantage[:, t + 1] * not_done[:, t])

    value_target = advantage[:, :time_steps] + value_old_state

    return advantage[:, :time_steps], value_target
