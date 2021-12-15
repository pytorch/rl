import torch

def generalized_advantage_estimate(
        gamma, lamda, value_old_state, value_new_state, reward, done
):
    """
    Get generalized advantage estimate of a trajectory
    gamma: exponential mean discount (scalar)
    lamda: trajectory discount (scalar)
    value_old_state: value function result with old_state input
    value_new_state: value function result with new_state input
    reward: agent reward of taking actions in the environment
    done: flag for end of episode
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