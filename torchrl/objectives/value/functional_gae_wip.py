import timeit

import torch

from torchrl.objectives.value.functional import \
    vec_generalized_advantage_estimate, generalized_advantage_estimate
from torchrl.objectives.value.utils import _custom_conv1d


# dones = torch.zeros(100, dtype=torch.bool).bernoulli_(0.1)
# dones_and_truncated = dones.clone()
#
#
# dones_and_truncated[-1] = 1
# traj_ids = dones_and_truncated.cumsum(0)
# num_per_traj = torch.ones_like(dones).cumsum(0)[dones_and_truncated]
# num_per_traj[1:] -= num_per_traj[:-1].clone()
#
# rewards = torch.randn(100)
# print("num_per_traj", num_per_traj)
# reward_split = rewards.split(num_per_traj.tolist(), 0)
# rewards = torch.nn.utils.rnn.pad_sequence(reward_split, True)


def _flatten_batch(tensor):
    """Because we mark the end of each batch with a truncated signal, we can concatenate them.

    Args:
        tensor (torch.Tensor): a tensor of shape [B, T]

    """
    return tensor.flatten(0, 1)


def _get_num_per_traj(dones_and_truncated):
    """Because we mark the end of each batch with a truncated signal, we can concatenate them.

    Args:
        dones_and_truncated (torch.Tensor): A done or truncated mark of shape [B, T]

    Returns:
        A list of integers representing the number of steps in each trajectories

    """
    dones_and_truncated = dones_and_truncated.clone()
    dones_and_truncated[..., -1] = 1
    dones_and_truncated = _flatten_batch(dones_and_truncated)
    # traj_ids = dones_and_truncated.cumsum(0)
    num_per_traj = torch.ones_like(dones_and_truncated).cumsum(0)[
        dones_and_truncated]
    num_per_traj[1:] -= num_per_traj[:-1].clone()
    return num_per_traj


def _split_and_pad_sequence(tensor, splits):
    """Given a tensor of size [B, T] and the corresponding traj lengths (flattened), returns the padded trajectories [NPad, Tmax]"""
    tensor = _flatten_batch(tensor)
    if isinstance(splits, torch.Tensor):
        splits = splits.tolist()
    tensor_split = tensor.split(splits, 0)
    tensor_pad = torch.nn.utils.rnn.pad_sequence(tensor_split, True)
    return tensor_pad

def _split_and_pad_sequence2(tensor, splits):
    """Given a tensor of size [B, T] and the corresponding traj lengths (flattened), returns the padded trajectories [NPad, Tmax]"""
    tensor = _flatten_batch(tensor)
    max_val = max(splits)
    mask = torch.zeros(len(splits), max_val, dtype=torch.bool)
    mask.scatter_(index=max_val - torch.tensor(splits).unsqueeze(-1), dim=1, value=1)
    mask = mask.cumsum(-1).flip(-1).bool()
    empty_tensor = torch.zeros(len(splits), max_val, dtype=tensor.dtype)
    empty_tensor[mask] = tensor
    return empty_tensor


# get rewards
def _inv_pad_sequence(tensor, splits):
    """
    Examples:
        >>> rewards = torch.randn(100, 20)
        >>> num_per_traj = _get_num_per_traj(torch.zeros(100, 20).bernoulli_(0.1))
        >>> padded = _split_and_pad_sequence(rewards, num_per_traj.tolist())
        >>> reconstructed = _inv_pad_sequence(padded, num_per_traj)
        >>> assert (reconstructed==rewards).all()

    """
    offset = torch.ones_like(splits) * tensor.shape[-1]
    offset[0] = 0
    offset = offset.cumsum(0)
    z = torch.zeros(tensor.numel(), dtype=torch.bool)

    ones = offset + splits
    while ones[-1] == len(z):
        ones = ones[:-1]
    z[ones] = 1
    z[offset[1:]] = torch.bitwise_xor(
        z[offset[1:]],
        torch.ones_like(z[offset[1:]])
    )  # make sure that the longest is accounted for
    idx = z.cumsum(0) % 2 == 0
    return tensor.view(-1)[idx]

def gae(reward, state_value, next_state_value, done, gamma, lmbda):
    """Generalized Advantage estimate

    Args:
        reward (torch.Tensor): a [B, T] tensor containing rewards
        state_value (torch.Tensor): a [B, T] tensor containing state values
        next_state_value (torch.Tensor): a [B, T] tensor containing next state values
        done (torch.Tensor): a [B, T] boolean tensor containing the done states
        gamma (scalar): the gamma decay
        lmbda (scalar): the lambda decay

    """
    gammalmbda = gamma * lmbda
    not_done = 1 - done.int()
    td0 = reward + not_done * gamma * next_state_value - state_value

    num_per_traj = _get_num_per_traj(done)
    td0_flat = _split_and_pad_sequence2(td0, num_per_traj)

    gammalmbdas = torch.ones_like(td0_flat[0])
    gammalmbdas[1:] = gammalmbda
    gammalmbdas[1:] = gammalmbdas[1:].cumprod(0)
    gammalmbdas = gammalmbdas.unsqueeze(-1)

    advantage = _custom_conv1d(td0_flat.unsqueeze(1), gammalmbdas)
    advantage = advantage.squeeze(1)
    advantage = _inv_pad_sequence(advantage, num_per_traj).view_as(reward)

    value_target = advantage + state_value
    return advantage, value_target

N = 20
T = 200
reward, state_value, next_state_value = [torch.randn(N, T) for _ in range(3)]
done = torch.zeros(N, T, dtype=torch.bool).bernoulli_(0.1)
gamma = 0.99
lmbda = 0.95
v1 = gae(reward, state_value, next_state_value, done, gamma, lmbda)

v2 = vec_generalized_advantage_estimate(
    gamma=gamma,
    lmbda=lmbda,
    state_value=state_value,
    next_state_value=next_state_value,
    reward=reward,
    done=done,
    time_dim=-1
)

print(v1[0] / v2[0])
print(v1[1] / v2[1])

print(timeit.timeit(
    "gae(reward, state_value, next_state_value, done, gamma, lmbda)", globals={
        "gae": gae, "reward": reward, "state_value": state_value,
        "next_state_value": next_state_value, "done": done, "gamma": gamma,
        "lmbda": lmbda}, number=1_000
    )
)
print(timeit.timeit(
    """generalized_advantage_estimate(    gamma=gamma,
    lmbda=lmbda,
    state_value=state_value,
    next_state_value=next_state_value,
    reward=reward,
    done=done,
    time_dim=-1
)""",
    globals={
        "generalized_advantage_estimate": generalized_advantage_estimate,
        "reward": reward, "state_value": state_value,
        "next_state_value": next_state_value, "done": done, "gamma": gamma,
        "lmbda": lmbda},
    number=1_000
    )
)
print(timeit.timeit(
    """vec_generalized_advantage_estimate(    gamma=gamma,
    lmbda=lmbda,
    state_value=state_value,
    next_state_value=next_state_value,
    reward=reward,
    done=done,
    time_dim=-1
)""",
    globals={
"vec_generalized_advantage_estimate": vec_generalized_advantage_estimate, "reward": reward, "state_value": state_value, "next_state_value": next_state_value, "done": done, "gamma": gamma, "lmbda": lmbda}, number = 1_000)
)
