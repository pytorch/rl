import timeit
import pytest
from functools import wraps

import torch

from torchrl.objectives.value.functional import (
    generalized_advantage_estimate,
    vec_generalized_advantage_estimate,
)
from torchrl.objectives.value.utils import _custom_conv1d

from tensordict import MemmapTensor, TensorDictBase


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
    return tensor.flatten(0, -1)


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
    num_per_traj = torch.ones_like(dones_and_truncated).cumsum(0)[dones_and_truncated]
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
    """
    Given a tensor of size [B, T] and the corresponding traj lengths (flattened), returns the padded trajectories [NPad, Tmax]

    [r00, r01, r02, r03, r10, r11] -> [[r00, r01, r02, r03], [r10, r11, 0, 0]]
    """
    tensor = _flatten_batch(tensor)
    max_val = max(splits)
    mask = torch.zeros(len(splits), max_val, dtype=torch.bool)
    mask.scatter_(index=max_val - splits.unsqueeze(-1), dim=1, value=1)
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
        z[offset[1:]], torch.ones_like(z[offset[1:]])
    )  # make sure that the longest is accounted for
    idx = z.cumsum(0) % 2 == 0
    return tensor.view(-1)[idx]

def gae(reward, state_value, next_state_value, done, gamma, lmbda, time_dim=-1):
    """Generalized Advantage estimate

    Args:
        reward (torch.Tensor): a [B, T] tensor containing rewards
        state_value (torch.Tensor): a [B, T] tensor containing state values (value function)
        next_state_value (torch.Tensor): a [B, T] tensor containing next state values (value function)
        done (torch.Tensor): a [B, T] boolean tensor containing the done states
        gamma (scalar): the gamma decay (trajectory discount)
        lmbda (scalar): the lambda decay (exponential mean discount)
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

def _transpose_time(fun):
    """Checks the time_dim argument of the function to allow for any dim.

    If not -2, makes a transpose of all the multi-dim input tensors to bring
    time at -2, and does the opposite transform for the outputs.
    """

    @wraps(fun)
    def transposed_fun(*args, time_dim=-2, **kwargs):
        def transpose_tensor(tensor):
            if isinstance(tensor, (torch.Tensor, MemmapTensor)) and tensor.ndim >= 2:
                tensor = tensor.transpose(time_dim, -2)
            return tensor

        if time_dim != -2:
            args = [transpose_tensor(arg) for arg in args]
            kwargs = {k: transpose_tensor(item) for k, item in kwargs.items()}
            out = fun(*args, time_dim=-2, **kwargs)
            if isinstance(out, torch.Tensor):
                return transpose_tensor(out)
            return tuple(transpose_tensor(_out) for _out in out)
        return fun(*args, time_dim=time_dim, **kwargs)

    return transposed_fun

@_transpose_time
def fast_gae(reward: torch.Tensor, state_value: torch.Tensor, next_state_value: torch.Tensor, done: torch.Tensor, gamma: float, lmbda:float, time_dim: int=-2):
    """Generalized Advantage estimate

    Args:
        reward (torch.Tensor): a [*B, T, *F] tensor containing rewards
        state_value (torch.Tensor): a [*B, T, *F] tensor containing state values (value function)
        next_state_value (torch.Tensor): a [*B, T, *F] tensor containing next state values (value function)
        done (torch.Tensor): a [B, T] boolean tensor containing the done states
        gamma (scalar): the gamma decay (trajectory discount)
        lmbda (scalar): the lambda decay (exponential mean discount)

    """

    # TODO: will be called from vec_generalized_advantage_estimate
    # which will transpose input such that time dimesion is penultimate
    # -> put time dimension last and revert afterwards
    # rework such that fast_gae work with time dimension penultimate nativley
    # or refactor vec_generalized_advantage_estimate so that it does not transpose
    # time dimension before calling fast_gae
    done = done.transpose(-2, -1)
    reward = reward.transpose(-2, -1)
    state_value = state_value.transpose(-2, -1)
    next_state_value = next_state_value.transpose(-2, -1)

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

    done = done.transpose(-1, -2)
    reward = reward.transpose(-1, -2)
    state_value = state_value.transpose(-1, -2)
    next_state_value = next_state_value.transpose(-1, -2)
    advantage = advantage.transpose(-1, -2)


    value_target = advantage + state_value
    return advantage, value_target


# wip test _get_num_per_traj(done)

#done = torch.zeros(2, 5, 1, dtype=torch.bool)
#print(f"{done.shape = }")
#print(f"{done = } {_get_num_per_traj(done) = }")

torch.manual_seed(44)
N = (2, )
T = 5
F = (2, 2)
time_dim=-3

done = torch.zeros(*N, T, *F, dtype=torch.bool).bernoulli_(0.1)
reward, state_value, next_state_value = [torch.ones(*N, T, *F) for _ in range(3)]

#state_value, next_state_value, reward = [torch.as_tensor([[[1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0]]]) for _ in range(3)]
#done = torch.as_tensor([[[False, False], [False, False], [True, False], [False, False], [False, False]]])
gamma = 0.99
lmbda = 0.95

#print(f"{reward = }")
#print(f"{state_value = }")
#print(f"{next_state_value = }")

v1 = fast_gae(reward, state_value, next_state_value, done, gamma, lmbda, time_dim=-3)

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

print(
    timeit.timeit(
        "gae(reward, state_value, next_state_value, done, gamma, lmbda)",
        globals={
            "gae": gae,
            "reward": reward[..., 0],
            "state_value": state_value[..., 0],
            "next_state_value": next_state_value[..., 0],
            "done": done[..., 0],
            "gamma": gamma,
            "lmbda": lmbda
        },
        number=1_000,
    )
)

print(
    timeit.timeit(
        "fast_gae(reward, state_value, next_state_value, done, gamma, lmbda)",
        globals={
            "fast_gae": fast_gae,
            "reward": reward,
            "state_value": state_value,
            "next_state_value": next_state_value,
            "done": done,
            "gamma": gamma,
            "lmbda": lmbda,
            "time_dim": time_dim,
        },
        number=1_000,
    )
)

print(
    timeit.timeit(
        """generalized_advantage_estimate(gamma=gamma,
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


print(
    timeit.timeit(
        """vec_generalized_advantage_estimate(gamma=gamma,
    lmbda=lmbda,
    state_value=state_value,
    next_state_value=next_state_value,
    reward=reward,
    done=done,
    time_dim=-1
)""",
    globals={
            "vec_generalized_advantage_estimate": vec_generalized_advantage_estimate,
            "reward": reward,
            "state_value": state_value,
            "next_state_value": next_state_value,
            "done": done,
            "gamma": gamma,
            "lmbda": lmbda,
            "time_dim": time_dim
        },
        number=1_000,
    )
)
