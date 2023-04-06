# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch


def _custom_conv1d(tensor: torch.Tensor, filter: torch.Tensor):
    """Computes a conv1d filter over a value.

    This is usually used to compute a discounted return:

    Tensor:                         Filter                      Result (discounted return)
    [ r_0,                          [ 1.0,                      [ r_0 + g r_1 + g^2 r_2 + r^3 r_3,
      r_1,                            g,                          r_1 + g r_2 + g^2 r_3,
      r_2,                            g^2,                        r_2 + g r_3,
      r_3,                            g^3 ]                       r_3 ]
      0,      |                        |
      0,      |  zero padding          | direction of filter
      0 ]     |                        v

    This function takes care of applying the one-sided zero padding. In this example,
    `Filter_dim` = :obj:`Time` = 4, but in practice Filter_dim can be <= to :obj:`Time`.

    Args:
        tensor (torch.Tensor): a [ Batch x 1 x Time ] floating-point tensor
        filter (torch.Tensor): a [ Filter_dim x 1 ] floating-point filter

    Returns: a filtered tensor of the same shape as the input tensor.

    """
    if filter.ndimension() > 2:
        # filter will have shape batch_dims x timesteps x filter_dim x 1
        # reshape to batch_dims x timesteps x 1 x filter_dim ready for convolving
        filter = filter.view(*filter.shape[:-2], 1, filter.shape[-2])

        # because time is represented on two different dimensions, we don't
        # need all convolutions, just those lying along a diagonal
        # rather than compute them all and discard, we stack just the slices
        # of val_pad that we care about, and apply the filter manually

        # STACK VERSION: val_pad is computed as in the block below
        # batched_val_pad = torch.stack(
        #     [val_pad[..., i : i + filter.shape[-1]] for i in range(tensor.shape[-1])],
        #     dim=1,
        # )

        # roll version
        T = tensor.shape[-1]
        device = tensor.device
        batched_val_pad = (
            roll_by_gather(
                tensor.expand(tensor.shape[0], filter.shape[-1], T).transpose(-2, -1),
                0,
                -torch.arange(filter.shape[-1], device=device),
            )
            .flip(-1)
            .triu(filter.shape[-1] - T)
            .flip(-1)
            .unsqueeze(-2)
        )

        # this is just a batched matrix multiplication, but einsum makes it
        # easy to keep the many dimensions under control. Here b = batch,
        # t = timestep, s = singleton, j is the filter dimension that should
        # get summed out. we swap the order of s and t here rather than
        # reshape / create a view later.
        # this is essentially identical to (batched_val_pad @ filter.transpose(-2, -1)).squeeze().unsqueeze(-2)
        # out = (batched_val_pad @ filter.transpose(-2, -1)).squeeze().unsqueeze(-2)
        out = torch.einsum("btsj,btsj->bst", batched_val_pad, filter)
    else:
        val_pad = torch.cat(
            [
                tensor,
                torch.zeros(
                    tensor.shape[0], 1, filter.shape[-2] - 1, device=tensor.device
                ),
            ],
            -1,
        )

        # shape = val.shape
        filter = filter.squeeze(-1).unsqueeze(0).unsqueeze(0)  # 1 x 1 x T
        out = torch.conv1d(val_pad, filter)
    # out = out.view(shape)
    if out.shape != tensor.shape:
        raise RuntimeError(
            f"wrong output shape: input shape: {tensor.shape}, output shape: {out.shape}"
        )
    return out


def roll_by_gather(mat: torch.Tensor, dim: int, shifts: torch.LongTensor):
    """Rolls a batched matrix along the last or last but one dimension.

    Args:
        mat (torch.Tensor): A batched matrix to roll
        dim (int): 0 or -2 indicates the last but one dimension,
            1 or -1 the last dimension.
        shifts (torch.LongTensor): A tensor containing the shifts. Must have the same number of
            elements as the unchosen dimension.

    Examples:
        >>> x = torch.arange(12).view(3, 4)
        >>> roll_by_gather(x, 0, -torch.arange(4))  # shifts the values in each column
        tensor([[ 0,  5, 10,  3],
                [ 4,  9,  2,  7],
                [ 8,  1,  6, 11]])
        >>> roll_by_gather(x, 1, -torch.arange(3))  # shifts the values in each row
        tensor([[ 0,  1,  2,  3],
                [ 5,  6,  7,  4],
                [10, 11,  8,  9]])

    """
    # assumes 2D array
    *batch, n_rows, n_cols = mat.shape
    device = mat.device

    if dim in (0, -2):
        arange1 = (
            torch.arange(n_rows, device=device).unsqueeze(-1).expand((n_rows, n_cols))
        )
        arange2 = (arange1 - shifts) % n_rows
        return torch.gather(mat, -2, arange2.expand(*batch, *arange2.shape))
    elif dim in (1, -1):
        arange1 = torch.arange(n_cols, device=device).expand((n_rows, n_cols))
        arange2 = (arange1 - shifts.unsqueeze(-1)) % n_cols
        return torch.gather(mat, -1, arange2.expand(*batch, n_rows, n_cols))
    else:
        raise NotImplementedError(f"dim {dim} is not supported.")


def _make_gammas_tensor(gamma: torch.Tensor, T: int, rolling_gamma: bool):
    """Prepares a decay tensor for a matrix multiplication.

    Given a tensor gamma of size [*batch, T, D],
    it will return a new tensor with size [*batch, T, T+1, D].
    In the rolling_gamma case, a rolling of the gamma values will be performed
    along the T axis, e.g.:
    [[ 1, g1, g2, g3],
    [ 1, g2, g3, 0],
    [ 1, g3, 0, 0]]

    Args:
        gamma (torch.tensor): the gamma tensor to be prepared.
        T (int): the time length
        rolling_gamma (bool): if ``True``, the gamma value is set for each step
            independently. If False, the gamma value at (i, t) will be used for the
            trajectory following (i, t).

    Returns: the prepared gamma decay tensor

    """
    # some reshaping code vendored from vec_td_lambda_return_estimate
    gamma = gamma.transpose(-2, -1).contiguous()
    gamma = gamma.view(-1, T)
    dtype = gamma.dtype
    device = gamma.device
    if rolling_gamma:
        # # loop
        # gammas = gamma.unsqueeze(-2).expand(gamma.shape[0], T, T).contiguous()
        # for i in range(1, T):
        #     s = gammas[:, i].clone()
        #     gammas[:, i] = 0
        #     gammas[:, i, :-i] = s[:, i:]
        # gammas = torch.cumprod(gammas.unsqueeze(-1), -2)
        # gammas_cont = torch.ones(gammas.shape[0], T, T, 1)
        # gammas_cont[..., 1:, :] = gammas[..., :-1, :]
        # gammas = gammas_cont

        # vectorized version
        gammas = torch.ones(gamma.shape[0], T, T + 1, 1, dtype=dtype, device=device)
        s0 = gamma.unsqueeze(-1).expand(gamma.shape[0], T, T)
        s1 = roll_by_gather(s0, 0, shifts=-torch.arange(T, device=device))

        # we should triu here, but it's useless since there is a triu on the values
        # happening in _custom_conv1d
        # s2 = s1.flip(-1).triu().flip(-1).transpose(-2, -1)
        s2 = s1.transpose(-2, -1)
        gammas[..., 1:, :] = s2.unsqueeze(-1)
    else:
        gammas = torch.ones(*gamma.shape, T + 1, 1, device=device, dtype=dtype)
        gammas[..., 1:, :] = gamma[..., None, None]
    return gammas


def _flatten_batch(tensor):
    """Because we mark the end of each batch with a truncated signal, we can concatenate them.

    Args:
        tensor (torch.Tensor): a tensor of shape [*B, T]

    """

    return tensor.flatten(0, -1)


def _get_num_per_traj(dones_and_truncated):
    """Because we mark the end of each batch with a truncated signal, we can concatenate them.

    Args:
        dones_and_truncated (torch.Tensor): A done or truncated mark of shape [*B, T]

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
    """
    Given a tensor of size [*B, T] and the corresponding traj lengths (flattened), returns the padded trajectories [NPad, Tmax]

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


def _fast_gae(reward: torch.Tensor, state_value: torch.Tensor, next_state_value: torch.Tensor, done: torch.Tensor, gamma: float, lmbda:float):
    """Fast vectorized Generalized Advantage Estimate 

    TODO: description; time_dim must be -2; will be called by vec_generalized_advantage_estimate

    Args:
        reward (torch.Tensor): a [*B, T, F] tensor containing rewards
        state_value (torch.Tensor): a [*B, T, F] tensor containing state values (value function)
        next_state_value (torch.Tensor): a [*B, T, F] tensor containing next state values (value function)
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
    td0_flat = _split_and_pad_sequence(td0, num_per_traj)

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
