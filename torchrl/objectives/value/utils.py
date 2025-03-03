# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import torch

from tensordict import TensorDictBase
from tensordict.utils import expand_right


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
        val_pad = torch.nn.functional.pad(tensor, [0, filter.shape[-2] - 1])

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


def _flatten_batch(tensor, time_dim=-1):
    """Because we mark the end of each batch with a truncated signal, we can concatenate them.

    Args:
        tensor (torch.Tensor): a tensor of shape [*B, T, *F]
        time_dim (int, optional): the time dimension T. Defaults to -1.

    """
    return tensor.flatten(0, time_dim)


def _get_num_per_traj(done):
    """Because we mark the end of each batch with a truncated signal, we can concatenate them.

    Args:
        done (torch.Tensor): A done or truncated mark of shape [*B, T]

    Returns:
        A list of integers representing the number of steps in each trajectory

    """
    done = done.clone()
    done[..., -1] = True
    # TODO: find a way of copying once only, eg not using reshape
    num_per_traj = torch.where(done.reshape(-1))[0] + 1
    num_per_traj[1:] = num_per_traj[1:] - num_per_traj[:-1]
    return num_per_traj


def _split_and_pad_sequence(
    tensor: torch.Tensor | TensorDictBase,
    splits: torch.Tensor,
    return_mask=False,
    time_dim=-1,
):
    """Given a tensor of size [*B, T, F] and the corresponding traj lengths (flattened), returns the padded trajectories [NPad, Tmax, *other].

    Compatible with tensordict inputs.

    Examples:
        >>> from tensordict import TensorDict
        >>> is_init = torch.zeros(4, 5, dtype=torch.bool)
        >>> is_init[:, 0] = True
        >>> is_init[0, 3] = True
        >>> is_init[1, 2] = True
        >>> tensordict = TensorDict({
        ...     "is_init": is_init,
        ...     "obs": torch.arange(20).view(4, 5).unsqueeze(-1).expand(4, 5, 3),
        ... }, [4, 5])
        >>> splits = _get_num_per_traj_init(is_init)
        >>> print(splits)
        tensor([3, 2, 2, 3, 5, 5])
        >>> td = _split_and_pad_sequence(tensordict, splits)
        >>> print(td)
        TensorDict(
            fields={
                is_init: Tensor(shape=torch.Size([6, 5]), device=cpu, dtype=torch.bool, is_shared=False),
                obs: Tensor(shape=torch.Size([6, 5, 3]), device=cpu, dtype=torch.int64, is_shared=False)},
            batch_size=torch.Size([6, 5]),
            device=None,
            is_shared=False)
        >>> print(td["obs"])
        tensor([[[ 0,  0,  0],
                 [ 1,  1,  1],
                 [ 2,  2,  2],
                 [ 0,  0,  0],
                 [ 0,  0,  0]],
        <BLANKLINE>
                [[ 3,  3,  3],
                 [ 4,  4,  4],
                 [ 0,  0,  0],
                 [ 0,  0,  0],
                 [ 0,  0,  0]],
        <BLANKLINE>
                [[ 5,  5,  5],
                 [ 6,  6,  6],
                 [ 0,  0,  0],
                 [ 0,  0,  0],
                 [ 0,  0,  0]],
        <BLANKLINE>
                [[ 7,  7,  7],
                 [ 8,  8,  8],
                 [ 9,  9,  9],
                 [ 0,  0,  0],
                 [ 0,  0,  0]],
        <BLANKLINE>
                [[10, 10, 10],
                 [11, 11, 11],
                 [12, 12, 12],
                 [13, 13, 13],
                 [14, 14, 14]],
        <BLANKLINE>
                [[15, 15, 15],
                 [16, 16, 16],
                 [17, 17, 17],
                 [18, 18, 18],
                 [19, 19, 19]]])

    """
    max_seq_len = torch.max(splits)
    shape = (len(splits), max_seq_len)

    # int16 supports length up to 32767
    dtype = (
        torch.int16
        if tensor.size(time_dim) < torch.iinfo(torch.int16).max
        else torch.int32
    )
    arange = torch.arange(max_seq_len, device=tensor.device, dtype=dtype).unsqueeze(0)
    mask = arange < splits.unsqueeze(1)

    tensor = _flatten_batch(tensor, time_dim=time_dim)

    def _fill_tensor(tensor):
        empty_tensor = torch.zeros(
            *shape,
            *tensor.shape[1:],
            dtype=tensor.dtype,
            device=tensor.device,
        )
        mask_expand = expand_right(mask, (*mask.shape, *tensor.shape[1:]))
        # We need to use masked-scatter to accommodate vmap
        return torch.masked_scatter(empty_tensor, mask_expand, tensor.reshape(-1))
        # empty_tensor[mask_expand] = tensor.reshape(-1)
        # return empty_tensor

    if isinstance(tensor, TensorDictBase):
        tensor = tensor.apply(_fill_tensor, batch_size=list(shape))
    else:
        tensor = _fill_tensor(tensor)
    if return_mask:
        return tensor, mask
    return tensor


def _inv_pad_sequence(
    tensor: torch.Tensor | TensorDictBase,
    splits: torch.Tensor,
    mask: torch.Tensor = None,
):
    """Inverse a pad_sequence operation.

    If tensor is of shape [B, T], than splits must be of of shape [B] with all elements
    and integer between [1, T].
    The result will be flattened along the batch dimension(s) and must be reshaped into
    the original shape (if necessary).

    Examples:
        >>> rewards = torch.randn(100, 20)
        >>> num_per_traj = _get_num_per_traj(torch.zeros(100, 20).bernoulli_(0.1))
        >>> padded = _split_and_pad_sequence(rewards, num_per_traj)
        >>> reconstructed = _inv_pad_sequence(padded, num_per_traj)
        >>> assert (reconstructed==rewards).all()
    """
    if splits.numel() == 1:
        return tensor

    if mask is None:
        # int16 supports length up to 32767
        dtype = (
            torch.int16
            if tensor.shape[-1] < torch.iinfo(torch.int16).max
            else torch.int32
        )
        arange = torch.arange(
            tensor.shape[-1], device=tensor.device, dtype=dtype
        ).unsqueeze(0)
        mask = arange < splits.unsqueeze(1)

    return tensor[mask]


def _get_num_per_traj_init(is_init):
    """Like _get_num_per_traj, but with is_init signal."""
    done = torch.zeros_like(is_init)
    done[..., :-1][is_init[..., 1:]] = 1
    return _get_num_per_traj(done)
