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
    `Filter_dim` = `Time` = 4, but in practice Filter_dim can be <= to `Time`.

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
        batched_val_pad = (
            roll_by_gather(
                tensor.expand(tensor.shape[0], T + 1, T).transpose(-2, -1),
                0,
                -torch.arange(T + 1),
            )
            .flip(-1)
            .triu(1)
            .flip(-1)
            .unsqueeze(-2)
        )

        # this is just a batched matrix multiplication, but einsum makes it
        # easy to keep the many dimensions under control. Here b = batch,
        # t = timestep, s = singleton, j is the filter dimension that should
        # get summed out. we swap the order of s and t here rather than
        # reshape / create a view later.
        # this is essentially identical to (batched_val_pad @ filter.transpose(-2, -1)).squeeze().unsqueeze(-2)
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
        raise RuntimeError("wrong output shape")
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
        # print(mat)
        arange1 = (
            torch.arange(n_rows, device=device).unsqueeze(-1).expand((n_rows, n_cols))
        )
        # print(arange1)
        arange2 = (arange1 - shifts) % n_rows
        # print(arange2)
        return torch.gather(mat, -2, arange2.expand(*batch, *arange2.shape))
    elif dim in (1, -1):
        arange1 = torch.arange(n_cols, device=device).expand((n_rows, n_cols))
        arange2 = (arange1 - shifts.unsqueeze(-1)) % n_cols
        return torch.gather(mat, -1, arange2.expand(*batch, n_rows, n_cols))
    else:
        raise NotImplementedError(f"dim {dim} is not supported.")


def _make_gammas_tensor(gamma: torch.Tensor, T: int, rolling_gamma: bool):
    # some reshaping code vendored from vec_td_lambda_return_estimate
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
        s0 = gamma.unsqueeze(-1).expand(gamma.shape[0], T, T).contiguous()
        s1 = roll_by_gather(s0, 0, shifts=-torch.arange(T))
        s2 = s1.flip(-1).triu().flip(-1).transpose(-2, -1)
        gammas[..., 1:, :] = s2.unsqueeze(-1)
        # torch.testing.assert_close(gammas, gammas2)

    else:
        gammas = torch.ones(*gamma.shape, T + 1, 1, device=device, dtype=dtype)
        gammas[..., 1:, :] = gamma[..., None, None]
    return gammas
