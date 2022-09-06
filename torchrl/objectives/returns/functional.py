# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple

import torch

__all__ = [
    "generalized_advantage_estimate",
    "vec_generalized_advantage_estimate",
    "vec_td_lambda_return_estimate",
    "vec_td_lambda_advantage_estimate",
    "td_lambda_return_estimate",
    "td_lambda_advantage_estimate",
    "td_advantage_estimate",
]


def generalized_advantage_estimate(
    gamma: float,
    lmbda: float,
    state_value: torch.Tensor,
    next_state_value: torch.Tensor,
    reward: torch.Tensor,
    done: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Get generalized advantage estimate of a trajectory
    Refer to "HIGH-DIMENSIONAL CONTINUOUS CONTROL USING GENERALIZED ADVANTAGE ESTIMATION"
    https://arxiv.org/pdf/1506.02438.pdf for more context.

    Args:
        gamma (scalar): exponential mean discount.
        lmbda (scalar): trajectory discount.
        state_value (Tensor): value function result with old_state input.
            must be a [Batch x TimeSteps x 1] or [Batch x TimeSteps] tensor
        next_state_value (Tensor): value function result with new_state input.
            must be a [Batch x TimeSteps x 1] or [Batch x TimeSteps] tensor
        reward (Tensor): reward of taking actions in the environment.
            must be a [Batch x TimeSteps x 1] or [Batch x TimeSteps] tensor
        done (Tensor): boolean flag for end of episode.
    """
    for tensor in (next_state_value, state_value, reward, done):
        if tensor.shape[-1] != 1:
            raise RuntimeError(
                "Last dimension of generalized_advantage_estimate inputs must be a singleton dimension."
            )
    dtype = next_state_value.dtype
    device = state_value.device

    not_done = 1 - done.to(dtype)
    *batch_size, time_steps = not_done.shape[:-1]
    advantage = torch.empty(*batch_size, time_steps, 1, device=device, dtype=dtype)
    prev_advantage = 0
    for t in reversed(range(time_steps)):
        delta = (
            reward[..., t, :]
            + (gamma * next_state_value[..., t, :] * not_done[..., t, :])
            - state_value[..., t, :]
        )
        prev_advantage = advantage[..., t, :] = delta + (
            gamma * lmbda * prev_advantage * not_done[..., t, :]
        )

    value_target = advantage + state_value

    return advantage, value_target


def vec_generalized_advantage_estimate(
    gamma: float,
    lmbda: float,
    state_value: torch.Tensor,
    next_state_value: torch.Tensor,
    reward: torch.Tensor,
    done: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Get generalized advantage estimate of a trajectory
    Refer to "HIGH-DIMENSIONAL CONTINUOUS CONTROL USING GENERALIZED ADVANTAGE ESTIMATION"
    https://arxiv.org/pdf/1506.02438.pdf for more context.

    Args:
        gamma (scalar): exponential mean discount.
        lmbda (scalar): trajectory discount.
        state_value (Tensor): value function result with old_state input.
            must be a [Batch x TimeSteps x 1] or [Batch x TimeSteps] tensor
        next_state_value (Tensor): value function result with new_state input.
            must be a [Batch x TimeSteps x 1] or [Batch x TimeSteps] tensor
        reward (Tensor): reward of taking actions in the environment.
            must be a [Batch x TimeSteps x 1] or [Batch x TimeSteps] tensor
        done (Tensor): boolean flag for end of episode.
    """
    for tensor in (next_state_value, state_value, reward, done):
        if tensor.shape[-1] != 1:
            raise RuntimeError(
                "Last dimension of generalized_advantage_estimate inputs must be a singleton dimension."
            )
    device = state_value.device
    dtype = state_value.dtype
    not_done = 1 - done.to(dtype)
    *batch_size, time_steps = not_done.shape[:-1]

    gammalmbda = torch.full_like(not_done, gamma * lmbda) * not_done
    gammalmbda = gammalmbda.flatten(0, len(batch_size) - 1).squeeze(-1)
    gammalmbdas = torch.ones(*gammalmbda.shape, time_steps + 1, 1, device=device, dtype=dtype)
    gammalmbdas[..., 1:, :] = gammalmbda[..., None, :, None]

    gammalmbdas = torch.cumprod(gammalmbdas[..., :-1, :], -2)

    filter = gammalmbdas

    # first_below_thr = gammalmbdas < 1e-7
    # # if we have multiple gammas, we only want to truncate if _all_ of
    # # the geometric sequences fall below the threshold
    # first_below_thr = first_below_thr.all(axis=0)
    # if first_below_thr.any():
    #     gammalmbdas = gammalmbdas[..., :first_below_thr, :]

    td0 = reward + not_done * gamma * next_state_value - state_value

    if len(batch_size) > 1:
        td0 = td0.flatten(0, len(batch_size) - 1)

    advantage = _custom_conv1d(td0.transpose(-2, -1), filter)

    if len(batch_size) > 1:
        advantage = advantage.unflatten(0, batch_size)

    advantage = advantage.transpose(-2, -1)
    value_target = advantage + state_value
    return advantage, value_target


def td_advantage_estimate(
    gamma: float,
    state_value: torch.Tensor,
    next_state_value: torch.Tensor,
    reward: torch.Tensor,
    done: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Get generalized advantage estimate of a trajectory
    Refer to "HIGH-DIMENSIONAL CONTINUOUS CONTROL USING GENERALIZED ADVANTAGE ESTIMATION"
    https://arxiv.org/pdf/1506.02438.pdf for more context.

    Args:
        gamma (scalar): exponential mean discount.
        state_value (Tensor): value function result with old_state input.
            must be a [Batch x TimeSteps x 1] or [Batch x TimeSteps] tensor
        next_state_value (Tensor): value function result with new_state input.
            must be a [Batch x TimeSteps x 1] or [Batch x TimeSteps] tensor
        reward (Tensor): reward of taking actions in the environment.
            must be a [Batch x TimeSteps x 1] or [Batch x TimeSteps] tensor
        done (Tensor): boolean flag for end of episode.
    """
    for tensor in (next_state_value, state_value, reward, done):
        if tensor.shape[-1] != 1:
            raise RuntimeError(
                "Last dimension of generalized_advantage_estimate inputs must be a singleton dimension."
            )
    not_done = 1 - done.to(next_state_value.dtype)
    advantage = reward + gamma * not_done * next_state_value - state_value
    return advantage


def td_lambda_return_estimate(
    gamma: float,
    lmbda: float,
    next_state_value: torch.Tensor,
    reward: torch.Tensor,
    done: torch.Tensor,
) -> torch.Tensor:
    """TD(lambda) return estimate.

    Args:
        gamma (scalar): exponential mean discount.
        lmbda (scalar): trajectory discount.
        next_state_value (Tensor): value function result with new_state input.
            must be a [Batch x TimeSteps x 1] or [Batch x TimeSteps] tensor
        reward (Tensor): reward of taking actions in the environment.
            must be a [Batch x TimeSteps x 1] or [Batch x TimeSteps] tensor
        done (Tensor): boolean flag for end of episode.
    """
    for tensor in (next_state_value, reward, done):
        if tensor.shape[-1] != 1:
            raise RuntimeError(
                "Last dimension of generalized_advantage_estimate inputs must be a singleton dimension."
            )
    not_done = 1 - done.to(next_state_value.dtype)
    next_state_value = not_done * next_state_value

    returns = torch.empty_like(next_state_value)

    g = next_state_value[..., -1, :]
    T = returns.shape[-2]

    for i in reversed(range(T)):
        g = returns[..., i, :] = reward[..., i, :] + gamma * (
            (1 - lmbda) * next_state_value[..., i, :] + lmbda * g
        )
    return returns


def td_lambda_advantage_estimate(
    gamma: float,
    lmbda: float,
    state_value: torch.Tensor,
    next_state_value: torch.Tensor,
    reward: torch.Tensor,
    done: torch.Tensor,
) -> torch.Tensor:
    """TD(lambda) advantage estimate.

    Args:
        gamma (scalar): exponential mean discount.
        lmbda (scalar): trajectory discount.
        state_value (Tensor): value function result with old_state input.
            must be a [Batch x TimeSteps x 1] or [Batch x TimeSteps] tensor
        next_state_value (Tensor): value function result with new_state input.
            must be a [Batch x TimeSteps x 1] or [Batch x TimeSteps] tensor
        reward (Tensor): reward of taking actions in the environment.
            must be a [Batch x TimeSteps x 1] or [Batch x TimeSteps] tensor
        done (Tensor): boolean flag for end of episode.
    """
    if not state_value.shape == next_state_value.shape:
        raise RuntimeError("shape of state_value and next_state_value must match")
    returns = td_lambda_return_estimate(gamma, lmbda, next_state_value, reward, done)
    advantage = returns - state_value
    return advantage


def _custom_conv1d(tensor, filter):
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
    val_pad = torch.cat(
        [
            tensor,
            torch.zeros(tensor.shape[0], 1, filter.shape[-2] - 1, device=tensor.device),
        ],
        -1,
    )

    if filter.ndimension() > 2:
        # filter will have shape batch_dims x timesteps x filter_dim x 1
        # reshape to batch_dims x timesteps x 1 x filter_dim ready for convolving
        filter = filter.view(*filter.shape[:-2], 1, filter.shape[-2])

        # because time is represented on two different dimensions, we don't
        # need all convolutions, just those lying along a diagonal
        # rather than compute them all and discard, we stack just the slices
        # of val_pad that we care about, and apply the filter manually
        batched_val_pad = torch.stack(
            [val_pad[..., i : i + filter.shape[-1]] for i in range(tensor.shape[-1])],
            dim=1,
        )

        # this is just a batched matrix multiplication, but einsum makes it
        # easy to keep the many dimensions under control. Here b = batch,
        # t = timestep, s = singleton, j is the filter dimension that should
        # get summed out. we swap the order of s and t here rather than
        # reshape / create a view later.
        # this is essentially identical to (batched_val_pad @ filter.transpose(-2, -1)).squeeze().unsqueeze(-2)
        out = torch.einsum("btsj,btsj->bst", batched_val_pad, filter)
    else:
        # shape = val.shape
        filter = filter.squeeze(-1).unsqueeze(0).unsqueeze(0)  # 1 x 1 x T
        out = torch.conv1d(val_pad, filter)
    # out = out.view(shape)
    if not out.shape == tensor.shape:
        raise RuntimeError("wrong output shape")
    return out


def roll_by_gather(mat, dim, shifts: torch.LongTensor):
    # assumes 2D array
    *batch, n_rows, n_cols = mat.shape

    if dim == 0:
        # print(mat)
        arange1 = torch.arange(n_rows).view((n_rows, 1)).repeat((1, n_cols))
        # print(arange1)
        arange2 = (arange1 - shifts) % n_rows
        # print(arange2)
        return torch.gather(mat, -2, arange2.expand(*batch, *arange2.shape))
    elif dim == 1:
        arange1 = torch.arange(n_cols).view((1, n_cols)).repeat((n_rows, 1))
        # print(arange1)
        arange2 = (arange1 - shifts) % n_cols
        # print(arange2)
        return torch.gather(mat, -1, arange2.expand(*batch, n_rows, n_cols))


def vec_td_lambda_advantage_estimate(
    gamma, lmbda, state_value, next_state_value, reward, done
):
    """Vectorized TD(lambda) advantage estimate.

    Args:
        gamma (scalar, Tensor): exponential mean discount. If tensor-valued,
            must be a [Batch x TimeSteps x 1] tensor.
        lmbda (scalar): trajectory discount.
        state_value (Tensor): value function result with old_state input.
            must be a [Batch x TimeSteps x 1] or [Batch x TimeSteps] tensor
        next_state_value (Tensor): value function result with new_state input.
            must be a [Batch x TimeSteps x 1] or [Batch x TimeSteps] tensor
        reward (Tensor): reward of taking actions in the environment.
            must be a [Batch x TimeSteps x 1] or [Batch x TimeSteps] tensor
        done (Tensor): boolean flag for end of episode.
    """
    return (
        vec_td_lambda_return_estimate(gamma, lmbda, next_state_value, reward, done)
        - state_value
    )

def make_gammas_tensor(gamma, T, device, rolling_gamma):
    # some reshaping code vendored from vec_td_lambda_return_estimate
    gamma = gamma.view(-1, T)
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
        gammas = torch.ones(gamma.shape[0], T, T, 1)
        s0 = gamma.unsqueeze(-1).expand(gamma.shape[0], T, T).contiguous()
        s1 = roll_by_gather(s0, 0, shifts=-torch.arange(T))
        s2 = s1.flip(-1).triu().flip(-1).transpose(-2, -1)
        gammas[..., 1:, :] = s2[..., :-1].unsqueeze(-1)
        # torch.testing.assert_close(gammas, gammas2)

    else:
        gammas = torch.ones(*gamma.shape, T + 1, 1, device=device)
        gammas[..., 1:, :] = gamma[..., None, None]
    return gammas

def vec_td_lambda_return_estimate(gamma, lmbda, next_state_value, reward, done, rolling_gamma: Optional[bool]=None):
    """Vectorized TD(lambda) return estimate.

    Args:
        gamma (scalar, Tensor): exponential mean discount. If tensor-valued,
            must be a [Batch x TimeSteps x 1] tensor.
        lmbda (scalar): trajectory discount.
        next_state_value (Tensor): value function result with new_state input.
            must be a [Batch x TimeSteps x 1] tensor
        reward (Tensor): reward of taking actions in the environment.
            must be a [Batch x TimeSteps x 1] or [Batch x TimeSteps] tensor
        done (Tensor): boolean flag for end of episode.
        rolling_gamma (bool, optional): if True, it is assumed that each gamma
            if a gamma tensor is tied to a single event:
              gamma = [g1, g2, g3, g4]
              value = [v1, v2, v3, v4]
              return = [
                v1 + g1 v2 + g1 g2 v3 + g1 g2 g3 v4,
                v2 + g2 v3 + g2 g3 v4,
                v3 + g3 v4,
                v4,
              ]
            if False, it is assumed that each gamma is tied to the upcoming
            trajectory:
              gamma = [g1, g2, g3, g4]
              value = [v1, v2, v3, v4]
              return = [
                v1 + g1 v2 + g1**2 v3 + g**3 v4,
                v2 + g2 v3 + g2**2 v4,
                v3 + g3 v4,
                v4,
              ]
            Default is True.

    """

    shape = next_state_value.shape
    if not shape[-1] == 1:
        raise RuntimeError("last dimension of inputs shape must be singleton")

    T = shape[-2]

    next_state_value = next_state_value.view(-1, 1, T)
    reward = reward.view(-1, 1, T)
    done = done.view(-1, 1, T)

    """Vectorized version of td_lambda_advantage_estimate"""
    device = reward.device
    not_done = 1 - done.to(next_state_value.dtype)
    next_state_value = not_done * next_state_value

    first_below_thr_gamma = None

    if isinstance(gamma, torch.Tensor) and gamma.ndimension() > 0:
        gammas = make_gammas_tensor(gamma, T, device, rolling_gamma)
    else:
        gammas = torch.ones(T + 1, 1, device=device)
        gammas[1:] = gamma

    gammas = torch.cumprod(gammas, -2)

    lambdas = torch.ones(T + 1, 1, device=device)
    lambdas[1:] = lmbda
    lambdas = torch.cumprod(lambdas, -2)

    first_below_thr = gammas < 1e-7
    while first_below_thr.ndimension() > 2:
        # if we have multiple gammas, we only want to truncate if _all_ of
        # the geometric sequences fall below the threshold
        first_below_thr = first_below_thr.all(axis=0)
    if first_below_thr.any():
        first_below_thr_gamma = first_below_thr.nonzero()[0, 0]
    first_below_thr = lambdas < 1e-7
    if first_below_thr.any() and first_below_thr_gamma is not None:
        first_below_thr = max(first_below_thr_gamma, first_below_thr.nonzero()[0, 0])
        gammas = gammas[..., :first_below_thr, :]
        lambdas = lambdas[:first_below_thr]

    gammas, gammas_prime = gammas[..., :-1, :], gammas[..., 1:, :]
    lambdas, lambdas_prime = lambdas[:-1], lambdas[1:]

    rs = _custom_conv1d(reward, gammas * lambdas)
    vs = _custom_conv1d(next_state_value, gammas_prime * lambdas)
    gam_lam = gammas_prime * lambdas_prime
    mask = gam_lam.flip(-2)
    if mask.shape[-2] < next_state_value.shape[-1]:
        mask = torch.cat(
            # [torch.zeros_like(next_state_value[..., : -mask.shape[-2], :]), mask], -2
            [
                torch.zeros(
                    *mask.shape[:-2],
                    next_state_value.shape[-1] - mask.shape[-2],
                    1,
                    device=device
                ),
                mask,
            ],
            -2,
        )
    if gammas.ndimension() > 2:
        vs2 = (
            _custom_conv1d(next_state_value, gam_lam)
            - mask.squeeze(-1)[..., -1:, :] * next_state_value[..., -1:]
        )
    else:
        vs2 = (
            _custom_conv1d(next_state_value, gam_lam)
            - mask.squeeze(-1) * next_state_value[..., -1:]
        )
    return (rs + vs - vs2).view(shape)
