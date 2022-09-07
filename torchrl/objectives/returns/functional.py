# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple, Optional

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

from torchrl.objectives.returns.utils import _custom_conv1d, _make_gammas_tensor


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
    dtype = state_value.dtype
    not_done = 1 - done.to(dtype)
    *batch_size, time_steps = not_done.shape[:-1]

    gammalmbdas = torch.full_like(not_done, gamma * lmbda) * not_done
    gammalmbdas = _make_gammas_tensor(gammalmbdas, time_steps, True)
    gammalmbdas = gammalmbdas.cumprod(-2)
    # first_below_thr = gammalmbdas < 1e-7
    # # if we have multiple gammas, we only want to truncate if _all_ of
    # # the geometric sequences fall below the threshold
    # first_below_thr = first_below_thr.all(axis=0)
    # if first_below_thr.any():
    #     gammalmbdas = gammalmbdas[..., :first_below_thr, :]

    td0 = reward + not_done * gamma * next_state_value - state_value

    if len(batch_size) > 1:
        td0 = td0.flatten(0, len(batch_size) - 1)
    elif not len(batch_size):
        td0 = td0.unsqueeze(0)

    advantage = _custom_conv1d(td0.transpose(-2, -1), gammalmbdas)

    if len(batch_size) > 1:
        advantage = advantage.unflatten(0, batch_size)
    elif not len(batch_size):
        advantage = advantage.squeeze(0)

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
    rolling_gamma: bool = None,
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

    if not (isinstance(gamma, torch.Tensor) and gamma.shape == not_done.shape):
        gamma = torch.full_like(next_state_value, gamma)
        rolling_gamma = True
    elif rolling_gamma is None:
        rolling_gamma = True

    if not (isinstance(lmbda, torch.Tensor) and lmbda.shape == not_done.shape):
        lmbda = torch.full_like(next_state_value, lmbda)
        rolling_gamma = True
    elif rolling_gamma is None:
        rolling_gamma = True

    if rolling_gamma:
        for i in reversed(range(T)):
            g = returns[..., i, :] = reward[..., i, :] + gamma[..., i, :] * (
                (1 - lmbda[..., i, :]) * next_state_value[..., i, :]
                + lmbda[..., i, :] * g
            )
    else:
        for i in range(T):
            g = next_state_value[..., -1, :]
            for j in reversed(range(i, T)):
                g = returns[..., i, :] = reward[..., j, :] + gamma[..., i, :] * (
                    (1 - lmbda[..., i, :]) * next_state_value[..., j, :]
                    + lmbda[..., i, :] * g
                )

    return returns


def td_lambda_advantage_estimate(
    gamma: float,
    lmbda: float,
    state_value: torch.Tensor,
    next_state_value: torch.Tensor,
    reward: torch.Tensor,
    done: torch.Tensor,
    rolling_gamma: bool = None,
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
    if not state_value.shape == next_state_value.shape:
        raise RuntimeError("shape of state_value and next_state_value must match")
    returns = td_lambda_return_estimate(
        gamma, lmbda, next_state_value, reward, done, rolling_gamma
    )
    advantage = returns - state_value
    return advantage


def vec_td_lambda_advantage_estimate(
    gamma,
    lmbda,
    state_value,
    next_state_value,
    reward,
    done,
    rolling_gamma: bool = None,
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
    return (
        vec_td_lambda_return_estimate(
            gamma, lmbda, next_state_value, reward, done, rolling_gamma
        )
        - state_value
    )


def vec_td_lambda_return_estimate(
    gamma, lmbda, next_state_value, reward, done, rolling_gamma: Optional[bool] = None
):
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
        if rolling_gamma is None:
            rolling_gamma = True
        gammas = _make_gammas_tensor(gamma, T, rolling_gamma)
    else:
        if rolling_gamma is not None:
            raise RuntimeError(
                "rolling_gamma cannot be set if a non-tensor gamma is provided"
            )
        gammas = torch.ones(T + 1, 1, device=device)
        gammas[1:] = gamma

    gammas = torch.cumprod(gammas, -2)

    lambdas = torch.ones(T + 1, 1, device=device)
    lambdas[1:] = lmbda
    lambdas = torch.cumprod(lambdas, -2)

    if not isinstance(gamma, torch.Tensor) or gamma.numel() <= 0:
        first_below_thr = gammas < 1e-7
        while first_below_thr.ndimension() > 2:
            # if we have multiple gammas, we only want to truncate if _all_ of
            # the geometric sequences fall below the threshold
            first_below_thr = first_below_thr.all(axis=0)
        if first_below_thr.any():
            first_below_thr_gamma = first_below_thr.nonzero()[0, 0]
        first_below_thr = lambdas < 1e-7
        if first_below_thr.any() and first_below_thr_gamma is not None:
            first_below_thr = max(
                first_below_thr_gamma, first_below_thr.nonzero()[0, 0]
            )
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
                    device=device,
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
