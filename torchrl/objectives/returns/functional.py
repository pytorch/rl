# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple

import torch

__all__ = [
    "generalized_advantage_estimate",
    "td_lambda_advantage_estimate",
    "td_advantage_estimate",
]


def generalized_advantage_estimate(
    gamma: float,
    lamda: float,
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
        lamda (scalar): trajectory discount.
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
    *batch_size, time_steps = not_done.shape[:-1]
    device = state_value.device
    advantage = torch.empty(*batch_size, time_steps, 1, device=device)
    prev_advantage = 0
    for t in reversed(range(time_steps)):
        delta = (
            reward[..., t, :]
            + (gamma * next_state_value[..., t, :] * not_done[..., t, :])
            - state_value[..., t, :]
        )
        prev_advantage = advantage[..., t, :] = delta + (
            gamma * lamda * prev_advantage * not_done[..., t, :]
        )

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


def td_lambda_advantage_estimate(
    gamma: float,
    lamda: float,
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
        lamda (scalar): trajectory discount.
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
    next_state_value = not_done * next_state_value

    returns = torch.empty_like(state_value)

    g = next_state_value[..., -1, :]
    T = returns.shape[-2]

    for i in reversed(range(T)):
        g = returns[..., i, :] = reward[..., i, :] + gamma * (
            (1 - lamda) * next_state_value[..., i, :] + lamda * g
        )
    advantage = returns - state_value
    return advantage


def _custom_conv1d(val, w):
    val_pad = torch.cat(
        [
            val,
            torch.zeros_like(w[1:]),
        ],
        0,
    )

    shape = val.shape
    w = w.squeeze(-1).unsqueeze(0).unsqueeze(0)
    val_pad = val_pad.squeeze(-1).unsqueeze(0).unsqueeze(0)
    out = torch.conv1d(val_pad, w)
    out = out.view(shape)
    return out


def vec_td_lambda_advantage_estimate(
    gamma, lamda, state_value, next_state_value, reward, done
):
    """Vectorized version of td_lambda_advantage_estimate"""
    device = reward.device
    not_done = 1 - done.to(next_state_value.dtype)
    next_state_value = not_done * next_state_value

    T = reward.shape[-2]

    first_below_thr_gamma = None

    gammas = torch.ones(T + 1, 1, device=device)
    gammas[1:] = gamma
    gammas = torch.cumprod(gammas, -2)

    lambdas = torch.ones(T + 1, 1, device=device)
    lambdas[1:] = lamda
    lambdas = torch.cumprod(lambdas, -2)

    first_below_thr = gammas < 1e-7
    if first_below_thr.any():
        first_below_thr_gamma = first_below_thr.nonzero()[0, 0]
    first_below_thr = lambdas < 1e-7
    if first_below_thr.any() and first_below_thr_gamma is not None:
        first_below_thr = max(first_below_thr_gamma, first_below_thr.nonzero()[0, 0])
        gammas = gammas[:first_below_thr]
        lambdas = lambdas[:first_below_thr]

    gammas, gammas_prime = gammas[:-1], gammas[1:]
    lambdas, lambdas_prime = lambdas[:-1], lambdas[1:]

    rs = _custom_conv1d(reward, gammas * lambdas)
    vs = _custom_conv1d(next_state_value, gammas_prime * lambdas)
    whatever = gammas_prime * lambdas_prime
    mask = whatever.flip(-2)
    if mask.shape[-2] < next_state_value.shape[-2]:
        mask = torch.cat(
            [torch.zeros_like(next_state_value[..., : -mask.shape[-2], :]), mask], -2
        )
    vs2 = _custom_conv1d(next_state_value, whatever) - mask * next_state_value[-1]
    return rs + vs - vs2 - state_value
