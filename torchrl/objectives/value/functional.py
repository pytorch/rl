# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from functools import wraps
from typing import Optional, Tuple

import torch
from tensordict import MemmapTensor, TensorDictBase

__all__ = [
    "generalized_advantage_estimate",
    "vec_generalized_advantage_estimate",
    "td0_advantage_estimate",
    "td0_return_estimate",
    "td1_return_estimate",
    "vec_td1_return_estimate",
    "td1_advantage_estimate",
    "vec_td1_advantage_estimate",
    "td_lambda_return_estimate",
    "vec_td_lambda_return_estimate",
    "td_lambda_advantage_estimate",
    "vec_td_lambda_advantage_estimate",
]

from torchrl.objectives.value.utils import _custom_conv1d, _make_gammas_tensor


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


########################################################################
# GAE
# ---


@_transpose_time
def generalized_advantage_estimate(
    gamma: float,
    lmbda: float,
    state_value: torch.Tensor,
    next_state_value: torch.Tensor,
    reward: torch.Tensor,
    done: torch.Tensor,
    time_dim: int = -2,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generalized advantage estimate of a trajectory.

    Refer to "HIGH-DIMENSIONAL CONTINUOUS CONTROL USING GENERALIZED ADVANTAGE ESTIMATION"
    https://arxiv.org/pdf/1506.02438.pdf for more context.

    Args:
        gamma (scalar): exponential mean discount.
        lmbda (scalar): trajectory discount.
        state_value (Tensor): value function result with old_state input.
        next_state_value (Tensor): value function result with new_state input.
        reward (Tensor): reward of taking actions in the environment.
        done (Tensor): boolean flag for end of episode.
        time_dim (int): dimension where the time is unrolled. Defaults to -2.

    All tensors (values, reward and done) must have shape
    ``[*Batch x TimeSteps x *F]``, with ``*F`` feature dimensions.

    """
    if not (next_state_value.shape == state_value.shape == reward.shape == done.shape):
        raise RuntimeError(
            "All input tensors (value, reward and done states) must share a unique shape."
        )
    dtype = next_state_value.dtype
    device = state_value.device
    lastdim = next_state_value.shape[-1]

    not_done = 1 - done.to(dtype)
    *batch_size, time_steps = not_done.shape[:-1]
    advantage = torch.empty(
        *batch_size, time_steps, lastdim, device=device, dtype=dtype
    )
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


@_transpose_time
def vec_generalized_advantage_estimate(
    gamma: float,
    lmbda: float,
    state_value: torch.Tensor,
    next_state_value: torch.Tensor,
    reward: torch.Tensor,
    done: torch.Tensor,
    time_dim: int = -2,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Vectorized Generalized advantage estimate of a trajectory.

    Refer to "HIGH-DIMENSIONAL CONTINUOUS CONTROL USING GENERALIZED ADVANTAGE ESTIMATION"
    https://arxiv.org/pdf/1506.02438.pdf for more context.

    Args:
        gamma (scalar): exponential mean discount.
        lmbda (scalar): trajectory discount.
        state_value (Tensor): value function result with old_state input.
        next_state_value (Tensor): value function result with new_state input.
        reward (Tensor): reward of taking actions in the environment.
        done (Tensor): boolean flag for end of episode.
        time_dim (int): dimension where the time is unrolled. Defaults to -2.

    All tensors (values, reward and done) must have shape
    ``[*Batch x TimeSteps x *F]``, with ``*F`` feature dimensions.

    """
    if not (next_state_value.shape == state_value.shape == reward.shape == done.shape):
        raise RuntimeError(
            "All input tensors (value, reward and done states) must share a unique shape."
        )
    dtype = state_value.dtype
    not_done = 1 - done.to(dtype)
    *batch_size, time_steps, lastdim = not_done.shape

    value = gamma * lmbda
    if isinstance(value, torch.Tensor):
        # create tensor while ensuring that gradients are passed
        gammalmbdas = torch.ones_like(not_done) * not_done * value
    else:
        gammalmbdas = torch.full_like(not_done, value) * not_done
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

    td0_r = td0.transpose(-2, -1)
    shapes = td0_r.shape[:2]
    if lastdim != 1:
        # then we flatten again the first dims and reset a singleton in between
        td0_r = td0_r.flatten(0, 1).unsqueeze(1)
    advantage = _custom_conv1d(td0_r, gammalmbdas)
    if lastdim != 1:
        advantage = advantage.squeeze(1).unflatten(0, shapes)

    if len(batch_size) > 1:
        advantage = advantage.unflatten(0, batch_size)
    elif not len(batch_size):
        advantage = advantage.squeeze(0)

    advantage = advantage.transpose(-2, -1)
    value_target = advantage + state_value
    return advantage, value_target


########################################################################
# TD(0)
# -----


def td0_advantage_estimate(
    gamma: float,
    state_value: torch.Tensor,
    next_state_value: torch.Tensor,
    reward: torch.Tensor,
    done: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """TD(0) advantage estimate of a trajectory.

    Also known as bootstrapped Temporal Difference or one-step return.

    Args:
        gamma (scalar): exponential mean discount.
        state_value (Tensor): value function result with old_state input.
        next_state_value (Tensor): value function result with new_state input.
        reward (Tensor): reward of taking actions in the environment.
        done (Tensor): boolean flag for end of episode.

    All tensors (values, reward and done) must have shape
    ``[*Batch x TimeSteps x *F]``, with ``*F`` feature dimensions.

    """
    if not (next_state_value.shape == state_value.shape == reward.shape == done.shape):
        raise RuntimeError(
            "All input tensors (value, reward and done states) must share a unique shape."
        )
    returns = td0_return_estimate(gamma, next_state_value, reward, done)
    advantage = returns - state_value
    return advantage


def td0_return_estimate(
    gamma: float,
    next_state_value: torch.Tensor,
    reward: torch.Tensor,
    done: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """TD(0) discounted return estimate of a trajectory.

    Also known as bootstrapped Temporal Difference or one-step return.

    Args:
        gamma (scalar): exponential mean discount.
        next_state_value (Tensor): value function result with new_state input.
            must be a [Batch x TimeSteps x 1] or [Batch x TimeSteps] tensor
        reward (Tensor): reward of taking actions in the environment.
            must be a [Batch x TimeSteps x 1] or [Batch x TimeSteps] tensor
        done (Tensor): boolean flag for end of episode.

    All tensors (values, reward and done) must have shape
    ``[*Batch x TimeSteps x *F]``, with ``*F`` feature dimensions.

    """
    if not (next_state_value.shape == reward.shape == done.shape):
        raise RuntimeError(
            "All input tensors (value, reward and done states) must share a unique shape."
        )
    not_done = 1 - done.to(next_state_value.dtype)
    advantage = reward + gamma * not_done * next_state_value
    return advantage


########################################################################
# TD(1)
# ----------


@_transpose_time
def td1_return_estimate(
    gamma: float,
    next_state_value: torch.Tensor,
    reward: torch.Tensor,
    done: torch.Tensor,
    rolling_gamma: bool = None,
    time_dim: int = -2,
) -> torch.Tensor:
    r"""TD(1) return estimate.

    Args:
        gamma (scalar): exponential mean discount.
        next_state_value (Tensor): value function result with new_state input.
        reward (Tensor): reward of taking actions in the environment.
        done (Tensor): boolean flag for end of episode.
        rolling_gamma (bool, optional): if ``True``, it is assumed that each gamma
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
        time_dim (int): dimension where the time is unrolled. Defaults to -2.

    All tensors (values, reward and done) must have shape
    ``[*Batch x TimeSteps x *F]``, with ``*F`` feature dimensions.

    """
    if not (next_state_value.shape == reward.shape == done.shape):
        raise RuntimeError(
            "All input tensors (value, reward and done states) must share a unique shape."
        )
    not_done = 1 - done.to(next_state_value.dtype)

    returns = torch.empty_like(next_state_value)

    T = returns.shape[-2]

    single_gamma = False
    if not (isinstance(gamma, torch.Tensor) and gamma.shape == not_done.shape):
        single_gamma = True
        gamma = torch.full_like(next_state_value, gamma)

    if rolling_gamma is None:
        rolling_gamma = True
    elif not rolling_gamma and single_gamma:
        raise RuntimeError(
            "rolling_gamma=False is expected only with time-sensitive gamma values"
        )

    if rolling_gamma:
        gamma = gamma * not_done
        g = next_state_value[..., -1, :]
        for i in reversed(range(T)):
            g = returns[..., i, :] = reward[..., i, :] + gamma[..., i, :] * g
    else:
        for k in range(T):
            g = next_state_value[..., -1, :]
            _gamma = gamma[..., k, :]
            nd = not_done
            _gamma = _gamma.unsqueeze(-2) * nd
            for i in reversed(range(k, T)):
                g = reward[..., i, :] + _gamma[..., i, :] * g
            returns[..., k, :] = g
    return returns


def td1_advantage_estimate(
    gamma: float,
    state_value: torch.Tensor,
    next_state_value: torch.Tensor,
    reward: torch.Tensor,
    done: torch.Tensor,
    rolling_gamma: bool = None,
    time_dim: int = -2,
) -> torch.Tensor:
    """TD(1) advantage estimate.

    Args:
        gamma (scalar): exponential mean discount.
        state_value (Tensor): value function result with old_state input.
        next_state_value (Tensor): value function result with new_state input.
        reward (Tensor): reward of taking actions in the environment.
        done (Tensor): boolean flag for end of episode.
        rolling_gamma (bool, optional): if ``True``, it is assumed that each gamma
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
        time_dim (int): dimension where the time is unrolled. Defaults to -2.

    All tensors (values, reward and done) must have shape
    ``[*Batch x TimeSteps x *F]``, with ``*F`` feature dimensions.

    """
    if not (next_state_value.shape == state_value.shape == reward.shape == done.shape):
        raise RuntimeError(
            "All input tensors (value, reward and done states) must share a unique shape."
        )
    if not state_value.shape == next_state_value.shape:
        raise RuntimeError("shape of state_value and next_state_value must match")
    returns = td1_return_estimate(
        gamma, next_state_value, reward, done, rolling_gamma, time_dim=time_dim
    )
    advantage = returns - state_value
    return advantage


@_transpose_time
def vec_td1_return_estimate(
    gamma,
    next_state_value,
    reward,
    done,
    rolling_gamma: Optional[bool] = None,
    time_dim: int = -2,
):
    """Vectorized TD(1) return estimate.

    Args:
        gamma (scalar, Tensor): exponential mean discount. If tensor-valued,
        next_state_value (Tensor): value function result with new_state input.
        reward (Tensor): reward of taking actions in the environment.
        done (Tensor): boolean flag for end of episode.
        rolling_gamma (bool, optional): if ``True``, it is assumed that each gamma
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
        time_dim (int): dimension where the time is unrolled. Defaults to -2.

    All tensors (values, reward and done) must have shape
    ``[*Batch x TimeSteps x *F]``, with ``*F`` feature dimensions.

    """
    return vec_td_lambda_return_estimate(
        gamma=gamma,
        next_state_value=next_state_value,
        reward=reward,
        done=done,
        rolling_gamma=rolling_gamma,
        lmbda=1,
        time_dim=time_dim,
    )


def vec_td1_advantage_estimate(
    gamma,
    state_value,
    next_state_value,
    reward,
    done,
    rolling_gamma: bool = None,
    time_dim: int = -2,
):
    """Vectorized TD(1) advantage estimate.

    Args:
        gamma (scalar, Tensor): exponential mean discount. If tensor-valued,
        state_value (Tensor): value function result with old_state input.
        next_state_value (Tensor): value function result with new_state input.
        reward (Tensor): reward of taking actions in the environment.
        done (Tensor): boolean flag for end of episode.
        rolling_gamma (bool, optional): if ``True``, it is assumed that each gamma
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
        time_dim (int): dimension where the time is unrolled. Defaults to -2.

    All tensors (values, reward and done) must have shape
    ``[*Batch x TimeSteps x *F]``, with ``*F`` feature dimensions.

    """
    if not (next_state_value.shape == state_value.shape == reward.shape == done.shape):
        raise RuntimeError(
            "All input tensors (value, reward and done states) must share a unique shape."
        )
    return (
        vec_td1_return_estimate(
            gamma, next_state_value, reward, done, rolling_gamma, time_dim=time_dim
        )
        - state_value
    )


########################################################################
# TD(lambda)
# ----------


@_transpose_time
def td_lambda_return_estimate(
    gamma: float,
    lmbda: float,
    next_state_value: torch.Tensor,
    reward: torch.Tensor,
    done: torch.Tensor,
    rolling_gamma: bool = None,
    time_dim: int = -2,
) -> torch.Tensor:
    r"""TD(:math:`\lambda`) return estimate.

    Args:
        gamma (scalar): exponential mean discount.
        lmbda (scalar): trajectory discount.
        next_state_value (Tensor): value function result with new_state input.
        reward (Tensor): reward of taking actions in the environment.
        done (Tensor): boolean flag for end of episode.
        rolling_gamma (bool, optional): if ``True``, it is assumed that each gamma
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
        time_dim (int): dimension where the time is unrolled. Defaults to -2.

    All tensors (values, reward and done) must have shape
    ``[*Batch x TimeSteps x *F]``, with ``*F`` feature dimensions.

    """
    if not (next_state_value.shape == reward.shape == done.shape):
        raise RuntimeError(
            "All input tensors (value, reward and done states) must share a unique shape."
        )

    not_done = 1 - done.to(next_state_value.dtype)

    returns = torch.empty_like(next_state_value)

    *batch, T, lastdim = returns.shape

    # if gamma is not a tensor of the same shape as other inputs, we use rolling_gamma = True
    single_gamma = False
    if not (isinstance(gamma, torch.Tensor) and gamma.shape == not_done.shape):
        single_gamma = True
        gamma = torch.full_like(next_state_value, gamma)

    single_lambda = False
    if not (isinstance(lmbda, torch.Tensor) and lmbda.shape == not_done.shape):
        single_lambda = True
        lmbda = torch.full_like(next_state_value, lmbda)

    if rolling_gamma is None:
        rolling_gamma = True
    elif not rolling_gamma and single_gamma and single_lambda:
        raise RuntimeError(
            "rolling_gamma=False is expected only with time-sensitive gamma or lambda values"
        )

    if rolling_gamma:
        gamma = gamma * not_done
        g = next_state_value[..., -1, :]
        for i in reversed(range(T)):
            g = returns[..., i, :] = reward[..., i, :] + gamma[..., i, :] * (
                (1 - lmbda[..., i, :]) * next_state_value[..., i, :]
                + lmbda[..., i, :] * g
            )
    else:
        for k in range(T):
            g = next_state_value[..., -1, :]
            _gamma = gamma[..., k, :]
            _lambda = lmbda[..., k, :]
            nd = not_done
            _gamma = _gamma.unsqueeze(-2) * nd
            for i in reversed(range(k, T)):
                g = reward[..., i, :] + _gamma[..., i, :] * (
                    (1 - _lambda) * next_state_value[..., i, :] + _lambda * g
                )
            returns[..., k, :] = g

    return returns


def td_lambda_advantage_estimate(
    gamma: float,
    lmbda: float,
    state_value: torch.Tensor,
    next_state_value: torch.Tensor,
    reward: torch.Tensor,
    done: torch.Tensor,
    rolling_gamma: bool = None,
    time_dim: int = -2,
) -> torch.Tensor:
    r"""TD(:math:`\lambda`) advantage estimate.

    Args:
        gamma (scalar): exponential mean discount.
        lmbda (scalar): trajectory discount.
        state_value (Tensor): value function result with old_state input.
        next_state_value (Tensor): value function result with new_state input.
        reward (Tensor): reward of taking actions in the environment.
        done (Tensor): boolean flag for end of episode.
        rolling_gamma (bool, optional): if ``True``, it is assumed that each gamma
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
        time_dim (int): dimension where the time is unrolled. Defaults to -2.

    All tensors (values, reward and done) must have shape
    ``[*Batch x TimeSteps x *F]``, with ``*F`` feature dimensions.

    """
    if not (next_state_value.shape == state_value.shape == reward.shape == done.shape):
        raise RuntimeError(
            "All input tensors (value, reward and done states) must share a unique shape."
        )
    if not state_value.shape == next_state_value.shape:
        raise RuntimeError("shape of state_value and next_state_value must match")
    returns = td_lambda_return_estimate(
        gamma, lmbda, next_state_value, reward, done, rolling_gamma, time_dim=time_dim
    )
    advantage = returns - state_value
    return advantage


@_transpose_time
def vec_td_lambda_return_estimate(
    gamma,
    lmbda,
    next_state_value,
    reward,
    done,
    rolling_gamma: Optional[bool] = None,
    time_dim: int = -2,
):
    r"""Vectorized TD(:math:`\lambda`) return estimate.

    Args:
        gamma (scalar, Tensor): exponential mean discount. If tensor-valued,
            must be a [Batch x TimeSteps x 1] tensor.
        lmbda (scalar): trajectory discount.
        next_state_value (Tensor): value function result with new_state input.
            must be a [Batch x TimeSteps x 1] tensor
        reward (Tensor): reward of taking actions in the environment.
            must be a [Batch x TimeSteps x 1] or [Batch x TimeSteps] tensor
        done (Tensor): boolean flag for end of episode.
        rolling_gamma (bool, optional): if ``True``, it is assumed that each gamma
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
        time_dim (int): dimension where the time is unrolled. Defaults to -2.

    All tensors (values, reward and done) must have shape
    ``[*Batch x TimeSteps x *F]``, with ``*F`` feature dimensions.

    """
    if not (next_state_value.shape == reward.shape == done.shape):
        raise RuntimeError(
            "All input tensors (value, reward and done states) must share a unique shape."
        )
    shape = next_state_value.shape

    *batch, T, lastdim = shape

    next_state_value = next_state_value.transpose(-2, -1).unsqueeze(-2)
    if len(batch):
        next_state_value = next_state_value.flatten(0, len(batch))

    reward = reward.transpose(-2, -1).unsqueeze(-2)
    if len(batch):
        reward = reward.flatten(0, len(batch))

    """Vectorized version of td_lambda_advantage_estimate"""
    device = reward.device
    not_done = 1 - done.to(next_state_value.dtype)

    first_below_thr_gamma = None

    # 3 use cases: (1) there is one gamma per time step, (2) there is a single gamma but
    # some intermediate dones and (3) there is a single gamma and no done.
    # (3) can be treated much faster than (1) and (2) (lower mem footprint)
    if (isinstance(gamma, torch.Tensor) and gamma.numel() > 1) or done.any():
        if rolling_gamma is None:
            rolling_gamma = True
        if rolling_gamma:
            gamma = gamma * not_done
        gammas = _make_gammas_tensor(gamma, T, rolling_gamma)

        if not rolling_gamma:
            done_follows_done = done[..., 1:, :][done[..., :-1, :]].all()
            if not done_follows_done:
                raise NotImplementedError(
                    "When using rolling_gamma=False and vectorized TD(lambda), "
                    "make sure that conseducitve trajectories are separated as different batch "
                    "items. Propagating a gamma value across trajectories is not permitted with "
                    "this method. Check that you need to use rolling_gamma=False, and if so "
                    "consider using the non-vectorized version of the return computation or splitting "
                    "your trajectories."
                )
            else:
                gammas[..., 1:, :] = gammas[..., 1:, :] * not_done.view(-1, 1, T, 1)

    else:
        if rolling_gamma is not None:
            raise RuntimeError(
                "rolling_gamma cannot be set if a non-tensor gamma is provided"
            )
        gammas = torch.ones(T + 1, 1, device=device)
        gammas[1:] = gamma

    gammas_cp = torch.cumprod(gammas, -2)

    lambdas = torch.ones(T + 1, 1, device=device)
    lambdas[1:] = lmbda
    lambdas_cp = torch.cumprod(lambdas, -2)

    if not isinstance(gamma, torch.Tensor) or gamma.numel() <= 0:
        first_below_thr = gammas_cp < 1e-7
        while first_below_thr.ndimension() > 2:
            # if we have multiple gammas, we only want to truncate if _all_ of
            # the geometric sequences fall below the threshold
            first_below_thr = first_below_thr.all(axis=0)
        if first_below_thr.any():
            first_below_thr_gamma = first_below_thr.nonzero()[0, 0]
        first_below_thr = lambdas_cp < 1e-7
        if first_below_thr.any() and first_below_thr_gamma is not None:
            first_below_thr = max(
                first_below_thr_gamma, first_below_thr.nonzero()[0, 0]
            )
            gammas_cp = gammas_cp[..., :first_below_thr, :]
            lambdas_cp = lambdas_cp[:first_below_thr]

    gammas = gammas[..., 1:, :]
    lambdas = lambdas[1:]

    dec = gammas_cp * lambdas_cp
    if rolling_gamma in (None, True):
        if gammas.ndimension() == 4 and gammas.shape[1] > 1:
            gammas = gammas[:, :1]
        if lambdas.ndimension() == 4 and lambdas.shape[1] > 1:
            lambdas = lambdas[:, :1]
        v3 = (gammas * lambdas).squeeze(-1) * next_state_value
        v3[..., :-1] = 0
        out = _custom_conv1d(
            reward + (gammas * (1 - lambdas)).squeeze(-1) * next_state_value + v3, dec
        )
        return out.view(*batch, lastdim, T).transpose(-2, -1)
    else:
        v1 = _custom_conv1d(reward, dec)

        if gammas.ndimension() == 4 and gammas.shape[1] > 1:
            gammas = gammas[:, :, :1].transpose(1, 2)
        if lambdas.ndimension() == 4 and lambdas.shape[1] > 1:
            lambdas = lambdas[:, :, :1].transpose(1, 2)

        v2 = _custom_conv1d(
            next_state_value * not_done.view_as(next_state_value),
            dec * (gammas * (1 - lambdas)).transpose(1, 2),
        )
        v3 = next_state_value * not_done.view_as(next_state_value)
        v3[..., :-1] = 0
        v3 = _custom_conv1d(v3, dec * (gammas * lambdas).transpose(1, 2))
        return (v1 + v2 + v3).view(*batch, lastdim, T).transpose(-2, -1)


def vec_td_lambda_advantage_estimate(
    gamma,
    lmbda,
    state_value,
    next_state_value,
    reward,
    done,
    rolling_gamma: bool = None,
    time_dim: int = -2,
):
    r"""Vectorized TD(:math:`\lambda`) advantage estimate.

    Args:
        gamma (scalar, Tensor): exponential mean discount. If tensor-valued,
        lmbda (scalar): trajectory discount.
        state_value (Tensor): value function result with old_state input.
        next_state_value (Tensor): value function result with new_state input.
        reward (Tensor): reward of taking actions in the environment.
        done (Tensor): boolean flag for end of episode.
        rolling_gamma (bool, optional): if ``True``, it is assumed that each gamma
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
        time_dim (int): dimension where the time is unrolled. Defaults to -2.

    All tensors (values, reward and done) must have shape
    ``[*Batch x TimeSteps x *F]``, with ``*F`` feature dimensions.

    """
    if not (next_state_value.shape == state_value.shape == reward.shape == done.shape):
        raise RuntimeError(
            "All input tensors (value, reward and done states) must share a unique shape."
        )
    return (
        vec_td_lambda_return_estimate(
            gamma,
            lmbda,
            next_state_value,
            reward,
            done,
            rolling_gamma,
            time_dim=time_dim,
        )
        - state_value
    )


########################################################################
# Reward to go
# ------------


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
    num_per_traj = torch.ones_like(dones_and_truncated).cumsum(0)[dones_and_truncated]
    num_per_traj[1:] -= num_per_traj[:-1].clone()
    return num_per_traj


def _get_num_per_traj_init(is_init):
    """Like _get_num_per_traj, but with is_init signal."""
    done = torch.zeros_like(is_init)
    done[..., :-1][is_init[..., 1:]] = 1
    return _get_num_per_traj(done)


def _split_and_pad_sequence(tensor, splits):
    """Given a tensor of size [B, T, *other] and the corresponding traj lengths (flattened), returns the padded trajectories [NPad, Tmax, *other].

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
    tensor = _flatten_batch(tensor)
    max_val = max(splits)
    mask = torch.zeros(len(splits), max_val, dtype=torch.bool, device=tensor.device)
    mask.scatter_(
        index=max_val - torch.tensor(splits, device=tensor.device).unsqueeze(-1),
        dim=1,
        value=1,
    )
    mask = mask.cumsum(-1).flip(-1).bool()

    def _fill_tensor(tensor):
        empty_tensor = torch.zeros(
            len(splits),
            max_val,
            *tensor.shape[1:],
            dtype=tensor.dtype,
            device=tensor.device,
        )
        empty_tensor[mask] = tensor
        return empty_tensor

    if isinstance(tensor, TensorDictBase):
        tensor = tensor.apply(_fill_tensor, batch_size=[len(splits), max_val])
    else:
        tensor = _fill_tensor(tensor)
    return tensor


def _inv_pad_sequence(tensor, splits):
    """Inverses a pad_sequence operation.

    Examples:
        >>> rewards = torch.randn(100, 20)
        >>> num_per_traj = _get_num_per_traj(torch.zeros(100, 20).bernoulli_(0.1))
        >>> padded = _split_and_pad_sequence(rewards, num_per_traj.tolist())
        >>> reconstructed = _inv_pad_sequence(padded, num_per_traj)
        >>> assert (reconstructed==rewards).all()


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
        >>> td = _split_and_pad_sequence(tensordict, splits)
        >>> assert (_inv_pad_sequence(td, splits).view(tensordict.shape) == tensordict).all()

    """
    offset = torch.ones_like(splits) * tensor.shape[-1]
    offset[0] = 0
    offset = offset.cumsum(0)
    z = torch.zeros(tensor.numel(), dtype=torch.bool, device=offset.device)

    ones = offset + splits
    ones = ones[ones < tensor.numel()]
    # while ones[-1] == tensor.numel():
    #     ones = ones[:-1]
    z[ones] = 1
    z_idx = z[offset[1:]]
    z[offset[1:]] = torch.bitwise_xor(
        z_idx, torch.ones_like(z_idx)
    )  # make sure that the longest is accounted for
    idx = z.cumsum(0) % 2 == 0
    return tensor.reshape(-1)[idx]


@_transpose_time
def reward2go(
    reward,
    done,
    gamma,
    time_dim: int = -2,
):
    """Compute the discounted cumulative sum of rewards given multiple trajectories and the episode ends.

    Args:
        reward (torch.Tensor): A tensor containing the rewards
            received at each time step over multiple trajectories.
        done (torch.Tensor): A tensor with done (or truncated) states.
        gamma (float, optional): The discount factor to use for computing the
            discounted cumulative sum of rewards. Defaults to 1.0.
        time_dim (int): dimension where the time is unrolled. Defaults to -2.

    Returns:
        torch.Tensor: A tensor of shape [B, T] containing the discounted cumulative
            sum of rewards (reward-to-go) at each time step.

    Examples:
        >>> reward = torch.ones(1, 10)
        >>> done = torch.zeros(1, 10, dtype=torch.bool)
        >>> done[:, [3, 7]] = True
        >>> reward2go(reward, done, 0.99, time_dim=-1)
        tensor([[3.9404],
                [2.9701],
                [1.9900],
                [1.0000],
                [3.9404],
                [2.9701],
                [1.9900],
                [1.0000],
                [1.9900],
                [1.0000]])

    """
    shape = reward.shape
    if shape != done.shape:
        raise ValueError(
            f"reward and done must share the same shape, got {reward.shape} and {done.shape}"
        )
    # place time at dim -1
    reward = reward.transpose(-2, -1)
    done = done.transpose(-2, -1)
    # flatten if needed
    if reward.ndim > 2:
        reward = reward.flatten(0, -2)
        done = done.flatten(0, -2)

    num_per_traj = _get_num_per_traj(done)
    td0_flat = _split_and_pad_sequence(reward, num_per_traj)
    gammas = torch.ones_like(td0_flat[0])
    gammas[1:] = gamma
    gammas[1:] = gammas[1:].cumprod(0)
    gammas = gammas.unsqueeze(-1)
    cumsum = _custom_conv1d(td0_flat.unsqueeze(1), gammas)
    cumsum = _inv_pad_sequence(cumsum, num_per_traj)
    cumsum = cumsum.view_as(reward)
    if cumsum.shape != shape:
        cumsum = cumsum.view(shape)
    return cumsum
