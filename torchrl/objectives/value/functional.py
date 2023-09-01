# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from functools import wraps
from typing import Optional, Tuple, Union

import torch

from tensordict import MemmapTensor

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

from torchrl.objectives.value.utils import (
    _custom_conv1d,
    _get_num_per_traj,
    _inv_pad_sequence,
    _make_gammas_tensor,
    _split_and_pad_sequence,
)

SHAPE_ERR = (
    "All input tensors (value, reward and done states) must share a unique shape."
)


def _transpose_time(fun):
    """Checks the time_dim argument of the function to allow for any dim.

    If not -2, makes a transpose of all the multi-dim input tensors to bring
    time at -2, and does the opposite transform for the outputs.
    """
    ERROR = (
        "The tensor shape and the time dimension are not compatible: "
        "got {} and time_dim={}."
    )

    @wraps(fun)
    def transposed_fun(*args, time_dim=-2, **kwargs):
        def transpose_tensor(tensor):
            if (
                not isinstance(tensor, (torch.Tensor, MemmapTensor))
                or tensor.numel() <= 1
            ):
                return tensor, False
            if time_dim >= 0:
                timedim = time_dim - tensor.ndim
            else:
                timedim = time_dim
            if timedim < -tensor.ndim or timedim >= 0:
                raise RuntimeError(ERROR.format(tensor.shape, timedim))
            if tensor.ndim >= 2:
                single_dim = False
                tensor = tensor.transpose(timedim, -2)
            elif tensor.ndim == 1 and timedim == -1:
                single_dim = True
                tensor = tensor.unsqueeze(-1)
            else:
                raise RuntimeError(ERROR.format(tensor.shape, timedim))
            return tensor, single_dim

        if time_dim != -2:
            args, single_dim = zip(*(transpose_tensor(arg) for arg in args))
            single_dim = any(single_dim)
            for k, item in kwargs.items():
                item, sd = transpose_tensor(item)
                single_dim = single_dim or sd
                kwargs[k] = item
            out = fun(*args, time_dim=-2, **kwargs)
            if isinstance(out, torch.Tensor):
                out = transpose_tensor(out)[0]
                if single_dim:
                    out = out.squeeze(-2)
                return out
            if single_dim:
                return tuple(transpose_tensor(_out)[0].squeeze(-2) for _out in out)
            return tuple(transpose_tensor(_out)[0] for _out in out)
        out = fun(*args, time_dim=time_dim, **kwargs)
        if isinstance(out, tuple):
            for _out in out:
                if _out.ndim < 2:
                    raise RuntimeError(ERROR.format(_out.shape, time_dim))
        else:
            if out.ndim < 2:
                raise RuntimeError(ERROR.format(out.shape, time_dim))
        return out

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
        raise RuntimeError(SHAPE_ERR)
    dtype = next_state_value.dtype
    device = state_value.device

    not_done = (~done).int()
    *batch_size, time_steps, lastdim = not_done.shape
    advantage = torch.empty(
        *batch_size, time_steps, lastdim, device=device, dtype=dtype
    )
    prev_advantage = 0
    gnotdone = gamma * not_done
    delta = reward + (gnotdone * next_state_value) - state_value
    discount = lmbda * gnotdone
    for t in reversed(range(time_steps)):
        prev_advantage = advantage[..., t, :] = delta[..., t, :] + (
            prev_advantage * discount[..., t, :]
        )

    value_target = advantage + state_value

    return advantage, value_target


def _geom_series_like(t, r, thr):
    """Creates a geometric series of the form [1, gammalmbda, gammalmbda**2] with the shape of `t`.

    Drops all elements which are smaller than `thr`.
    """
    if isinstance(r, torch.Tensor):
        r = r.item()

    if r == 0.0:
        return torch.zeros_like(t)
    elif r >= 1.0:
        lim = t.numel()
    else:
        lim = int(math.log(thr) / math.log(r))

    rs = torch.full_like(t[:lim], r)
    rs[0] = 1.0
    rs = rs.cumprod(0)
    rs = rs.unsqueeze(-1)
    return rs


def _fast_vec_gae(
    reward: torch.Tensor,
    state_value: torch.Tensor,
    next_state_value: torch.Tensor,
    done: torch.Tensor,
    gamma: float,
    lmbda: float,
    thr: float = 1e-7,
):
    """Fast vectorized Generalized Advantage Estimate when gamma and lmbda are scalars.

    In contrast to `vec_generalized_advantage_estimate` this function does not need
    to allocate a big tensor of the form [B, T, T].

    Args:
        reward (torch.Tensor): a [*B, T, F] tensor containing rewards
        state_value (torch.Tensor): a [*B, T, F] tensor containing state values (value function)
        next_state_value (torch.Tensor): a [*B, T, F] tensor containing next state values (value function)
        done (torch.Tensor): a [B, T] boolean tensor containing the done states
        gamma (scalar): the gamma decay (trajectory discount)
        lmbda (scalar): the lambda decay (exponential mean discount)
        thr (float): threshold for the filter. Below this limit, components will ignored.
            Defaults to 1e-7.

    All tensors (values, reward and done) must have shape
    ``[*Batch x TimeSteps x F]``, with ``F`` feature dimensions.

    """
    # _gen_num_per_traj and _split_and_pad_sequence need
    # time dimension at last position
    done = done.transpose(-2, -1)
    reward = reward.transpose(-2, -1)
    state_value = state_value.transpose(-2, -1)
    next_state_value = next_state_value.transpose(-2, -1)

    gammalmbda = gamma * lmbda
    not_done = (~done).int()
    td0 = reward + not_done * gamma * next_state_value - state_value

    num_per_traj = _get_num_per_traj(done)
    td0_flat, mask = _split_and_pad_sequence(td0, num_per_traj, return_mask=True)

    gammalmbdas = _geom_series_like(td0_flat[0], gammalmbda, thr=thr)

    advantage = _custom_conv1d(td0_flat.unsqueeze(1), gammalmbdas)
    advantage = advantage.squeeze(1)
    advantage = advantage[mask].view_as(reward)

    value_target = advantage + state_value

    advantage = advantage.transpose(-1, -2)
    value_target = value_target.transpose(-1, -2)

    return advantage, value_target


@_transpose_time
def vec_generalized_advantage_estimate(
    gamma: Union[float, torch.Tensor],
    lmbda: Union[float, torch.Tensor],
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
        raise RuntimeError(SHAPE_ERR)
    dtype = state_value.dtype
    not_done = (~done).to(dtype)
    *batch_size, time_steps, lastdim = not_done.shape

    value = gamma * lmbda

    if isinstance(value, torch.Tensor) and value.numel() > 1:
        # create tensor while ensuring that gradients are passed
        gammalmbdas = not_done * value
    else:
        # when gamma and lmbda are scalars, use fast_vec_gae implementation
        return _fast_vec_gae(
            reward=reward,
            state_value=state_value,
            next_state_value=next_state_value,
            done=done,
            gamma=gamma,
            lmbda=lmbda,
        )

    gammalmbdas = _make_gammas_tensor(gammalmbdas, time_steps, True)
    gammalmbdas = gammalmbdas.cumprod(-2)

    first_below_thr = gammalmbdas < 1e-7
    # if we have multiple gammas, we only want to truncate if _all_ of
    # the geometric sequences fall below the threshold
    first_below_thr = first_below_thr.flatten(0, 1).all(0).all(-1)
    if first_below_thr.any():
        first_below_thr = torch.where(first_below_thr)[0][0].item()
        gammalmbdas = gammalmbdas[..., :first_below_thr, :]

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
        raise RuntimeError(SHAPE_ERR)
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
        raise RuntimeError(SHAPE_ERR)
    not_done = (~done).int()
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
        raise RuntimeError(SHAPE_ERR)
    not_done = (~done).int()

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
        raise RuntimeError(SHAPE_ERR)
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
        raise RuntimeError(SHAPE_ERR)
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
        raise RuntimeError(SHAPE_ERR)

    not_done = (~done).int()

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
        raise RuntimeError(SHAPE_ERR)
    if not state_value.shape == next_state_value.shape:
        raise RuntimeError("shape of state_value and next_state_value must match")
    returns = td_lambda_return_estimate(
        gamma, lmbda, next_state_value, reward, done, rolling_gamma, time_dim=time_dim
    )
    advantage = returns - state_value
    return advantage


def _fast_td_lambda_return_estimate(
    gamma: Union[torch.Tensor, float],
    lmbda: float,
    next_state_value: torch.Tensor,
    reward: torch.Tensor,
    done: torch.Tensor,
    thr: float = 1e-7,
):
    """Fast vectorized TD lambda return estimate.

    In contrast to the generalized `vec_td_lambda_return_estimate` this function does not need
    to allocate a big tensor of the form [B, T, T], but it only works with gamma/lmbda being scalars.

    Args:
        gamma (scalar): the gamma decay, can be a tensor with a single element (trajectory discount)
        lmbda (scalar): the lambda decay (exponential mean discount)
        next_state_value (torch.Tensor): a [*B, T, F] tensor containing next state values (value function)
        reward (torch.Tensor): a [*B, T, F] tensor containing rewards
        done (torch.Tensor): a [B, T] boolean tensor containing the done states
        thr (float): threshold for the filter. Below this limit, components will ignored.
            Defaults to 1e-7.

    All tensors (values, reward and done) must have shape
    ``[*Batch x TimeSteps x F]``, with ``F`` feature dimensions.

    """
    device = reward.device
    done = done.transpose(-2, -1)
    reward = reward.transpose(-2, -1)
    next_state_value = next_state_value.transpose(-2, -1)

    gamma_tensor = torch.tensor([gamma], device=device)
    gammalmbda = gamma_tensor * lmbda

    not_done = (~done).int()
    num_per_traj = _get_num_per_traj(done)
    nvalue_ndone = not_done * next_state_value

    t = nvalue_ndone * gamma_tensor * (1 - lmbda) + reward
    v3 = torch.zeros_like(t, device=device)
    v3[..., -1] = nvalue_ndone[..., -1].clone()

    t_flat, mask = _split_and_pad_sequence(
        t + v3 * gammalmbda, num_per_traj, return_mask=True
    )

    gammalmbdas = _geom_series_like(t_flat[0], gammalmbda, thr=thr)

    ret_flat = _custom_conv1d(t_flat.unsqueeze(1), gammalmbdas)
    ret = ret_flat.squeeze(1)[mask]

    return ret.view_as(reward).transpose(-1, -2)


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
        raise RuntimeError(SHAPE_ERR)

    gamma_thr = 1e-7
    shape = next_state_value.shape

    *batch, T, lastdim = shape

    def _is_scalar(tensor):
        return not isinstance(tensor, torch.Tensor) or tensor.numel() == 1

    # There are two use-cases: if gamma/lmbda are scalars we can use the
    # fast implementation, if not we must construct a gamma tensor.
    if _is_scalar(gamma) and _is_scalar(lmbda):
        return _fast_td_lambda_return_estimate(
            gamma=gamma,
            lmbda=lmbda,
            next_state_value=next_state_value,
            reward=reward,
            done=done,
            thr=gamma_thr,
        )

    next_state_value = next_state_value.transpose(-2, -1).unsqueeze(-2)
    if len(batch):
        next_state_value = next_state_value.flatten(0, len(batch))

    reward = reward.transpose(-2, -1).unsqueeze(-2)
    if len(batch):
        reward = reward.flatten(0, len(batch))

    """Vectorized version of td_lambda_advantage_estimate"""
    device = reward.device
    not_done = (~done).int()

    if rolling_gamma is None:
        rolling_gamma = True
    if rolling_gamma:
        gamma = gamma * not_done
    gammas = _make_gammas_tensor(gamma, T, rolling_gamma)

    if not rolling_gamma:
        done_follows_done = done[..., 1:, :][done[..., :-1, :]].all()
        if not done_follows_done:
            raise NotImplementedError(
                "When using rolling_gamma=False and vectorized TD(lambda) with time-dependent gamma, "
                "make sure that conseducitve trajectories are separated as different batch "
                "items. Propagating a gamma value across trajectories is not permitted with "
                "this method. Check that you need to use rolling_gamma=False, and if so "
                "consider using the non-vectorized version of the return computation or splitting "
                "your trajectories."
            )
        else:
            gammas[..., 1:, :] = gammas[..., 1:, :] * not_done.view(-1, 1, T, 1)

    gammas_cp = torch.cumprod(gammas, -2)

    lambdas = torch.ones(T + 1, 1, device=device)
    lambdas[1:] = lmbda
    lambdas_cp = torch.cumprod(lambdas, -2)

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
        raise RuntimeError(SHAPE_ERR)
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
    gammas = _geom_series_like(td0_flat[0], gamma, thr=1e-7)
    cumsum = _custom_conv1d(td0_flat.unsqueeze(1), gammas)
    cumsum = cumsum.squeeze(1)
    cumsum = _inv_pad_sequence(cumsum, num_per_traj)
    cumsum = cumsum.view_as(reward)
    if cumsum.shape != shape:
        cumsum = cumsum.view(shape)
    return cumsum
