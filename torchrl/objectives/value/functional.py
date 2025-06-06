# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import math
import warnings
from functools import wraps

import torch

try:
    from torch.compiler import is_dynamo_compiling
except ImportError:
    from torch._dynamo import is_compiling as is_dynamo_compiling

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
    "vtrace_advantage_estimate",
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
    def transposed_fun(*args, **kwargs):
        time_dim = kwargs.pop("time_dim", -2)

        def transpose_tensor(tensor):
            if not isinstance(tensor, torch.Tensor) or tensor.numel() <= 1:
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
            single_dim = False
            if args:
                args, single_dim = zip(*(transpose_tensor(arg) for arg in args))
                single_dim = any(single_dim)
            for k, item in list(kwargs.items()):
                item, sd = transpose_tensor(item)
                single_dim = single_dim or sd
                kwargs[k] = item
            # We don't pass time_dim because it isn't supposed to be used thereafter
            out = fun(*args, **kwargs)
            if isinstance(out, torch.Tensor):
                out = transpose_tensor(out)[0]
                if single_dim:
                    out = out.squeeze(-2)
                return out
            if single_dim:
                return tuple(transpose_tensor(_out)[0].squeeze(-2) for _out in out)
            return tuple(transpose_tensor(_out)[0] for _out in out)
        # We don't pass time_dim because it isn't supposed to be used thereafter
        out = fun(*args, **kwargs)
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
    terminated: torch.Tensor | None = None,
    *,
    time_dim: int = -2,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generalized advantage estimate of a trajectory.

    Refer to "HIGH-DIMENSIONAL CONTINUOUS CONTROL USING GENERALIZED ADVANTAGE ESTIMATION"
    https://arxiv.org/pdf/1506.02438.pdf for more context.

    Args:
        gamma (scalar): exponential mean discount.
        lmbda (scalar): trajectory discount.
        state_value (Tensor): value function result with old_state input.
        next_state_value (Tensor): value function result with new_state input.
        reward (Tensor): reward of taking actions in the environment.
        done (Tensor): boolean flag for end of trajectory.
        terminated (Tensor): boolean flag for the end of episode. Defaults to ``done``
            if not provided.
        time_dim (int): dimension where the time is unrolled. Defaults to -2.

    All tensors (values, reward and done) must have shape
    ``[*Batch x TimeSteps x *F]``, with ``*F`` feature dimensions.

    """
    if terminated is None:
        terminated = done.clone()
    if not (
        next_state_value.shape
        == state_value.shape
        == reward.shape
        == done.shape
        == terminated.shape
    ):
        raise RuntimeError(SHAPE_ERR)
    dtype = next_state_value.dtype
    device = state_value.device
    not_done = (~done).int()
    not_terminated = (~terminated).int()
    *batch_size, time_steps, lastdim = not_done.shape
    advantage = torch.empty(
        *batch_size, time_steps, lastdim, device=device, dtype=dtype
    )
    prev_advantage = 0
    g_not_terminated = gamma * not_terminated
    delta = reward + (g_not_terminated * next_state_value) - state_value
    discount = lmbda * gamma * not_done
    for t in reversed(range(time_steps)):
        prev_advantage = advantage[..., t, :] = delta[..., t, :] + (
            prev_advantage * discount[..., t, :]
        )

    value_target = advantage + state_value

    return advantage, value_target


def _geom_series_like(t, r, thr):
    """Creates a geometric series of the form [1, gammalmbda, gammalmbda**2] with the shape of `t`.

    Drops all elements which are smaller than `thr` (unless in compile mode).
    """
    if is_dynamo_compiling():
        if isinstance(r, torch.Tensor):
            rs = r.expand_as(t)
        else:
            rs = torch.full_like(t, r)
    else:
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
    terminated: torch.Tensor,
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
        done (torch.Tensor): a [B, T] boolean tensor containing the done states.
        terminated (torch.Tensor): a [B, T] boolean tensor containing the terminated states.
        gamma (scalar): the gamma decay (trajectory discount)
        lmbda (scalar): the lambda decay (exponential mean discount)
        thr (:obj:`float`): threshold for the filter. Below this limit, components will ignored.
            Defaults to 1e-7.

    All tensors (values, reward and done) must have shape
    ``[*Batch x TimeSteps x F]``, with ``F`` feature dimensions.

    """
    # _get_num_per_traj and _split_and_pad_sequence need
    # time dimension at last position
    done = done.transpose(-2, -1)
    terminated = terminated.transpose(-2, -1)
    reward = reward.transpose(-2, -1)
    state_value = state_value.transpose(-2, -1)
    next_state_value = next_state_value.transpose(-2, -1)

    gammalmbda = gamma * lmbda
    not_terminated = (~terminated).int()
    td0 = reward + not_terminated * gamma * next_state_value - state_value

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
    gamma: float | torch.Tensor,
    lmbda: float | torch.Tensor,
    state_value: torch.Tensor,
    next_state_value: torch.Tensor,
    reward: torch.Tensor,
    done: torch.Tensor,
    terminated: torch.Tensor | None = None,
    *,
    time_dim: int = -2,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Vectorized Generalized advantage estimate of a trajectory.

    Refer to "HIGH-DIMENSIONAL CONTINUOUS CONTROL USING GENERALIZED ADVANTAGE ESTIMATION"
    https://arxiv.org/pdf/1506.02438.pdf for more context.

    Args:
        gamma (scalar): exponential mean discount.
        lmbda (scalar): trajectory discount.
        state_value (Tensor): value function result with old_state input.
        next_state_value (Tensor): value function result with new_state input.
        reward (Tensor): reward of taking actions in the environment.
        done (Tensor): boolean flag for end of trajectory.
        terminated (Tensor): boolean flag for the end of episode. Defaults to ``done``
            if not provided.
        time_dim (int): dimension where the time is unrolled. Defaults to -2.

    All tensors (values, reward and done) must have shape
    ``[*Batch x TimeSteps x *F]``, with ``*F`` feature dimensions.

    """
    if terminated is None:
        terminated = done.clone()
    if not (
        next_state_value.shape
        == state_value.shape
        == reward.shape
        == done.shape
        == terminated.shape
    ):
        raise RuntimeError(SHAPE_ERR)
    dtype = state_value.dtype
    *batch_size, time_steps, lastdim = terminated.shape

    value = gamma * lmbda

    if isinstance(value, torch.Tensor) and value.numel() > 1:
        # create tensor while ensuring that gradients are passed
        not_done = (~done).to(dtype)
        gammalmbdas = not_done * value
    else:
        # when gamma and lmbda are scalars, use fast_vec_gae implementation
        return _fast_vec_gae(
            reward=reward,
            state_value=state_value,
            next_state_value=next_state_value,
            done=done,
            terminated=terminated,
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

    not_terminated = (~terminated).to(dtype)
    td0 = reward + not_terminated * gamma * next_state_value - state_value

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
    terminated: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """TD(0) advantage estimate of a trajectory.

    Also known as bootstrapped Temporal Difference or one-step return.

    Args:
        gamma (scalar): exponential mean discount.
        state_value (Tensor): value function result with old_state input.
        next_state_value (Tensor): value function result with new_state input.
        reward (Tensor): reward of taking actions in the environment.
        done (Tensor): boolean flag for end of trajectory.
        terminated (Tensor): boolean flag for the end of episode. Defaults to ``done``
            if not provided.

    All tensors (values, reward and done) must have shape
    ``[*Batch x TimeSteps x *F]``, with ``*F`` feature dimensions.

    """
    if terminated is None:
        terminated = done.clone()
    if not (
        next_state_value.shape
        == state_value.shape
        == reward.shape
        == done.shape
        == terminated.shape
    ):
        raise RuntimeError(SHAPE_ERR)
    returns = td0_return_estimate(gamma, next_state_value, reward, terminated)
    advantage = returns - state_value
    return advantage


def td0_return_estimate(
    gamma: float,
    next_state_value: torch.Tensor,
    reward: torch.Tensor,
    terminated: torch.Tensor | None = None,
    *,
    done: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    # noqa: D417
    """TD(0) discounted return estimate of a trajectory.

    Also known as bootstrapped Temporal Difference or one-step return.

    Args:
        gamma (scalar): exponential mean discount.
        next_state_value (Tensor): value function result with new_state input.
            must be a [Batch x TimeSteps x 1] or [Batch x TimeSteps] tensor
        reward (Tensor): reward of taking actions in the environment.
            must be a [Batch x TimeSteps x 1] or [Batch x TimeSteps] tensor
        terminated (Tensor): boolean flag for the end of episode. Defaults to ``done``
            if not provided.

    Keyword Args:
        done (Tensor): Deprecated. Use ``terminated`` instead.

    All tensors (values, reward and done) must have shape
    ``[*Batch x TimeSteps x *F]``, with ``*F`` feature dimensions.

    """
    if done is not None and terminated is None:
        terminated = done.clone()
        warnings.warn(
            "done for td0_return_estimate is deprecated. Pass ``terminated`` instead."
        )
    if not (next_state_value.shape == reward.shape == terminated.shape):
        raise RuntimeError(SHAPE_ERR)
    not_terminated = (~terminated).int()
    advantage = reward + gamma * not_terminated * next_state_value
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
    terminated: torch.Tensor | None = None,
    rolling_gamma: bool | None = None,
    *,
    time_dim: int = -2,
) -> torch.Tensor:
    r"""TD(1) return estimate.

    Args:
        gamma (scalar): exponential mean discount.
        next_state_value (Tensor): value function result with new_state input.
        reward (Tensor): reward of taking actions in the environment.
        done (Tensor): boolean flag for end of trajectory.
        terminated (Tensor): boolean flag for the end of episode. Defaults to ``done``
            if not provided.
        rolling_gamma (bool, optional): if ``True``, it is assumed that each gamma
            of a gamma tensor is tied to a single event:

              >>> gamma = [g1, g2, g3, g4]
              >>> value = [v1, v2, v3, v4]
              >>> return = [
              ...   v1 + g1 v2 + g1 g2 v3 + g1 g2 g3 v4,
              ...   v2 + g2 v3 + g2 g3 v4,
              ...   v3 + g3 v4,
              ...   v4,
              ... ]

            if ``False``, it is assumed that each gamma is tied to the upcoming
            trajectory:

              >>> gamma = [g1, g2, g3, g4]
              >>> value = [v1, v2, v3, v4]
              >>> return = [
              ...   v1 + g1 v2 + g1**2 v3 + g**3 v4,
              ...   v2 + g2 v3 + g2**2 v4,
              ...   v3 + g3 v4,
              ...   v4,
              ... ]

            Default is ``True``.
        time_dim (int): dimension where the time is unrolled. Defaults to -2.

    All tensors (values, reward and done) must have shape
    ``[*Batch x TimeSteps x *F]``, with ``*F`` feature dimensions.

    """
    if terminated is None:
        terminated = done.clone()
    if not (next_state_value.shape == reward.shape == done.shape == terminated.shape):
        raise RuntimeError(SHAPE_ERR)
    not_done = (~done).int()
    not_terminated = (~terminated).int()

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

    done_but_not_terminated = (done & ~terminated).int()
    if rolling_gamma:
        gamma = gamma * not_terminated
        g = next_state_value[..., -1, :]
        for i in reversed(range(T)):
            # if not done (and hence not terminated), get the bootstrapped value
            # if done but not terminated, get nex_val
            # if terminated, take nothing (gamma = 0)
            dnt = done_but_not_terminated[..., i, :]
            g = returns[..., i, :] = reward[..., i, :] + gamma[..., i, :] * (
                (1 - dnt) * g + dnt * next_state_value[..., i, :]
            )
    else:
        for k in range(T):
            g = 0
            _gamma = gamma[..., k, :]
            nd = not_terminated
            _gamma = _gamma.unsqueeze(-2) * nd
            for i in reversed(range(k, T)):
                dnt = done_but_not_terminated[..., i, :]
                g = reward[..., i, :] + _gamma[..., i, :] * (
                    (1 - dnt) * g + dnt * next_state_value[..., i, :]
                )
            returns[..., k, :] = g
    return returns


def td1_advantage_estimate(
    gamma: float,
    state_value: torch.Tensor,
    next_state_value: torch.Tensor,
    reward: torch.Tensor,
    done: torch.Tensor,
    terminated: torch.Tensor | None = None,
    rolling_gamma: bool | None = None,
    time_dim: int = -2,
) -> torch.Tensor:
    """TD(1) advantage estimate.

    Args:
        gamma (scalar): exponential mean discount.
        state_value (Tensor): value function result with old_state input.
        next_state_value (Tensor): value function result with new_state input.
        reward (Tensor): reward of taking actions in the environment.
        done (Tensor): boolean flag for end of trajectory.
        terminated (Tensor): boolean flag for the end of episode. Defaults to ``done``
            if not provided.
        rolling_gamma (bool, optional): if ``True``, it is assumed that each gamma
            of a gamma tensor is tied to a single event:

              >>> gamma = [g1, g2, g3, g4]
              >>> value = [v1, v2, v3, v4]
              >>> return = [
              ...   v1 + g1 v2 + g1 g2 v3 + g1 g2 g3 v4,
              ...   v2 + g2 v3 + g2 g3 v4,
              ...   v3 + g3 v4,
              ...   v4,
              ... ]

            if ``False``, it is assumed that each gamma is tied to the upcoming
            trajectory:

              >>> gamma = [g1, g2, g3, g4]
              >>> value = [v1, v2, v3, v4]
              >>> return = [
              ...   v1 + g1 v2 + g1**2 v3 + g**3 v4,
              ...   v2 + g2 v3 + g2**2 v4,
              ...   v3 + g3 v4,
              ...   v4,
              ... ]

            Default is ``True``.
        time_dim (int): dimension where the time is unrolled. Defaults to -2.

    All tensors (values, reward and done) must have shape
    ``[*Batch x TimeSteps x *F]``, with ``*F`` feature dimensions.

    """
    if terminated is None:
        terminated = done.clone()
    if not (
        next_state_value.shape
        == state_value.shape
        == reward.shape
        == done.shape
        == terminated.shape
    ):
        raise RuntimeError(SHAPE_ERR)
    if not state_value.shape == next_state_value.shape:
        raise RuntimeError("shape of state_value and next_state_value must match")
    returns = td1_return_estimate(
        gamma,
        next_state_value,
        reward,
        done,
        terminated=terminated,
        rolling_gamma=rolling_gamma,
        time_dim=time_dim,
    )
    advantage = returns - state_value
    return advantage


@_transpose_time
def vec_td1_return_estimate(
    gamma,
    next_state_value,
    reward,
    done: torch.Tensor,
    terminated: torch.Tensor | None = None,
    rolling_gamma: bool | None = None,
    time_dim: int = -2,
):
    """Vectorized TD(1) return estimate.

    Args:
        gamma (scalar, Tensor): exponential mean discount. If tensor-valued,
        next_state_value (Tensor): value function result with new_state input.
        reward (Tensor): reward of taking actions in the environment.
        done (Tensor): boolean flag for end of trajectory.
        terminated (Tensor): boolean flag for the end of episode. Defaults to ``done``
            if not provided.
        rolling_gamma (bool, optional): if ``True``, it is assumed that each gamma
            of the gamma tensor is tied to a single event:

              >>> gamma = [g1, g2, g3, g4]
              >>> value = [v1, v2, v3, v4]
              >>> return = [
              ...   v1 + g1 v2 + g1 g2 v3 + g1 g2 g3 v4,
              ...   v2 + g2 v3 + g2 g3 v4,
              ...   v3 + g3 v4,
              ...   v4,
              ... ]

            if ``False``, it is assumed that each gamma is tied to the upcoming
            trajectory:

              >>> gamma = [g1, g2, g3, g4]
              >>> value = [v1, v2, v3, v4]
              >>> return = [
              ...   v1 + g1 v2 + g1**2 v3 + g**3 v4,
              ...   v2 + g2 v3 + g2**2 v4,
              ...   v3 + g3 v4,
              ...   v4,
              ... ]

            Default is ``True``.
        time_dim (int): dimension where the time is unrolled. Defaults to ``-2``.

    All tensors (values, reward and done) must have shape
    ``[*Batch x TimeSteps x *F]``, with ``*F`` feature dimensions.

    """
    return vec_td_lambda_return_estimate(
        gamma=gamma,
        next_state_value=next_state_value,
        reward=reward,
        done=done,
        terminated=terminated,
        rolling_gamma=rolling_gamma,
        lmbda=1,
        time_dim=time_dim,
    )


def vec_td1_advantage_estimate(
    gamma,
    state_value,
    next_state_value,
    reward,
    done: torch.Tensor,
    terminated: torch.Tensor | None = None,
    rolling_gamma: bool | None = None,
    time_dim: int = -2,
):
    """Vectorized TD(1) advantage estimate.

    Args:
        gamma (scalar, Tensor): exponential mean discount. If tensor-valued,
        state_value (Tensor): value function result with old_state input.
        next_state_value (Tensor): value function result with new_state input.
        reward (Tensor): reward of taking actions in the environment.
        done (Tensor): boolean flag for end of trajectory.
        terminated (Tensor): boolean flag for the end of episode. Defaults to ``done``
            if not provided.
        rolling_gamma (bool, optional): if ``True``, it is assumed that each gamma
            of a gamma tensor is tied to a single event:

              >>> gamma = [g1, g2, g3, g4]
              >>> value = [v1, v2, v3, v4]
              >>> return = [
              ...   v1 + g1 v2 + g1 g2 v3 + g1 g2 g3 v4,
              ...   v2 + g2 v3 + g2 g3 v4,
              ...   v3 + g3 v4,
              ...   v4,
              ... ]

            if ``False``, it is assumed that each gamma is tied to the upcoming
            trajectory:

              >>> gamma = [g1, g2, g3, g4]
              >>> value = [v1, v2, v3, v4]
              >>> return = [
              ...   v1 + g1 v2 + g1**2 v3 + g**3 v4,
              ...   v2 + g2 v3 + g2**2 v4,
              ...   v3 + g3 v4,
              ...   v4,
              ... ]

            Default is ``True``.
        time_dim (int): dimension where the time is unrolled. Defaults to -2.

    All tensors (values, reward and done) must have shape
    ``[*Batch x TimeSteps x *F]``, with ``*F`` feature dimensions.

    """
    if terminated is None:
        terminated = done.clone()
    if not (
        next_state_value.shape
        == state_value.shape
        == reward.shape
        == done.shape
        == terminated.shape
    ):
        raise RuntimeError(SHAPE_ERR)
    return (
        vec_td1_return_estimate(
            gamma,
            next_state_value,
            reward,
            done=done,
            terminated=terminated,
            rolling_gamma=rolling_gamma,
            time_dim=time_dim,
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
    terminated: torch.Tensor | None = None,
    rolling_gamma: bool | None = None,
    *,
    time_dim: int = -2,
) -> torch.Tensor:
    r"""TD(:math:`\lambda`) return estimate.

    Args:
        gamma (scalar): exponential mean discount.
        lmbda (scalar): trajectory discount.
        next_state_value (Tensor): value function result with new_state input.
        reward (Tensor): reward of taking actions in the environment.
        done (Tensor): boolean flag for end of trajectory.
        terminated (Tensor): boolean flag for the end of episode. Defaults to ``done``
            if not provided.
        rolling_gamma (bool, optional): if ``True``, it is assumed that each gamma
            of a gamma tensor is tied to a single event:

              >>> gamma = [g1, g2, g3, g4]
              >>> value = [v1, v2, v3, v4]
              >>> return = [
              ...   v1 + g1 v2 + g1 g2 v3 + g1 g2 g3 v4,
              ...   v2 + g2 v3 + g2 g3 v4,
              ...   v3 + g3 v4,
              ...   v4,
              ... ]

            if ``False``, it is assumed that each gamma is tied to the upcoming
            trajectory:

              >>> gamma = [g1, g2, g3, g4]
              >>> value = [v1, v2, v3, v4]
              >>> return = [
              ...   v1 + g1 v2 + g1**2 v3 + g**3 v4,
              ...   v2 + g2 v3 + g2**2 v4,
              ...   v3 + g3 v4,
              ...   v4,
              ... ]

            Default is ``True``.
        time_dim (int): dimension where the time is unrolled. Defaults to -2.

    All tensors (values, reward and done) must have shape
    ``[*Batch x TimeSteps x *F]``, with ``*F`` feature dimensions.

    """
    if terminated is None:
        terminated = done.clone()
    if not (next_state_value.shape == reward.shape == done.shape == terminated.shape):
        raise RuntimeError(SHAPE_ERR)

    not_terminated = (~terminated).int()

    returns = torch.empty_like(next_state_value)
    next_state_value = next_state_value * not_terminated

    *batch, T, lastdim = returns.shape

    # if gamma is not a tensor of the same shape as other inputs, we use rolling_gamma = True
    single_gamma = False
    if not (isinstance(gamma, torch.Tensor) and gamma.shape == done.shape):
        single_gamma = True
        gamma = torch.full_like(next_state_value, gamma)

    single_lambda = False
    if not (isinstance(lmbda, torch.Tensor) and lmbda.shape == done.shape):
        single_lambda = True
        lmbda = torch.full_like(next_state_value, lmbda)

    if rolling_gamma is None:
        rolling_gamma = True
    elif not rolling_gamma and single_gamma and single_lambda:
        raise RuntimeError(
            "rolling_gamma=False is expected only with time-sensitive gamma or lambda values"
        )
    if rolling_gamma:
        g = next_state_value[..., -1, :]
        for i in reversed(range(T)):
            dn = done[..., i, :].int()
            nv = next_state_value[..., i, :]
            lmd = lmbda[..., i, :]
            # if done, the bootstrapped gain is the next value, otherwise it's the
            # value we computed during the previous iter
            g = g * (1 - dn) + nv * dn
            g = returns[..., i, :] = reward[..., i, :] + gamma[..., i, :] * (
                (1 - lmd) * nv + lmd * g
            )
    else:
        for k in range(T):
            g = next_state_value[..., -1, :]
            _gamma = gamma[..., k, :]
            _lambda = lmbda[..., k, :]
            for i in reversed(range(k, T)):
                dn = done[..., i, :].int()
                nv = next_state_value[..., i, :]
                g = g * (1 - dn) + nv * dn
                g = reward[..., i, :] + _gamma * ((1 - _lambda) * nv + _lambda * g)
            returns[..., k, :] = g

    return returns


def td_lambda_advantage_estimate(
    gamma: float,
    lmbda: float,
    state_value: torch.Tensor,
    next_state_value: torch.Tensor,
    reward: torch.Tensor,
    done: torch.Tensor,
    terminated: torch.Tensor | None = None,
    rolling_gamma: bool | None = None,
    # not a kwarg because used directly
    time_dim: int = -2,
) -> torch.Tensor:
    r"""TD(:math:`\lambda`) advantage estimate.

    Args:
        gamma (scalar): exponential mean discount.
        lmbda (scalar): trajectory discount.
        state_value (Tensor): value function result with old_state input.
        next_state_value (Tensor): value function result with new_state input.
        reward (Tensor): reward of taking actions in the environment.
        done (Tensor): boolean flag for end of trajectory.
        terminated (Tensor): boolean flag for the end of episode. Defaults to ``done``
            if not provided.
        rolling_gamma (bool, optional): if ``True``, it is assumed that each gamma
            of a gamma tensor is tied to a single event:

              >>> gamma = [g1, g2, g3, g4]
              >>> value = [v1, v2, v3, v4]
              >>> return = [
              ...   v1 + g1 v2 + g1 g2 v3 + g1 g2 g3 v4,
              ...   v2 + g2 v3 + g2 g3 v4,
              ...   v3 + g3 v4,
              ...   v4,
              ... ]

            if ``False``, it is assumed that each gamma is tied to the upcoming
            trajectory:

              >>> gamma = [g1, g2, g3, g4]
              >>> value = [v1, v2, v3, v4]
              >>> return = [
              ...   v1 + g1 v2 + g1**2 v3 + g**3 v4,
              ...   v2 + g2 v3 + g2**2 v4,
              ...   v3 + g3 v4,
              ...   v4,
              ... ]

            Default is ``True``.
        time_dim (int): dimension where the time is unrolled. Defaults to -2.

    All tensors (values, reward and done) must have shape
    ``[*Batch x TimeSteps x *F]``, with ``*F`` feature dimensions.

    """
    if terminated is None:
        terminated = done.clone()
    if not (
        next_state_value.shape
        == state_value.shape
        == reward.shape
        == done.shape
        == terminated.shape
    ):
        raise RuntimeError(SHAPE_ERR)
    if not state_value.shape == next_state_value.shape:
        raise RuntimeError("shape of state_value and next_state_value must match")
    returns = td_lambda_return_estimate(
        gamma,
        lmbda,
        next_state_value,
        reward,
        done,
        terminated=terminated,
        rolling_gamma=rolling_gamma,
        time_dim=time_dim,
    )
    advantage = returns - state_value
    return advantage


def _fast_td_lambda_return_estimate(
    gamma: torch.Tensor | float,
    lmbda: float,
    next_state_value: torch.Tensor,
    reward: torch.Tensor,
    done: torch.Tensor,
    terminated: torch.Tensor,
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
        done (Tensor): boolean flag for end of trajectory.
        terminated (Tensor): boolean flag for end of episode.
        thr (:obj:`float`): threshold for the filter. Below this limit, components will ignored.
            Defaults to 1e-7.

    All tensors (values, reward and done) must have shape
    ``[*Batch x TimeSteps x F]``, with ``F`` feature dimensions.

    """
    device = reward.device
    done = done.transpose(-2, -1)
    terminated = terminated.transpose(-2, -1)
    reward = reward.transpose(-2, -1)
    next_state_value = next_state_value.transpose(-2, -1)

    # the only valid next states are those where the trajectory does not terminate
    next_state_value = (~terminated).int() * next_state_value

    gamma_tensor = torch.tensor([gamma], device=device)
    gammalmbda = gamma_tensor * lmbda

    num_per_traj = _get_num_per_traj(done)

    done = done.clone()
    done[..., -1] = 1
    not_done = (~done).int()

    t = reward + next_state_value * gamma_tensor * (1 - not_done * lmbda)

    t_flat, mask = _split_and_pad_sequence(t, num_per_traj, return_mask=True)

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
    terminated: torch.Tensor | None = None,
    rolling_gamma: bool | None = None,
    *,
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
        done (Tensor): boolean flag for end of trajectory.
        terminated (Tensor): boolean flag for the end of episode. Defaults to ``done``
            if not provided.
        rolling_gamma (bool, optional): if ``True``, it is assumed that each gamma
            of a gamma tensor is tied to a single event:

              >>> gamma = [g1, g2, g3, g4]
              >>> value = [v1, v2, v3, v4]
              >>> return = [
              ...   v1 + g1 v2 + g1 g2 v3 + g1 g2 g3 v4,
              ...   v2 + g2 v3 + g2 g3 v4,
              ...   v3 + g3 v4,
              ...   v4,
              ... ]

            if ``False``, it is assumed that each gamma is tied to the upcoming
            trajectory:

              >>> gamma = [g1, g2, g3, g4]
              >>> value = [v1, v2, v3, v4]
              >>> return = [
              ...   v1 + g1 v2 + g1**2 v3 + g**3 v4,
              ...   v2 + g2 v3 + g2**2 v4,
              ...   v3 + g3 v4,
              ...   v4,
              ... ]

            Default is ``True``.
        time_dim (int): dimension where the time is unrolled. Defaults to -2.

    All tensors (values, reward and done) must have shape
    ``[*Batch x TimeSteps x *F]``, with ``*F`` feature dimensions.

    """
    if terminated is None:
        terminated = done.clone()
    if not (next_state_value.shape == reward.shape == done.shape == terminated.shape):
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
            terminated=terminated,
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
    not_terminated = (~terminated).int().transpose(-2, -1).unsqueeze(-2)
    if len(batch):
        not_terminated = not_terminated.flatten(0, len(batch))
    next_state_value = next_state_value * not_terminated

    if rolling_gamma is None:
        rolling_gamma = True
    if not rolling_gamma:
        terminated_follows_terminated = terminated[..., 1:, :][
            terminated[..., :-1, :]
        ].all()
        if not terminated_follows_terminated:
            raise NotImplementedError(
                "When using rolling_gamma=False and vectorized TD(lambda) with time-dependent gamma, "
                "make sure that conseducitve trajectories are separated as different batch "
                "items. Propagating a gamma value across trajectories is not permitted with "
                "this method. Check that you need to use rolling_gamma=False, and if so "
                "consider using the non-vectorized version of the return computation or splitting "
                "your trajectories."
            )

    if rolling_gamma:
        # Make the coefficient table
        gammas = _make_gammas_tensor(gamma * not_done, T, rolling_gamma)
        gammas_cp = torch.cumprod(gammas, -2)
        lambdas = torch.ones(T + 1, 1, device=device)
        lambdas[1:] = lmbda
        lambdas_cp = torch.cumprod(lambdas, -2)
        lambdas = lambdas[1:]
        dec = gammas_cp * lambdas_cp

        gammas = _make_gammas_tensor(gamma, T, rolling_gamma)
        gammas = gammas[..., 1:, :]
        if gammas.ndimension() == 4 and gammas.shape[1] > 1:
            gammas = gammas[:, :1]
        if lambdas.ndimension() == 4 and lambdas.shape[1] > 1:
            lambdas = lambdas[:, :1]

        not_done = not_done.transpose(-2, -1).unsqueeze(-2)
        if len(batch):
            not_done = not_done.flatten(0, len(batch))
        # lambdas = lambdas * not_done

        v3 = (gammas * lambdas).squeeze(-1) * next_state_value * not_done
        v3[..., :-1] = 0
        out = _custom_conv1d(
            reward
            + gammas.squeeze(-1)
            * next_state_value
            * (1 - lambdas.squeeze(-1) * not_done)
            + v3,
            dec,
        )

        return out.view(*batch, lastdim, T).transpose(-2, -1)
    else:
        raise NotImplementedError(
            "The vectorized version of TD(lambda) with rolling_gamma=False is currently not available. "
            "To use this feature, use the non-vectorized version of TD(lambda). You can expect "
            "good speed improvements by decorating the function with torch.compile!"
        )


def vec_td_lambda_advantage_estimate(
    gamma,
    lmbda,
    state_value,
    next_state_value,
    reward,
    done,
    terminated: torch.Tensor | None = None,
    rolling_gamma: bool | None = None,
    # not a kwarg because used directly
    time_dim: int = -2,
):
    r"""Vectorized TD(:math:`\lambda`) advantage estimate.

    Args:
        gamma (scalar, Tensor): exponential mean discount. If tensor-valued,
        lmbda (scalar): trajectory discount.
        state_value (Tensor): value function result with old_state input.
        next_state_value (Tensor): value function result with new_state input.
        reward (Tensor): reward of taking actions in the environment.
        done (Tensor): boolean flag for end of trajectory.
        terminated (Tensor): boolean flag for the end of episode. Defaults to ``done``
            if not provided.
        rolling_gamma (bool, optional): if ``True``, it is assumed that each gamma
            of a gamma tensor is tied to a single event:

              >>> gamma = [g1, g2, g3, g4]
              >>> value = [v1, v2, v3, v4]
              >>> return = [
              ...   v1 + g1 v2 + g1 g2 v3 + g1 g2 g3 v4,
              ...   v2 + g2 v3 + g2 g3 v4,
              ...   v3 + g3 v4,
              ...   v4,
              ... ]

            if ``False``, it is assumed that each gamma is tied to the upcoming
            trajectory:

              >>> gamma = [g1, g2, g3, g4]
              >>> value = [v1, v2, v3, v4]
              >>> return = [
              ...   v1 + g1 v2 + g1**2 v3 + g**3 v4,
              ...   v2 + g2 v3 + g2**2 v4,
              ...   v3 + g3 v4,
              ...   v4,
              ... ]

            Default is ``True``.
        time_dim (int): dimension where the time is unrolled. Defaults to -2.

    All tensors (values, reward and done) must have shape
    ``[*Batch x TimeSteps x *F]``, with ``*F`` feature dimensions.

    """
    if terminated is None:
        terminated = done.clone()
    if not (
        next_state_value.shape
        == state_value.shape
        == reward.shape
        == done.shape
        == terminated.shape
    ):
        raise RuntimeError(SHAPE_ERR)
    return (
        vec_td_lambda_return_estimate(
            gamma,
            lmbda,
            next_state_value,
            reward,
            done=done,
            terminated=terminated,
            rolling_gamma=rolling_gamma,
            time_dim=time_dim,
        )
        - state_value
    )


########################################################################
# V-Trace
# -----


@_transpose_time
def vtrace_advantage_estimate(
    gamma: float,
    log_pi: torch.Tensor,
    log_mu: torch.Tensor,
    state_value: torch.Tensor,
    next_state_value: torch.Tensor,
    reward: torch.Tensor,
    done: torch.Tensor,
    terminated: torch.Tensor | None = None,
    rho_thresh: float | torch.Tensor = 1.0,
    c_thresh: float | torch.Tensor = 1.0,
    # not a kwarg because used directly
    time_dim: int = -2,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Computes V-Trace off-policy actor critic targets.

    Refer to "IMPALA: Scalable Distributed Deep-RL with Importance Weighted  Actor-Learner Architectures"
    https://arxiv.org/abs/1802.01561 for more context.

    Args:
        gamma (scalar): exponential mean discount.
        log_pi (Tensor): collection actor log probability of taking actions in the environment.
        log_mu (Tensor): current actor log probability of taking actions in the environment.
        state_value (Tensor): value function result with state input.
        next_state_value (Tensor): value function result with next_state input.
        reward (Tensor): reward of taking actions in the environment.
        done (Tensor): boolean flag for end of episode.
        terminated (torch.Tensor): a [B, T] boolean tensor containing the terminated states.
        rho_thresh (Union[float, Tensor]): rho clipping parameter for importance weights.
        c_thresh (Union[float, Tensor]): c clipping parameter for importance weights.
        time_dim (int): dimension where the time is unrolled. Defaults to -2.

    All tensors (values, reward and done) must have shape
    ``[*Batch x TimeSteps x *F]``, with ``*F`` feature dimensions.
    """
    if not (next_state_value.shape == state_value.shape == reward.shape == done.shape):
        raise RuntimeError(SHAPE_ERR)

    device = state_value.device

    if not isinstance(rho_thresh, torch.Tensor):
        rho_thresh = torch.tensor(rho_thresh, device=device)
    if not isinstance(c_thresh, torch.Tensor):
        c_thresh = torch.tensor(c_thresh, device=device)

    c_thresh = c_thresh.to(device)
    rho_thresh = rho_thresh.to(device)

    not_done = (~done).int()
    not_terminated = not_done if terminated is None else (~terminated).int()
    *batch_size, time_steps, lastdim = not_done.shape
    done_discounts = gamma * not_done
    terminated_discounts = gamma * not_terminated

    rho = (log_pi - log_mu).exp()
    clipped_rho = rho.clamp_max(rho_thresh)
    deltas = clipped_rho * (
        reward + terminated_discounts * next_state_value - state_value
    )
    clipped_c = rho.clamp_max(c_thresh)

    vs_minus_v_xs = [torch.zeros_like(next_state_value[..., -1, :])]
    for i in reversed(range(time_steps)):
        discount_t, c_t, delta_t = (
            done_discounts[..., i, :],
            clipped_c[..., i, :],
            deltas[..., i, :],
        )
        vs_minus_v_xs.append(delta_t + discount_t * c_t * vs_minus_v_xs[-1])
    vs_minus_v_xs = torch.stack(vs_minus_v_xs[1:], dim=time_dim)
    vs_minus_v_xs = torch.flip(vs_minus_v_xs, dims=[time_dim])
    vs = vs_minus_v_xs + state_value
    vs_t_plus_1 = torch.cat(
        [vs[..., 1:, :], next_state_value[..., -1:, :]], dim=time_dim
    )
    advantages = clipped_rho * (
        reward + terminated_discounts * vs_t_plus_1 - state_value
    )

    return advantages, vs


########################################################################
# Reward to go
# ------------


@_transpose_time
def reward2go(
    reward,
    done,
    gamma,
    *,
    time_dim: int = -2,
):
    """Compute the discounted cumulative sum of rewards given multiple trajectories and the episode ends.

    Args:
        reward (torch.Tensor): A tensor containing the rewards
            received at each time step over multiple trajectories.
        done (Tensor): boolean flag for end of episode. Differs from
            truncated, where the episode did not end but was interrupted.
        gamma (:obj:`float`, optional): The discount factor to use for computing the
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
    # flatten if needed
    if reward.ndim > 2:
        # we know time dim is at -2, let's put it at -3
        rflip = reward.transpose(-2, -3)
        rflip_shape = rflip.shape[-2:]
        r2go = reward2go(
            rflip.flatten(-2, -1), done.transpose(-2, -3).flatten(-2, -1), gamma=gamma
        ).unflatten(-1, rflip_shape)
        return r2go.transpose(-2, -3)

    # place time at dim -1
    reward = reward.transpose(-2, -1)
    done = done.transpose(-2, -1)

    num_per_traj = _get_num_per_traj(done)
    td0_flat = _split_and_pad_sequence(reward, num_per_traj)
    gammas = _geom_series_like(td0_flat[0], gamma, thr=1e-7)
    cumsum = _custom_conv1d(td0_flat.unsqueeze(1), gammas)
    cumsum = cumsum.squeeze(1)
    cumsum = _inv_pad_sequence(cumsum, num_per_traj)
    cumsum = cumsum.reshape_as(reward)
    cumsum = cumsum.transpose(-2, -1)
    if cumsum.shape != shape:
        try:
            cumsum = cumsum.reshape(shape)
        except RuntimeError:
            raise RuntimeError(
                f"Wrong shape for output reward2go: {cumsum.shape} when {shape} was expected."
            )
    return cumsum
