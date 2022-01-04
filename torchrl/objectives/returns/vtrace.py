import math
from numbers import Number
from typing import Union

import torch


def c_val(log_pi: torch.Tensor, log_mu: torch.Tensor, c: Number = 1):
    return (log_pi - log_mu).clamp_max(math.log(c)).exp()


def dv_val(rewards, vals, gamma, rho_bar, log_pi, log_mu):
    rho = c_val(log_pi, log_mu, rho_bar)
    next_vals = torch.cat([vals[:, 1:], torch.zeros_like(vals[:, :1])], 1)
    dv = rho * (rewards + gamma * next_vals - vals)
    return dv, rho


def vtrace(rewards: torch.Tensor, vals: torch.Tensor, log_pi: torch.Tensor, log_mu: torch.Tensor,
           gamma: Union[torch.Tensor, Number], rho_bar: Number = 1.0, c_bar: Number = 1.0):
    T = vals.shape[1]
    if not isinstance(gamma, torch.Tensor):
        gamma = torch.full_like(vals, gamma)

    dv, rho = dv_val(rewards, vals, gamma, rho_bar, log_pi, log_mu)
    c = c_val(log_pi, log_mu, c_bar)

    v_out = []
    v_out.append(vals[:, -1] + dv[:, -1])
    for t in range(T - 2, -1, -1):
        _v_out = vals[:, t] + dv[:, t] + gamma[:, t] * c[:, t] * (v_out[-1] - vals[:, t + 1])
        v_out.append(_v_out)
    v_out = torch.stack(list(reversed(v_out)), 1)
    return v_out, rho
