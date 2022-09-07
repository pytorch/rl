# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Callable, Union

import torch
from torch import nn
from torch.nn import functional as F

__all__ = ["mappings", "inv_softplus", "biased_softplus"]


def inv_softplus(bias: Union[float, torch.Tensor]) -> Union[float, torch.Tensor]:
    """
    inverse softplus function.

    """
    is_tensor = True
    if not isinstance(bias, torch.Tensor):
        is_tensor = False
        bias = torch.tensor(bias)
    out = bias.expm1().clamp_min(1e-6).log()
    if not is_tensor and out.numel() == 1:
        return out.item()
    return out


class biased_softplus(nn.Module):
    """
    A biased softplus layer.
    Args:
        bias (scalar): 'bias' of the softplus transform. If bias=1.0, then a _bias shift will be computed such that
            softplus(0.0 + _bias) = bias.
        min_val (scalar): minimum value of the transform.
            default: 0.1
    """

    def __init__(self, bias: float, min_val: float = 0.01):
        super().__init__()
        self.bias = inv_softplus(bias - min_val)
        self.min_val = min_val

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.softplus(x + self.bias) + self.min_val


class minmax_softplus(nn.Module):
    def __init__(self, min_logvar=-10, max_logvar=0.5):
        super().__init__()
        self.min_logvar = min_logvar
        self.max_logvar = max_logvar

    def forward(self, x):
        logvar = 2 * torch.log(x)
        logvar = self.max_logvar - F.softplus(self.max_logvar - logvar)
        logvar = self.min_logvar + F.softplus(logvar - self.min_logvar)
        x = torch.exp(logvar / 2)
        return x


def expln(x):
    """
    A smooth, continuous positive mapping presented in "State-Dependent
    Exploration for Policy Gradient Methods"
    https://people.idsia.ch/~juergen/ecml2008rueckstiess.pdf

    """
    out = torch.empty_like(x)
    idx_neg = x <= 0
    out[idx_neg] = x[idx_neg].exp()
    out[~idx_neg] = x[~idx_neg].log1p() + 1
    return out


def mappings(key: str) -> Callable:
    """
    Given an input string, return a surjective function f(x): R -> R^+

    Args:
        key (str): one of "softplus", "exp", "relu", "expln",
        or "biased_softplus".

    Returns:
         a Callable

    """
    _mappings = {
        "softplus": torch.nn.functional.softplus,
        "exp": torch.exp,
        "relu": torch.relu,
        "biased_softplus": biased_softplus(1.0),
        "minmax_softplus": minmax_softplus(),
        "expln": expln,
    }
    if key in _mappings:
        return _mappings[key]
    elif key.startswith("biased_softplus"):
        return biased_softplus(float(key.split("_")[-1]))
    else:
        raise NotImplementedError(f"Unknown mapping {key}")
