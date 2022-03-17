from typing import Callable

import torch
from torch import nn

__all__ = ["mappings", "inv_softplus", "biased_softplus"]


def inv_softplus(bias: float):
    """
    inverse softplus function.

    """
    return torch.tensor(bias).expm1().clamp_min(1e-6).log().item()


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


def mappings(key: str) -> Callable:
    """
    Given an input string, return a surjective function f(x): R -> R^+

    Args:
        key (str): one of "softplus", "exp", "relu", or "biased_softplus".

    Returns:
         a Callable

    """
    _mappings = {
        "softplus": torch.nn.functional.softplus,
        "exp": torch.exp,
        "relu": torch.relu,
        "biased_softplus": biased_softplus(1.0),
    }
    if key in _mappings:
        return _mappings[key]
    elif key.startswith("biased_softplus"):
        return biased_softplus(float(key.split("_")[-1]))
    else:
        raise NotImplementedError(f"Unknown mapping {key}")
