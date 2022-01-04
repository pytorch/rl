from numbers import Number
from typing import Callable

import torch
from torch import nn

__all__ = ["mappings"]


def inv_softplus(bias: Number):
    return torch.tensor(bias).expm1().clamp_min(1e-6).log().item()


class biased_softplus(nn.Module):
    def __init__(self, bias: Number, min_val: Number=0.1):
        super().__init__()
        self.bias = inv_softplus(bias-min_val)
        self.min_val = min_val

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.softplus(x + self.bias)+self.min_val


def mappings(key: str) -> Callable:
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
