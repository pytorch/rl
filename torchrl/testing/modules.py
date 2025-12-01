from __future__ import annotations

import torch
from torch import nn


class BiasModule(nn.Module):
    """Simple bias module to check weight synchronization correctness."""

    def __init__(self, value: float = 0.0):
        super().__init__()
        self.bias = nn.Parameter(torch.tensor(value, dtype=torch.float))

    def forward(self, x):
        return x + self.bias
