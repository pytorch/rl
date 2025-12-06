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


class NonSerializableBiasModule(BiasModule):
    """Bias module that intentionally fails to serialize.

    This is used in tests to simulate a policy that cannot be pickled.
    """

    def __getstate__(self):
        # Simulate a non-serializable policy by raising on pickling
        raise RuntimeError("NonSerializableBiasModule cannot be pickled")
