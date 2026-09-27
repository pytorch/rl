# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimplicialNormalization(nn.Module):
    """Apply softmax independently to fixed-size feature simplices.

    Args:
        dim (int): Number of features in each simplex. The last input
            dimension must be divisible by ``dim``.
    """

    def __init__(self, dim: int):
        super().__init__()
        if isinstance(dim, bool) or not isinstance(dim, int) or dim <= 0:
            raise ValueError(f"dim must be a positive integer, got {dim!r}.")
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize the last dimension of ``x`` in groups of ``dim``."""
        if x.shape[-1] % self.dim:
            raise ValueError(
                f"The last input dimension must be divisible by dim={self.dim}, "
                f"got {x.shape[-1]}."
            )
        shape = x.shape
        x = x.reshape(*shape[:-1], -1, self.dim)
        x = F.softmax(x, dim=-1)
        return x.reshape(shape)
