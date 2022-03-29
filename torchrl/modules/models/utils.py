# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, Sequence

import torch
from torch import nn

from .exploration import NoisyLazyLinear, NoisyLinear

LazyMapping = {
    nn.Linear: nn.LazyLinear,
    NoisyLinear: NoisyLazyLinear,
}

__all__ = [
    "SqueezeLayer",
    "Squeeze2dLayer",
]


class SqueezeLayer(nn.Module):
    """
    Squeezing layer.
    Squeezes some given singleton dimensions of an input tensor.

    Args:
         dims (iterable): dimensions to be squeezed
            default: (-1,)

    """

    def __init__(self, dims: Sequence[int] = (-1,)):
        super().__init__()
        self.dims = dims

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        for dim in self.dims:
            input = input.squeeze(dim)
        return input


class Squeeze2dLayer(SqueezeLayer):
    """
    Squeezing layer for convolutional neural networks.
    Squeezes the last two singleton dimensions of an input tensor.

    """

    def __init__(self):
        super().__init__((-1, -2))


class SquashDims(nn.Module):
    """
    A squashing layer.
    Flattens the N last dimensions of an input tensor.

    Args:
        ndims_in (int): number of dimensions to be flattened.
            default = 3
    """

    def __init__(self, ndims_in: int = 3):
        super().__init__()
        self.ndims_in = ndims_in

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value = value.flatten(-self.ndims_in, -1)
        return value


def _find_depth(depth: Optional[int], *list_or_ints: Sequence):
    if depth is None:
        for item in list_or_ints:
            if isinstance(item, (list, tuple)):
                depth = len(item)
    if depth is None:
        raise Exception(
            f"depth=None requires one of the input args (kernel_sizes, strides, num_cells) to be a a list or tuple. Got {tuple(type(item) for item in list_or_ints)}"
        )
    return depth
