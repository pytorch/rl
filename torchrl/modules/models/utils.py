import torch
from torch import nn

from .noisy import NoisyLinear, NoisyLazyLinear

LazyMapping = {
    nn.Linear: nn.LazyLinear,
    NoisyLinear: NoisyLazyLinear,
}

__all__ = [
    "SqueezeLayer",
    "Squeeze2dLayer",
]


class SqueezeLayer(nn.Module):
    def __init__(self, dims=(-1,)):
        super().__init__()
        self.dims = dims

    def forward(self, input: torch.Tensor):
        for dim in self.dims:
            input = input.squeeze(dim)
        return input


class Squeeze2dLayer(nn.Module):
    def forward(self, x):
        return x.squeeze(-1).squeeze(-1)


class SquashDims(nn.Module):
    def __init__(self, ndims_in=3):
        super().__init__()
        self.ndims_in = ndims_in

    def forward(self, value):
        value = value.view(*value.shape[:-self.ndims_in], -1)
        return value


def _find_depth(depth, *list_or_ints):
    if depth is None:
        for item in list_or_ints:
            if isinstance(item, (list, tuple)):
                if depth is not None:
                    assert depth == len(item)
                    continue
                depth = len(item)
    if depth is None:
        raise Exception(
            f"depth=None requires one of the input args (kernel_sizes, strides, num_cells) to be a a list or tuple. Got {tuple(type(item) for item in list_or_ints)}"
        )
    return depth
