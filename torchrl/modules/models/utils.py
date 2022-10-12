# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import inspect
from typing import Optional, Sequence, Type

import torch
from torch import nn

from torchrl.data import DEVICE_TYPING
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
        for dim in dims:
            if dim >= 0:
                raise RuntimeError("dims must all be < 0")
        self.dims = dims

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        for dim in self.dims:
            if input.shape[dim] != 1:
                raise RuntimeError(
                    f"Tried to squeeze an input over dims {self.dims} with shape {input.shape}"
                )
            input = input.squeeze(dim)
        return input


class Squeeze2dLayer(SqueezeLayer):
    """
    Squeezing layer for convolutional neural networks.
    Squeezes the last two singleton dimensions of an input tensor.

    """

    def __init__(self):
        super().__init__((-2, -1))


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
    """
    Find depth based on a sequence of inputs and a depth indicator.
    If the depth is None, it is inferred by the length of one (or more) matching
    lists of integers.
    Raises an exception if depth does not match the list lengths or if lists lengths
    do not match.

    Args:
        depth (int, optional): depth of the network
        *list_or_ints (lists of int or int): if depth is None, at least one of
            these inputs must be a list of ints of the length of the desired
            network.
    """
    if depth is None:
        for item in list_or_ints:
            if isinstance(item, (list, tuple)):
                depth = len(item)
    if depth is None:
        raise Exception(
            f"depth=None requires one of the input args (kernel_sizes, strides, "
            f"num_cells) to be a a list or tuple. Got {tuple(type(item) for item in list_or_ints)}"
        )
    return depth


def create_on_device(
    module_class: Type[nn.Module], device: Optional[DEVICE_TYPING], *args, **kwargs
) -> nn.Module:
    """
    Create a new instance of `module_class` on `device`.

    The new instance is created directly on the device if its constructor supports this.

    Args:
        module_class (Type[nn.Module]): the class of module to be created.
        device (DEVICE_TYPING): device to create the module on.
        *args: positional arguments to be passed to the module constructor.
        **kwargs: keyword arguments to be passed to the module constructor.
    """
    fullargspec = inspect.getfullargspec(module_class.__init__)
    if "device" in fullargspec.args or "device" in fullargspec.kwonlyargs:
        return module_class(*args, device=device, **kwargs)
    else:
        return module_class(*args, **kwargs).to(device)
        # .to() is always available for nn.Module, and does nothing if the Module contains no parameters or buffers
