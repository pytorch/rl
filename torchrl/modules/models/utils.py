# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import inspect
import warnings
from typing import Callable, Sequence, Type

import torch
from torch import nn

from torchrl.data.utils import DEVICE_TYPING

from torchrl.modules.models.exploration import NoisyLazyLinear, NoisyLinear

LazyMapping = {
    nn.Linear: nn.LazyLinear,
    NoisyLinear: NoisyLazyLinear,
}


class SqueezeLayer(nn.Module):
    """Squeezing layer.

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

    def forward(self, input: torch.Tensor) -> torch.Tensor:  # noqa: D102
        for dim in self.dims:
            if input.shape[dim] != 1:
                raise RuntimeError(
                    f"Tried to squeeze an input over dims {self.dims} with shape {input.shape}"
                )
            input = input.squeeze(dim)
        return input


class Squeeze2dLayer(SqueezeLayer):
    """Squeezing layer for convolutional neural networks.

    Squeezes the last two singleton dimensions of an input tensor.

    """

    def __init__(self):
        super().__init__((-2, -1))


class SquashDims(nn.Module):
    """A squashing layer.

    Flattens the N last dimensions of an input tensor.

    Args:
        ndims_in (int): number of dimensions to be flattened.
            default = 3

    Examples:
        >>> from torchrl.modules.models.utils import SquashDims
        >>> import torch
        >>> x = torch.randn(1, 2, 3, 4)
        >>> print(SquashDims()(x).shape)
        torch.Size([1, 24])

    """

    def __init__(self, ndims_in: int = 3):
        super().__init__()
        self.ndims_in = ndims_in

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value = value.flatten(-self.ndims_in, -1)
        return value


def _find_depth(depth: int | None, *list_or_ints: Sequence):
    """Find depth based on a sequence of inputs and a depth indicator.

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
        raise ValueError(
            f"depth=None requires one of the input args (kernel_sizes, strides, "
            f"num_cells) to be a a list or tuple. Got {tuple(type(item) for item in list_or_ints)}"
        )
    return depth


def create_on_device(
    module_class: Type[nn.Module] | Callable,
    device: DEVICE_TYPING | None,
    *args,
    **kwargs,
) -> nn.Module:
    """Create a new instance of :obj:`module_class` on :obj:`device`.

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
        result = module_class(*args, **kwargs)
        if hasattr(result, "to"):
            result = result.to(device)
        return result


def _reset_parameters_recursive(module, warn_if_no_op: bool = True) -> bool:
    """Recursively resets the parameters of a :class:`~torch.nn.Module` in-place.

    Args:
        module (torch.nn.Module): the module to reset.
        warn_if_no_op (bool, optional): whether to raise a warning in case this is a no-op.
            Defaults to ``True``.

    Returns: whether any parameter has been reset.

    """
    any_reset = False
    for layer in module.children():
        if hasattr(layer, "reset_parameters"):
            layer.reset_parameters()
            any_reset |= True
        any_reset |= _reset_parameters_recursive(layer, warn_if_no_op=False)
    if warn_if_no_op and not any_reset:
        warnings.warn(
            "_reset_parameters_recursive was called without the parameters argument and did not find any parameters to reset"
        )
    return any_reset
