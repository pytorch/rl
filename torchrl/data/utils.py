# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import typing
from typing import Any, Callable, List, Sequence, Tuple, Union

import numpy as np
import torch
from torch import Tensor

numpy_to_torch_dtype_dict = {
    np.dtype("bool"): torch.bool,
    np.dtype("uint8"): torch.uint8,
    np.dtype("int8"): torch.int8,
    np.dtype("int16"): torch.int16,
    np.dtype("int32"): torch.int32,
    np.dtype("int64"): torch.int64,
    np.dtype("float16"): torch.float16,
    np.dtype("float32"): torch.float32,
    np.dtype("float64"): torch.float64,
    np.dtype("complex64"): torch.complex64,
    np.dtype("complex128"): torch.complex128,
}
torch_to_numpy_dtype_dict = {
    value: key for key, value in numpy_to_torch_dtype_dict.items()
}
DEVICE_TYPING = Union[torch.device, str, int]
if hasattr(typing, "get_args"):
    DEVICE_TYPING_ARGS = typing.get_args(DEVICE_TYPING)
else:
    DEVICE_TYPING_ARGS = (torch.device, str, int)

INDEX_TYPING = Union[None, int, slice, str, Tensor, List[Any], Tuple[Any, ...]]


class CloudpickleWrapper(object):
    """A wrapper for functions that allow for serialization in multiprocessed settings."""

    def __init__(self, fn: Callable, **kwargs):
        if fn.__class__.__name__ == "EnvCreator":
            raise RuntimeError(
                "CloudpickleWrapper usage with EnvCreator class is "
                "prohibited as it breaks the transmission of shared tensors."
            )
        self.fn = fn
        self.kwargs = kwargs

    def __getstate__(self):
        import cloudpickle

        return cloudpickle.dumps((self.fn, self.kwargs))

    def __setstate__(self, ob: bytes):
        import pickle

        self.fn, self.kwargs = pickle.loads(ob)

    def __call__(self, **kwargs) -> Any:
        kwargs = {k: item for k, item in kwargs.items()}
        kwargs.update(self.kwargs)
        return self.fn(**kwargs)


def expand_as_right(
    tensor: Union[torch.Tensor, "MemmapTensor", "TensorDictBase"],  # noqa: F821
    dest: Union[torch.Tensor, "MemmapTensor", "TensorDictBase"],  # noqa: F821
):
    """Expand a tensor on the right to match another tensor shape.

    Args:
        tensor: tensor to be expanded
        dest: tensor providing the target shape

    Returns:
         a tensor with shape matching the dest input tensor shape.

    Examples:
        >>> tensor = torch.zeros(3,4)
        >>> dest = torch.zeros(3,4,5)
        >>> print(expand_as_right(tensor, dest).shape)
        torch.Size([3,4,5])

    """
    if dest.ndimension() < tensor.ndimension():
        raise RuntimeError(
            "expand_as_right requires the destination tensor to have less "
            f"dimensions than the input tensor, got"
            f" tensor.ndimension()={tensor.ndimension()} and "
            f"dest.ndimension()={dest.ndimension()}"
        )
    if not (tensor.shape == dest.shape[: tensor.ndimension()]):
        raise RuntimeError(
            f"tensor shape is incompatible with dest shape, "
            f"got: tensor.shape={tensor.shape}, dest={dest.shape}"
        )
    for _ in range(dest.ndimension() - tensor.ndimension()):
        tensor = tensor.unsqueeze(-1)
    return tensor.expand(dest.shape)


def expand_right(
    tensor: Union[torch.Tensor, "MemmapTensor"], shape: Sequence[int]  # noqa: F821
) -> torch.Tensor:
    """Expand a tensor on the right to match a desired shape.

    Args:
        tensor: tensor to be expanded
        shape: target shape

    Returns:
         a tensor with shape matching the target shape.

    Examples:
        >>> tensor = torch.zeros(3,4)
        >>> shape = (3,4,5)
        >>> print(expand_right(tensor, shape).shape)
        torch.Size([3,4,5])

    """
    tensor_expand = tensor
    while tensor_expand.ndimension() < len(shape):
        tensor_expand = tensor_expand.unsqueeze(-1)
    tensor_expand = tensor_expand.expand(*shape)
    return tensor_expand
