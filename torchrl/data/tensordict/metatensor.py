# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import functools
from numbers import Number
from typing import Callable, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from torchrl.data.utils import DEVICE_TYPING, INDEX_TYPING
from .memmap import MemmapTensor
from .utils import _getitem_batch_size, _get_shape

META_HANDLED_FUNCTIONS = dict()


def implements_for_meta(torch_function) -> Callable:
    """Register a torch function override for ScalarTensor."""

    @functools.wraps(torch_function)
    def decorator(func):
        META_HANDLED_FUNCTIONS[torch_function] = func
        return func

    return decorator


class MetaTensor:
    """MetaTensor is a custom class that stores the meta-information about a tensor without requiring to access the tensor.

    This is intended to be used with tensors that have a high access cost.
    MetaTensor supports more operations than tensors on 'meta' device (
    `torch.tensor(..., device='meta')`).
    For instance, MetaTensor supports some operations on its shape and device,
    such as :obj:`mt.to(device)`, :obj:`mt.view(*new_shape)`, :obj:`mt.expand(
    *expand_shape)` etc.

    Args:
        shape (iterable of integers): shape of the tensor. If the first
            element of "shape" is a torch.Tensor, the
            MetaTensor is built with this tensor specs.
        device (int, str or torch.device): device on which the tensor is
            stored.
        dtype (torch.dtype): tensor dtype.
        requires_grad (bool): tensor requires_grad.

    Examples:
        >>> meta1 = MetaTensor(3,4, device=torch.device("cpu"))
        >>> meta2 = MetaTensor(torch.randn(3,4,device="cuda:0",
        ...    dtype=torch.double))
        >>> assert meta1.device != meta2.device
        >>> assert meta1.dtype != meta2.dtype
        >>> assert meta1.expand(2, 3, 4).shape == torch.Size([2, 3, 4])
        >>> assert torch.stack([MetaTensor(3,4) for _ in range(10)],
        ...    1).shape == torch.Size([3, 10, 4])
    """

    def __init__(
        self,
        *shape: Union[int, torch.Tensor, "MemmapTensor"],
        device: Optional[DEVICE_TYPING] = "cpu",
        dtype: torch.dtype = None,
        requires_grad: bool = False,
        _is_shared: Optional[bool] = None,
        _is_memmap: Optional[bool] = None,
        _is_tensordict: Optional[bool] = None,
        _repr_tensordict: Optional[str] = None,
    ):
        if len(shape) == 1 and not isinstance(shape[0], (Number,)):
            tensor = shape[0]
            shape = _get_shape(tensor)
            if _is_shared is None:
                _is_shared = tensor.is_shared()
            if _is_memmap is None:
                _is_memmap = isinstance(tensor, MemmapTensor)
            # FIXME: using isinstance(tensor, TensorDictBase) would likely be
            # better here, but creates circular import without more refactoring
            device = tensor.device if not tensor.is_meta else device
            if _is_tensordict is None:
                _is_tensordict = not _is_memmap and not isinstance(tensor, torch.Tensor)
            if not _is_tensordict:
                dtype = tensor.dtype
            else:
                dtype = None
                _repr_tensordict = str(tensor)

            requires_grad = (
                tensor.requires_grad
                if isinstance(tensor, torch.Tensor)
                else requires_grad
            )

        if not isinstance(shape, torch.Size):
            shape = torch.Size(shape)
        self.shape = shape
        self.device = device
        self.dtype = dtype if dtype is not None else torch.get_default_dtype()
        self.requires_grad = requires_grad
        self._ndim = len(shape)
        self._numel = None
        self._is_shared = bool(_is_shared)
        self._is_memmap = bool(_is_memmap)
        self._is_tensordict = bool(_is_tensordict)
        self._repr_tensordict = _repr_tensordict
        if _is_tensordict:
            name = "TensorDict"
        elif _is_memmap:
            name = "MemmapTensor"
        elif _is_shared and device.type != "cuda":
            name = "SharedTensor"
        else:
            name = "Tensor"
        self.class_name = name

    def get_repr(self):
        if self.is_tensordict():
            return self._repr_tensordict
        else:
            return f"{self.class_name}({self.shape}, dtype={self.dtype})"

    def memmap_(self) -> MetaTensor:
        """Changes the storage of the MetaTensor to memmap.

        Returns:
            self

        """
        self._is_memmap = True
        self.class_name = "MemmapTensor"
        return self

    def share_memory_(self) -> MetaTensor:
        """Changes the storage of the MetaTensor to shared memory.

        Returns:
            self

        """
        self._is_shared = True
        self.class_name = "SharedTensor" if self.device.type != "cuda" else "Tensor"
        return self

    def is_shared(self) -> bool:
        return self._is_shared

    def is_memmap(self) -> bool:
        return self._is_memmap

    def is_tensordict(self) -> bool:
        return self._is_tensordict

    def numel(self) -> int:
        if self._numel is None:
            self._numel = np.prod(self.shape)
        return self._numel

    def ndimension(self) -> int:
        return self._ndim

    def clone(self) -> MetaTensor:
        """Clones the meta-tensor.

        Returns: a new MetaTensor with the same specs.

        """
        return MetaTensor(
            *self.shape,
            device=self.device,
            dtype=self.dtype,
            requires_grad=self.requires_grad,
            _is_shared=self.is_shared(),
            _is_memmap=self.is_memmap(),
            _is_tensordict=self.is_tensordict(),
            _repr_tensordict=self._repr_tensordict,
        )

    def _to_meta(self) -> torch.Tensor:
        return torch.empty(*self.shape, dtype=self.dtype, device="meta")

    def __getitem__(self, item: INDEX_TYPING) -> MetaTensor:
        shape = _getitem_batch_size(self.shape, item)
        return MetaTensor(
            *shape,
            dtype=self.dtype,
            device=self.device,
            requires_grad=self.requires_grad,
            _is_shared=self.is_shared(),
            _is_memmap=self.is_memmap(),
            _is_tensordict=self.is_tensordict(),
            _repr_tensordict=self._repr_tensordict,
        )

    @classmethod
    def __torch_function__(
        cls,
        func: Callable,
        types,
        args: Tuple = (),
        kwargs: Optional[dict] = None,
    ):
        if kwargs is None:
            kwargs = {}
        if func not in META_HANDLED_FUNCTIONS or not all(
            issubclass(t, (torch.Tensor, MetaTensor)) for t in types
        ):
            return NotImplemented
        return META_HANDLED_FUNCTIONS[func](*args, **kwargs)

    def expand(self, *shape: int) -> MetaTensor:
        shape = torch.Size([*shape, *self.shape])
        return MetaTensor(
            *shape,
            device=self.device,
            dtype=self.dtype,
            requires_grad=self.requires_grad,
        )

    def __repr__(self) -> str:
        return (
            f"MetaTensor(shape={self.shape}, device={self.device}, "
            f"dtype={self.dtype})"
        )

    def unsqueeze(self, dim: int) -> MetaTensor:
        """Unsqueezes the meta-tensor along the desired dim."""
        clone = self.clone()
        new_shape = []
        shape = [i for i in clone.shape]
        for i in range(len(shape) + 1):
            if i == dim:
                new_shape.append(1)
            else:
                new_shape.append(shape[0])
                shape = shape[1:]
        clone.shape = torch.Size(new_shape)
        return clone

    def squeeze(self, dim: Optional[int] = None) -> MetaTensor:
        """Squeezes the meta-tensor along the desired dim."""
        clone = self.clone()
        shape = clone.shape
        if dim is None:
            new_shape = [i for i in shape if i != 1]
        else:
            new_shape = []
            for i in range(len(shape)):
                if i == dim and shape[0] == 1:
                    shape = shape[1:]
                    continue
                else:
                    new_shape.append(shape[0])
                    shape = shape[1:]
        clone.shape = torch.Size(new_shape)
        return clone

    def permute(self, dims: int) -> MetaTensor:
        """Permutes the dims of the meta-tensor."""
        clone = self.clone()
        new_shape = [self.shape[dim] for dim in dims]
        clone.shape = torch.Size(new_shape)
        return clone

    def view(
        self,
        *shape: Sequence,
        size: Optional[Union[List, Tuple, torch.Size]] = None,
    ) -> MetaTensor:
        """Returns a view of a reshaped meta-tensor."""
        if len(shape) == 0 and size is not None:
            return self.view(*size)
        elif len(shape) == 1 and isinstance(shape[0], (list, tuple, torch.Size)):
            return self.view(*shape[0])
        elif not isinstance(shape, torch.Size):
            shape = torch.Size(shape)
        new_shape = torch.zeros(
            self.shape, device="meta", requires_grad=self.requires_grad
        ).view(*shape)
        return MetaTensor(
            new_shape,
            device=self.device,
            dtype=self.dtype,
            requires_grad=self.requires_grad,
            _is_shared=self.is_shared(),
            _is_memmap=self.is_memmap(),
            _is_tensordict=self.is_tensordict(),
        )


def _stack_meta(
    list_of_meta_tensors: Sequence[MetaTensor],
    dim: int = 0,
    dtype: torch.dtype = torch.float,
    device: DEVICE_TYPING = "cpu",
    requires_grad: bool = False,
    safe: bool = False,
) -> MetaTensor:
    if not len(list_of_meta_tensors):
        raise RuntimeError("empty list of meta tensors is not supported")
    is_tensordict = list_of_meta_tensors[0].is_tensordict()
    shape = list_of_meta_tensors[0].shape
    if safe:
        for tensor in list_of_meta_tensors:
            if tensor.shape != shape:
                raise RuntimeError(
                    f"Stacking meta tensors of different shapes is not "
                    f"allowed, got shapes {shape} and {tensor.shape}"
                )
            if is_tensordict and not tensor.is_tensordict():
                raise RuntimeError(
                    "Stacking meta tensors from tensordict and non-tensordict "
                    "inputs is not allowed."
                )
            if tensor.dtype != dtype:
                raise TypeError(
                    f"Stacking meta tensors of different dtype is not "
                    f"allowed, got shapes {dtype} and {tensor.dtype}"
                )
    shape = [s for s in shape]
    shape.insert(dim, len(list_of_meta_tensors))
    return MetaTensor(
        *shape,
        dtype=dtype,
        device=device,
        _is_tensordict=is_tensordict,
        requires_grad=requires_grad,
    )


@implements_for_meta(torch.stack)
def stack_meta(
    list_of_meta_tensors: Sequence[MetaTensor],
    dim: int = 0,
    safe: bool = False,
) -> MetaTensor:
    """Stacks similar meta-tensors into a single meta-tensor."""
    dtype = (
        list_of_meta_tensors[0].dtype
        if len(list_of_meta_tensors)
        else torch.get_default_dtype()
    )
    device = (
        list_of_meta_tensors[0].device
        if len(list_of_meta_tensors)
        else torch.device("cpu")
    )
    requires_grad = any(tensor.requires_grad for tensor in list_of_meta_tensors)
    return _stack_meta(
        list_of_meta_tensors,
        dim=dim,
        dtype=dtype,
        device=device,
        requires_grad=requires_grad,
        safe=safe,
    )
