from __future__ import annotations

import functools
from numbers import Number
from typing import Optional, Union, List, Tuple, Callable, Iterable

import numpy as np
import torch

from .memmap import MemmapTensor
from .utils import _getitem_batch_size
from ..utils import DEVICE_TYPING

META_HANDLED_FUNCTIONS = dict()


def implements_for_meta(torch_function) -> Callable:
    """Register a torch function override for ScalarTensor"""

    @functools.wraps(torch_function)
    def decorator(func):
        META_HANDLED_FUNCTIONS[torch_function] = func
        return func

    return decorator


class MetaTensor:
    def __init__(self, *shape,
                 device: Optional[DEVICE_TYPING] = 'cpu',
                 dtype: torch.dtype = torch.float,
                 _is_shared: bool = False,
                 _is_memmap: bool = False
                 ):

        if len(shape) == 1 and not isinstance(shape[0], (Number,)):
            tensor = shape[0]
            shape = tensor.shape
            _is_shared = tensor.is_shared() if tensor.device != torch.device('meta') else _is_shared
            _is_memmap = isinstance(tensor, MemmapTensor) if tensor.device != torch.device('meta') else _is_memmap
            device = tensor.device if tensor.device != torch.device('meta') else device
            dtype = tensor.dtype
        if not isinstance(shape, torch.Size):
            shape = torch.Size(shape)
        self.shape = shape
        self.device = torch.device(device) if not isinstance(device, torch.device) else device
        self.dtype = dtype
        self._ndim = len(shape)
        self._numel = np.prod(shape)
        self._is_shared = _is_shared
        self._is_memmap = _is_memmap
        if _is_memmap:
            name = "MemmapTensor"
        elif _is_shared:
            name = "SharedTensor"
        else:
            name = "Tensor"
        self.class_name = name

    def memmap_(self) -> MetaTensor:
        self._is_memmap = True
        self.class_name = "MemmapTensor"
        return self

    def share_memory_(self) -> MetaTensor:
        self._is_shared = True
        self.class_name = "SharedTensor"
        return self

    def is_shared(self) -> bool:
        return self._is_shared

    def is_memmap(self) -> bool:
        return self._is_memmap

    def numel(self) -> int:
        return self._numel

    def ndimension(self) -> int:
        return self._ndim

    def clone(self) -> MetaTensor:
        return MetaTensor(*self.shape, device=self.device, dtype=self.dtype)

    def _to_meta(self) -> torch.Tensor:
        return torch.empty(*self.shape, dtype=self.dtype, device='meta')

    def __getitem__(self, item: INDEX_TYPING) -> MetaTensor:
        shape = _getitem_batch_size(self.shape, item)
        return MetaTensor(*shape, dtype=self.dtype, device=self.device, _is_shared=self.is_shared())

    def __torch_function__(self, func: Callable, types, args: Tuple = (), kwargs: Optional[dict] = None):
        if kwargs is None:
            kwargs = {}
        if func not in META_HANDLED_FUNCTIONS or not all(
                issubclass(t, (torch.Tensor, MetaTensor)) for t in types
        ):
            return NotImplemented
        return META_HANDLED_FUNCTIONS[func](*args, **kwargs)

    def expand(self, *shape: Iterable) -> MetaTensor:
        shape = torch.Size([*shape, *self.shape])
        return MetaTensor(shape, device=self.device, dtype=self.dtype)

    def __repr__(self) -> str:
        return f"MetaTensor(shape={self.shape}, device={self.device}, dtype={self.dtype})"

    def unsqueeze(self, dim: int) -> MetaTensor:
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
        clone = self.clone()
        shape = [i for i in clone.shape]
        if dim is None:
            new_shape = [i for i in shape if i != 1]
        else:
            new_shape = []
        for i in range(len(shape)):
            if i == dim and shape[0] == 1:
                continue
            else:
                new_shape.append(shape[0])
            shape = shape[1:]
        clone.shape = torch.Size(new_shape)
        return clone

    def view(self, *shape: Iterable, size: Optional[Union[List, Tuple, torch.Size]] = None) -> MetaTensor:
        if len(shape) == 0 and size is not None:
            return self.view(*size)
        elif len(shape) == 1 and isinstance(shape[0], (list, tuple, torch.Size)):
            return self.view(*shape[0])
        elif not isinstance(shape, torch.Size):
            shape = torch.Size(shape)
        new_shape = torch.zeros(self.shape, device='meta').view(*shape)
        return MetaTensor(new_shape, device=self.device, dtype=self.dtype)


def _stack_meta(list_of_meta_tensors: Iterable[MetaTensor], dim: int = 0) -> MetaTensor:
    if not len(list_of_meta_tensors):
        return torch.tensor([], device='meta')
    shape0 = list_of_meta_tensors[0].shape
    dtype0 = list_of_meta_tensors[0].dtype
    for tensor in list_of_meta_tensors:
        assert tensor.device == torch.device(
            'meta'), f"Got a tensor with device {tensor.device} when expecting meta tensor"
        assert tensor.shape == shape0, f"Stacking meta tensors of different shapes is not allowed, got shapes {shape0} and {tensor.shape}"
        assert tensor.dtype == dtype0, f"Stacking meta tensors of different dtype is not allowed, got shapes {dtype0} and {tensor.dtype}"
    shape = []
    for i in range(len(shape0) + 1):
        if i == dim:
            shape.append(len(list_of_meta_tensors))
        else:
            shape.append(shape0[0])
            shape0 = shape0[1:]
    return torch.zeros(*shape, device='meta', dtype=dtype0)


@implements_for_meta(torch.stack)
def stack_meta(list_of_meta_tensors: Iterable[MetaTensor], dim: int = 0) -> MetaTensor:
    dtype = list_of_meta_tensors[0].dtype if len(list_of_meta_tensors) else torch.float
    device = list_of_meta_tensors[0].device if len(list_of_meta_tensors) else torch.device('cpu')
    _meta = _stack_meta([t._to_meta() for t in list_of_meta_tensors], dim=dim)
    return MetaTensor(_meta, dtype=dtype, device=device)
