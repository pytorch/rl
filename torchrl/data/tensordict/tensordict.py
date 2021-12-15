from __future__ import annotations

import math
import tempfile
import textwrap
from collections.abc import Iterable
from copy import deepcopy, copy
from numbers import Number
from typing import Optional, Union, Tuple, List, Callable
from collections import OrderedDict
from warnings import warn

import numpy as np
import torch

from .memmap import MemmapTensor
from .metatensor import MetaTensor
from ..batchers.utils import expand_as_right

__all__ = ["TensorDict", "SubTensorDict", "merge_tensor_dicts", "LazyStackedTensorDict", "SavedTensorDict"]

from .utils import _sub_index, _td_fields, _getitem_batch_size, _check_keys

TD_HANDLED_FUNCTIONS = dict()

_accepted_classes = (torch.Tensor, MemmapTensor)


class _TensorDict:
    _safe = False

    def __init__(self, source: Optional[dict] = None):
        raise NotImplementedError

    @property
    def shape(self) -> torch.Size:
        return self.batch_size

    @property
    def batch_size(self) -> torch.Size:
        raise NotImplementedError

    @property
    def batch_dims(self) -> int:
        return len(self.batch_size)

    @property
    def device(self) -> torch.device:
        raise NotImplementedError

    def is_shared(self, no_check=True) -> bool:
        return all([item.is_shared() for key, item in self.items_meta()])

    def is_memmap(self) -> bool:
        return all([item.is_memmap() for key, item in self.items_meta()])

    def numel(self) -> int:
        return math.prod(self.batch_size)

    def _check_batch_size(self):
        bs = [value.shape[:self.batch_dims] for key, value in self.items_meta()]
        if len(bs):
            assert bs[0] == self.batch_size, (
                "batch_size provided during initialization violates batch size of registered tensors, "
                f"got self._batch_size={self.batch_size} and tensor.shape[:batch_dim]={bs[0]}"
            )
        if len(bs) > 1:
            for _bs in bs[1:]:
                assert (
                        _bs == bs[0]
                ), f"batch_size are incongruent, got {_bs} and {bs[0]} -- expected {self.batch_size}"

    def _check_is_shared(self):
        raise NotImplementedError(f"{self.__class__.__name__}")

    def _check_device(self):
        raise NotImplementedError(f"{self.__class__.__name__}")

    def set(self, key, item, **kwargs) -> _TensorDict:
        raise NotImplementedError(f"{self.__class__.__name__}")

    def set_(self, key, item) -> _TensorDict:
        raise NotImplementedError(f"{self.__class__.__name__}")

    def get(self, key: str, default: Optional[torch.Tensor] = None) -> torch.Tensor:
        raise NotImplementedError(f"{self.__class__.__name__}")

    def _get_meta(self, key, **kwargs) -> MetaTensor:
        raise NotImplementedError(f"{self.__class__.__name__}")

    def apply(self, fn: Callable):
        for key, item in self.items():
            item_trsf = fn(item)
            if item_trsf is not None:
                self.set(key, item_trsf, inplace=True)
        return self

    def update(self, input_dict_or_td: Union[dict, _TensorDict], clone: bool=False, inplace: bool=True, **kwargs) -> _TensorDict:
        """updates the TensorDict with values from either a dictionary or another TensorDict.
        Returns self.
        If it is the same tensordict, this is a no-op.
        kwargs to TensorDict.set (specifically, if one wants inplace=False) can be passed to this method.
        """
        if input_dict_or_td is self:
            # no op
            return self
        for key, value in input_dict_or_td.items():
            assert isinstance(value, _accepted_classes)
            if clone:
                value = value.clone()
            self.set(key, value, inplace=inplace, **kwargs)
        return self

    def update_(self, input_dict_or_td: Union[dict, _TensorDict], clone: bool=False) -> _TensorDict:
        """unlike TensorDict.update, this function will return an error if the key is unknown to the TensorDict"""
        if input_dict_or_td is self:
            # no op
            return self
        for key, value in input_dict_or_td.items():
            assert isinstance(value, _accepted_classes)
            if clone:
                value = value.clone()
            self.set_(key, value)
        return self

    def update_at_(self, input_dict_or_td: Union[dict, _TensorDict], idx: Union[List, torch.Size], clone: bool=False) -> _TensorDict:
        for key, value in input_dict_or_td.items():
            assert isinstance(value, _accepted_classes)
            if clone:
                value = value.clone()
            self.set_at_(
                key, value, idx,
            )
        return self

    def _process_tensor(self, tensor: torch.Tensor, check_device: bool=True, check_tensor_shape: bool=True, ) -> torch.Tensor:
        # TODO: move to _TensorDict?
        if not isinstance(tensor, _accepted_classes):
            tensor = torch.tensor(tensor, device=self.device)
        if check_device and self.device and tensor.device is not self.device:
            tensor = tensor.to(self.device)
        if check_tensor_shape and tensor.shape[: self.batch_dims] != self.batch_size:
            raise RuntimeError(
                f"batch dimension mismatch, got self.batch_size={self.batch_size} "
                f"and tensor.shape[:self.batch_dims]={tensor.shape[: self.batch_dims]}"
            )
        if tensor.ndimension() == 0:
            tensor = tensor.view(1)
        return tensor

    def pin_memory(self) -> _TensorDict:
        raise NotImplementedError(f"{self.__class__.__name__}")

    def items(self) -> Iterable:
        raise NotImplementedError(f"{self.__class__.__name__}")

    def items_meta(self) -> Iterable:
        raise NotImplementedError(f"{self.__class__.__name__}")

    def keys(self) -> Iterable:
        raise NotImplementedError(f"{self.__class__.__name__}")

    def expand(self, *shape: Iterable) -> _TensorDict:
        return TensorDict(
            source={key: value.expand(*shape, *value.shape) for key, value in self.items()},
            batch_size=[*shape, *self.batch_size]
        )

    def __eq__(self, other: _TensorDict) -> _TensorDict:
        assert isinstance(other, _TensorDict)
        keys1 = set(self.keys())
        keys2 = set(other.keys())
        assert not len(keys1.difference(keys2)) and len(keys1) == len(
            keys2
        ), f"keys in {self} and {other} mismatch, got {keys1} and {keys2}"
        d = dict()
        for (key, item1) in self.items():
            d[key] = item1 == other.get(key)
        return TensorDict(batch_size=self.batch_size, source=d)

    def del_(self, key):
        raise NotImplementedError(f"{self.__class__.__name__}")

    def select(self, *keys):
        raise NotImplementedError(f"{self.__class__.__name__}")

    def set_at_(self, key, tensor, idx):
        raise NotImplementedError(f"{self.__class__.__name__}")

    def copy_(self, tensor_dict):
        return self.update_(tensor_dict)

    def copy_at_(self, tensor_dict, idx):
        return self.update_at_(tensor_dict, idx)

    def get_at(self, key, idx, default=None):
        try:
            return self.get(key)[idx]
        except KeyError:
            if default is not None:
                return default
            raise KeyError(
                f"key {key} not found in {self.__class__.__name__} with keys {sorted(list(self.keys()))}"
            )

    def share_memory_(self):
        raise NotImplementedError(f"{self.__class__.__name__}")

    def memmap_(self):
        raise NotImplementedError(f"{self.__class__.__name__}")

    def zero_(self):
        for key in self.keys():
            self.fill_(key, 0)
        return self

    def unbind(self, dim):
        idx = [(tuple(slice(None) for _ in range(dim)) + (i,)) for i in range(self.shape[dim])]
        return tuple(self[_idx] for _idx in idx)

    def clone(self, recursive=True):
        """clones a tensordict.
        For SubTensorDict objects, this will return a truncated version of the tensordict without access to parent tensordict."""
        return TensorDict(
            source={key: value.clone() if recursive else value for key, value in self.items()},
            batch_size=self.batch_size,
        )

    def __torch_function__(self, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        if func not in TD_HANDLED_FUNCTIONS or not all(
                issubclass(t, (torch.Tensor, _TensorDict)) for t in types
        ):
            return NotImplemented
        return TD_HANDLED_FUNCTIONS[func](*args, **kwargs)

    def to(self, dest, **kwargs):
        raise NotImplementedError

    def cpu(self):
        return self.to('cpu')

    def cuda(self):
        return self.to('cuda')

    def masked_fill_(self, mask, value):
        raise NotImplementedError

    def masked_select(self, mask):
        d = dict()
        for key, value in self.items():
            mask_expand = mask.squeeze(-1)
            value_select = value[mask_expand]
            d[key] = value_select
        return TensorDict(
            device=self.device,
            source=d,
            batch_size=torch.Size([mask.sum()])
        )

    def contiguous(self):
        return self

    def to_dict(self):
        return {key: value for key, value in self.items()}

    def unsqueeze(self, dim):
        if dim < 0:
            dim = self.batch_dims + dim + 1
        return UnsqueezedTensorDict(source=self, custom_op="unsqueeze", inv_op="squeeze", custom_op_kwargs={"dim": dim},
                                    inv_op_kwargs={"dim": dim})

    def squeeze(self, dim):
        if dim < 0:
            dim = self.batch_dims + dim
        if dim >= self.batch_dims or self.batch_size[dim] != 1:
            return self
        return SqueezedTensorDict(source=self, custom_op="squeeze", inv_op="unsqueeze", custom_op_kwargs={"dim": dim},
                                  inv_op_kwargs={"dim": dim})

    def view(self, *shape, size: Optional[Union[List, Tuple, torch.Size]] = None):
        if len(shape) == 0 and size is not None:
            return self.view(*size)
        elif len(shape) == 1 and isinstance(shape[0], (list, tuple, torch.Size)):
            return self.view(*shape[0])
        elif not isinstance(shape, torch.Size):
            shape = torch.Size(shape)
        return ViewedTensorDict(source=self, custom_op='view', inv_op='view',
                                custom_op_kwargs={'size': shape}, inv_op_kwargs={'size': self.batch_size})

    def select(self, *keys, inplace=False):
        raise NotImplementedError(f"{self.__class__.__name__}")

    def __repr__(self):
        fields = _td_fields(self)
        return f"{type(self).__name__}(" \
               f"\n\tfields={{{fields}}}, " \
               f"\n\tbatch_size={self.batch_size}, " \
               f"\n\tdevice={self.device})"

    def get_sub_tensor_dict(
            self, idx,
    ):
        sub_td = SubTensorDict(
            source=self,
            idx=idx,
        )
        return sub_td

    def __iter__(self):
        if not self.batch_dims:
            raise StopIteration
        length = self.batch_size[0]
        for i in range(length):
            yield self[i]

    def __len__(self):
        """

        Returns: Number of keys in _TensorDict instance.

        """
        return len(list(self.keys()))

    def __getitem__(self, idx: Union[torch.Tensor, slice, Number, Tuple]):
        if isinstance(idx, Number):
            idx = (idx,)
        elif isinstance(idx, torch.Tensor) and idx.dtype == torch.bool:
            return self.masked_select(idx)
        assert len(self.batch_size), "indexing a tensordict with td.batch_dims==0 is not permitted"
        if not self.is_memmap():
            return TensorDict(
                source={key: item[idx] for key, item in self.items()},
                batch_size=_getitem_batch_size(self.batch_size, idx),
            )
        return self.get_sub_tensor_dict(idx)

    def __setitem__(self, index: Union[List, Tuple, torch.Tensor, int, slice], value: _TensorDict):
        indexed_bs = _getitem_batch_size(self.batch_size, index)
        if value.batch_size != indexed_bs:
            raise RuntimeError(f"indexed destination TensorDict batch size is {indexed_bs} "
                               f"(batch_size = {self.batch_size}, index={index}), "
                               f"which differs from the source batch size {value.batch_size}")
        for key, item in value.items():
            self.set_at_(key, item, index)

    def rename_key(self, key1: str, key2: str, **kwargs):
        raise NotImplementedError

    def fill_(self, key, value):
        meta_tensor = self._get_meta(key)
        shape = meta_tensor.shape
        device = meta_tensor.device
        dtype = meta_tensor.dtype
        value = torch.full(shape, value, device=device, dtype=dtype)
        self.set_(key, value)
        return self

    def empty(self):
        return self.select()

class TensorDict(_TensorDict):
    _safe = True

    def __init__(
            self,
            source: Optional[Union[_TensorDict, dict]] = None,
            batch_size: Optional[Union[list, tuple, torch.Size, int]] = None,
            device: Optional[Union[str, torch.device, int]] = None,
    ):
        self._tensor_dict = dict()
        self._tensor_dict_meta = OrderedDict()
        if source is None:
            source = dict()

        if isinstance(batch_size, (Number, Iterable,)):
            if not isinstance(batch_size, torch.Size):
                if isinstance(batch_size, Number):
                    batch_size = torch.Size([batch_size])
                else:
                    batch_size = torch.Size(batch_size)
            self._batch_size = batch_size
            self._batch_dims = len(batch_size)
        elif isinstance(source, _TensorDict):
            self._batch_size = source.batch_size
        else:
            raise Exception("batch size was not specified when creating the TensorDict instance and it could not be "
                            "retrieved from source.")

        if isinstance(device, (int, str)):
            device = torch.device(device)
        map_item_to_device = device is not None
        self._device = device
        if source is not None:
            for key, value in source.items():
                assert isinstance(value, _accepted_classes)
                if map_item_to_device:
                    value = value.to(device)
                self.set(key, value)
        self._check_batch_size()
        self._check_device()

    def _batch_dims_get(self):
        if self._safe and hasattr(self, '_batch_dims'):
            assert len(self.batch_size) == self._batch_dims
        return len(self.batch_size)

    def _batch_dims_set(self, value):
        raise RuntimeError(f"Setting batch dims on {self.__class__.__name__} instances is not allowed.")

    batch_dims = property(_batch_dims_get, _batch_dims_set)

    def is_shared(self, no_check=False):
        if no_check:
            for key, item in self.items_meta():
                return item.is_shared()
        return self._check_is_shared()

    def is_memmap(self):
        return self._check_is_memmap()

    def _device_get(self):
        device = self._device
        if device is None and len(self):
            device = next(self.items_meta())[1].device
        if not isinstance(device, torch.device) and device is not None:
            device = torch.device(device)
        self._device = device
        return device

    def _device_set(self, value):
        raise RuntimeError(f"Setting device on {self.__class__.__name__} instances is not allowed. "
                           f"Please call {self.__class__.__name__}.to(device) instead.")

    device = property(_device_get, _device_set)

    def _batch_size_get(self):
        return self._batch_size

    def _batch_size_set(self, value):
        raise RuntimeError(f"Setting batch size on {self.__class__.__name__} instances is not allowed.")

    batch_size = property(_batch_size_get, _batch_size_set)

    # Checks
    def _check_is_shared(self):
        share_list = [value.is_shared() for key, value in self.items_meta()]
        shared_str = ", ".join([f"{key}: {value.is_shared()}" for key, value in self.items_meta()])
        assert all(share_list) or not any(
            share_list
        ), f"tensors must be either all shared or not, but mixed features is not allowed. Found: {shared_str}"
        return all(share_list) and len(share_list) > 0

    def _check_is_memmap(self):
        memmap_list = [value.is_memmap() for key, value in self.items_meta()]
        memmap_str = ", ".join([f"{key}: {value.is_memmap()}" for key, value in self.items_meta()])
        assert all(memmap_list) or not any(
            memmap_list
        ), f"tensors must be either all MemmapTensor or not, but mixed features is not allowed. Found: {memmap_str}"
        return all(memmap_list) and len(memmap_list) > 0

    def _check_device(self):
        devices = {key: value.device for key, value in self.items_meta()}
        if len(devices):
            assert (
                    len(np.unique([str(device) for key, device in devices.items()])) == 1
            ), f"expected tensors to be on a single device, found {devices}"
            device = devices[list(devices.keys())[0]]
            assert torch.device(
                device) == self.device, f"expected {self.__class__.__name__}.device to be identical to tensors " \
                                        f"device, found {self.__class__.__name__}.device={self.device} and {device}"

    def _process_tensor(self, tensor, check_device=True, check_tensor_shape=True, check_shared=True):
        # TODO: move to _TensorDict?
        if not isinstance(tensor, _accepted_classes):
            tensor = torch.tensor(tensor, device=self.device)
        if check_device and self.device and tensor.device is not self.device:
            tensor = tensor.to(self.device)
        if check_shared:
            if self.is_shared():
                tensor = tensor.share_memory_()
            elif self.is_memmap():
                tensor = MemmapTensor(tensor)
            elif tensor.is_shared() and len(self):
                tensor = tensor.clone()
        if check_tensor_shape and tensor.shape[: self.batch_dims] != self.batch_size:
            raise RuntimeError(
                f"batch dimension mismatch, got self.batch_size={self.batch_size} "
                f"and tensor.shape[:self.batch_dims]={tensor.shape[: self.batch_dims]}"
            )
        # minimum ndimension is 1
        if tensor.ndimension() == 0:
            tensor = tensor.view(1)
        return tensor

    def pin_memory(self):
        if self.device == torch.device("cpu"):
            for key, value in self.items():
                if value.dtype in (torch.half, torch.float, torch.double):
                    value.pin_memory()
        return self

    def expand(self, *shape, inplace=False):
        """expand every tensor with (*shape, *tensor.shape) and returns the same tensordict with new tensors with expanded shapes."""
        _batch_size = torch.Size([*shape, *self.batch_size])
        _batch_dims = len(_batch_size)
        d = {key: value.expand(*shape, *value.shape) for key, value in self.items()}
        if inplace:
            d_meta = {key: value.expand(*shape, *value.shape) for key, value in self.items_meta()}
            self._tensor_dict = d
            self._tensor_dict_meta = d_meta
            self._batch_size = _batch_size
            self._batch_dims = _batch_dims
        return TensorDict(source=d, batch_size=_batch_size)

    def all(self, dim=None):
        if dim is not None:
            raise NotImplementedError("TODO: td.all(dim=dim) is a missing feature")
        return all([value.all() for key, value in self.items()])

    def set(self, key, tensor, inplace=True, _run_checks=True):
        """Sets a value in the TensorDict. If inplace=True (default), if the key already exists, set will call set_ (in place setting). 
        """
        tensor = self._process_tensor(
            tensor, check_tensor_shape=_run_checks, check_shared=_run_checks,
        )  # check_tensor_shape=_run_checks
        if key in self._tensor_dict and inplace:
            return self.set_(key, tensor)
        self._tensor_dict[key] = tensor
        self._tensor_dict_meta[key] = MetaTensor(tensor)
        return self

    def del_(self, key):
        del self._tensor_dict[key]
        del self._tensor_dict_meta[key]
        return self

    def rename_key(self, old_name, new_name, safe=False):
        if safe:
            assert new_name not in self.keys(), f"key {new_name} already present in TensorDict."
        self.set(new_name, self.get(old_name))
        self.del_(old_name)
        return self

    def set_(self, key, tensor):
        tensor = self._process_tensor(tensor, check_device=False, check_shared=False)
        if key in self.keys():
            target_shape = self._get_meta(key).shape
            assert (tensor.shape == target_shape), \
                f"calling set_(\"{key}\", tensor) with tensors of " \
                f"different shape: got tensor.shape={tensor.shape} and get(\"{key}\").shape={target_shape}"
            self._tensor_dict[key].copy_(tensor)
        else:
            raise AttributeError(
                f"key {key} not found in tensordict, "
                f"call td.set({key}, value) for populating tensordict with new key-value pair"
            )
        return self

    def set_at_(self, key, tensor, idx):
        tensor = self._process_tensor(tensor, check_tensor_shape=False, check_device=False)
        assert (key in self.keys()), f"did not find key {key} in {self.__class__.__name__}"
        tensor_in = self._tensor_dict[key]
        if isinstance(idx, tuple) and len(idx) and isinstance(idx[0], tuple):
            warn("Multiple indexing can lead to unexpected behaviours when setting items,"
                 "for instance `td[idx1][idx2] = other` may not write to the desired location if idx1 is a list/tensor.")
            tensor_in = _sub_index(tensor_in, idx)
            tensor_in.copy_(tensor)
        else:
            tensor_in[idx] = tensor
        return self

    def get(self, key: str, default: Optional[torch.Tensor] = None):
        try:
            return self._tensor_dict[key]
        except KeyError:
            if default is not None:
                return default
            raise KeyError(
                f"key {key} not found in {self.__class__.__name__} with keys {sorted(list(self.keys()))}"
            )

    def _get_meta(self, key):
        try:
            return self._tensor_dict_meta[key]
        except KeyError:
            raise KeyError(
                f"key {key} not found in {self.__class__.__name__} with keys {sorted(list(self.keys()))}"
            )

    def share_memory_(self):
        assert not self.is_memmap(), "memmap and shared memory are mutually exclusive features."
        if not len(self._tensor_dict):
            raise Exception(
                "share_memory_ must be called when the TensorDict is (partially) populated. Set a tensor first."
            )
        for key, value in self.items():
            value.share_memory_()
        for key, value in self.items_meta():
            value.share_memory_()
        return self

    def memmap_(self):
        assert not self.is_memmap(), "memmap and shared memory are mutually exclusive features."
        if not len(self._tensor_dict):
            raise Exception(
                "memmap_() must be called when the TensorDict is (partially) populated. Set a tensor first."
            )
        for key, value in self.items():
            self._tensor_dict[key] = MemmapTensor(value)
        for key, value in self.items_meta():
            value.memmap_()
        return self

    def to(self, dest, **kwargs):
        if isinstance(dest, type) and issubclass(dest, _TensorDict):
            td = dest(
                source=self,
                **kwargs,
            )
            return td
        elif isinstance(dest, (torch.device, str, int)):
            # must be device
            if not isinstance(dest, torch.device):
                dest = torch.device(dest)
            if dest == self.device:
                return self

            self_copy = TensorDict(source={
                key: value.to(dest) for key, value in self.items()}, batch_size=self.batch_size)
            if self._safe:
                # sanity check
                self_copy._check_device()
                self_copy._check_is_shared()
            return self_copy
        else:
            raise NotImplementedError(
                f"dest must be a string, torch.device or a TensorDict instance, {dest} not allowed"
            )

    def masked_fill_(self, mask, value):
        for key, value in self.items():
            mask_expand = expand_as_right(mask, value)
            value.masked_fill_(mask_expand, value)
        return self

    def contiguous(self):
        return self

    def select(self, *keys, inplace=False):
        d = {key: value for (key, value) in self.items() if key in keys}
        if inplace:
            self._tensor_dict = d
            self._tensor_dict_meta = {key: value for (key, value) in self.items_meta() if key in keys}
            return self
        return TensorDict(
            device=self.device,
            batch_size=self.batch_size,
            source=d
        )

    def items(self):
        for k in self._tensor_dict:
            yield k, self.get(k)

    def items_meta(self):
        for k in self._tensor_dict_meta:
            yield k, self._get_meta(k)

    def keys(self):
        return self._tensor_dict.keys()

    def __repr__(self):
        fields = _td_fields(self)
        return f"{type(self).__name__}(" \
               f"\n\tfields={{{fields}}}, " \
               f"\n\tshared={self.is_shared()}, " \
               f"\n\tbatch_size={self.batch_size})"


import functools


def implements_for_td(torch_function):
    """Register a torch function override for ScalarTensor"""

    @functools.wraps(torch_function)
    def decorator(func):
        TD_HANDLED_FUNCTIONS[torch_function] = func
        return func

    return decorator


# @implements_for_td(torch.testing.assert_allclose) TODO
def assert_allclose_td(
        actual, expected, rtol=None, atol=None, equal_nan=True, msg=""
) -> None:
    assert isinstance(actual, _TensorDict) and isinstance(
        expected, _TensorDict
    ), "assert_allclose inputs must be of TensorDict dtype"
    set1 = set(actual.keys())
    set2 = set(expected.keys())
    if not (len(set1.difference(set2)) == 0 and len(set2) == len(set1)):
        raise KeyError(
            "actual and expected tensordict keys mismatch, "
            f"keys {(set1 - set2).union(set2 - set1)} appear in one but not the other."
        )
    keys = sorted(list(actual.keys()))
    for key in keys:
        input1 = actual.get(key)
        input2 = expected.get(key)
        mse = (
            (input1.to(torch.float) - input2.to(torch.float))
                .pow(2)
                .sum()
                .div(input1.numel())
                .sqrt()
                .item()
        )

        default_msg = f"key {key} does not match, got mse = {mse:4.4f}"
        if len(msg):
            msg = "\t".join([default_msg, msg])
        else:
            msg = default_msg
        torch.testing.assert_allclose(
            input1, input2, rtol=rtol, atol=atol, equal_nan=equal_nan, msg=msg
        )
    return True


@implements_for_td(torch.unbind)
def unbind(td, *args, **kwargs):
    return td.unbind(*args, **kwargs)


@implements_for_td(torch.clone)
def clone(td, *args, **kwargs):
    return td.clone(*args, **kwargs)


@implements_for_td(torch.cat)
def cat(list_of_tensor_dicts, dim=0, device=None, out=None):
    assert len(list_of_tensor_dicts), "list_of_tensor_dicts cannot be empty"
    if not device:
        device = list_of_tensor_dicts[0].device
    assert (
            dim >= 0
    ), f"negative dim in torch.dim(list_of_tensor_dicts, dim=dim) not allowed, got dim={dim}"

    batch_size = list(list_of_tensor_dicts[0].batch_size)
    assert dim < len(
        batch_size
    ), f"dim must be in the range 0 <= dim < len(batch_size), got dim={dim} and batch_size={batch_size}"
    batch_size[dim] = sum([td.batch_size[dim] for td in list_of_tensor_dicts])
    batch_size = torch.Size(batch_size)

    # check that all tensordict match
    keys = _check_keys(list_of_tensor_dicts, strict=True)
    if out is None:
        out_td = TensorDict(device=device, batch_size=batch_size)
        for key in keys:
            out_td.set(key, torch.cat([td.get(key) for td in list_of_tensor_dicts], dim))
        return out_td
    else:
        out_td = out
        assert out.batch_size == batch_size, (
            "out.batch_size and cat batch size must match, "
            f"got out.batch_size={out.batch_size} and batch_size={batch_size}"
        )

        for key in keys:
            out_td.set_(key, torch.cat([td.get(key) for td in list_of_tensor_dicts], dim))
        return out_td


@implements_for_td(torch.stack)
def stack(list_of_tensor_dicts, dim=0, out=None):
    assert len(list_of_tensor_dicts), "list_of_tensor_dicts cannot be empty"
    assert (
            dim >= 0
    ), f"negative dim in torch.stack(list_of_tensor_dicts, dim=dim) not allowed, got dim={dim}"
    batch_size = list_of_tensor_dicts[0].batch_size
    if len(list_of_tensor_dicts) > 1:
        for td in list_of_tensor_dicts[1:]:
            assert td.batch_size == list_of_tensor_dicts[0].batch_size, (
                "stacking tensor_dicts requires them to have congruent batch sizes, "
                f"got td1.batch_size={td.batch_size} and td2.batch_size{list_of_tensor_dicts[0].batch_size}"
            )
    # check that all tensordict match
    keys = _check_keys(list_of_tensor_dicts)
    batch_size = list(batch_size)
    batch_size.insert(dim, len(list_of_tensor_dicts))
    batch_size = torch.Size(batch_size)

    if out is None:
        return LazyStackedTensorDict(*list_of_tensor_dicts, stack_dim=dim, )
    else:
        out_td = out
        assert out.batch_size == batch_size, (
            "out.batch_size and stacked batch size must match, "
            f"got out.batch_size={out.batch_size} and batch_size={batch_size}"
        )
        for key in keys:
            out_td.set(
                key, torch.stack([td.get(key) for td in list_of_tensor_dicts], dim), inplace=True,
            )
        return out_td


# @implements_for_td(torch.nn.utils.rnn.pad_sequence)
def pad_sequence_td(
        list_of_tensor_dicts, batch_first=True, padding_value=0.0, out=None, device=None
):
    assert len(list_of_tensor_dicts), "list_of_tensor_dicts cannot be empty"
    # check that all tensordict match
    keys = _check_keys(list_of_tensor_dicts)
    if out is None:
        out_td = TensorDict(device=device)
        for key in keys:
            out_td.set(
                key,
                torch.nn.utils.rnn.pad_sequence(
                    [td.get(key) for td in list_of_tensor_dicts],
                    batch_first=batch_first,
                    padding_value=padding_value,
                ),
            )
        return out_td
    else:
        out_td = out
        for key in keys:
            out_td.set_(
                key,
                torch.nn.utils.rnn.pad_sequence(
                    [td.get(key) for td in list_of_tensor_dicts],
                    batch_first=batch_first,
                    padding_value=padding_value,
                ),
            )
        return out_td


class SubTensorDict(_TensorDict):
    """A TensorDict that only sees an index of the stored tensors
    """
    _safe = False

    def __init__(self, source: _TensorDict, idx: Union[int, torch.Tensor, Iterable, slice]):
        assert isinstance(source, _TensorDict)
        self._source = source
        if not isinstance(idx, (tuple, list)):
            idx = (idx,)
        else:
            idx = tuple(idx)
        self.idx = idx
        self._batch_size = self.batch_size

    @property
    def batch_size(self):
        batch_size = _getitem_batch_size(self._source.batch_size, self.idx)
        return batch_size

    @property
    def device(self):
        return self._source.device

    def _preallocate(self, key, tensor):
        return self._source.set(key, tensor)

    def set(
            self, key, tensor, inplace=True, _run_checks=True,
    ):
        if inplace and key in self.keys():
            return self.set_(key, tensor)

        tensor = self._process_tensor(tensor, check_device=False, check_tensor_shape=False)
        parent = self.get_parent_tensor_dict()
        tensor_expand = torch.zeros(
            *parent.batch_size, *tensor.shape[self.batch_dims:], dtype=tensor.dtype, device=self.device
        )

        if self.is_shared():
            tensor_expand.share_memory_()
        elif self.is_memmap():
            tensor_expand = MemmapTensor(tensor_expand)

        parent.set(key, tensor_expand, _run_checks=_run_checks)
        self.set_(key, tensor)
        return self

    def keys(self):
        return self._source.keys()

    def set_(self, key, tensor):
        assert key in self.keys()
        assert tensor.shape[:self.batch_dims] == self.batch_size
        return self._source.set_at_(key, tensor, self.idx)

    def to(self, dest, **kwargs):
        if isinstance(dest, type) and issubclass(dest, _TensorDict):
            return dest(
                source=self.clone(),
            )
        elif isinstance(dest, (torch.device, str, int)):
            if not isinstance(dest, torch.device):
                dest = torch.device(dest)
            if dest == self.device:
                return self
            td = self.clone()
            # must be device
            return td.to(dest, **kwargs)
        else:
            raise NotImplementedError(
                f"dest must be a string, torch.device or a TensorDict instance, {dest} not allowed"
            )

    def get(self, key: str, default: Optional[torch.Tensor] = None):
        return self._source.get_at(key, self.idx)

    def _get_meta(self, key):
        return self._source._get_meta(key)[self.idx]

    def set_at_(self, key, value, idx, discard_idx_attr=False):
        if not isinstance(idx, tuple):
            idx = (idx,)
        if discard_idx_attr:
            return self._source.set_at_(key, value, idx)
        else:
            return self._source.set_at_(key, value, (self.idx, idx))

    def get_at(self, key, idx, discard_idx_attr=False):
        if not isinstance(idx, tuple):
            idx = (idx,)
        if discard_idx_attr:
            return self._source.get_at(key, idx)
        else:
            return self._source.get_at(key, self.idx)[idx]

    def update_(self, input_dict, clone=False):
        return self.update_at_(
            input_dict, idx=self.idx, discard_idx_attr=True, clone=clone
        )

    def update_at_(self, input_dict, idx, discard_idx_attr=False, clone=False):
        for key, value in input_dict.items():
            assert isinstance(value, _accepted_classes)
            if clone:
                value = value.clone()
            self.set_at_(
                key, value, idx, discard_idx_attr=discard_idx_attr,
            )
        return self

    def get_parent_tensor_dict(self, ):
        assert isinstance(self._source, _TensorDict), (
            f"SubTensorDict was initialized with a source of type {self._source.__class__.__name__}, "
            "parent tensordict not accessible"
        )
        return self._source

    def del_(self, key):
        self._source = self._source.del_(key)
        return self

    def contiguous(self):
        return TensorDict(batch_size=self.batch_size, source={key: value for key, value in self.items()})

    def items(self):
        for k in self.keys():
            yield k, self.get(k)

    def items_meta(self):
        for key, value in self._source.items_meta():
            yield key, value[self.idx]

    def select(self, *keys, inplace=False):
        if inplace:
            self._source = self._source.select(*keys)
            return self
        return self._source.select(*keys)[self.idx]

    def expand(self, *shape, inplace=False):
        new_source = self._source.expand(*shape)
        idx = tuple(slice(None) for _ in shape) + tuple(self.idx)
        if inplace:
            self._source = new_source
            self.idx = idx
        return new_source[idx]

    def is_shared(self, no_check=True):
        return self._source.is_shared(no_check=no_check)

    def is_memmap(self):
        return self._source.is_memmap()

    def rename_key(self, key1: str, key2: str, **kwargs):
        self._source.rename_key(key1, key2, **kwargs)
        return self

    def pin_memory(self):
        self._source.pin_memory()
        return self


def merge_tensor_dicts(*tensor_dicts):
    assert (
            len(tensor_dicts) >= 2
    ), f"at least 2 tensor_dicts must be provided, got {len(tensor_dicts)}"
    d = tensor_dicts[0].to_dict()
    for td in tensor_dicts[1:]:
        d.update(td.to_dict())
    return TensorDict(device=td.device).update(d)


class LazyStackedTensorDict(_TensorDict):
    _safe = False

    def __init__(self, *tensor_dicts: List[TensorDict], stack_dim=0):
        # sanity check
        N = len(tensor_dicts)
        assert isinstance(tensor_dicts[0], _TensorDict), f"Expected input to be _TensorDict instance" \
                                                         f" but got {type(tensor_dicts[0])} instead."
        assert stack_dim >= 0, f"stack_dim must be non negative, got stack_dim={stack_dim}"
        assert N, "at least one tensordict must be provided to StackedTensorDict to be instantiated"
        batch_size = tensor_dicts[0].batch_size
        device = tensor_dicts[0].device
        keys = set(tensor_dicts[0].keys())

        for i, td in enumerate(tensor_dicts[1:]):
            assert isinstance(td, _TensorDict), f"Expected input to be _TensorDict instance" \
                                                f" but got {type(tensor_dicts[0])} instead."
            _bs = td.batch_size
            _device = td.device
            assert device == _device, f"devices differ, got {device} and {_device}"
            _keys = set(td.keys())
            assert _bs == batch_size, f"batch sizes in tensor_dicts differs, StackedTensorDict cannot be created. Got " \
                                      f"td[0].batch_size={batch_size} and td[i].batch_size={_bs} "
        self.tensor_dicts = list(tensor_dicts)
        self.stack_dim = stack_dim
        self._batch_size = self._compute_batch_size(batch_size, stack_dim, N)
        self._batch_dims = len(self._batch_size)
        self._update_valid_keys()

    @property
    def device(self):
        device_set = {td.device for td in self.tensor_dicts}
        assert len(device_set) == 1, f"found multiple devices in {self.__class__.__name__}: {device_set}"
        return self.tensor_dicts[0].device

    @property
    def batch_size(self):
        return self._batch_size

    def is_shared(self, no_check=True):
        return all(td.is_shared(no_check=no_check) for td in self.tensor_dicts)

    def is_memmap(self):
        return all(td.is_memmap() for td in self.tensor_dicts)

    def get_valid_keys(self):
        self._update_valid_keys()
        return self._valid_keys

    def set_valid_keys(self, keys):
        raise RuntimeError('setting valid keys is not permitted. valid keys are defined as the intersection of all the '
                           'key sets from the TensorDicts in a stack and cannot be defined explicitely.')

    valid_keys = property(get_valid_keys, set_valid_keys)

    @staticmethod
    def _compute_batch_size(batch_size, stack_dim, N):
        s_list = list(reversed(list(batch_size)))
        s = torch.Size([s_list.pop() if i != stack_dim else N for i in range(len(batch_size) + 1)])
        return s

    def set(self, key, tensor, **kwargs):
        assert self.batch_size == tensor.shape[:self.batch_dims]
        tensor = self._process_tensor(tensor, check_device=False, check_tensor_shape=False)
        tensor = tensor.unbind(self.stack_dim)
        for td, _item in zip(self.tensor_dicts, tensor):
            td.set(key, _item, **kwargs)
        return self

    def set_(self, key, tensor, **kwargs):

        assert self.batch_size == tensor.shape[:self.batch_dims]
        assert key in self.valid_keys, 'setting a value in-place on a stack of TensorDict is only permitted if all ' \
                                       'members of the stack have this key in their register.'
        tensor = self._process_tensor(tensor, check_device=False, check_tensor_shape=False)
        tensor = tensor.unbind(self.stack_dim)
        for td, _item in zip(self.tensor_dicts, tensor):
            td.set_(key, _item, **kwargs)
        return self

    def set_at_(self, key, tensor, idx):
        sub_td = self[idx]
        sub_td.set_(key, tensor)
        return self

    def get(self, key: str, default: Optional[torch.Tensor] = None, **kwargs):
        if not (key in self.valid_keys):
            raise KeyError(f"key {key} not found in {list(self._valid_keys)}")
        tensors = [td.get(key, default=default, **kwargs) for td in self.tensor_dicts]
        shapes = set(tensor.shape for tensor in tensors)
        assert len(
            shapes) == 1, f'found more than one unique shape in the tensors to be stacked ({shapes}). This is likely due to ' \
                          'a modification of one of the stacked TensorDicts, where a key has been updated/created with ' \
                          'an uncompatible shape.'
        return torch.stack(tensors, self.stack_dim)

    def _get_meta(self, key, **kwargs):
        assert key in self.valid_keys, f"key {key} not found in {list(self._valid_keys)}"
        return torch.stack([td._get_meta(key, **kwargs) for td in self.tensor_dicts], self.stack_dim)

    def contiguous(self):
        return super().clone()

    def clone(self, recursive=True):
        if recursive:
            return LazyStackedTensorDict(*[td.clone() for td in self.tensor_dicts], stack_dim=self.stack_dim)
        return LazyStackedTensorDict(*[td for td in self.tensor_dicts], stack_dim=self.stack_dim)

    def pin_memory(self):
        for td in self.tensor_dicts:
            td.pin_memory()
        return self

    def to(self, dest, **kwargs):
        if isinstance(dest, type) and issubclass(dest, _TensorDict):
            return dest(source=self)
        elif isinstance(dest, (torch.device, str, int)):
            if not isinstance(dest, torch.device):
                dest = torch.device(dest)
            if dest == self.device:
                return self
            tds = [td.to(dest) for td in self.tensor_dicts]
            self_copy = copy(self)
            self_copy.tensor_dicts = tds
            return self_copy
        else:
            raise NotImplementedError(
                f"dest must be a string, torch.device or a TensorDict instance, {dest} not allowed"
            )

    def items(self):
        for key in self.keys():
            item = self.get(key)
            yield key, item

    def items_meta(self):
        for key in self.keys():
            item = self._get_meta(key)
            yield key, item

    def keys(self):
        for key in self.valid_keys:
            yield key

    def _update_valid_keys(self):
        valid_keys = set(self.tensor_dicts[0].keys())
        for td in self.tensor_dicts[1:]:
            valid_keys = valid_keys.intersection(td.keys())
        self._valid_keys = valid_keys

    def select(self, *keys, inplace=False):
        assert len(self.valid_keys.intersection(keys)) == len(keys)
        tensor_dicts = [td.select(*keys, inplace=inplace) for td in self.tensor_dicts]
        if inplace:
            return self
        return LazyStackedTensorDict(
            *tensor_dicts,
            stack_dim=self.stack_dim,
        )

    def __getitem__(self, item: Union[torch.Tensor, slice, Number, Tuple]):

        if isinstance(item, torch.Tensor) and item.dtype == torch.bool:
            return self.masked_select(item)
        elif (isinstance(item, (Number,)) or (
                isinstance(item, torch.Tensor) and item.ndimension() == 0)) and self.stack_dim == 0:
            return self.tensor_dicts[item]
        elif isinstance(item, (torch.Tensor, list)) and self.stack_dim == 0:
            return LazyStackedTensorDict(*[self.tensor_dicts[_item] for _item in item], stack_dim=self.stack_dim)
        elif isinstance(item, slice) and self.stack_dim == 0:
            return LazyStackedTensorDict(*self.tensor_dicts[item], stack_dim=self.stack_dim)
        elif isinstance(item, (slice, Number)):
            new_stack_dim = self.stack_dim - 1 if isinstance(item, Number) else self.stack_dim
            return LazyStackedTensorDict(*[td[item] for td in self.tensor_dicts], stack_dim=new_stack_dim)
        elif isinstance(item, tuple):
            _sub_item = tuple(_item for i, _item in enumerate(item) if i == self.stack_dim)
            if len(_sub_item):
                tensor_dicts = self.tensor_dicts[_sub_item[0]]
                if isinstance(tensor_dicts, _TensorDict):
                    return tensor_dicts
            else:
                tensor_dicts = self.tensor_dicts
            # select sub tensor_dicts
            _sub_item = tuple(_item for i, _item in enumerate(item) if i != self.stack_dim)
            if len(_sub_item):
                tensor_dicts = [td[_sub_item] for td in tensor_dicts]
            new_stack_dim = self.stack_dim - sum([isinstance(_item, Number) for _item in item[:self.stack_dim]])
            return torch.stack(list(tensor_dicts), dim=new_stack_dim)
        else:
            raise NotImplementedError(f"selecting StackedTensorDicts with type {item.__class__.__name__} is not "
                                      f"supported yet")

    def del_(self, *args, **kwargs):
        return torch.stack([td.del_(*args, **kwargs) for td in self.tensor_dicts], self.stack_dim)

    def share_memory_(self):
        for td in self.tensor_dicts:
            td.share_memory_()
        return self

    def memmap_(self):
        for td in self.tensor_dicts:
            td.memmap_()
        return self

    def is_shared(self, no_check=False):
        are_shared = [td.is_shared(no_check=no_check) for td in self.tensor_dicts]
        assert all(are_shared) or not any(are_shared), f"tensor_dicts shared status mismatch, got {sum(are_shared)} " \
                                                       f"shared tensor_dicts and {len(are_shared) - sum(are_shared)} non " \
                                                       f"shared tensordict "
        return all(are_shared)

    def is_memmap(self, no_check=False):
        are_memmap = [td.is_memmap() for td in self.tensor_dicts]
        assert all(are_memmap) or not any(are_memmap), f"tensor_dicts memmap status mismatch, got {sum(are_memmap)} " \
                                                       f"memmap tensor_dicts and {len(are_memmap) - sum(are_memmap)} non " \
                                                       f"memmap tensordict "
        return all(are_memmap)

    def expand(self, *shape, inplace=False):
        stack_dim = self.stack_dim + len(shape)
        tensor_dicts = [td.expand(*shape) for td in self.tensor_dicts]
        if inplace:
            self.tensor_dicts = tensor_dicts
            self.stack_dim = stack_dim
            return self
        return torch.stack(tensor_dicts, stack_dim)

    def update(self, input_dict_or_td, clone=False, **kwargs):
        if input_dict_or_td is self:
            # no op
            return self
        for key, value in input_dict_or_td.items():
            assert isinstance(value, _accepted_classes)
            if clone:
                value = value.clone()
            self.set(key, value, **kwargs)
        return self

    def update_(self, input_dict_or_td, clone=False, **kwargs):
        if input_dict_or_td is self:
            # no op
            return self
        for key, value in input_dict_or_td.items():
            assert isinstance(value, _accepted_classes)
            if clone:
                value = value.clone()
            self.set_(key, value, **kwargs)
        return self

    def rename_key(self, key1: str, key2: str, **kwargs):
        for td in self.tensor_dicts:
            td.rename_key(key1, key2, **kwargs)
        return self


class SavedTensorDict(_TensorDict):
    _safe = False

    def __init__(self, source: _TensorDict, device=None):
        assert isinstance(source, _TensorDict), f"Expected source to be a _TensorDict instance, " \
                                                f"but got {type(source)} instead."
        self.file = tempfile.NamedTemporaryFile()
        self.filename = self.file.name
        if source.is_memmap():
            source = source.clone()
        self._device = device if device else \
            source.device if hasattr(source, 'device') else \
                source[list(source.keys())[0]].device if len(source) else \
                    'cpu'
        td = source
        self._save(td)

    def _save(self, td):
        self._keys = list(td.keys())
        self._device = td.device
        self._batch_size = td.batch_size
        self._td_fields = _td_fields(td)
        self._tensor_dict_meta = {key: value for key, value in td.items_meta()}
        torch.save(td, self.filename)

    def _load(self) -> _TensorDict:
        return torch.load(self.filename, self._device)

    def _get_meta(self, key):
        return self._tensor_dict_meta.get(key)

    @property
    def batch_size(self) -> torch.Size:
        return self._batch_size

    @property
    def device(self) -> torch.device:
        return self._device

    def keys(self) -> Iterable:
        for k in self._keys:
            yield k

    def get(self, key: str, default: Optional[torch.Tensor] = None) -> torch.Tensor:
        td = self._load()
        return td.get(key, default=default)

    def set(self, key, value, **kwargs) -> _TensorDict:
        td = self._load()
        td.set(key, value, **kwargs)
        self._save(td)
        return self

    def expand(self, *shape, inplace=False):
        td = self._load()
        td = td.expand(*shape)
        if inplace:
            self._save(td)
            return self
        return td.to(SavedTensorDict)

    def set_(self, key, value):
        return self.set(key, value)

    def set_at_(self, key, tensor, idx):
        td = self._load()
        td.set_at_(key, tensor, idx)
        self._save(td)
        return self

    def update(self, input_dict_or_td, clone=False, **kwargs):
        if input_dict_or_td is self:
            # no op
            return self
        td = self._load()
        for key, value in input_dict_or_td.items():
            assert isinstance(value, _accepted_classes)
            if clone:
                value = value.clone()
            td.set(key, value, **kwargs)
        self._save(td)
        return self

    def update_(self, input_dict_or_td, clone=False):
        if input_dict_or_td is self:
            return self
        return self.update(input_dict_or_td, clone=clone)

    def __del__(self):
        if hasattr(self, 'file'):
            print('deleting')
            self.file.close()

    def is_shared(self, no_check=False):
        return False

    def is_memmap(self, no_check=False):
        return False

    def share_memory_(self):
        raise RuntimeError("SavedTensorDict cannot be put in shared memory.")

    def memmap__(self):
        raise RuntimeError("SavedTensorDict and memmap are mutually exclusive features.")

    def items(self):
        return self.contiguous().items()

    def items_meta(self):
        return self._tensor_dict_meta.items()

    def contiguous(self):
        return self._load()

    def clone(self, recursive=True):
        raise NotImplementedError("should copy files to another file")
        # td = self._load()
        # if recursive:
        #     td = td.clone()
        # return td

    def select(self, *keys, inplace=False):
        _source = self.clone().select(*keys)
        if inplace:
            self._save(_source)
            return self
        return SavedTensorDict(source=_source)

    def rename_key(self, key1: str, key2: str, **kwargs):
        td = self._load()
        td.rename_key(key1, key2, **kwargs)
        self._save(td)
        return self

    def __repr__(self):
        return f'SavedTensorDict(\n\tfields={{{self._td_fields}}}, \n\tbatch_size={self.batch_size}, \n\tfile={self.filename})'

    def to(self, dest, **kwargs):
        if isinstance(dest, type) and issubclass(dest, _TensorDict):
            td = dest(
                source=self.clone(),
                **kwargs, )
            return td
        elif isinstance(dest, (torch.device, str, int)):
            # must be device
            if not isinstance(dest, torch.device):
                dest = torch.device(dest)
            if dest == self.device:
                return self
            self_copy = copy(self)
            self_copy._device = dest
            return self_copy
        else:
            raise NotImplementedError(
                f"dest must be a string, torch.device or a TensorDict instance, {dest} not allowed"
            )

    def del_(self, key):
        td = self._load()
        td = td.del_(key)
        self._save(td)
        return self

    def pin_memory(self):
        raise RuntimeError("pin_memory requires tensordicts that live in memory.")

    def __reduce__(self, *args, **kwargs):
        if hasattr(self, 'file'):
            file = self.file
            del self.file
            self_copy = copy(self)
            self.file = file
            return super(SavedTensorDict, self_copy).__reduce__(*args, **kwargs)
        return super().__reduce__(*args, **kwargs)

    def __getitem__(self, idx: Union[torch.Tensor, slice, Number, Tuple]):
        if isinstance(idx, Number):
            idx = (idx,)
        elif isinstance(idx, torch.Tensor) and idx.dtype == torch.bool:
            return self.masked_select(idx)
        assert len(self.batch_size), "indexing a tensordict with td.batch_dims==0 is not permitted"
        return self.get_sub_tensor_dict(idx)


class _CustomOpTensorDict(_TensorDict):
    def __init__(
            self,
            source: _TensorDict,
            custom_op: str,
            inv_op: Optional[str] = None,
            custom_op_kwargs: Optional[dict] = None,
            inv_op_kwargs: Optional[dict] = None
    ):
        """
        Encodes lazy operations on tensors contained in a TensorDict.

        Args:
            source:
            custom_op:
            inv_op:
            custom_op_kwargs:
            inv_op_kwargs:
        """
        assert isinstance(source, _TensorDict), f"Expected source to be a _TensorDict isntance, " \
                                                f"but got {type(source)} instead."
        self._source = source
        self.custom_op = custom_op
        self.inv_op = inv_op
        self.custom_op_kwargs = custom_op_kwargs if custom_op_kwargs is not None else {}
        self.inv_op_kwargs = inv_op_kwargs if inv_op_kwargs is not None else {}

    def _update_custom_op_kwargs(self, source_meta_tensor: MetaTensor) -> dict:
        """
        Allows for a transformation to be customized for a certain shape, device or dtype.
        By default, this is a no-op on self.custom_op_kwargs

        Args:
            source_meta_tensor: corresponding MetaTensor

        Returns: a dictionary with the kwargs of the operation to execute for the tensor

        """
        return self.custom_op_kwargs

    def _update_inv_op_kwargs(self, source_meta_tensor: MetaTensor) -> dict:
        """
        Allows for an inverse transformation to be customized for a certain shape, device or dtype.
        By default, this is a no-op on self.inv_op_kwargs

        Args:
            source_meta_tensor: corresponding MetaTensor

        Returns: a dictionary with the kwargs of the operation to execute for the tensor

        """
        return self.inv_op_kwargs

    @property
    def device(self):
        return self._source.device

    def _get_meta(self, key, **kwargs):
        item = self._source._get_meta(key, **kwargs)
        return getattr(item, self.custom_op)(**self._update_custom_op_kwargs(item))

    def items_meta(self):
        for key, value in self._source.items_meta():
            yield key, self._get_meta(key)

    def items(self):
        for key in self._source.keys():
            yield key, self.get(key)

    @property
    def batch_size(self):
        return getattr(MetaTensor(*self._source.batch_size), self.custom_op)(**self.custom_op_kwargs).shape

    def get(self, key: str, default: Optional[torch.Tensor] = None, _return_original_tensor: bool = False):
        source_meta_tensor = self._source._get_meta(key)
        item = self._source.get(key, default)
        transformed_tensor = getattr(item, self.custom_op)(**self._update_custom_op_kwargs(source_meta_tensor))
        if not _return_original_tensor:
            return transformed_tensor
        return transformed_tensor, item

    def set(self, key, tensor, **kwargs):
        if self.inv_op is None:
            raise Exception(
                f"{self.__class__.__name__} does not support setting values. "
                f"Consider calling .contiguous() before calling this method.")
        tensor = self._process_tensor(tensor, check_device=False, check_tensor_shape=False)
        if key in self.keys():
            source_meta_tensor = self._source._get_meta(key)
        else:
            source_meta_tensor = MetaTensor(*tensor.shape, device=tensor.device, dtype=tensor.dtype)
        tensor = getattr(tensor, self.inv_op)(**self._update_inv_op_kwargs(source_meta_tensor))
        self._source.set(key, tensor, **kwargs)
        return self

    def set_(self, key, tensor, **kwargs):
        if self.inv_op is None:
            raise Exception(
                f"{self.__class__.__name__} does not support setting values. "
                f"Consider calling .contiguous() before calling this method.")
        meta_tensor = self._source._get_meta(key)
        tensor = getattr(tensor, self.inv_op)(**self._update_inv_op_kwargs(meta_tensor))
        self._source.set_(key, tensor, **kwargs)
        return self

    def set_at_(self, key, tensor, idx):
        transformed_tensor, original_tensor = self.get(key, _return_original_tensor=True)
        assert transformed_tensor.data_ptr() == original_tensor.data_ptr(), \
            f"{self} original tensor and transformed do not point to the same storage. Setting values in place is not " \
            f"currently supported in this setting, consider calling `td.clone()` before `td.set_at_(...)`"
        transformed_tensor[idx] = tensor
        return self

    def __repr__(self):
        custom_op_kwargs_str = ", ".join([f"{key}={value}" for key, value in self.custom_op_kwargs.items()])
        indented_source = textwrap.indent(f"source={self._source}", "\t")
        return f"{self.__class__.__name__}(\n{indented_source}, \n\top={self.custom_op}({custom_op_kwargs_str}))"

    def keys(self):
        return self._source.keys()

    def select(self, *keys, inplace=False):
        if inplace:
            self._source.select(*keys, inplace=inplace)
            return self
        try:
            return type(self)(
                source=self._source.select(*keys),
                custom_op=self.custom_op,
                inv_op=self.inv_op,
                custom_op_kwargs=self.custom_op_kwargs,
                inv_op_kwargs=self.inv_op_kwargs,
            )
        except TypeError:
            self_copy = deepcopy(self)
            self_copy._source = self._source.select(*keys)
            return self_copy

    def clone(self, recursive=True):
        if not recursive:
            return copy(self)
        return TensorDict(
            source=self.to_dict(),
            batch_size=self.batch_size)

    def contiguous(self):
        return self.clone()

    def rename_key(self, key1, key2, **kwargs):
        self._source.rename_key(key1, key2, **kwargs)
        return self

    def del_(self, key):
        self._source = self._source.del_(key)
        return self

    def to(self, dest, **kwargs):
        if isinstance(dest, type) and issubclass(dest, _TensorDict):
            return dest(source=self.clone())
        elif isinstance(dest, (torch.device, str, int)):
            if torch.device(dest) == self.device:
                return self
            td = self._source.to(dest)
            self_copy = copy(self)
            self_copy._source = td
            return self_copy
        else:
            raise NotImplementedError(
                f"dest must be a string, torch.device or a TensorDict instance, {dest} not allowed"
            )

    def pin_memory(self):
        self._source.pin_memory()
        return self


class UnsqueezedTensorDict(_CustomOpTensorDict):
    def squeeze(self, dim):
        if dim < 0:
            dim = self.batch_dims + dim
        if dim == self.custom_op_kwargs.get("dim"):
            return self._source
        return super().squeeze(dim)


class SqueezedTensorDict(_CustomOpTensorDict):
    def unsqueeze(self, dim):
        if dim < 0:
            dim = self.batch_dims + dim + 1
        if dim == self.inv_op_kwargs.get("dim"):
            return self._source
        return super().unsqueeze(dim)


class ViewedTensorDict(_CustomOpTensorDict):
    def _update_custom_op_kwargs(self, source_meta_tensor: MetaTensor) -> dict:
        new_dim = torch.Size([*self.custom_op_kwargs.get("size"), *source_meta_tensor.shape[self._source.batch_dims:]])
        new_dict = deepcopy(self.custom_op_kwargs)
        new_dict.update({'size': new_dim})
        return new_dict

    def _update_inv_op_kwargs(self, source_meta_tensor: MetaTensor) -> dict:
        new_dim = torch.Size([*self.inv_op_kwargs.get("size"), *source_meta_tensor.shape[self._source.batch_dims:]])
        new_dict = deepcopy(self.inv_op_kwargs)
        new_dict.update({'size': new_dim})
        return new_dict

    def view(self, *shape, size: Optional[Union[List, Tuple, torch.Size]] = None):
        if len(shape) == 0 and size is not None:
            return self.view(*size)
        elif len(shape) == 1 and isinstance(shape[0], (list, tuple, torch.Size)):
            return self.view(*shape[0])
        elif not isinstance(shape, torch.Size):
            shape = torch.Size(shape)
        if shape == self._source.batch_size:
            return self._source
        return super().view(*shape)
