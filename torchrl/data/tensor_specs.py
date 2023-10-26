# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import abc
import math
import warnings
from collections.abc import Iterable
from copy import deepcopy
from dataclasses import dataclass
from functools import wraps
from textwrap import indent
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    ItemsView,
    KeysView,
    List,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
    ValuesView,
)

import numpy as np
import torch
from tensordict import unravel_key
from tensordict.tensordict import LazyStackedTensorDict, TensorDict, TensorDictBase
from tensordict.utils import _getitem_batch_size, NestedKey

from torchrl._utils import get_binary_env_var

DEVICE_TYPING = Union[torch.device, str, int]

INDEX_TYPING = Union[int, torch.Tensor, np.ndarray, slice, List]

SHAPE_INDEX_TYPING = Union[
    int,
    range,
    List[int],
    np.ndarray,
    slice,
    None,
    torch.Tensor,
    type(...),
    Tuple[
        int,
        range,
        List[int],
        np.ndarray,
        slice,
        None,
        torch.Tensor,
        type(...),
        Tuple[Any],
    ],
]

# By default, we do not check that an obs is in the domain. THis should be done when validating the env beforehand
_CHECK_SPEC_ENCODE = get_binary_env_var("CHECK_SPEC_ENCODE")

_DEFAULT_SHAPE = torch.Size((1,))

DEVICE_ERR_MSG = "device of empty CompositeSpec is not defined."
NOT_IMPLEMENTED_ERROR = NotImplementedError(
    "method is not currently implemented."
    " If you are interested in this feature please submit"
    " an issue at https://github.com/pytorch/rl/issues"
)


def _default_dtype_and_device(
    dtype: Union[None, torch.dtype],
    device: Union[None, str, int, torch.device],
) -> Tuple[torch.dtype, torch.device]:
    if dtype is None:
        dtype = torch.get_default_dtype()
    if device is None:
        device = torch.device("cpu")
    device = torch.device(device)
    return dtype, device


def _validate_idx(shape: list[int], idx: int, axis: int = 0):
    """Raise an IndexError if idx is out of bounds for shape[axis].

    Args:
        shape (list[int]): Input shape
        idx (int): Index, may be negative
        axis (int): Shape axis to check
    """
    if idx >= shape[axis] or idx < 0 and -idx > shape[axis]:
        raise IndexError(
            f"index {idx} is out of bounds for axis {axis} with size {shape[axis]}"
        )


def _validate_iterable(
    idx: Iterable[Any], expected_type: type, iterable_classname: str
):
    """Raise an IndexError if the iterable contains a type different from the expected type or Iterable.

    Args:
        idx (Iterable[Any]): Iterable, may contain nested iterables
        expected_type (type): Required item type in the Iterable (e.g. int)
        iterable_classname (str): Iterable type as a string (e.g. 'List'). Logging purpose only.
    """
    for item in idx:
        if isinstance(item, Iterable):
            _validate_iterable(item, expected_type, iterable_classname)
        else:
            if not isinstance(item, expected_type):
                raise IndexError(
                    f"{iterable_classname} indexing expects {expected_type} indices"
                )


def _slice_indexing(shape: list[int], idx: slice) -> List[int]:
    """Given an input shape and a slice index, returns the new indexed shape.

    Args:
        shape (list[int]): Input shape
        idx (slice): Index
    Returns:
        Indexed shape
    Examples:
        >>> _slice_indexing([3, 4], slice(None, 2))
        [2, 4]
        >>> list(torch.rand(3, 4)[:2].shape)
        [2, 4]
    """
    if idx.step == 0:
        raise ValueError("slice step cannot be zero")
    # Slicing an empty shape returns the shape
    if len(shape) == 0:
        return shape

    if idx.start is None:
        start = 0
    else:
        start = idx.start if idx.start >= 0 else max(shape[0] + idx.start, 0)

    if idx.stop is None:
        stop = shape[0]
    else:
        stop = idx.stop if idx.stop >= 0 else max(shape[0] + idx.stop, 0)

    step = 1 if idx.step is None else idx.step
    if step > 0:
        if start >= stop:
            n_items = 0
        else:
            stop = min(stop, shape[0])
            n_items = math.ceil((stop - start) / step)
    else:
        if start <= stop:
            n_items = 0
        else:
            start = min(start, shape[0] - 1)
            n_items = math.ceil((stop - start) / step)
    return [n_items] + shape[1:]


def _shape_indexing(
    shape: Union[list[int], torch.Size, tuple[int]], idx: SHAPE_INDEX_TYPING
) -> List[int]:
    """Given an input shape and an index, returns the size of the resulting indexed spec.

    This function includes indexing checks and may raise IndexErrors.

    Args:
        shape (list[int], torch.Size, tuple[int): Input shape
        idx (SHAPE_INDEX_TYPING): Index
    Returns:
        Shape of the resulting spec
    Examples:
        >>> idx = (2, ..., None)
        >>> DiscreteTensorSpec(2, shape=(3, 4))[idx].shape
        torch.Size([4, 1])
        >>> _shape_indexing([3, 4], idx)
        torch.Size([4, 1])
    """
    if not isinstance(shape, list):
        shape = list(shape)

    if idx is Ellipsis or (
        isinstance(idx, slice) and (idx.step is idx.start is idx.stop is None)
    ):
        return shape

    if idx is None:
        return [1] + shape

    if len(shape) == 0 and (
        isinstance(idx, int)
        or isinstance(idx, range)
        or isinstance(idx, list)
        and len(idx) > 0
    ):
        raise IndexError(
            f"cannot use integer indices on 0-dimensional shape. `{idx}` received"
        )

    if isinstance(idx, int):
        _validate_idx(shape, idx)
        return shape[1:]

    if isinstance(idx, range):
        if len(idx) > 0 and (idx.start >= shape[0] or idx.stop > shape[0]):
            raise IndexError(f"index out of bounds for axis 0 with size {shape[0]}")
        return [len(idx)] + shape[1:]

    if isinstance(idx, slice):
        return _slice_indexing(shape, idx)

    if isinstance(idx, tuple):
        # Supports int, None, slice and ellipsis indices
        # Index on the current shape dimension
        shape_idx = 0
        none_dims = 0
        ellipsis = False
        prev_is_list = False
        shape_len = len(shape)
        for item_idx, item in enumerate(idx):
            if item is None:
                shape = shape[:shape_idx] + [1] + shape[shape_idx:]
                shape_idx += 1
                none_dims += 1
            elif isinstance(item, int):
                _validate_idx(shape, item, shape_idx)
                del shape[shape_idx]
            elif isinstance(item, slice):
                shape[shape_idx] = _slice_indexing([shape[shape_idx]], item)[0]
                shape_idx += 1
            elif item is Ellipsis:
                if ellipsis:
                    raise IndexError("an index can only have a single ellipsis (`...`)")
                # Move to the end of the shape, subtracted by the number of future indices impacting the dimensions (i.e. all except None and ...)
                shape_idx = len(shape) - len(
                    [i for i in idx[item_idx + 1 :] if not (i is None or i is Ellipsis)]
                )
                ellipsis = True
            elif any(
                isinstance(item, _type)
                for _type in [list, tuple, range, np.ndarray, torch.Tensor]
            ):
                while isinstance(idx, tuple) and len(idx) == 1:
                    idx = idx[0]

                # Nested tuples are handled as a list. Numpy behavior
                if isinstance(item, tuple):
                    item = list(item)

                if prev_is_list and isinstance(item, list):
                    del shape[shape_idx]
                    continue

                if isinstance(item, list):
                    prev_is_list = True

                if shape_idx >= len(shape):
                    raise IndexError("Raise IndexError: too many indices for array")

                res = _shape_indexing([shape[shape_idx]], item)
                shape = shape[:shape_idx] + res + shape[shape_idx + 1 :]
                shape_idx += len(res)
            else:
                raise IndexError(
                    f"tuple indexing only supports integers, ranges, slices (`:`), ellipsis (`...`), new axis (`None`), tuples, list, tensor and ndarray indices. {str(type(idx))} received"
                )

        if len(idx) - none_dims - int(ellipsis) > shape_len:
            raise IndexError(
                f"shape is {shape_len}-dimensional, but {len(idx) - none_dims - int(ellipsis)} dimensions were indexed"
            )
        return shape

    if isinstance(idx, list):
        # int indexing only
        _validate_iterable(idx, int, "list")
        for item in np.array(idx).reshape(-1):
            _validate_idx(shape, item, 0)
        return list(np.array(idx).shape) + shape[1:]

    if isinstance(idx, np.ndarray) or isinstance(idx, torch.Tensor):
        # Out of bounds check
        for item in idx.reshape(-1):
            _validate_idx(shape, item)
        return list(_getitem_batch_size(shape, idx))


class invertible_dict(dict):
    """An invertible dictionary.

    Examples:
        >>> my_dict = invertible_dict(a=3, b=2)
        >>> inv_dict = my_dict.invert()
        >>> assert {2, 3} == set(inv_dict.keys())
    """

    def __init__(self, *args, inv_dict=None, **kwargs):
        if inv_dict is None:
            inv_dict = {}
        super().__init__(*args, **kwargs)
        self.inv_dict = inv_dict

    def __setitem__(self, k, v):
        if v in self.inv_dict or k in self:
            raise Exception("overwriting in invertible_dict is not permitted")
        self.inv_dict[v] = k
        return super().__setitem__(k, v)

    def update(self, d):
        raise NotImplementedError

    def invert(self):
        d = invertible_dict()
        for k, value in self.items():
            d[value] = k
        return d

    def inverse(self):
        return self.inv_dict


class Box:
    """A box of values."""

    def __iter__(self):
        raise NotImplementedError

    def to(self, dest: Union[torch.dtype, DEVICE_TYPING]) -> ContinuousBox:
        raise NotImplementedError

    def __repr__(self):
        return f"{self.__class__.__name__}()"

    def clone(self) -> DiscreteBox:
        return deepcopy(self)


@dataclass(repr=False)
class ContinuousBox(Box):
    """A continuous box of values, in between a minimum (self.low) and a maximum (self.high)."""

    _low: torch.Tensor
    _high: torch.Tensor
    device: torch.device = None

    # We store the tensors on CPU to avoid overloading CUDA with tensors that are rarely used.
    @property
    def low(self):
        return self._low.to(self.device)

    @property
    def high(self):
        return self._high.to(self.device)

    @low.setter
    def low(self, value):
        self.device = value.device
        self._low = value.cpu()

    @high.setter
    def high(self, value):
        self.device = value.device
        self._high = value.cpu()

    @property
    def minimum(self):
        warnings.warn(
            f"{type(self)}.minimum is going to be deprecated in favour of {type(self)}.low",
            category=DeprecationWarning,
        )
        return self._low.to(self.device)

    @property
    def maximum(self):
        warnings.warn(
            f"{type(self)}.maximum is going to be deprecated in favour of {type(self)}.high",
            category=DeprecationWarning,
        )
        return self._high.to(self.device)

    @low.setter
    def low(self, value):
        self.device = value.device
        self._low = value.cpu()

    @high.setter
    def high(self, value):
        self.device = value.device
        self._high = value.cpu()

    def __post_init__(self):
        self.low = self.low.clone()
        self.high = self.high.clone()

    def __iter__(self):
        yield self.low
        yield self.high

    def to(self, dest: Union[torch.dtype, DEVICE_TYPING]) -> ContinuousBox:
        return self.__class__(self.low.to(dest), self.high.to(dest))

    def clone(self) -> ContinuousBox:
        return self.__class__(self.low.clone(), self.high.clone())

    def __repr__(self):
        min_str = indent(
            f"\nlow=Tensor(shape={self.low.shape}, device={self.low.device}, dtype={self.low.dtype}, contiguous={self.high.is_contiguous()})",
            " " * 4,
        )
        max_str = indent(
            f"\nhigh=Tensor(shape={self.high.shape}, device={self.high.device}, dtype={self.high.dtype}, contiguous={self.high.is_contiguous()})",
            " " * 4,
        )
        return f"{self.__class__.__name__}({min_str},{max_str})"

    def __eq__(self, other):
        return (
            type(self) == type(other)
            and self.low.dtype == other.low.dtype
            and self.high.dtype == other.high.dtype
            and torch.equal(self.low, other.low)
            and torch.equal(self.high, other.high)
        )


@dataclass(repr=False)
class DiscreteBox(Box):
    """A box of discrete values."""

    n: int
    register = invertible_dict()

    def to(self, dest: Union[torch.dtype, DEVICE_TYPING]) -> DiscreteBox:
        return deepcopy(self)

    def __repr__(self):
        return f"{self.__class__.__name__}(n={self.n})"


@dataclass(repr=False)
class BoxList(Box):
    """A box of discrete values."""

    boxes: List

    def to(self, dest: Union[torch.dtype, DEVICE_TYPING]) -> BoxList:
        return BoxList([box.to(dest) for box in self.boxes])

    def __iter__(self):
        for elt in self.boxes:
            yield elt

    def __repr__(self):
        return f"{self.__class__.__name__}(boxes={self.boxes})"

    def __len__(self):
        return len(self.boxes)

    @staticmethod
    def from_nvec(nvec: torch.Tensor):
        if nvec.ndim == 0:
            return DiscreteBox(nvec.item())
        else:
            return BoxList([BoxList.from_nvec(n) for n in nvec.unbind(-1)])


@dataclass(repr=False)
class BinaryBox(Box):
    """A box of n binary values."""

    n: int

    def to(self, dest: Union[torch.dtype, DEVICE_TYPING]) -> ContinuousBox:
        return deepcopy(self)

    def __repr__(self):
        return f"{self.__class__.__name__}(n={self.n})"


@dataclass(repr=False)
class TensorSpec:
    """Parent class of the tensor meta-data containers for observation, actions and rewards.

    Args:
        shape (torch.Size): size of the tensor
        space (Box): Box instance describing what kind of values can be
            expected
        device (torch.device): device of the tensor
        dtype (torch.dtype): dtype of the tensor

    """

    shape: torch.Size
    space: Union[None, Box]
    device: torch.device = torch.device("cpu")
    dtype: torch.dtype = torch.float
    domain: str = ""

    SPEC_HANDLED_FUNCTIONS = {}

    @classmethod
    def implements_for_spec(cls, torch_function: Callable) -> Callable:
        """Register a torch function override for TensorSpec."""

        @wraps(torch_function)
        def decorator(func):
            cls.SPEC_HANDLED_FUNCTIONS[torch_function] = func
            return func

        return decorator

    def encode(
        self, val: Union[np.ndarray, torch.Tensor], *, ignore_device=False
    ) -> torch.Tensor:
        """Encodes a value given the specified spec, and return the corresponding tensor.

        Args:
            val (np.ndarray or torch.Tensor): value to be encoded as tensor.

        Keyword Args:
            ignore_device (bool, optional): if ``True``, the spec device will
                be ignored. This is used to group tensor casting within a call
                to ``TensorDict(..., device="cuda")`` which is faster.

        Returns:
            torch.Tensor matching the required tensor specs.

        """
        if not isinstance(val, torch.Tensor):
            if isinstance(val, list):
                if len(val) == 1:
                    # gym used to return lists of images since 0.26.0
                    # We convert these lists in np.array or take the first element
                    # if there is just one.
                    # See https://github.com/pytorch/rl/pull/403/commits/73d77d033152c61d96126ccd10a2817fecd285a1
                    val = val[0]
                else:
                    val = np.array(val)
            if isinstance(val, np.ndarray) and not all(
                stride > 0 for stride in val.strides
            ):
                val = val.copy()
            if not ignore_device:
                val = torch.tensor(val, device=self.device, dtype=self.dtype)
            else:
                val = torch.tensor(val, dtype=self.dtype)
            if val.shape != self.shape:
                # if val.shape[-len(self.shape) :] != self.shape:
                # option 1: add a singleton dim at the end
                if val.shape == self.shape and self.shape[-1] == 1:
                    val = val.unsqueeze(-1)
                else:
                    try:
                        val = val.reshape(self.shape)
                    except Exception as err:
                        raise RuntimeError(
                            f"Shape mismatch: the value has shape {val.shape} which "
                            f"is incompatible with the spec shape {self.shape}."
                        ) from err
        if _CHECK_SPEC_ENCODE:
            self.assert_is_in(val)
        return val

    def __ne__(self, other):
        return not (self == other)

    def __setattr__(self, key, value):
        if key == "shape":
            value = torch.Size(value)
        super().__setattr__(key, value)

    def to_numpy(self, val: torch.Tensor, safe: bool = None) -> np.ndarray:
        """Returns the np.ndarray correspondent of an input tensor.

        Args:
            val (torch.Tensor): tensor to be transformed_in to numpy.
            safe (bool): boolean value indicating whether a check should be
                performed on the value against the domain of the spec.
                Defaults to the value of the ``CHECK_SPEC_ENCODE`` environment variable.

        Returns:
            a np.ndarray

        """
        if safe is None:
            safe = _CHECK_SPEC_ENCODE
        if safe:
            self.assert_is_in(val)
        return val.detach().cpu().numpy()

    @property
    def ndim(self):
        return self.ndimension()

    def ndimension(self):
        return len(self.shape)

    @abc.abstractmethod
    def index(self, index: INDEX_TYPING, tensor_to_index: torch.Tensor) -> torch.Tensor:
        """Indexes the input tensor.

        Args:
            index (int, torch.Tensor, slice or list): index of the tensor
            tensor_to_index: tensor to be indexed

        Returns:
            indexed tensor

        """
        raise NotImplementedError

    @abc.abstractmethod
    def expand(self, *shape):
        """Returns a new Spec with the extended shape.

        Args:
            *shape (tuple or iterable of int): the new shape of the Spec. Must comply with the current shape:
                its length must be at least as long as the current shape length,
                and its last values must be complient too; ie they can only differ
                from it if the current dimension is a singleton.

        """
        raise NotImplementedError

    def squeeze(self, dim: int | None = None):
        """Returns a new Spec with all the dimensions of size ``1`` removed.

        When ``dim`` is given, a squeeze operation is done only in that dimension.

        Args:
            dim (int or None): the dimension to apply the squeeze operation to

        """
        shape = _squeezed_shape(self.shape, dim)
        if shape is None:
            return self
        return self.__class__(shape=shape, device=self.device, dtype=self.dtype)

    def unsqueeze(self, dim: int):
        shape = _unsqueezed_shape(self.shape, dim)
        return self.__class__(shape=shape, device=self.device, dtype=self.dtype)

    def _project(self, val: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @abc.abstractmethod
    def is_in(self, val: torch.Tensor) -> bool:
        """If the value :obj:`val` is in the box defined by the TensorSpec, returns True, otherwise False.

        Args:
            val (torch.Tensor): value to be checked

        Returns:
            boolean indicating if values belongs to the TensorSpec box

        """
        raise NotImplementedError

    def project(self, val: torch.Tensor) -> torch.Tensor:
        """If the input tensor is not in the TensorSpec box, it maps it back to it given some heuristic.

        Args:
            val (torch.Tensor): tensor to be mapped to the box.

        Returns:
            a torch.Tensor belonging to the TensorSpec box.

        """
        if not self.is_in(val):
            return self._project(val)
        return val

    def assert_is_in(self, value: torch.Tensor) -> None:
        """Asserts whether a tensor belongs to the box, and raises an exception otherwise.

        Args:
            value (torch.Tensor): value to be checked.

        """
        if not self.is_in(value):
            raise AssertionError(
                f"Encoding failed because value is not in space. "
                f"Consider calling project(val) first. value was = {value} "
                f"and spec was {self}."
            )

    def type_check(self, value: torch.Tensor, key: str = None) -> None:
        """Checks the input value dtype against the TensorSpec dtype and raises an exception if they don't match.

        Args:
            value (torch.Tensor): tensor whose dtype has to be checked
            key (str, optional): if the TensorSpec has keys, the value
                dtype will be checked against the spec pointed by the
                indicated key.

        """
        if value.dtype is not self.dtype:
            raise TypeError(
                f"value.dtype={value.dtype} but"
                f" {self.__class__.__name__}.dtype={self.dtype}"
            )

    @abc.abstractmethod
    def rand(self, shape=None) -> torch.Tensor:
        """Returns a random tensor in the box. The sampling will be uniform unless the box is unbounded.

        Args:
            shape (torch.Size): shape of the random tensor

        Returns:
            a random tensor sampled in the TensorSpec box.

        """
        raise NotImplementedError

    def zero(self, shape=None) -> torch.Tensor:
        """Returns a zero-filled tensor in the box.

        Args:
            shape (torch.Size): shape of the zero-tensor

        Returns:
            a zero-filled tensor sampled in the TensorSpec box.

        """
        if shape is None:
            shape = torch.Size([])
        return torch.zeros((*shape, *self.shape), dtype=self.dtype, device=self.device)

    @abc.abstractmethod
    def to(self, dest: Union[torch.dtype, DEVICE_TYPING]) -> "TensorSpec":
        raise NotImplementedError

    @abc.abstractmethod
    def clone(self) -> "TensorSpec":
        raise NotImplementedError

    def __repr__(self):
        shape_str = indent("shape=" + str(self.shape), " " * 4)
        space_str = indent("space=" + str(self.space), " " * 4)
        device_str = indent("device=" + str(self.device), " " * 4)
        dtype_str = indent("dtype=" + str(self.dtype), " " * 4)
        domain_str = indent("domain=" + str(self.domain), " " * 4)
        sub_string = ",\n".join(
            [shape_str, space_str, device_str, dtype_str, domain_str]
        )
        string = f"{self.__class__.__name__}(\n{sub_string})"
        return string

    @classmethod
    def __torch_function__(
        cls,
        func: Callable,
        types,
        args: Tuple = (),
        kwargs: Optional[dict] = None,
    ) -> Callable:
        if kwargs is None:
            kwargs = {}
        if func not in cls.SPEC_HANDLED_FUNCTIONS or not all(
            issubclass(t, (TensorSpec,)) for t in types
        ):
            return NotImplemented(
                f"func {func} for spec {cls} with handles {cls.SPEC_HANDLED_FUNCTIONS}"
            )
        return cls.SPEC_HANDLED_FUNCTIONS[func](*args, **kwargs)

    def unbind(self, dim: int):
        raise NotImplementedError


T = TypeVar("T")


class _LazyStackedMixin(Generic[T]):
    def __init__(self, *specs: tuple[T, ...], dim: int) -> None:
        self._specs = list(specs)
        self.dim = dim
        if self.dim < 0:
            self.dim = len(self.shape) + self.dim

    def __getitem__(self, item):
        is_key = isinstance(item, str) or (
            isinstance(item, tuple) and all(isinstance(_item, str) for _item in item)
        )
        if is_key:
            return torch.stack(
                [composite_spec[item] for composite_spec in self._specs], dim=self.dim
            )
        elif isinstance(item, tuple):
            # quick check that the index is along the stacked dim
            # case 1: index is a tuple, and the first arg is an ellipsis. Then dim must be the last dim of all composite_specs
            if item[0] is Ellipsis:
                if len(item) == 1:
                    return self
                elif self.dim == len(self.shape) - 1 and len(item) == 2:
                    # we can return
                    return self._specs[item[1]]
                elif len(item) > 2:
                    # check that there is only one non-slice index
                    assigned = False
                    dim_idx = self.dim
                    for i, _item in enumerate(item[1:]):
                        if (
                            isinstance(_item, slice)
                            and not (
                                _item.start is None
                                and _item.stop is None
                                and _item.step is None
                            )
                        ) or not isinstance(_item, slice):
                            if assigned:
                                raise RuntimeError(
                                    "Found more than one meaningful index in a stacked composite spec."
                                )
                            item = _item
                            dim_idx = i + 1
                            assigned = True
                        if not assigned:
                            return self
                        if dim_idx != self.dim:
                            raise RuntimeError(
                                f"Indexing occured along dimension {dim_idx} but stacking was done along dim {self.dim}."
                            )
                        out = self._specs[item]
                        if isinstance(out, TensorSpec):
                            return out
                        return torch.stack(list(out), 0)
                else:
                    raise IndexError(
                        f"Indexing a {self.__class__.__name__} with [..., idx] is only permitted if the stack dimension is the last dimension. "
                        f"Got self.dim={self.dim} and self.shape={self.shape}."
                    )
            elif len(item) >= 2 and item[-1] is Ellipsis:
                return self[item[:-1]]
            elif any(_item is Ellipsis for _item in item):
                raise IndexError("Cannot index along multiple dimensions.")
            # Ellipsis is now ruled out
            elif any(_item is None for _item in item):
                raise IndexError(
                    f"Cannot index a {self.__class__.__name__} with None values"
                )
            # Must be an index with slices then
            else:
                for i, _item in enumerate(item):
                    if i == self.dim:
                        out = self._specs[_item]
                        if isinstance(out, TensorSpec):
                            return out
                        return torch.stack(list(out), 0)
                    elif isinstance(_item, slice):
                        # then the slice must be trivial
                        if not (_item.step is _item.start is _item.stop is None):
                            raise IndexError(
                                f"Got a non-trivial index at dim {i} when only the dim {self.dim} could be indexed."
                            )
                else:
                    return self
        else:
            if not self.dim == 0:
                raise IndexError(
                    f"Trying to index a {self.__class__.__name__} along dimension 0 when the stack dimension is {self.dim}."
                )
            out = self._specs[item]
            if isinstance(out, TensorSpec):
                return out
            return torch.stack(list(out), 0)

    def clone(self) -> T:
        return torch.stack([spec.clone() for spec in self._specs], self.stack_dim)

    @property
    def stack_dim(self):
        return self.dim

    def zero(self, shape=None) -> TensorDictBase:
        if shape is not None:
            dim = self.dim + len(shape)
        else:
            dim = self.dim
        return torch.stack([spec.zero(shape) for spec in self._specs], dim)

    def rand(self, shape=None) -> TensorDictBase:
        if shape is not None:
            dim = self.dim + len(shape)
        else:
            dim = self.dim
        return torch.stack([spec.rand(shape) for spec in self._specs], dim)

    def to(self, dest: Union[torch.dtype, DEVICE_TYPING]) -> T:
        return torch.stack([spec.to(dest) for spec in self._specs], self.dim)

    def unbind(self, dim: int):
        if dim == self.stack_dim:
            return self._specs
        shape = self.shape
        if dim < 0 or dim > self.ndim - 1 or shape[dim] == -1:
            raise ValueError(
                f"Provided dim {dim} is not valid for unbinding shape {shape}"
            )
        else:
            raise ValueError(
                f"A {type(self)} instance can only be unbound along its stack dimension. Expected {self.stack_dim}, received {dim} instead."
            )

    def unsqueeze(self, dim: int):
        if dim < 0:
            new_dim = dim + len(self.shape) + 1
        else:
            new_dim = dim
        if new_dim > len(self.shape) or new_dim < 0:
            raise ValueError(f"Cannot unsqueeze along dim {dim}.")
        if new_dim > self.dim:
            # unsqueeze 2, stack is on 1 => unsqueeze 1, stack along 1
            new_stack_dim = self.dim
            new_dim = new_dim - 1
        else:
            # unsqueeze 0, stack is on 1 => unsqueeze 0, stack on 1
            new_stack_dim = self.dim + 1
        return torch.stack(
            [spec.unsqueeze(new_dim) for spec in self._specs], dim=new_stack_dim
        )

    def squeeze(self, dim: int = None):
        if dim is None:
            size = self.shape
            if len(size) == 1 or size.count(1) == 0:
                return self
            first_singleton_dim = size.index(1)

            squeezed_dict = self.squeeze(first_singleton_dim)
            return squeezed_dict.squeeze(dim=None)

        if dim < 0:
            new_dim = self.ndim + dim
        else:
            new_dim = dim

        if self.shape and (new_dim >= self.ndim or new_dim < 0):
            raise RuntimeError(
                f"squeezing is allowed for dims comprised between 0 and "
                f"spec.ndim only. Got dim={dim} and shape"
                f"={self.shape}."
            )

        if new_dim >= self.ndim or self.shape[new_dim] != 1:
            return self

        if new_dim == self.dim:
            return self._specs[0]
        if new_dim > self.dim:
            # squeeze 2, stack is on 1 => squeeze 1, stack along 1
            new_stack_dim = self.dim
            new_dim = new_dim - 1
        else:
            # squeeze 0, stack is on 1 => squeeze 0, stack on 1
            new_stack_dim = self.dim - 1
        return torch.stack(
            [spec.squeeze(new_dim) for spec in self._specs], dim=new_stack_dim
        )


class LazyStackedTensorSpec(_LazyStackedMixin[TensorSpec], TensorSpec):
    """A lazy representation of a stack of tensor specs.

    Stacks tensor-specs together along one dimension.
    When random samples are drawn, a stack of samples is returned if possible.
    If not, an error is thrown.

    Indexing is allowed but only along the stack dimension.

    This class is aimed to be used in multi-task and multi-agent settings, where
    heterogeneous specs may occur (same semantic but different shape).

    """

    def __eq__(self, other):
        if not isinstance(other, LazyStackedTensorSpec):
            return False
        if len(self._specs) != len(other._specs):
            return False
        for _spec1, _spec2 in zip(self._specs, other._specs):
            if _spec1 != _spec2:
                return False
        return True

    def __len__(self):
        return self.shape[0]

    def to_numpy(self, val: torch.Tensor, safe: bool = None) -> dict:
        if safe is None:
            safe = _CHECK_SPEC_ENCODE
        if safe:
            if val.shape[self.dim] != len(self._specs):
                raise ValueError(
                    "Size of LazyStackedTensorSpec and val differ along the stacking "
                    "dimension"
                )
            for spec, v in zip(self._specs, torch.unbind(val, dim=self.dim)):
                spec.assert_is_in(v)
        return val.detach().cpu().numpy()

    def __repr__(self):
        shape_str = "shape=" + str(self.shape)
        device_str = "device=" + str(self.device)
        dtype_str = "dtype=" + str(self.dtype)
        domain_str = "domain=" + str(self._specs[0].domain)
        sub_string = ", ".join([shape_str, device_str, dtype_str, domain_str])
        string = f"LazyStacked{self._specs[0].__class__.__name__}(\n    {sub_string})"
        return string

    @property
    def device(self) -> DEVICE_TYPING:
        return self._specs[0].device

    @property
    def ndim(self):
        return self.ndimension()

    def ndimension(self):
        return len(self.shape)

    @property
    def shape(self):
        first_shape = self._specs[0].shape
        shape = []
        for i in range(len(first_shape)):
            homo_dim = True
            for spec in self._specs:
                if spec.shape[i] != first_shape[i]:
                    homo_dim = False
                    break
            shape.append(first_shape[i] if homo_dim else -1)

        dim = self.dim
        if dim < 0:
            dim = len(shape) + dim + 1
        shape.insert(dim, len(self._specs))
        return torch.Size(shape)

    def expand(self, *shape):
        if len(shape) == 1 and not isinstance(shape[0], (int,)):
            return self.expand(*shape[0])
        expand_shape = shape[: -len(self.shape)]
        existing_shape = self.shape
        shape_check = shape[-len(self.shape) :]
        for _i, (size1, size2) in enumerate(zip(existing_shape, shape_check)):
            if size1 != size2 and size1 != 1:
                raise RuntimeError(
                    f"Expanding a non-singletom dimension: existing shape={size1} vs expand={size2}"
                )
            elif size1 != size2 and size1 == 1 and _i == self.dim:
                # if we're expanding along the stack dim we just need to clone the existing spec
                return torch.stack(
                    [self._specs[0].clone() for _ in range(size2)], self.dim
                ).expand(*shape)
        if _i != len(self.shape) - 1:
            raise RuntimeError(
                f"Trying to expand non-congruent shapes: received {shape} when the shape is {self.shape}."
            )
        # remove the stack dim from the expanded shape, which we know to match
        shape_check = [s for i, s in enumerate(shape_check) if i != self.dim]
        specs = []
        for spec in self._specs:
            spec_shape = []
            for dim_check, spec_dim in zip(shape_check, spec.shape):
                spec_shape.append(dim_check if dim_check != -1 else spec_dim)
            unstack_shape = list(expand_shape) + list(spec_shape)
            specs.append(spec.expand(unstack_shape))
        return torch.stack(
            specs,
            self.dim + len(expand_shape),
        )

    def type_check(self, value: torch.Tensor, key: str = None) -> None:
        raise NOT_IMPLEMENTED_ERROR

    def is_in(self, val) -> bool:
        raise NOT_IMPLEMENTED_ERROR

    @property
    def space(self):
        raise NOT_IMPLEMENTED_ERROR

    def _project(self, val: TensorDictBase) -> TensorDictBase:
        raise NOT_IMPLEMENTED_ERROR

    def encode(
        self, val: Union[np.ndarray, torch.Tensor], *, ignore_device=False
    ) -> torch.Tensor:
        raise NOT_IMPLEMENTED_ERROR


@dataclass(repr=False)
class OneHotDiscreteTensorSpec(TensorSpec):
    """A unidimensional, one-hot discrete tensor spec.

    By default, TorchRL assumes that categorical variables are encoded as
    one-hot encodings of the variable. This allows for simple indexing of
    tensors, e.g.

        >>> batch, size = 3, 4
        >>> action_value = torch.arange(batch*size)
        >>> action_value = action_value.view(batch, size).to(torch.float)
        >>> action = (action_value == action_value.max(-1,
        ...    keepdim=True)[0]).to(torch.long)
        >>> chosen_action_value = (action * action_value).sum(-1)
        >>> print(chosen_action_value)
        tensor([ 3.,  7., 11.])

    The last dimension of the shape (variable domain) cannot be indexed.

    Args:
        n (int): number of possible outcomes.
        shape (torch.Size, optional): total shape of the sampled tensors.
            If provided, the last dimension must match n.
        device (str, int or torch.device, optional): device of the tensors.
        dtype (str or torch.dtype, optional): dtype of the tensors.
        user_register (bool): experimental feature. If True, every integer
            will be mapped onto a binary vector in the order in which they
            appear. This feature is designed for environment with no
            a-priori definition of the number of possible outcomes (e.g.
            discrete outcomes are sampled from an arbitrary set, whose
            elements will be mapped in a register to a series of unique
            one-hot binary vectors).

    """

    shape: torch.Size
    space: DiscreteBox
    device: torch.device = torch.device("cpu")
    dtype: torch.dtype = torch.float
    domain: str = ""

    # SPEC_HANDLED_FUNCTIONS = {}

    def __init__(
        self,
        n: int,
        shape: Optional[torch.Size] = None,
        device: Optional[DEVICE_TYPING] = None,
        dtype: Optional[Union[str, torch.dtype]] = torch.bool,
        use_register: bool = False,
        mask: torch.Tensor | None = None,
    ):
        dtype, device = _default_dtype_and_device(dtype, device)
        self.use_register = use_register
        space = DiscreteBox(n)
        if shape is None:
            shape = torch.Size((space.n,))
        else:
            shape = torch.Size(shape)
            if not len(shape) or shape[-1] != space.n:
                raise ValueError(
                    f"The last value of the shape must match n for transform of type {self.__class__}. "
                    f"Got n={space.n} and shape={shape}."
                )
        super().__init__(shape, space, device, dtype, "discrete")
        self.update_mask(mask)

    def update_mask(self, mask):
        if mask is not None:
            try:
                mask = mask.expand(self.shape)
            except RuntimeError as err:
                raise RuntimeError("Cannot expand mask to the desired shape.") from err
            if mask.dtype != torch.bool:
                raise ValueError("Only boolean masks are accepted.")
        self.mask = mask

    def to(self, dest: Union[torch.dtype, DEVICE_TYPING]) -> CompositeSpec:
        if isinstance(dest, torch.dtype):
            dest_dtype = dest
            dest_device = self.device
        else:
            dest_dtype = self.dtype
            dest_device = torch.device(dest)
        if dest_device == self.device and dest_dtype == self.dtype:
            return self
        return self.__class__(
            n=self.space.n,
            shape=self.shape,
            device=dest_device,
            dtype=dest_dtype,
            use_register=self.use_register,
            mask=self.mask.to(dest) if self.mask is not None else None,
        )

    def clone(self) -> OneHotDiscreteTensorSpec:
        return self.__class__(
            n=self.space.n,
            shape=self.shape,
            device=self.device,
            dtype=self.dtype,
            use_register=self.use_register,
            mask=self.mask.clone() if self.mask is not None else None,
        )

    def expand(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list, torch.Size)):
            shape = shape[0]
        if any(val < 0 for val in shape):
            raise ValueError(
                f"{self.__class__.__name__}.expand does not support negative shapes."
            )
        if any(s1 != s2 and s2 != 1 for s1, s2 in zip(shape[-self.ndim :], self.shape)):
            raise ValueError(
                f"The last {self.ndim} of the expanded shape {shape} must match the"
                f"shape of the {self.__class__.__name__} spec in expand()."
            )
        mask = self.mask
        if mask is not None:
            mask = mask.expand(shape)
        return self.__class__(
            n=shape[-1],
            shape=shape,
            device=self.device,
            dtype=self.dtype,
            mask=mask,
        )

    def squeeze(self, dim=None):
        if self.shape[-1] == 1 and dim in (len(self.shape), -1, None):
            raise ValueError(f"Final dimension of {type(self)} must remain unchanged")

        shape = _squeezed_shape(self.shape, dim)
        if shape is None:
            return self
        mask = self.mask
        if mask is not None:
            mask = mask.reshape(shape)
        return self.__class__(
            n=shape[-1],
            shape=shape,
            device=self.device,
            dtype=self.dtype,
            use_register=self.use_register,
            mask=mask,
        )

    def unsqueeze(self, dim: int):
        if dim in (len(self.shape), -1):
            raise ValueError(f"Final dimension of {type(self)} must remain unchanged")

        shape = _unsqueezed_shape(self.shape, dim)
        mask = self.mask
        if mask is not None:
            mask = mask.reshape(shape)
        return self.__class__(
            n=shape[-1],
            shape=shape,
            device=self.device,
            dtype=self.dtype,
            use_register=self.use_register,
            mask=mask,
        )

    def unbind(self, dim: int):
        if dim in (len(self.shape), -1):
            raise ValueError(f"Final dimension of {type(self)} must remain unchanged")
        orig_dim = dim
        if dim < 0:
            dim = len(self.shape) + dim
        if dim < 0:
            raise ValueError(
                f"Cannot unbind along dim {orig_dim} with shape {self.shape}."
            )
        shape = tuple(s for i, s in enumerate(self.shape) if i != dim)
        mask = self.mask
        if mask is not None:
            mask = mask.unbind(dim)
        else:
            mask = (None,) * self.shape[dim]
        return tuple(
            self.__class__(
                n=shape[-1],
                shape=shape,
                device=self.device,
                dtype=self.dtype,
                use_register=self.use_register,
                mask=mask[i],
            )
            for i in range(self.shape[dim])
        )

    def rand(self, shape=None) -> torch.Tensor:
        if shape is None:
            shape = self.shape[:-1]
        else:
            shape = torch.Size([*shape, *self.shape[:-1]])
        mask = self.mask
        if mask is None:
            n = self.space.n
            m = torch.randint(n, shape, device=self.device)
        else:
            mask = mask.expand(*shape, mask.shape[-1])
            if mask.ndim > 2:
                mask_flat = torch.flatten(mask, 0, -2)
            else:
                mask_flat = mask
            shape_out = mask.shape[:-1]
            m = torch.multinomial(mask_flat.float(), 1).reshape(shape_out)
        out = torch.nn.functional.one_hot(m, self.space.n).to(self.dtype)
        # torch.zeros((*shape, self.space.n), device=self.device, dtype=self.dtype)
        # out.scatter_(-1, m, 1)
        return out

    def encode(
        self,
        val: Union[np.ndarray, torch.Tensor],
        space: Optional[DiscreteBox] = None,
        *,
        ignore_device: bool = False,
    ) -> torch.Tensor:
        if not isinstance(val, torch.Tensor):
            if ignore_device:
                val = torch.tensor(val)
            else:
                val = torch.tensor(val, device=self.device)

        if space is None:
            space = self.space

        if self.use_register:
            if val not in space.register:
                space.register[val] = len(space.register)
            val = space.register[val]

        if (val >= space.n).any():
            raise AssertionError("Value must be less than action space.")

        val = torch.nn.functional.one_hot(val.long(), space.n).to(self.dtype)
        return val

    def to_numpy(self, val: torch.Tensor, safe: bool = None) -> np.ndarray:
        if safe is None:
            safe = _CHECK_SPEC_ENCODE
        if safe:
            if not isinstance(val, torch.Tensor):
                raise NotImplementedError
            self.assert_is_in(val)
        val = val.long().argmax(-1).cpu().numpy()
        if self.use_register:
            inv_reg = self.space.register.inverse()
            vals = []
            for _v in val.view(-1):
                vals.append(inv_reg[int(_v)])
            return np.array(vals).reshape(tuple(val.shape))
        return val

    def index(self, index: INDEX_TYPING, tensor_to_index: torch.Tensor) -> torch.Tensor:
        if not isinstance(index, torch.Tensor):
            raise ValueError(
                f"Only tensors are allowed for indexing using "
                f"{self.__class__.__name__}.index(...)"
            )
        index = index.nonzero().squeeze()
        index = index.expand((*tensor_to_index.shape[:-1], index.shape[-1]))
        return tensor_to_index.gather(-1, index)

    def __getitem__(self, idx: SHAPE_INDEX_TYPING):
        """Indexes the current TensorSpec based on the provided index.

        The last dimension of the spec corresponding to the variable domain cannot be indexed.
        """
        indexed_shape = _shape_indexing(self.shape[:-1], idx)
        return self.__class__(
            n=self.space.n,
            shape=torch.Size(indexed_shape + [self.shape[-1]]),
            device=self.device,
            dtype=self.dtype,
            use_register=self.use_register,
            mask=self.mask[idx] if self.mask is not None else None,
        )

    def _project(self, val: torch.Tensor) -> torch.Tensor:
        if self.mask is None:
            out = torch.multinomial(val.to(torch.float), 1).squeeze(-1)
            out = torch.nn.functional.one_hot(out, self.space.n).to(self.dtype)
            return out
        shape = self.mask.shape
        shape = torch.broadcast_shapes(shape, val.shape)
        mask_expand = self.mask.expand(shape)
        gathered = mask_expand & val
        oob = ~gathered.any(-1)
        new_val = torch.multinomial(mask_expand[oob].float(), 1)
        val = val.clone()
        val[oob] = 0
        val[oob] = torch.scatter(val[oob], -1, new_val, 1)
        return val

    def is_in(self, val: torch.Tensor) -> bool:
        if self.mask is None:
            return (val.sum(-1) == 1).all()
        shape = self.mask.shape
        shape = torch.broadcast_shapes(shape, val.shape)
        mask_expand = self.mask.expand(shape)
        gathered = mask_expand & val
        return gathered.any(-1).all()

    def __eq__(self, other):
        if not hasattr(other, "mask"):
            return False
        mask_equal = (self.mask is None and other.mask is None) or (
            isinstance(self.mask, torch.Tensor)
            and isinstance(other.mask, torch.Tensor)
            and (self.mask.shape == other.mask.shape)
            and (self.mask == other.mask).all()
        )
        return (
            type(self) == type(other)
            and self.shape == other.shape
            and self.space == other.space
            and self.device == other.device
            and self.dtype == other.dtype
            and self.domain == other.domain
            and self.use_register == other.use_register
            and mask_equal
        )

    def to_categorical(self, val: torch.Tensor, safe: bool = None) -> torch.Tensor:
        """Converts a given one-hot tensor in categorical format.

        Args:
            val (torch.Tensor, optional): One-hot tensor to convert in categorical format.
            safe (bool): boolean value indicating whether a check should be
                performed on the value against the domain of the spec.
                Defaults to the value of the ``CHECK_SPEC_ENCODE`` environment variable.

        Returns:
            The categorical tensor.
        """
        if safe is None:
            safe = _CHECK_SPEC_ENCODE
        if safe:
            self.assert_is_in(val)
        return val.long().argmax(-1)

    def to_categorical_spec(self) -> DiscreteTensorSpec:
        """Converts the spec to the equivalent categorical spec."""
        return DiscreteTensorSpec(
            self.space.n,
            device=self.device,
            shape=self.shape[:-1],
            mask=self.mask,
        )


@dataclass(repr=False)
class BoundedTensorSpec(TensorSpec):
    """A bounded continuous tensor spec.

    Args:
        low (np.ndarray, torch.Tensor or number): lower bound of the box.
        high (np.ndarray, torch.Tensor or number): upper bound of the box.
        device (str, int or torch.device, optional): device of the tensors.
        dtype (str or torch.dtype, optional): dtype of the tensors.

    """

    # SPEC_HANDLED_FUNCTIONS = {}
    DEPRECATED_KWARGS = (
        "The `minimum` and `maximum` keyword arguments are now "
        "deprecated in favour of `low` and `high`."
    )
    CONFLICTING_KWARGS = (
        "The keyword arguments {} and {} conflict. Only one of these can be passed."
    )

    def __init__(
        self,
        low: Union[float, torch.Tensor, np.ndarray] = None,
        high: Union[float, torch.Tensor, np.ndarray] = None,
        shape: Optional[Union[torch.Size, int]] = None,
        device: Optional[DEVICE_TYPING] = None,
        dtype: Optional[Union[torch.dtype, str]] = None,
        **kwargs,
    ):
        if "maximum" in kwargs:
            if high is not None:
                raise TypeError(self.CONFLICTING_KWARGS.format("high", "maximum"))
            high = kwargs.pop("maximum")
            warnings.warn(self.DEPRECATED_KWARGS, category=DeprecationWarning)
        if "minimum" in kwargs:
            if low is not None:
                raise TypeError(self.CONFLICTING_KWARGS.format("low", "minimum"))
            low = kwargs.pop("minimum")
            warnings.warn(self.DEPRECATED_KWARGS, category=DeprecationWarning)
        if len(kwargs):
            raise TypeError(f"Got unrecognised kwargs {tuple(kwargs.keys())}.")

        dtype, device = _default_dtype_and_device(dtype, device)
        if dtype is None:
            dtype = torch.get_default_dtype()
        if device is None:
            device = torch._get_default_device()

        if not isinstance(low, torch.Tensor):
            low = torch.tensor(low, dtype=dtype, device=device)
        if not isinstance(high, torch.Tensor):
            high = torch.tensor(high, dtype=dtype, device=device)
        if high.device != device:
            high = high.to(device)
        if low.device != device:
            low = low.to(device)
        if dtype is not None and low.dtype is not dtype:
            low = low.to(dtype)
        if dtype is not None and high.dtype is not dtype:
            high = high.to(dtype)
        err_msg = (
            "BoundedTensorSpec requires the shape to be explicitely (via "
            "the shape argument) or implicitely defined (via either the "
            "minimum or the maximum or both). If the maximum and/or the "
            "minimum have a non-singleton shape, they must match the "
            "provided shape if this one is set explicitely."
        )
        if shape is not None and not isinstance(shape, torch.Size):
            if isinstance(shape, int):
                shape = torch.Size([shape])
            else:
                shape = torch.Size(list(shape))

        if high.ndimension():
            if shape is not None and shape != high.shape:
                raise RuntimeError(err_msg)
            shape = high.shape
            low = low.expand(shape).clone()
        elif low.ndimension():
            if shape is not None and shape != low.shape:
                raise RuntimeError(err_msg)
            shape = low.shape
            high = high.expand(shape).clone()
        elif shape is None:
            raise RuntimeError(err_msg)
        else:
            low = low.expand(shape).clone()
            high = high.expand(shape).clone()

        if low.numel() > high.numel():
            high = high.expand_as(low).clone()
        elif high.numel() > low.numel():
            low = low.expand_as(high).clone()
        if shape is None:
            shape = low.shape
        else:
            if isinstance(shape, float):
                shape = torch.Size([shape])
            elif not isinstance(shape, torch.Size):
                shape = torch.Size(shape)
            shape_err_msg = f"low and shape mismatch, got {low.shape} and {shape}"
            if len(low.shape) != len(shape):
                raise RuntimeError(shape_err_msg)
            if not all(_s == _sa for _s, _sa in zip(shape, low.shape)):
                raise RuntimeError(shape_err_msg)
        self.shape = shape

        super().__init__(
            shape, ContinuousBox(low, high, device=device), device, dtype, "continuous"
        )

    def expand(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list, torch.Size)):
            shape = shape[0]
        if any(val < 0 for val in shape):
            raise ValueError(
                f"{self.__class__.__name__}.expand does not support negative shapes."
            )
        if any(s1 != s2 and s2 != 1 for s1, s2 in zip(shape[-self.ndim :], self.shape)):
            raise ValueError(
                f"The last {self.ndim} of the expanded shape {shape} must match the"
                f"shape of the {self.__class__.__name__} spec in expand()."
            )
        return self.__class__(
            low=self.space.low.expand(shape).clone(),
            high=self.space.high.expand(shape).clone(),
            shape=shape,
            device=self.device,
            dtype=self.dtype,
        )

    def squeeze(self, dim: int | None = None):
        shape = _squeezed_shape(self.shape, dim)
        if shape is None:
            return self

        if dim is None:
            low = self.space.low.squeeze().clone()
            high = self.space.high.squeeze().clone()
        else:
            low = self.space.low.squeeze(dim).clone()
            high = self.space.high.squeeze(dim).clone()

        return self.__class__(
            low=low,
            high=high,
            shape=shape,
            device=self.device,
            dtype=self.dtype,
        )

    def unsqueeze(self, dim: int):
        shape = _unsqueezed_shape(self.shape, dim)
        return self.__class__(
            low=self.space.low.unsqueeze(dim).clone(),
            high=self.space.high.unsqueeze(dim).clone(),
            shape=shape,
            device=self.device,
            dtype=self.dtype,
        )

    def unbind(self, dim: int):
        if dim in (len(self.shape), -1):
            raise ValueError(f"Final dimension of {type(self)} must remain unchanged")
        orig_dim = dim
        if dim < 0:
            dim = len(self.shape) + dim
        if dim < 0:
            raise ValueError(
                f"Cannot unbind along dim {orig_dim} with shape {self.shape}."
            )
        shape = tuple(s for i, s in enumerate(self.shape) if i != dim)
        low = self.space.low.unbind(dim)
        high = self.space.high.unbind(dim)
        return tuple(
            self.__class__(
                low=low[i],
                high=high[i],
                shape=shape,
                device=self.device,
                dtype=self.dtype,
            )
            for i in range(self.shape[dim])
        )

    def rand(self, shape=None) -> torch.Tensor:
        if shape is None:
            shape = torch.Size([])
        a, b = self.space
        if self.dtype in (torch.float, torch.double, torch.half):
            shape = [*shape, *self.shape]
            out = (
                torch.zeros(shape, dtype=self.dtype, device=self.device).uniform_()
                * (b - a)
                + a
            )
            if (out > b).any():
                out[out > b] = b.expand_as(out)[out > b]
            if (out < a).any():
                out[out < a] = a.expand_as(out)[out < a]
            return out
        else:
            if self.space.high.dtype == torch.bool:
                maxi = self.space.high.int()
            else:
                maxi = self.space.high
            if self.space.low.dtype == torch.bool:
                mini = self.space.low.int()
            else:
                mini = self.space.low
            interval = maxi - mini
            r = torch.rand(torch.Size([*shape, *self.shape]), device=interval.device)
            r = interval * r
            r = self.space.low + r
            r = r.to(self.dtype).to(self.device)
            return r

    def _project(self, val: torch.Tensor) -> torch.Tensor:
        low = self.space.low.to(val.device)
        high = self.space.high.to(val.device)
        try:
            val = val.clamp_(low.item(), high.item())
        except ValueError:
            low = low.expand_as(val)
            high = high.expand_as(val)
            val[val < low] = low[val < low]
            val[val > high] = high[val > high]
        except RuntimeError:
            low = low.expand_as(val)
            high = high.expand_as(val)
            val[val < low] = low[val < low]
            val[val > high] = high[val > high]
        return val

    def is_in(self, val: torch.Tensor) -> bool:
        try:
            return (val >= self.space.low.to(val.device)).all() and (
                val <= self.space.high.to(val.device)
            ).all()
        except RuntimeError as err:
            if "The size of tensor a" in str(err):
                warnings.warn(f"Got a shape mismatch: {str(err)}")
                return False
            raise err

    def to(self, dest: Union[torch.dtype, DEVICE_TYPING]) -> CompositeSpec:
        if isinstance(dest, torch.dtype):
            dest_dtype = dest
            dest_device = self.device
        else:
            dest_dtype = self.dtype
            dest_device = torch.device(dest)
        if dest_device == self.device and dest_dtype == self.dtype:
            return self
        return self.__class__(
            low=self.space.low.to(dest),
            high=self.space.high.to(dest),
            shape=self.shape,
            device=dest_device,
            dtype=dest_dtype,
        )

    def clone(self) -> BoundedTensorSpec:
        return self.__class__(
            low=self.space.low.clone(),
            high=self.space.high.clone(),
            shape=self.shape,
            device=self.device,
            dtype=self.dtype,
        )

    def __getitem__(self, idx: SHAPE_INDEX_TYPING):
        """Indexes the current TensorSpec based on the provided index."""
        if _is_nested_list(idx):
            raise NotImplementedError(
                "Pending resolution of https://github.com/pytorch/pytorch/issues/100080."
            )

        indexed_shape = torch.Size(_shape_indexing(self.shape, idx))
        # Expand is required as pytorch.tensor indexing
        return self.__class__(
            low=self.space.low[idx].clone().expand(indexed_shape),
            high=self.space.high[idx].clone().expand(indexed_shape),
            shape=indexed_shape,
            device=self.device,
            dtype=self.dtype,
        )


def _is_nested_list(index, notuple=False):
    if not notuple and isinstance(index, tuple):
        for idx in index:
            if _is_nested_list(idx, notuple=True):
                return True
    elif isinstance(index, list):
        for idx in index:
            if isinstance(idx, list):
                return True
        else:
            return False
    return False


@dataclass(repr=False)
class UnboundedContinuousTensorSpec(TensorSpec):
    """An unbounded continuous tensor spec.

    Args:
        device (str, int or torch.device, optional): device of the tensors.
        dtype (str or torch.dtype, optional): dtype of the tensors
            (should be an floating point dtype such as float, double etc.)
    """

    # SPEC_HANDLED_FUNCTIONS = {}

    def __init__(
        self,
        shape: Union[torch.Size, int] = _DEFAULT_SHAPE,
        device: Optional[DEVICE_TYPING] = None,
        dtype: Optional[Union[str, torch.dtype]] = None,
    ):
        if isinstance(shape, int):
            shape = torch.Size([shape])

        dtype, device = _default_dtype_and_device(dtype, device)
        box = (
            ContinuousBox(
                torch.tensor(-np.inf, device=device).expand(shape),
                torch.tensor(np.inf, device=device).expand(shape),
            )
            if shape == _DEFAULT_SHAPE
            else None
        )
        super().__init__(
            shape=shape,
            space=box,
            device=device,
            dtype=dtype,
            domain="continuous",
        )

    def to(self, dest: Union[torch.dtype, DEVICE_TYPING]) -> CompositeSpec:
        if isinstance(dest, torch.dtype):
            dest_dtype = dest
            dest_device = self.device
        else:
            dest_dtype = self.dtype
            dest_device = torch.device(dest)
        if dest_device == self.device and dest_dtype == self.dtype:
            return self
        return self.__class__(shape=self.shape, device=dest_device, dtype=dest_dtype)

    def clone(self) -> UnboundedContinuousTensorSpec:
        return self.__class__(shape=self.shape, device=self.device, dtype=self.dtype)

    def rand(self, shape=None) -> torch.Tensor:
        if shape is None:
            shape = torch.Size([])
        shape = [*shape, *self.shape]
        return torch.randn(shape, device=self.device, dtype=self.dtype)

    def is_in(self, val: torch.Tensor) -> bool:
        return True

    def expand(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list, torch.Size)):
            shape = shape[0]
        if any(val < 0 for val in shape):
            raise ValueError(
                f"{self.__class__.__name__}.expand does not support negative shapes."
            )
        if any(s1 != s2 and s2 != 1 for s1, s2 in zip(shape[-self.ndim :], self.shape)):
            raise ValueError(
                f"The last {self.ndim} of the expanded shape {shape} must match the"
                f"shape of the {self.__class__.__name__} spec in expand()."
            )
        return self.__class__(shape=shape, device=self.device, dtype=self.dtype)

    def __getitem__(self, idx: SHAPE_INDEX_TYPING):
        """Indexes the current TensorSpec based on the provided index."""
        indexed_shape = torch.Size(_shape_indexing(self.shape, idx))
        return self.__class__(shape=indexed_shape, device=self.device, dtype=self.dtype)

    def unbind(self, dim: int):
        orig_dim = dim
        if dim < 0:
            dim = len(self.shape) + dim
        if dim < 0:
            raise ValueError(
                f"Cannot unbind along dim {orig_dim} with shape {self.shape}."
            )
        shape = tuple(s for i, s in enumerate(self.shape) if i != dim)
        return tuple(
            self.__class__(
                shape=shape,
                device=self.device,
                dtype=self.dtype,
            )
            for i in range(self.shape[dim])
        )


@dataclass(repr=False)
class UnboundedDiscreteTensorSpec(TensorSpec):
    """An unbounded discrete tensor spec.

    Args:
        device (str, int or torch.device, optional): device of the tensors.
        dtype (str or torch.dtype, optional): dtype of the tensors
            (should be an integer dtype such as long, uint8 etc.)
    """

    # SPEC_HANDLED_FUNCTIONS = {}

    def __init__(
        self,
        shape: Union[torch.Size, int] = _DEFAULT_SHAPE,
        device: Optional[DEVICE_TYPING] = None,
        dtype: Optional[Union[str, torch.dtype]] = None,
    ):
        if isinstance(shape, int):
            shape = torch.Size([shape])

        dtype, device = _default_dtype_and_device(dtype, device)
        if dtype == torch.bool:
            min_value = False
            max_value = True
        else:
            if dtype.is_floating_point:
                min_value = torch.finfo(dtype).min
                max_value = torch.finfo(dtype).max
            else:
                min_value = torch.iinfo(dtype).min
                max_value = torch.iinfo(dtype).max
        space = ContinuousBox(
            torch.full(shape, min_value, device=device),
            torch.full(shape, max_value, device=device),
        )

        super().__init__(
            shape=shape,
            space=space,
            device=device,
            dtype=dtype,
            domain="continuous",
        )

    def to(self, dest: Union[torch.dtype, DEVICE_TYPING]) -> CompositeSpec:
        if isinstance(dest, torch.dtype):
            dest_dtype = dest
            dest_device = self.device
        else:
            dest_dtype = self.dtype
            dest_device = torch.device(dest)
        if dest_device == self.device and dest_dtype == self.dtype:
            return self
        return self.__class__(shape=self.shape, device=dest_device, dtype=dest_dtype)

    def clone(self) -> UnboundedDiscreteTensorSpec:
        return self.__class__(shape=self.shape, device=self.device, dtype=self.dtype)

    def rand(self, shape=None) -> torch.Tensor:
        if shape is None:
            shape = torch.Size([])
        interval = self.space.high - self.space.low
        r = torch.rand(torch.Size([*shape, *interval.shape]), device=interval.device)
        r = r * interval
        r = self.space.low + r
        r = r.to(self.dtype)
        return r.to(self.device)

    def is_in(self, val: torch.Tensor) -> bool:
        return True

    def expand(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list, torch.Size)):
            shape = shape[0]
        if any(val < 0 for val in shape):
            raise ValueError(
                f"{self.__class__.__name__}.expand does not support negative shapes."
            )
        if any(s1 != s2 and s2 != 1 for s1, s2 in zip(shape[-self.ndim :], self.shape)):
            raise ValueError(
                f"The last {self.ndim} of the expanded shape {shape} must match the"
                f"shape of the {self.__class__.__name__} spec in expand()."
            )
        return self.__class__(shape=shape, device=self.device, dtype=self.dtype)

    def __getitem__(self, idx: SHAPE_INDEX_TYPING):
        """Indexes the current TensorSpec based on the provided index."""
        indexed_shape = torch.Size(_shape_indexing(self.shape, idx))
        return self.__class__(shape=indexed_shape, device=self.device, dtype=self.dtype)

    def unbind(self, dim: int):
        orig_dim = dim
        if dim < 0:
            dim = len(self.shape) + dim
        if dim < 0:
            raise ValueError(
                f"Cannot unbind along dim {orig_dim} with shape {self.shape}."
            )
        shape = tuple(s for i, s in enumerate(self.shape) if i != dim)
        return tuple(
            self.__class__(
                shape=shape,
                device=self.device,
                dtype=self.dtype,
            )
            for i in range(self.shape[dim])
        )


@dataclass(repr=False)
class MultiOneHotDiscreteTensorSpec(OneHotDiscreteTensorSpec):
    """A concatenation of one-hot discrete tensor spec.

    The last dimension of the shape (domain of the tensor elements) cannot be indexed.

    Args:
        nvec (iterable of integers): cardinality of each of the elements of
            the tensor.
        shape (torch.Size, optional): total shape of the sampled tensors.
            If provided, the last dimension must match sum(nvec).
        device (str, int or torch.device, optional): device of
            the tensors.
        dtype (str or torch.dtype, optional): dtype of the tensors.

    Examples:
        >>> ts = MultiOneHotDiscreteTensorSpec((3,2,3))
        >>> ts.is_in(torch.tensor([0,0,1,
        ...                        0,1,
        ...                        1,0,0]))
        True
        >>> ts.is_in(torch.tensor([1,0,1,
        ...                        0,1,
        ...                        1,0,0])) # False
        False

    """

    # SPEC_HANDLED_FUNCTIONS = {}

    def __init__(
        self,
        nvec: Sequence[int],
        shape: Optional[torch.Size] = None,
        device=None,
        dtype=torch.bool,
        use_register=False,
        mask: torch.Tensor | None = None,
    ):
        self.nvec = nvec
        dtype, device = _default_dtype_and_device(dtype, device)
        if shape is None:
            shape = torch.Size((sum(nvec),))
        else:
            shape = torch.Size(shape)
            if shape[-1] != sum(nvec):
                raise ValueError(
                    f"The last value of the shape must match sum(nvec) for transform of type {self.__class__}. "
                    f"Got sum(nvec)={sum(nvec)} and shape={shape}."
                )
        space = BoxList([DiscreteBox(n) for n in nvec])
        self.use_register = use_register
        super(OneHotDiscreteTensorSpec, self).__init__(
            shape,
            space,
            device,
            dtype,
            domain="discrete",
        )
        self.update_mask(mask)

    def update_mask(self, mask):
        if mask is not None:
            try:
                mask = mask.expand(*self.shape)
            except RuntimeError as err:
                raise RuntimeError("Cannot expand mask to the desired shape.") from err
            if mask.dtype != torch.bool:
                raise ValueError("Only boolean masks are accepted.")
        self.mask = mask

    def to(self, dest: Union[torch.dtype, DEVICE_TYPING]) -> CompositeSpec:
        if isinstance(dest, torch.dtype):
            dest_dtype = dest
            dest_device = self.device
        else:
            dest_dtype = self.dtype
            dest_device = torch.device(dest)
        if dest_device == self.device and dest_dtype == self.dtype:
            return self
        return self.__class__(
            nvec=deepcopy(self.nvec),
            shape=self.shape,
            device=dest_device,
            dtype=dest_dtype,
            mask=self.mask.to(dest) if self.mask is not None else None,
        )

    def clone(self) -> MultiOneHotDiscreteTensorSpec:
        return self.__class__(
            nvec=deepcopy(self.nvec),
            shape=self.shape,
            device=self.device,
            dtype=self.dtype,
            mask=self.mask.clone() if self.mask is not None else None,
        )

    def __eq__(self, other):
        if not hasattr(other, "mask"):
            return False
        mask_equal = (self.mask is None and other.mask is None) or (
            isinstance(self.mask, torch.Tensor)
            and isinstance(other.mask, torch.Tensor)
            and (self.mask.shape == other.mask.shape)
            and (self.mask == other.mask).all()
        )
        return (
            type(self) == type(other)
            and self.shape == other.shape
            and self.space == other.space
            and self.device == other.device
            and self.dtype == other.dtype
            and self.domain == other.domain
            and self.use_register == other.use_register
            and mask_equal
        )

    def rand(self, shape: Optional[torch.Size] = None) -> torch.Tensor:
        if shape is None:
            shape = self.shape[:-1]
        else:
            shape = torch.Size([*shape, *self.shape[:-1]])
        mask = self.mask

        if mask is None:
            x = torch.cat(
                [
                    torch.nn.functional.one_hot(
                        torch.randint(
                            space.n,
                            (
                                *shape,
                                1,
                            ),
                            device=self.device,
                        ),
                        space.n,
                    ).to(self.dtype)
                    for space in self.space
                ],
                -1,
            ).squeeze(-2)
            return x
        mask = mask.expand(*shape, mask.shape[-1])
        mask_splits = torch.split(mask, [space.n for space in self.space], -1)
        out = []
        for _mask in mask_splits:
            if mask.ndim > 2:
                mask_flat = torch.flatten(_mask, 0, -2)
            else:
                mask_flat = _mask
            shape_out = _mask.shape[:-1]
            m = torch.multinomial(mask_flat.float(), 1).reshape(shape_out)
            m = torch.nn.functional.one_hot(m, _mask.shape[-1]).to(self.dtype)
            out.append(m)
        return torch.cat(out, -1)

    def encode(
        self, val: Union[np.ndarray, torch.Tensor], *, ignore_device: bool = False
    ) -> torch.Tensor:
        if not isinstance(val, torch.Tensor):
            if not ignore_device:
                val = torch.tensor(val, device=self.device)
            else:
                val = torch.as_tensor(val)

        x = []
        for v, space in zip(val.unbind(-1), self.space):
            if not (v < space.n).all():
                raise RuntimeError(
                    f"value {v} is greater than the allowed max {space.n}"
                )
            x.append(
                super(MultiOneHotDiscreteTensorSpec, self).encode(
                    v, space, ignore_device=ignore_device
                )
            )
        return torch.cat(x, -1).reshape(self.shape)

    def _split(self, val: torch.Tensor) -> Optional[torch.Tensor]:
        split_sizes = [space.n for space in self.space]
        if val.ndim < 1 or val.shape[-1] != sum(split_sizes):
            return None
        return val.split(split_sizes, dim=-1)

    def index(self, index: INDEX_TYPING, tensor_to_index: torch.Tensor) -> torch.Tensor:
        if not isinstance(index, torch.Tensor):
            raise ValueError(
                f"Only tensors are allowed for indexing using"
                f" {self.__class__.__name__}.index(...)"
            )
        indices = self._split(index)
        tensor_to_index = self._split(tensor_to_index)

        out = []
        for _index, _tensor_to_index in zip(indices, tensor_to_index):
            _index = _index.nonzero().squeeze()
            _index = _index.expand((*_tensor_to_index.shape[:-1], _index.shape[-1]))
            out.append(_tensor_to_index.gather(-1, _index))
        return torch.cat(out, -1)

    def is_in(self, val: torch.Tensor) -> bool:
        vals = self._split(val)
        if vals is None:
            return False
        return all(spec.is_in(val) for val, spec in zip(vals, self._split_self()))

    def _project(self, val: torch.Tensor) -> torch.Tensor:
        vals = self._split(val)
        return torch.cat(
            [spec._project(val) for val, spec in zip(vals, self._split_self())], -1
        )

    def _split_self(self):
        result = []
        device = self.device
        dtype = self.dtype
        use_register = self.use_register
        mask = (
            self.mask.split([space.n for space in self.space], -1)
            if self.mask is not None
            else [None] * len(self.space)
        )
        for _mask, space in zip(mask, self.space):
            n = space.n
            shape = self.shape[:-1] + (n,)
            result.append(
                OneHotDiscreteTensorSpec(
                    n=n,
                    shape=shape,
                    device=device,
                    dtype=dtype,
                    use_register=use_register,
                    mask=_mask,
                )
            )
        return result

    def to_categorical(self, val: torch.Tensor, safe: bool = None) -> torch.Tensor:
        """Converts a given one-hot tensor in categorical format.

        Args:
            val (torch.Tensor, optional): One-hot tensor to convert in categorical format.
            safe (bool): boolean value indicating whether a check should be
                performed on the value against the domain of the spec.
                Defaults to the value of the ``CHECK_SPEC_ENCODE`` environment variable.

        Returns:
            The categorical tensor.
        """
        if safe is None:
            safe = _CHECK_SPEC_ENCODE
        if safe:
            self.assert_is_in(val)
        vals = self._split(val)
        return torch.stack([val.long().argmax(-1) for val in vals], -1)

    def to_categorical_spec(self) -> MultiDiscreteTensorSpec:
        """Converts the spec to the equivalent categorical spec."""
        return MultiDiscreteTensorSpec(
            [_space.n for _space in self.space],
            device=self.device,
            shape=[*self.shape[:-1], len(self.space)],
            mask=self.mask,
        )

    def expand(self, *shape):
        nvecs = [space.n for space in self.space]
        if len(shape) == 1 and isinstance(shape[0], (tuple, list, torch.Size)):
            shape = shape[0]
        if any(val < 0 for val in shape):
            raise ValueError(
                f"{self.__class__.__name__}.expand does not support negative shapes."
            )
        if any(s1 != s2 and s2 != 1 for s1, s2 in zip(shape[-self.ndim :], self.shape)):
            raise ValueError(
                f"The last {self.ndim} of the expanded shape {shape} must match the"
                f"shape of the {self.__class__.__name__} spec in expand()."
            )
        mask = self.mask.expand(shape) if self.mask is not None else None
        return self.__class__(
            nvec=nvecs,
            shape=shape,
            device=self.device,
            dtype=self.dtype,
            mask=mask,
        )

    def squeeze(self, dim=None):
        if self.shape[-1] == 1 and dim in (len(self.shape), -1, None):
            raise ValueError(f"Final dimension of {type(self)} must remain unchanged")

        shape = _squeezed_shape(self.shape, dim)
        if shape is None:
            return self
        mask = self.mask.reshape(shape) if self.mask is not None else None
        return self.__class__(
            nvec=self.nvec,
            shape=shape,
            device=self.device,
            dtype=self.dtype,
            mask=mask,
        )

    def unsqueeze(self, dim: int):
        if dim in (len(self.shape), -1):
            raise ValueError(f"Final dimension of {type(self)} must remain unchanged")
        shape = _unsqueezed_shape(self.shape, dim)
        mask = self.mask.reshape(shape) if self.mask is not None else None
        return self.__class__(
            nvec=self.nvec, shape=shape, device=self.device, dtype=self.dtype, mask=mask
        )

    def unbind(self, dim: int):
        if dim in (len(self.shape), -1):
            raise ValueError(f"Final dimension of {type(self)} must remain unchanged")
        orig_dim = dim
        if dim < 0:
            dim = len(self.shape) + dim
        if dim < 0:
            raise ValueError(
                f"Cannot unbind along dim {orig_dim} with shape {self.shape}."
            )
        shape = tuple(s for i, s in enumerate(self.shape) if i != dim)
        mask = self.mask
        if mask is None:
            mask = (None,) * self.shape[dim]
        else:
            mask = mask.unbind(dim)

        return tuple(
            self.__class__(
                nvec=self.nvec,
                shape=shape,
                device=self.device,
                dtype=self.dtype,
                mask=mask[i],
            )
            for i in range(self.shape[dim])
        )

    def __getitem__(self, idx: SHAPE_INDEX_TYPING):
        """Indexes the current TensorSpec based on the provided index.

        The last dimension of the spec corresponding to the domain of the tensor elements is non-indexable.
        """
        indexed_shape = _shape_indexing(self.shape[:-1], idx)
        return self.__class__(
            nvec=self.nvec,
            shape=torch.Size(indexed_shape + [self.shape[-1]]),
            device=self.device,
            dtype=self.dtype,
        )


class DiscreteTensorSpec(TensorSpec):
    """A discrete tensor spec.

    An alternative to OneHotTensorSpec for categorical variables in TorchRL. Instead of
    using multiplication, categorical variables perform indexing which can speed up
    computation and reduce memory cost for large categorical variables.
    The last dimension of the spec (length n of the binary vector) cannot be indexed

    Example:
        >>> batch, size = 3, 4
        >>> action_value = torch.arange(batch*size)
        >>> action_value = action_value.view(batch, size).to(torch.float)
        >>> action = torch.argmax(action_value, dim=-1).to(torch.long)
        >>> chosen_action_value = action_value[range(batch), action]
        >>> print(chosen_action_value)
        tensor([ 3.,  7., 11.])

    Args:
        n (int): number of possible outcomes.
        shape: (torch.Size, optional): shape of the variable, default is "torch.Size([])".
        device (str, int or torch.device, optional): device of the tensors.
        dtype (str or torch.dtype, optional): dtype of the tensors.

    """

    shape: torch.Size
    space: DiscreteBox
    device: torch.device = torch.device("cpu")
    dtype: torch.dtype = torch.float
    domain: str = ""

    # SPEC_HANDLED_FUNCTIONS = {}

    def __init__(
        self,
        n: int,
        shape: torch.Size | None = None,
        device: DEVICE_TYPING | None = None,
        dtype: str | torch.dtype = torch.long,
        mask: torch.Tensor | None = None,
    ):
        if shape is None:
            shape = torch.Size([])
        dtype, device = _default_dtype_and_device(dtype, device)
        space = DiscreteBox(n)
        super().__init__(shape, space, device, dtype, domain="discrete")
        self.update_mask(mask)

    def update_mask(self, mask):
        if mask is not None:
            try:
                mask = mask.expand(*self.shape, self.space.n)
            except RuntimeError as err:
                raise RuntimeError("Cannot expand mask to the desired shape.") from err
            if mask.dtype != torch.bool:
                raise ValueError("Only boolean masks are accepted.")
        self.mask = mask

    def rand(self, shape=None) -> torch.Tensor:
        if shape is None:
            shape = torch.Size([])
        if self.mask is None:
            return torch.randint(
                0,
                self.space.n,
                torch.Size([*shape, *self.shape]),
                device=self.device,
                dtype=self.dtype,
            )
        mask = self.mask
        mask = mask.expand(*shape, *mask.shape)
        if mask.ndim > 2:
            mask_flat = torch.flatten(mask, 0, -2)
        else:
            mask_flat = mask
        shape_out = mask.shape[:-1]
        out = torch.multinomial(mask_flat.float(), 1).reshape(shape_out)
        return out

    def _project(self, val: torch.Tensor) -> torch.Tensor:
        if val.dtype not in (torch.int, torch.long):
            val = torch.round(val)
        if self.mask is None:
            return val.clamp_(min=0, max=self.space.n - 1)
        shape = self.mask.shape
        shape = torch.Size([*torch.broadcast_shapes(shape[:-1], val.shape), shape[-1]])
        mask_expand = self.mask.expand(shape)
        gathered = mask_expand.gather(-1, val.unsqueeze(-1))
        oob = ~gathered.all(-1)
        new_val = torch.multinomial(mask_expand[oob].float(), 1).squeeze(-1)
        val = torch.masked_scatter(val, oob, new_val)
        return val

    def is_in(self, val: torch.Tensor) -> bool:
        if self.mask is None:
            return (0 <= val).all() and (val < self.space.n).all()
        shape = self.mask.shape
        shape = torch.Size([*torch.broadcast_shapes(shape[:-1], val.shape), shape[-1]])
        mask_expand = self.mask.expand(shape)
        gathered = mask_expand.gather(-1, val.unsqueeze(-1))
        return gathered.all()

    def __getitem__(self, idx: SHAPE_INDEX_TYPING):
        """Indexes the current TensorSpec based on the provided index."""
        indexed_shape = torch.Size(_shape_indexing(self.shape, idx))
        return self.__class__(
            n=self.space.n,
            shape=indexed_shape,
            device=self.device,
            dtype=self.dtype,
        )

    def __eq__(self, other):
        if not hasattr(other, "mask"):
            return False
        mask_equal = (self.mask is None and other.mask is None) or (
            isinstance(self.mask, torch.Tensor)
            and isinstance(other.mask, torch.Tensor)
            and (self.mask.shape == other.mask.shape)
            and (self.mask == other.mask).all()
        )
        return (
            type(self) == type(other)
            and self.shape == other.shape
            and self.space == other.space
            and self.device == other.device
            and self.dtype == other.dtype
            and self.domain == other.domain
            and mask_equal
        )

    def to_numpy(self, val: torch.Tensor, safe: bool = None) -> dict:
        if safe is None:
            safe = _CHECK_SPEC_ENCODE
        # if not val.shape and not safe:
        #     return val.item()
        return super().to_numpy(val, safe)

    def to_one_hot(self, val: torch.Tensor, safe: bool = None) -> torch.Tensor:
        """Encodes a discrete tensor from the spec domain into its one-hot correspondent.

        Args:
            val (torch.Tensor, optional): Tensor to one-hot encode.
            safe (bool): boolean value indicating whether a check should be
                performed on the value against the domain of the spec.
                Defaults to the value of the ``CHECK_SPEC_ENCODE`` environment variable.

        Returns:
            The one-hot encoded tensor.
        """
        if safe is None:
            safe = _CHECK_SPEC_ENCODE
        if safe:
            self.assert_is_in(val)
        return torch.nn.functional.one_hot(val, self.space.n)

    def to_one_hot_spec(self) -> OneHotDiscreteTensorSpec:
        """Converts the spec to the equivalent one-hot spec."""
        shape = [*self.shape, self.space.n]
        return OneHotDiscreteTensorSpec(
            n=self.space.n,
            shape=shape,
            device=self.device,
        )

    def expand(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list, torch.Size)):
            shape = shape[0]
        if any(val < 0 for val in shape):
            raise ValueError(
                f"{self.__class__.__name__}.expand does not support negative shapes."
            )
        if any(s1 != s2 and s2 != 1 for s1, s2 in zip(shape[-self.ndim :], self.shape)):
            raise ValueError(
                f"The last {self.ndim} of the expanded shape {shape} must match the"
                f"shape of the {self.__class__.__name__} spec in expand()."
            )
        return self.__class__(
            n=self.space.n, shape=shape, device=self.device, dtype=self.dtype
        )

    def squeeze(self, dim=None):
        shape = _squeezed_shape(self.shape, dim)
        mask = self.mask
        if mask is not None:
            mask = mask.view(*shape, mask.shape[-1])

        if shape is None:
            return self
        return self.__class__(
            n=self.space.n,
            shape=shape,
            device=self.device,
            dtype=self.dtype,
            mask=mask,
        )

    def unsqueeze(self, dim: int):
        shape = _unsqueezed_shape(self.shape, dim)
        mask = self.mask
        if mask is not None:
            mask = mask.view(*shape, mask.shape[-1])
        return self.__class__(
            n=self.space.n,
            shape=shape,
            device=self.device,
            dtype=self.dtype,
            mask=mask,
        )

    def unbind(self, dim: int):
        orig_dim = dim
        if dim < 0:
            dim = len(self.shape) + dim
        if dim < 0:
            raise ValueError(
                f"Cannot unbind along dim {orig_dim} with shape {self.shape}."
            )
        shape = tuple(s for i, s in enumerate(self.shape) if i != dim)
        mask = self.mask
        if mask is None:
            mask = (None,) * self.shape[dim]
        else:
            mask = mask.unbind(dim)
        return tuple(
            self.__class__(
                n=self.space.n,
                shape=shape,
                device=self.device,
                dtype=self.dtype,
                mask=mask[i],
            )
            for i in range(self.shape[dim])
        )

    def to(self, dest: Union[torch.dtype, DEVICE_TYPING]) -> CompositeSpec:
        if isinstance(dest, torch.dtype):
            dest_dtype = dest
            dest_device = self.device
        else:
            dest_dtype = self.dtype
            dest_device = torch.device(dest)
        if dest_device == self.device and dest_dtype == self.dtype:
            return self
        return self.__class__(
            n=self.space.n, shape=self.shape, device=dest_device, dtype=dest_dtype
        )

    def clone(self) -> DiscreteTensorSpec:
        return self.__class__(
            n=self.space.n,
            shape=self.shape,
            device=self.device,
            dtype=self.dtype,
            mask=self.mask.clone() if self.mask is not None else None,
        )


@dataclass(repr=False)
class BinaryDiscreteTensorSpec(DiscreteTensorSpec):
    """A binary discrete tensor spec.

    Args:
        n (int): length of the binary vector.
        shape (torch.Size, optional): total shape of the sampled tensors.
            If provided, the last dimension must match n.
        device (str, int or torch.device, optional): device of the tensors.
        dtype (str or torch.dtype, optional): dtype of the tensors. Defaults to torch.long.

    Examples:
        >>> spec = BinaryDiscreteTensorSpec(n=4, shape=(5, 4), device="cpu", dtype=torch.bool)
        >>> print(spec.zero())
    """

    def __init__(
        self,
        n: int,
        shape: Optional[torch.Size] = None,
        device: Optional[DEVICE_TYPING] = None,
        dtype: Union[str, torch.dtype] = torch.long,
    ):
        if shape is None or not len(shape):
            shape = torch.Size((n,))
        else:
            shape = torch.Size(shape)
            if shape[-1] != n:
                raise ValueError(
                    f"The last value of the shape must match n for spec {self.__class__}. "
                    f"Got n={n} and shape={shape}."
                )
        super().__init__(n=2, shape=shape, device=device, dtype=dtype)

    def expand(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list, torch.Size)):
            shape = shape[0]
        if any(val < 0 for val in shape):
            raise ValueError(
                f"{self.__class__.__name__}.expand does not support negative shapes."
            )
        if any(s1 != s2 and s2 != 1 for s1, s2 in zip(shape[-self.ndim :], self.shape)):
            raise ValueError(
                f"The last {self.ndim} of the expanded shape {shape} must match the"
                f"shape of the {self.__class__.__name__} spec in expand()."
            )
        return self.__class__(
            n=self.shape[-1], shape=shape, device=self.device, dtype=self.dtype
        )

    def squeeze(self, dim=None):
        shape = _squeezed_shape(self.shape, dim)
        if shape is None:
            return self
        return self.__class__(
            n=self.shape[-1], shape=shape, device=self.device, dtype=self.dtype
        )

    def unsqueeze(self, dim: int):
        shape = _unsqueezed_shape(self.shape, dim)
        return self.__class__(
            n=self.shape[-1], shape=shape, device=self.device, dtype=self.dtype
        )

    def unbind(self, dim: int):
        if dim in (len(self.shape) - 1, -1):
            raise ValueError(f"Final dimension of {type(self)} must remain unchanged")

        orig_dim = dim
        if dim < 0:
            dim = len(self.shape) + dim
        if dim < 0:
            raise ValueError(
                f"Cannot unbind along dim {orig_dim} with shape {self.shape}."
            )
        shape = tuple(s for i, s in enumerate(self.shape) if i != dim)
        return tuple(
            self.__class__(
                n=self.shape[-1], shape=shape, device=self.device, dtype=self.dtype
            )
            for i in range(self.shape[dim])
        )

    def to(self, dest: Union[torch.dtype, DEVICE_TYPING]) -> CompositeSpec:
        if isinstance(dest, torch.dtype):
            dest_dtype = dest
            dest_device = self.device
        else:
            dest_dtype = self.dtype
            dest_device = torch.device(dest)
        if dest_device == self.device and dest_dtype == self.dtype:
            return self
        return self.__class__(
            n=self.shape[-1], shape=self.shape, device=dest_device, dtype=dest_dtype
        )

    def clone(self) -> BinaryDiscreteTensorSpec:
        return self.__class__(
            n=self.shape[-1],
            shape=self.shape,
            device=self.device,
            dtype=self.dtype,
        )

    def __getitem__(self, idx: SHAPE_INDEX_TYPING):
        """Indexes the current TensorSpec based on the provided index.

        The last dimension of the spec (length n of the binary vector) cannot be indexed.
        """
        indexed_shape = _shape_indexing(self.shape[:-1], idx)
        return self.__class__(
            n=self.shape[-1],
            shape=torch.Size(indexed_shape + [self.shape[-1]]),
            device=self.device,
            dtype=self.dtype,
        )


@dataclass(repr=False)
class MultiDiscreteTensorSpec(DiscreteTensorSpec):
    """A concatenation of discrete tensor spec.

    Args:
        nvec (iterable of integers or torch.Tensor): cardinality of each of the elements of
            the tensor. Can have several axes.
        shape (torch.Size, optional): total shape of the sampled tensors.
            If provided, the last m dimensions must match nvec.shape.
        device (str, int or torch.device, optional): device of
            the tensors.
        dtype (str or torch.dtype, optional): dtype of the tensors.

    Examples:
        >>> ts = MultiDiscreteTensorSpec((3,2,3))
        >>> ts.is_in(torch.tensor([2, 0, 1]))
        True
        >>> ts.is_in(torch.tensor([2, 2, 1]))
        False
    """

    # SPEC_HANDLED_FUNCTIONS = {}

    def __init__(
        self,
        nvec: Union[Sequence[int], torch.Tensor, int],
        shape: Optional[torch.Size] = None,
        device: Optional[DEVICE_TYPING] = None,
        dtype: Optional[Union[str, torch.dtype]] = torch.long,
        mask: torch.Tensor | None = None,
    ):
        if not isinstance(nvec, torch.Tensor):
            nvec = torch.tensor(nvec)
        if nvec.ndim < 1:
            nvec = nvec.unsqueeze(0)
        self.nvec = nvec
        dtype, device = _default_dtype_and_device(dtype, device)
        if shape is None:
            shape = nvec.shape
        else:
            shape = torch.Size(shape)
            if shape[-1] != nvec.shape[-1]:
                raise ValueError(
                    f"The last value of the shape must match nvec.shape[-1] for transform of type {self.__class__}. "
                    f"Got nvec.shape[-1]={sum(nvec)} and shape={shape}."
                )

        self.nvec = self.nvec.expand(shape)

        space = BoxList.from_nvec(self.nvec)
        super(DiscreteTensorSpec, self).__init__(
            shape, space, device, dtype, domain="discrete"
        )
        self.update_mask(mask)

    def update_mask(self, mask):
        if mask is not None:
            try:
                mask = mask.expand(*self.shape[:-1], mask.shape[-1])
            except RuntimeError as err:
                raise RuntimeError("Cannot expand mask to the desired shape.") from err
            if mask.dtype != torch.bool:
                raise ValueError("Only boolean masks are accepted.")
        self.mask = mask

    def to(self, dest: Union[torch.dtype, DEVICE_TYPING]) -> CompositeSpec:
        if isinstance(dest, torch.dtype):
            dest_dtype = dest
            dest_device = self.device
        else:
            dest_dtype = self.dtype
            dest_device = torch.device(dest)
        if dest_device == self.device and dest_dtype == self.dtype:
            return self
        mask = self.mask.to(dest) if self.mask is not None else None
        return self.__class__(
            n=self.nvec.to(dest),
            shape=None,
            device=dest_device,
            dtype=dest_dtype,
            mask=mask,
        )

    def __eq__(self, other):
        if not hasattr(other, "mask"):
            return False
        mask_equal = (self.mask is None and other.mask is None) or (
            isinstance(self.mask, torch.Tensor)
            and isinstance(other.mask, torch.Tensor)
            and (self.mask.shape == other.mask.shape)
            and (self.mask == other.mask).all()
        )
        return (
            type(self) == type(other)
            and self.shape == other.shape
            and self.space == other.space
            and self.device == other.device
            and self.dtype == other.dtype
            and self.domain == other.domain
            and mask_equal
        )

    def clone(self) -> MultiDiscreteTensorSpec:
        return self.__class__(
            nvec=self.nvec.clone(),
            shape=None,
            device=self.device,
            dtype=self.dtype,
            mask=self.mask.clone() if self.mask is not None else None,
        )

    def _rand(self, space: Box, shape: torch.Size, i: int):
        x = []
        for _s in space:
            if isinstance(_s, BoxList):
                x.append(self._rand(_s, shape[:-1], i - 1))
            else:
                x.append(
                    torch.randint(
                        0,
                        _s.n,
                        shape,
                        device=self.device,
                        dtype=self.dtype,
                    )
                )
        return torch.stack(x, -1)

    def rand(self, shape: Optional[torch.Size] = None) -> torch.Tensor:
        if self.mask is not None:
            splits = self._split_self()
            return torch.stack([split.rand(shape) for split in splits], -1)
        if shape is None:
            shape = self.shape[:-1]
        else:
            shape = (
                *shape,
                *self.shape[:-1],
            )
        x = self._rand(space=self.space, shape=shape, i=self.nvec.ndim)
        if self.shape == torch.Size([1]):
            x = x.squeeze(-1)
        return x

    def _split_self(self):
        result = []
        device = self.device
        dtype = self.dtype
        nvec = self.nvec
        if nvec.ndim > 1:
            nvec = torch.flatten(nvec, 0, -2)[0]
            if (self.nvec != nvec).any():
                raise ValueError(
                    f"Only homogeneous MultiDiscrete specs can be masked, got nvec={self.nvec}."
                )
        nvec = nvec.tolist()
        mask = (
            self.mask.split(nvec, -1)
            if self.mask is not None
            else [None] * len(self.space)
        )
        for n, _mask in zip(nvec, mask):
            shape = self.shape[:-1]
            result.append(
                DiscreteTensorSpec(
                    n=n, shape=shape, device=device, dtype=dtype, mask=_mask
                )
            )
        return result

    def _project(self, val: torch.Tensor) -> torch.Tensor:
        if self.mask is not None:
            return torch.stack(
                [
                    spec._project(_val)
                    for (_val, spec) in zip(val.unbind(-1), self._split_self())
                ],
                -1,
            )

        val_is_scalar = val.ndim < 1
        if val_is_scalar:
            val = val.unsqueeze(0)
        if not self.dtype.is_floating_point:
            val = torch.round(val)
        val = val.type(self.dtype)
        val[val >= self.nvec] = (self.nvec.expand_as(val)[val >= self.nvec] - 1).type(
            self.dtype
        )
        return val.squeeze(0) if val_is_scalar else val

    def is_in(self, val: torch.Tensor) -> bool:
        if self.mask is not None:
            return all(
                spec.is_in(_val)
                for (_val, spec) in zip(val.unbind(-1), self._split_self())
            )

        if val.ndim < 1:
            val = val.unsqueeze(0)
        val_have_wrong_dim = (
            self.shape != torch.Size([1])
            and val.shape[-len(self.shape) :] != self.shape
        )
        if self.dtype != val.dtype or len(self.shape) > val.ndim or val_have_wrong_dim:
            return False
        val_device = val.device
        return (
            (
                (val >= torch.zeros(self.nvec.size(), device=val_device))
                & (val < self.nvec.to(val_device))
            )
            .all()
            .item()
        )

    def to_one_hot(
        self, val: torch.Tensor, safe: bool = None
    ) -> Union[MultiOneHotDiscreteTensorSpec, torch.Tensor]:
        """Encodes a discrete tensor from the spec domain into its one-hot correspondent.

        Args:
            val (torch.Tensor, optional): Tensor to one-hot encode.
            safe (bool): boolean value indicating whether a check should be
                performed on the value against the domain of the spec.
                Defaults to the value of the ``CHECK_SPEC_ENCODE`` environment variable.

        Returns:
            The one-hot encoded tensor.
        """
        if safe is None:
            safe = _CHECK_SPEC_ENCODE
        if safe:
            self.assert_is_in(val)
        return torch.cat(
            [
                torch.nn.functional.one_hot(val[..., i], n)
                for i, n in enumerate(self.nvec)
            ],
            -1,
        ).to(self.device)

    def to_one_hot_spec(self) -> MultiOneHotDiscreteTensorSpec:
        """Converts the spec to the equivalent one-hot spec."""
        nvec = [_space.n for _space in self.space]
        return MultiOneHotDiscreteTensorSpec(
            nvec,
            device=self.device,
            shape=[*self.shape[:-1], sum(nvec)],
            mask=self.mask,
        )

    def expand(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list, torch.Size)):
            shape = shape[0]
        if any(val < 0 for val in shape):
            raise ValueError(
                f"{self.__class__.__name__}.expand does not support negative shapes."
            )
        if any(s1 != s2 and s2 != 1 for s1, s2 in zip(shape[-self.ndim :], self.shape)):
            raise ValueError(
                f"The last {self.ndim} of the expanded shape {shape} must match the"
                f"shape of the {self.__class__.__name__} spec in expand()."
            )
        mask = (
            self.mask.expand(*shape, self.mask.shape[-1])
            if self.mask is not None
            else None
        )
        return self.__class__(
            nvec=self.nvec,
            shape=shape,
            device=self.device,
            dtype=self.dtype,
            mask=mask,
        )

    def squeeze(self, dim: int | None = None):
        if self.shape[-1] == 1 and dim in (len(self.shape), -1, None):
            raise ValueError(f"Final dimension of {type(self)} must remain unchanged")
        shape = _squeezed_shape(self.shape, dim)
        if shape is None:
            return self

        if dim is None:
            nvec = self.nvec.squeeze()
        else:
            nvec = self.nvec.squeeze(dim)
        mask = self.mask
        if mask is not None:
            mask = mask.view(*shape[:-1], mask.shape[-1])
        return self.__class__(
            nvec=nvec, shape=shape, device=self.device, dtype=self.dtype, mask=mask
        )

    def unsqueeze(self, dim: int):
        if dim in (len(self.shape), -1):
            raise ValueError(f"Final dimension of {type(self)} must remain unchanged")
        shape = _unsqueezed_shape(self.shape, dim)
        nvec = self.nvec.unsqueeze(dim)
        mask = self.mask
        if mask is not None:
            mask = mask.view(*shape[:-1], mask.shape[-1])
        return self.__class__(
            nvec=nvec,
            shape=shape,
            device=self.device,
            dtype=self.dtype,
            mask=mask,
        )

    def unbind(self, dim: int):
        if dim in (len(self.shape), -1):
            raise ValueError(f"Final dimension of {type(self)} must remain unchanged")
        orig_dim = dim
        if dim < 0:
            dim = len(self.shape) + dim
        if dim < 0:
            raise ValueError(
                f"Cannot unbind along dim {orig_dim} with shape {self.shape}."
            )
        shape = tuple(s for i, s in enumerate(self.shape) if i != dim)
        mask = self.mask
        nvec = self.nvec.unbind(dim)
        if mask is not None:
            mask = mask.unbind(dim)
        else:
            mask = (None,) * self.shape[dim]
        return tuple(
            self.__class__(
                nvec=nvec[i],
                shape=shape,
                device=self.device,
                dtype=self.dtype,
                mask=mask[i],
            )
            for i in range(self.shape[dim])
        )

    def __getitem__(self, idx: SHAPE_INDEX_TYPING):
        """Indexes the current TensorSpec based on the provided index."""
        if _is_nested_list(idx):
            raise NotImplementedError(
                "Pending resolution of https://github.com/pytorch/pytorch/issues/100080."
            )

        return self.__class__(
            nvec=self.nvec[idx].clone(),
            shape=None,
            device=self.device,
            dtype=self.dtype,
        )


class CompositeSpec(TensorSpec):
    """A composition of TensorSpecs.

    Args:
        *args: if an unnamed argument is passed, it must be a dictionary with keys
            matching the expected keys to be found in the :obj:`CompositeSpec` object.
            This is useful to build nested CompositeSpecs with tuple indices.
        **kwargs (key (str): value (TensorSpec)): dictionary of tensorspecs
            to be stored. Values can be None, in which case is_in will be assumed
            to be ``True`` for the corresponding tensors, and :obj:`project()` will have no
            effect. `spec.encode` cannot be used with missing values.

    Examples:
        >>> pixels_spec = BoundedTensorSpec(
        ...    torch.zeros(3,32,32),
        ...    torch.ones(3, 32, 32))
        >>> observation_vector_spec = BoundedTensorSpec(torch.zeros(33),
        ...    torch.ones(33))
        >>> composite_spec = CompositeSpec(
        ...     pixels=pixels_spec,
        ...     observation_vector=observation_vector_spec)
        >>> td = TensorDict({"pixels": torch.rand(10,3,32,32),
        ...    "observation_vector": torch.rand(10,33)}, batch_size=[10])
        >>> print("td (rand) is within bounds: ", composite_spec.is_in(td))
        td (rand) is within bounds:  True
        >>> td = TensorDict({"pixels": torch.randn(10,3,32,32),
        ...    "observation_vector": torch.randn(10,33)}, batch_size=[10])
        >>> print("td (randn) is within bounds: ", composite_spec.is_in(td))
        td (randn) is within bounds:  False
        >>> td_project = composite_spec.project(td)
        >>> print("td modification done in place: ", td_project is td)
        td modification done in place:  True
        >>> print("check td is within bounds after projection: ",
        ...    composite_spec.is_in(td_project))
        check td is within bounds after projection:  True
        >>> print("random td: ", composite_spec.rand([3,]))
        random td:  TensorDict(
            fields={
                observation_vector: Tensor(torch.Size([3, 33]), dtype=torch.float32),
                pixels: Tensor(torch.Size([3, 3, 32, 32]), dtype=torch.float32)},
            batch_size=torch.Size([3]),
            device=None,
            is_shared=False)


    Examples:
        >>> # we can build a nested composite spec using unnamed arguments
        >>> print(CompositeSpec({("a", "b"): None, ("a", "c"): None}))
        CompositeSpec(
            a: CompositeSpec(
                b: None,
                c: None))

    CompositeSpec supports nested indexing:
        >>> spec = CompositeSpec(obs=None)
        >>> spec["nested", "x"] = None
        >>> print(spec)
        CompositeSpec(
            nested: CompositeSpec(
                x: None),
            x: None)

    """

    shape: torch.Size
    domain: str = "composite"

    SPEC_HANDLED_FUNCTIONS = {}

    @classmethod
    def __new__(cls, *args, **kwargs):
        cls._device = torch.device("cpu")
        cls._locked = False
        return super().__new__(cls)

    @property
    def shape(self):
        return self._shape

    @shape.setter
    def shape(self, value: torch.Size):
        if self.locked:
            raise RuntimeError("Cannot modify shape of locked composite spec.")
        for key, spec in self.items():
            if isinstance(spec, CompositeSpec):
                if spec.shape[: len(value)] != value:
                    spec.shape = value
            elif spec is not None:
                if spec.shape[: len(value)] != value:
                    raise ValueError(
                        f"The shape of the spec and the CompositeSpec mismatch during shape resetting: the "
                        f"{self.ndim} first dimensions should match but got self['{key}'].shape={spec.shape} and "
                        f"CompositeSpec.shape={self.shape}."
                    )
        self._shape = torch.Size(value)

    def is_empty(self):
        """Whether the composite spec contains specs or not."""
        return len(self._specs) == 0

    @property
    def ndim(self):
        return self.ndimension()

    def ndimension(self):
        return len(self.shape)

    def set(self, name, spec):
        if self.locked:
            raise RuntimeError("Cannot modify a locked CompositeSpec.")
        if spec is not None:
            shape = spec.shape
            if shape[: self.ndim] != self.shape:
                raise ValueError(
                    "The shape of the spec and the CompositeSpec mismatch: the first "
                    f"{self.ndim} dimensions should match but got spec.shape={spec.shape} and "
                    f"CompositeSpec.shape={self.shape}."
                )
        self._specs[name] = spec

    def __init__(self, *args, shape=None, device=None, **kwargs):
        if shape is None:
            # Should we do this? Other specs have a default empty shape, maybe it would make sense to keep it
            # optional for composite (for clarity and easiness of use).
            # warnings.warn("shape=None for CompositeSpec will soon be deprecated. Make sure you set the "
            #               "batch size of your CompositeSpec as you would do for a tensordict.")
            shape = []
        self._shape = torch.Size(shape)
        self._specs = {}
        for key, value in kwargs.items():
            self.set(key, value)

        _device = torch.device(device) if device is not None else device
        if len(kwargs):
            for key, item in self.items():
                if item is None:
                    continue

                try:
                    item_device = item.device
                except RuntimeError as err:
                    cond1 = DEVICE_ERR_MSG in str(err)
                    if cond1:
                        item_device = _device
                    else:
                        raise err

                if _device is None:
                    _device = item_device
                elif item_device != _device:
                    raise RuntimeError(
                        f"Setting a new attribute ({key}) on another device "
                        f"({item.device} against {_device}). All devices of "
                        "CompositeSpec must match."
                    )
        self._device = _device
        if len(args):
            if len(args) > 1:
                raise RuntimeError(
                    "Got multiple arguments, when at most one is expected for CompositeSpec."
                )
            argdict = args[0]
            if not isinstance(argdict, (dict, CompositeSpec)):
                raise RuntimeError(
                    f"Expected a dictionary of specs, but got an argument of type {type(argdict)}."
                )
            for k, item in argdict.items():
                if isinstance(item, dict):
                    item = CompositeSpec(item, shape=shape)
                if item is not None:
                    if self._device is None:
                        try:
                            self._device = item.device
                        except RuntimeError as err:
                            if DEVICE_ERR_MSG in str(err):
                                self._device = item._device
                            else:
                                raise err
                self[k] = item

    @property
    def device(self) -> DEVICE_TYPING:
        if self._device is None:
            # try to replace device by the true device
            _device = None
            for value in self.values():
                if value is not None:
                    _device = value.device
            if _device is None:
                raise RuntimeError(
                    "device of empty CompositeSpec is not defined. "
                    "You can set it directly by calling "
                    "`spec.device = device`."
                )
            self._device = _device
        return self._device

    @device.setter
    def device(self, device: DEVICE_TYPING):
        device = torch.device(device)
        self.to(device)

    def __getitem__(self, idx):
        """Indexes the current CompositeSpec based on the provided index."""
        if isinstance(idx, (str, tuple)):
            idx_unravel = unravel_key(idx)
        else:
            idx_unravel = ()
        if idx_unravel:
            if isinstance(idx_unravel, tuple):
                return self[idx[0]][idx[1:]]
            if idx_unravel in {"shape", "device", "dtype", "space"}:
                raise AttributeError(f"CompositeSpec has no key {idx_unravel}")
            return self._specs[idx_unravel]

        indexed_shape = _shape_indexing(self.shape, idx)
        indexed_specs = {}
        for k, v in self._specs.items():
            _idx = idx
            if isinstance(idx, tuple):
                protected_dims = 0
                if any(
                    isinstance(v, spec_class)
                    for spec_class in [
                        BinaryDiscreteTensorSpec,
                        MultiDiscreteTensorSpec,
                        OneHotDiscreteTensorSpec,
                    ]
                ):
                    protected_dims = 1
                # TensorSpecs dims which are not part of the composite shape cannot be indexed
                _idx = idx + (slice(None),) * (
                    len(v.shape) - len(self.shape) - protected_dims
                )
            indexed_specs[k] = v[_idx] if v is not None else None

        try:
            device = self.device
        except RuntimeError:
            device = self._device

        return self.__class__(
            indexed_specs,
            shape=indexed_shape,
            device=device,
        )

    def __setitem__(self, key, value):
        if isinstance(key, tuple) and len(key) > 1:
            if key[0] not in self.keys(True):
                self[key[0]] = CompositeSpec(shape=self.shape)
            self[key[0]][key[1:]] = value
            return
        elif isinstance(key, tuple):
            self[key[0]] = value
            return
        elif not isinstance(key, str):
            raise TypeError(f"Got key of type {type(key)} when a string was expected.")
        if key in {"shape", "device", "dtype", "space"}:
            raise AttributeError(f"CompositeSpec[{key}] cannot be set")
        try:
            if value is not None and value.device != self.device:
                raise RuntimeError(
                    f"Setting a new attribute ({key}) on another device ({value.device} against {self.device}). "
                    f"All devices of CompositeSpec must match."
                )
        except RuntimeError as err:
            cond1 = DEVICE_ERR_MSG in str(err)
            cond2 = self._device is None
            if cond1 and cond2:
                try:
                    device_val = value.device
                    self.to(device_val)
                except RuntimeError as suberr:
                    if DEVICE_ERR_MSG in str(suberr):
                        pass
                    else:
                        raise suberr
            elif cond1:
                pass
            else:
                raise err

        self.set(key, value)

    def __iter__(self):
        for k in self._specs:
            yield k

    def __delitem__(self, key: str) -> None:
        if isinstance(key, tuple) and len(key) > 1:
            del self._specs[key[0]][key[1:]]
            return
        elif isinstance(key, tuple):
            del self._specs[key[0]]
            return
        elif not isinstance(key, str):
            raise TypeError(
                f"Got key of type {type(key)} when a string or a tuple of strings was expected."
            )

        if key in {"shape", "device", "dtype", "space"}:
            raise AttributeError(f"CompositeSpec has no key {key}")
        del self._specs[key]

    def encode(
        self, vals: Dict[str, Any], *, ignore_device: bool = False
    ) -> Dict[str, torch.Tensor]:
        if isinstance(vals, TensorDict):
            out = vals.select()  # create and empty tensordict similar to vals
        else:
            out = TensorDict({}, torch.Size([]), _run_checks=False)
        for key, item in vals.items():
            if item is None:
                raise RuntimeError(
                    "CompositeSpec.encode cannot be used with missing values."
                )
            try:
                out[key] = self[key].encode(item, ignore_device=ignore_device)
            except KeyError:
                raise KeyError(
                    f"The CompositeSpec instance with keys {self.keys()} does not have a '{key}' key."
                )
        return out

    def __repr__(self) -> str:
        sub_str = [
            indent(f"{k}: {str(item)}", 4 * " ") for k, item in self._specs.items()
        ]
        sub_str = ",\n".join(sub_str)
        return f"CompositeSpec(\n{sub_str}, device={self._device}, shape={self.shape})"

    def type_check(
        self,
        value: Union[torch.Tensor, TensorDictBase],
        selected_keys: Union[str, Optional[Sequence[str]]] = None,
    ):
        if isinstance(value, torch.Tensor) and isinstance(selected_keys, str):
            value = {selected_keys: value}
            selected_keys = [selected_keys]

        for _key in self.keys():
            if self[_key] is not None and (
                selected_keys is None or _key in selected_keys
            ):
                self._specs[_key].type_check(value[_key], _key)

    def is_in(self, val: Union[dict, TensorDictBase]) -> bool:
        for key, item in self._specs.items():
            if item is None or (isinstance(item, CompositeSpec) and item.is_empty()):
                continue
            val_item = val.get(key)
            if not item.is_in(val_item):
                return False
        return True

    def project(self, val: TensorDictBase) -> TensorDictBase:
        for key, item in self.items():
            if item is None:
                continue
            _val = val.get(key)
            if not self._specs[key].is_in(_val):
                val.set(key, self._specs[key].project(_val))
        return val

    def rand(self, shape=None) -> TensorDictBase:
        if shape is None:
            shape = torch.Size([])
        _dict = {
            key: self[key].rand(shape) for key in self.keys() if self[key] is not None
        }
        return TensorDict(
            _dict,
            batch_size=[*shape, *self.shape],
            device=self._device,
        )

    def keys(
        self,
        include_nested: bool = False,
        leaves_only: bool = False,
    ) -> KeysView:
        """Keys of the CompositeSpec.

        The keys argument reflect those of :class:`tensordict.TensorDict`.

        Args:
            include_nested (bool, optional): if ``False``, the returned keys will not be nested. They will
                represent only the immediate children of the root, and not the whole nested sequence, i.e. a
                :obj:`CompositeSpec(next=CompositeSpec(obs=None))` will lead to the keys
                :obj:`["next"]. Default is ``False``, i.e. nested keys will not
                be returned.
            leaves_only (bool, optional): if ``False``, the values returned
                will contain every level of nesting, i.e. a :obj:`CompositeSpec(next=CompositeSpec(obs=None))`
                will lead to the keys :obj:`["next", ("next", "obs")]`.
                Default is ``False``.
        """
        return _CompositeSpecKeysView(
            self, include_nested=include_nested, leaves_only=leaves_only
        )

    def items(
        self,
        include_nested: bool = False,
        leaves_only: bool = False,
    ) -> ItemsView:
        """Items of the CompositeSpec.

        Args:
            include_nested (bool, optional): if ``False``, the returned keys will not be nested. They will
                represent only the immediate children of the root, and not the whole nested sequence, i.e. a
                :obj:`CompositeSpec(next=CompositeSpec(obs=None))` will lead to the keys
                :obj:`["next"]. Default is ``False``, i.e. nested keys will not
                be returned.
            leaves_only (bool, optional): if ``False``, the values returned
                will contain every level of nesting, i.e. a :obj:`CompositeSpec(next=CompositeSpec(obs=None))`
                will lead to the keys :obj:`["next", ("next", "obs")]`.
                Default is ``False``.
        """
        if not include_nested and not leaves_only:
            yield from self._specs.items()
        else:
            yield from (
                (key, self[key])
                for key in self.keys(
                    include_nested=include_nested, leaves_only=leaves_only
                )
            )

    def values(
        self,
        include_nested: bool = False,
        leaves_only: bool = False,
    ) -> ValuesView:
        """Values of the CompositeSpec.

        Args:
            include_nested (bool, optional): if ``False``, the returned keys will not be nested. They will
                represent only the immediate children of the root, and not the whole nested sequence, i.e. a
                :obj:`CompositeSpec(next=CompositeSpec(obs=None))` will lead to the keys
                :obj:`["next"]. Default is ``False``, i.e. nested keys will not
                be returned.
            leaves_only (bool, optional): if ``False``, the values returned
                will contain every level of nesting, i.e. a :obj:`CompositeSpec(next=CompositeSpec(obs=None))`
                will lead to the keys :obj:`["next", ("next", "obs")]`.
                Default is ``False``.
        """
        if not include_nested and not leaves_only:
            yield from self._specs.values()
        else:
            yield from (
                self[key]
                for key in self.keys(
                    include_nested=include_nested, leaves_only=leaves_only
                )
            )

    def __len__(self):
        return len(self.keys())

    def to(self, dest: Union[torch.dtype, DEVICE_TYPING]) -> CompositeSpec:
        if not isinstance(dest, (str, int, torch.device)):
            raise ValueError(
                "Only device casting is allowed with specs of type CompositeSpec."
            )
        if self._device and self._device == torch.device(dest):
            return self

        _device = torch.device(dest)
        items = list(self.items())
        kwargs = {}
        for key, value in items:
            if value is None:
                kwargs[key] = value
                continue
            kwargs[key] = value.to(dest)
        return self.__class__(**kwargs, device=_device, shape=self.shape)

    def clone(self) -> CompositeSpec:
        try:
            device = self.device
        except RuntimeError:
            device = self._device
        return self.__class__(
            {
                key: item.clone() if item is not None else None
                for key, item in self.items()
            },
            device=device,
            shape=self.shape,
        )

    def empty(self):
        """Create a spec like self, but with no entries."""
        try:
            device = self.device
        except RuntimeError:
            device = self._device
        return self.__class__(
            {},
            device=device,
            shape=self.shape,
        )

    def to_numpy(self, val: TensorDict, safe: bool = None) -> dict:
        return {key: self[key].to_numpy(val) for key, val in val.items()}

    def zero(self, shape=None) -> TensorDictBase:
        if shape is None:
            shape = torch.Size([])
        try:
            device = self.device
        except RuntimeError:
            device = self._device
        return TensorDict(
            {
                key: self[key].zero(shape)
                for key in self.keys(True)
                if isinstance(key, str) and self[key] is not None
            },
            torch.Size([*shape, *self.shape]),
            device=device,
        )

    def __eq__(self, other):
        return (
            type(self) is type(other)
            and self._device == other._device
            and self._specs == other._specs
        )

    def update(self, dict_or_spec: Union[CompositeSpec, Dict[str, TensorSpec]]) -> None:
        for key, item in dict_or_spec.items():
            if key in self.keys(True) and isinstance(self[key], CompositeSpec):
                self[key].update(item)
                continue
            try:
                if isinstance(item, TensorSpec) and item.device != self.device:
                    item = deepcopy(item)
                    if self.device is not None:
                        item = item.to(self.device)
            except RuntimeError as err:
                if DEVICE_ERR_MSG in str(err):
                    try:
                        item_device = item.device
                        self.device = item_device
                    except RuntimeError as suberr:
                        if DEVICE_ERR_MSG in str(suberr):
                            pass
                        else:
                            raise suberr
                else:
                    raise err
            self[key] = item
        return self

    def expand(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list, torch.Size)):
            shape = shape[0]
        if any(val < 0 for val in shape):
            raise ValueError("CompositeSpec.expand does not support negative shapes.")
        if any(s1 != s2 and s2 != 1 for s1, s2 in zip(shape[-self.ndim :], self.shape)):
            raise ValueError(
                f"The last {self.ndim} of the expanded shape {shape} must match the"
                f"shape of the {self.__class__.__name__} spec in expand()."
            )
        try:
            device = self.device
        except RuntimeError:
            device = self._device
        out = CompositeSpec(
            {
                key: value.expand((*shape, *value.shape[self.ndim :]))
                if value is not None
                else None
                for key, value in tuple(self.items())
            },
            shape=shape,
            device=device,
        )
        return out

    def squeeze(self, dim: int | None = None):
        if dim is not None:
            if dim < 0:
                dim += len(self.shape)

            shape = _squeezed_shape(self.shape, dim)
            if shape is None:
                return self

            try:
                device = self.device
            except RuntimeError:
                device = self._device

            return CompositeSpec(
                {key: value.squeeze(dim) for key, value in self.items()},
                shape=shape,
                device=device,
            )

        if self.shape.count(1) == 0:
            return self

        # we can't just recursively apply squeeze with dim=None because we don't want
        # to squeeze non-batch dims of the values. Instead we find the first dim in
        # the batch dims with size 1, squeeze that, then recurse on the root spec
        out = self.squeeze(self.shape.index(1))
        return out.squeeze()

    def unsqueeze(self, dim: int):
        if dim < 0:
            dim += len(self.shape) + 1

        shape = _unsqueezed_shape(self.shape, dim)

        try:
            device = self.device
        except RuntimeError:
            device = self._device

        return CompositeSpec(
            {
                key: value.unsqueeze(dim) if value is not None else None
                for key, value in self.items()
            },
            shape=shape,
            device=device,
        )

    def unbind(self, dim: int):
        orig_dim = dim
        if dim < 0:
            dim = len(self.shape) + dim
        if dim < 0:
            raise ValueError(
                f"Cannot unbind along dim {orig_dim} with shape {self.shape}."
            )
        shape = (s for i, s in enumerate(self.shape) if i != dim)
        unbound_vals = {key: val.unbind(dim) for key, val in self.items()}
        return tuple(
            self.__class__(
                {key: val[i] for key, val in unbound_vals.items()},
                shape=shape,
                device=self.device,
            )
            for i in range(self.shape[dim])
        )

    def lock_(self, recurse=False):
        """Locks the CompositeSpec and prevents modification of its content.

        This is only a first-level lock, unless specified otherwise through the
        ``recurse`` arg.

        Leaf specs can always be modified in place, but they cannot be replaced
        in their CompositeSpec parent.

        Examples:
            >>> shape = [3, 4, 5]
            >>> spec = CompositeSpec(
            ...         a=CompositeSpec(
            ...         b=CompositeSpec(shape=shape[:3], device="cpu"), shape=shape[:2]
            ...     ),
            ...     shape=shape[:1],
            ... )
            >>> spec["a"] = spec["a"].clone()
            >>> recurse = False
            >>> spec.lock_(recurse=recurse)
            >>> try:
            ...     spec["a"] = spec["a"].clone()
            ... except RuntimeError:
            ...     print("failed!")
            failed!
            >>> try:
            ...     spec["a", "b"] = spec["a", "b"].clone()
            ...     print("succeeded!")
            ... except RuntimeError:
            ...     print("failed!")
            succeeded!
            >>> recurse = True
            >>> spec.lock_(recurse=recurse)
            >>> try:
            ...     spec["a", "b"] = spec["a", "b"].clone()
            ...     print("succeeded!")
            ... except RuntimeError:
            ...     print("failed!")
            failed!

        """
        self._locked = True
        if recurse:
            for value in self.values():
                if isinstance(value, CompositeSpec):
                    value.lock_(recurse)
        return self

    def unlock_(self, recurse=False):
        """Unlocks the CompositeSpec and allows modification of its content.

        This is only a first-level lock modification, unless specified
        otherwise through the ``recurse`` arg.

        """
        self._locked = False
        if recurse:
            for value in self.values():
                if isinstance(value, CompositeSpec):
                    value.unlock_(recurse)
        return self

    @property
    def locked(self):
        return self._locked


class LazyStackedCompositeSpec(_LazyStackedMixin[CompositeSpec], CompositeSpec):
    """A lazy representation of a stack of composite specs.

    Stacks composite specs together along one dimension.
    When random samples are drawn, a LazyStackedTensorDict is returned.

    Indexing is allowed but only along the stack dimension.

    This class is aimed to be used in multi-task and multi-agent settings, where
    heterogeneous specs may occur (same semantic but different shape).

    """

    def update(self, dict) -> None:
        for key, item in dict.items():
            if key in self.keys() and isinstance(
                item, (Dict, CompositeSpec, LazyStackedCompositeSpec)
            ):
                for spec, sub_item in zip(self._specs, item.unbind(self.dim)):
                    spec[key].update(sub_item)
                continue
            self[key] = item
        return self

    def __eq__(self, other):
        if not isinstance(other, LazyStackedCompositeSpec):
            return False
        if len(self._specs) != len(other._specs):
            return False
        if self.stack_dim != other.stack_dim:
            return False
        for _spec1, _spec2 in zip(self._specs, other._specs):
            if _spec1 != _spec2:
                return False
        return True

    def to_numpy(self, val: TensorDict, safe: bool = None) -> dict:
        if safe is None:
            safe = _CHECK_SPEC_ENCODE
        if safe:
            if val.shape[self.dim] != len(self._specs):
                raise ValueError(
                    "Size of LazyStackedCompositeSpec and val differ along the "
                    "stacking dimension"
                )
            for spec, v in zip(self._specs, torch.unbind(val, dim=self.dim)):
                spec.assert_is_in(v)
        return {key: self[key].to_numpy(val) for key, val in val.items()}

    def __len__(self):
        return self.shape[0]

    def values(
        self,
        include_nested: bool = False,
        leaves_only: bool = False,
    ):
        for key in self.keys(include_nested=include_nested, leaves_only=leaves_only):
            yield self[key]

    def items(
        self,
        include_nested: bool = False,
        leaves_only: bool = False,
    ):
        for key in self.keys(include_nested=include_nested, leaves_only=leaves_only):
            yield key, self[key]

    def keys(
        self,
        include_nested: bool = False,
        leaves_only: bool = False,
    ) -> KeysView:
        keys = self._specs[0].keys(
            include_nested=include_nested, leaves_only=leaves_only
        )
        keys = set(keys)
        for spec in self._specs[1:]:
            keys = keys.intersection(spec.keys(include_nested, leaves_only))
        return sorted(keys, key=str)

    def project(self, val: TensorDictBase) -> TensorDictBase:
        vals = []
        for spec, subval in zip(self._specs, val.unbind(self.dim)):
            if not spec.is_in(subval):
                vals.append(spec.project(subval))
            else:
                vals.append(subval)
        res = torch.stack(vals, dim=self.dim)
        if not isinstance(val, LazyStackedTensorDict):
            res = res.to_tensordict()
        return res

    def type_check(
        self,
        value: Union[torch.Tensor, TensorDictBase],
        selected_keys: Union[NestedKey, Optional[Sequence[NestedKey]]] = None,
    ):
        if selected_keys is None:
            if isinstance(value, torch.Tensor):
                raise ValueError(
                    "value must be of type TensorDictBase when key is None"
                )
            for spec, subvalue in zip(self._specs, value.unbind(self.dim)):
                spec.type_check(subvalue)
        else:
            if isinstance(value, torch.Tensor) and isinstance(selected_keys, str):
                value = {selected_keys: value}
                selected_keys = [selected_keys]
            for _key in self.keys():
                if self[_key] is not None and _key in selected_keys:
                    self[_key].type_check(value[_key], _key)

    def __repr__(self) -> str:
        sub_str = ",\n".join(
            [indent(f"{k}: {repr(item)}", 4 * " ") for k, item in self.items()]
        )
        sub_str = indent(f"fields={{\n{', '.join([sub_str])}}}", 4 * " ")
        exclusive_key_str = self.repr_exclusive_keys()
        device_str = indent(f"device={self._specs[0].device}", 4 * " ")
        shape_str = indent(f"shape={self.shape}", 4 * " ")
        stack_dim = indent(f"stack_dim={self.dim}", 4 * " ")
        string = ",\n".join(
            [sub_str, exclusive_key_str, device_str, shape_str, stack_dim]
        )
        return f"LazyStackedCompositeSpec(\n{string})"

    def repr_exclusive_keys(self):
        keys = set(self.keys())
        exclusive_keys = [
            ",\n".join(
                [
                    indent(f"{k}: {repr(spec[k])}", 4 * " ")
                    for k in spec.keys()
                    if k not in keys
                ]
            )
            for spec in self._specs
        ]
        exclusive_key_str = ",\n".join(
            [
                indent(f"{i} ->\n{line}", 4 * " ")
                for i, line in enumerate(exclusive_keys)
                if line != ""
            ]
        )

        return indent(f"exclusive_fields={{\n{exclusive_key_str}}}", 4 * " ")

    def is_in(self, val) -> bool:
        for spec, subval in zip(self._specs, val.unbind(self.dim)):
            if not spec.is_in(subval):
                return False
        return True

    def __delitem__(self, key: NestedKey):
        """Deletes a key from the stacked composite spec.

        This method will be executed if the key is present in at least one of the stacked specs,
        otherwise it will raise an error.

        Args:
            key (NestedKey): the key to delete.
        """
        at_least_one_deletion = False
        for spec in self._specs:
            try:
                del spec[key]
                at_least_one_deletion = True
            except KeyError:
                continue
        if not at_least_one_deletion:
            raise KeyError(
                f"Key {key} must be present in at least one of the stacked specs"
            )
        return self

    def __iter__(self):
        for k in self.keys():
            yield self[k]

    def __setitem__(self, key: NestedKey, value):
        key = unravel_key(key)
        is_key = isinstance(key, str) or (
            isinstance(key, tuple) and all(isinstance(_item, str) for _item in key)
        )
        if is_key:
            self.set(key, value)
        else:
            raise ValueError(
                f"{self.__class__} expects str or tuple of str as key to set values "
            )

    @property
    def device(self) -> DEVICE_TYPING:
        return self._specs[0].device

    @property
    def ndim(self):
        return self.ndimension()

    def ndimension(self):
        return len(self.shape)

    def set(self, name, spec):
        for sub_spec, sub_item in zip(self._specs, spec.unbind(self.dim)):
            sub_spec[name] = sub_item

    @property
    def shape(self):
        shape = list(self._specs[0].shape)
        dim = self.dim
        if dim < 0:
            dim = len(shape) + dim + 1
        shape.insert(dim, len(self._specs))
        return torch.Size(shape)

    def expand(self, *shape):
        if len(shape) == 1 and not isinstance(shape[0], (int,)):
            return self.expand(*shape[0])
        expand_shape = shape[: -len(self.shape)]
        existing_shape = self.shape
        shape_check = shape[-len(self.shape) :]
        for _i, (size1, size2) in enumerate(zip(existing_shape, shape_check)):
            if size1 != size2 and size1 != 1:
                raise RuntimeError(
                    f"Expanding a non-singletom dimension: existing shape={size1} vs expand={size2}"
                )
            elif size1 != size2 and size1 == 1 and _i == self.dim:
                # if we're expanding along the stack dim we just need to clone the existing spec
                return torch.stack(
                    [self._specs[0].clone() for _ in range(size2)], self.dim
                ).expand(*shape)
        if _i != len(self.shape) - 1:
            raise RuntimeError(
                f"Trying to expand non-congruent shapes: received {shape} when the shape is {self.shape}."
            )
        # remove the stack dim from the expanded shape, which we know to match
        unstack_shape = list(expand_shape) + [
            s for i, s in enumerate(shape_check) if i != self.dim
        ]
        return torch.stack(
            [spec.expand(unstack_shape) for spec in self._specs],
            self.dim + len(expand_shape),
        )

    def empty(self):
        return torch.stack([spec.empty() for spec in self._specs], dim=self.stack_dim)

    def encode(
        self, vals: Dict[str, Any], ignore_device: bool = False
    ) -> Dict[str, torch.Tensor]:
        raise NOT_IMPLEMENTED_ERROR


# for SPEC_CLASS in [BinaryDiscreteTensorSpec, BoundedTensorSpec, DiscreteTensorSpec, MultiDiscreteTensorSpec, MultiOneHotDiscreteTensorSpec, OneHotDiscreteTensorSpec, UnboundedContinuousTensorSpec, UnboundedDiscreteTensorSpec]:
@TensorSpec.implements_for_spec(torch.stack)
def _stack_specs(list_of_spec, dim, out=None):
    if out is not None:
        raise NotImplementedError(
            "In-place spec modification is not a feature of torchrl, hence "
            "torch.stack(list_of_specs, dim, out=spec) is not implemented."
        )
    if not len(list_of_spec):
        raise ValueError("Cannot stack an empty list of specs.")
    spec0 = list_of_spec[0]
    if isinstance(spec0, TensorSpec):
        device = spec0.device

        all_equal = True
        for spec in list_of_spec[1:]:
            if not isinstance(spec, spec0.__class__):
                raise RuntimeError(
                    "Stacking specs cannot occur: Found more than one type of specs in the list."
                )
            if device != spec.device:
                raise RuntimeError(f"Devices differ, got {device} and {spec.device}")
            if spec.dtype != spec0.dtype:
                raise RuntimeError(f"Dtypes differ, got {spec0.dtype} and {spec.dtype}")
            if spec.ndim != spec0.ndim:
                raise RuntimeError(f"Ndims differ, got {spec0.ndim} and {spec.ndim}")
            all_equal = all_equal and spec == spec0
        if all_equal:
            shape = list(spec0.shape)
            if dim < 0:
                dim += len(shape) + 1
            shape.insert(dim, len(list_of_spec))
            return spec0.clone().unsqueeze(dim).expand(shape)
        return LazyStackedTensorSpec(*list_of_spec, dim=dim)
    else:
        raise NotImplementedError


@CompositeSpec.implements_for_spec(torch.stack)
def _stack_composite_specs(list_of_spec, dim, out=None):
    if out is not None:
        raise NotImplementedError(
            "In-place spec modification is not a feature of torchrl, hence "
            "torch.stack(list_of_specs, dim, out=spec) is not implemented."
        )
    if not len(list_of_spec):
        raise ValueError("Cannot stack an empty list of specs.")
    spec0 = list_of_spec[0]
    if isinstance(spec0, CompositeSpec):
        device = spec0.device
        all_equal = True
        for spec in list_of_spec[1:]:
            if not isinstance(spec, CompositeSpec):
                raise RuntimeError(
                    "Stacking specs cannot occur: Found more than one type of spec in "
                    "the list."
                )
            if device != spec.device:
                raise RuntimeError(f"Devices differ, got {device} and {spec.device}")
            if spec.shape != spec0.shape:
                raise RuntimeError(f"Shapes differ, got {spec.shape} and {spec0.shape}")
            all_equal = all_equal and spec == spec0
        if all_equal:
            shape = list(spec0.shape)
            if dim < 0:
                dim += len(shape) + 1
            shape.insert(dim, len(list_of_spec))
            return spec0.clone().unsqueeze(dim).expand(shape)
        return LazyStackedCompositeSpec(*list_of_spec, dim=dim)
    else:
        raise NotImplementedError


@TensorSpec.implements_for_spec(torch.squeeze)
def _squeeze_spec(spec: TensorSpec, *args, **kwargs) -> TensorSpec:
    return spec.squeeze(*args, **kwargs)


@CompositeSpec.implements_for_spec(torch.squeeze)
def _squeeze_composite_spec(spec: CompositeSpec, *args, **kwargs) -> CompositeSpec:
    return spec.squeeze(*args, **kwargs)


@TensorSpec.implements_for_spec(torch.unsqueeze)
def _unsqueeze_spec(spec: TensorSpec, *args, **kwargs) -> TensorSpec:
    return spec.unsqueeze(*args, **kwargs)


@CompositeSpec.implements_for_spec(torch.unsqueeze)
def _unsqueeze_composite_spec(spec: CompositeSpec, *args, **kwargs) -> CompositeSpec:
    return spec.unsqueeze(*args, **kwargs)


def _keys_to_empty_composite_spec(keys):
    """Given a list of keys, creates a CompositeSpec tree where each leaf is assigned a None value."""
    if not len(keys):
        return
    c = CompositeSpec()
    for key in keys:
        if isinstance(key, str):
            c[key] = None
        elif key[0] in c.keys():
            if c[key[0]] is None:
                # if the value is None we just replace it
                c[key[0]] = _keys_to_empty_composite_spec([key[1:]])
            elif isinstance(c[key[0]], CompositeSpec):
                # if the value is Composite, we update it
                out = _keys_to_empty_composite_spec([key[1:]])
                if out is not None:
                    c[key[0]].update(out)
            else:
                raise RuntimeError("Conflicting keys")
        else:
            c[key[0]] = _keys_to_empty_composite_spec(key[1:])
    return c


def _squeezed_shape(shape: torch.Size, dim: int | None) -> torch.Size | None:
    if dim is None:
        if len(shape) == 1 or shape.count(1) == 0:
            return None
        new_shape = torch.Size([s for s in shape if s != 1])
    else:
        if dim < 0:
            dim += len(shape)

        if shape[dim] != 1:
            return None

        new_shape = torch.Size([s for i, s in enumerate(shape) if i != dim])
    return new_shape


def _unsqueezed_shape(shape: torch.Size, dim: int) -> torch.Size:
    n = len(shape)
    if dim < -(n + 1) or dim > n:
        raise ValueError(
            f"Dimension out of range, expected value in the range [{-(n+1)}, {n}], but "
            f"got {dim}"
        )
    if dim < 0:
        dim += n + 1

    new_shape = list(shape)
    new_shape.insert(dim, 1)
    return torch.Size(new_shape)


class _CompositeSpecKeysView:
    """Wrapper class that enables richer behaviour of `key in tensordict.keys()`."""

    def __init__(
        self,
        composite: CompositeSpec,
        include_nested,
        leaves_only,
    ):
        self.composite = composite
        self.leaves_only = leaves_only
        self.include_nested = include_nested

    def __iter__(self):
        for key, item in self.composite.items():
            if self.include_nested and isinstance(item, CompositeSpec):
                for subkey in item.keys(
                    include_nested=True, leaves_only=self.leaves_only
                ):
                    if not isinstance(subkey, tuple):
                        subkey = (subkey,)
                    yield (key, *subkey)
                if not self.leaves_only:
                    yield key
            elif not isinstance(item, CompositeSpec) or not self.leaves_only:
                yield key

    def __len__(self):
        i = 0
        for _ in self:
            i += 1
        return i

    def __repr__(self):
        return f"_CompositeSpecKeysView(keys={list(self)})"

    def __contains__(self, item):
        item = unravel_key(item)
        if len(item) == 1:
            item = item[0]
        for key in self.__iter__():
            if key == item:
                return True
        else:
            return False
