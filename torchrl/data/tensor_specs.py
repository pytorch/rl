# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import abc
import enum
import functools
import gc
import math
import warnings
import weakref
from collections.abc import Iterable
from copy import deepcopy
from dataclasses import dataclass, field
from functools import wraps
from textwrap import indent
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    List,
    overload,
    Sequence,
    Tuple,
    TypeVar,
    Union,
)

import numpy as np

import tensordict
import torch
from tensordict import (
    is_tensor_collection,
    LazyStackedTensorDict,
    NonTensorData,
    NonTensorStack,
    set_capture_non_tensor_stack,
    TensorDict,
    TensorDictBase,
    unravel_key,
)
from tensordict.base import NO_DEFAULT
from tensordict.utils import (
    _getitem_batch_size,
    expand_as_right,
    is_non_tensor,
    NestedKey,
)
from torchrl._utils import _make_ordinal_device, get_binary_env_var, implement_for

try:
    from torch.compiler import is_compiling
except ImportError:
    from torch._dynamo import is_compiling

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

DEVICE_ERR_MSG = "device of empty Composite is not defined."
NOT_IMPLEMENTED_ERROR = NotImplementedError(
    "method is not currently implemented."
    " If you are interested in this feature please submit"
    " an issue at https://github.com/pytorch/rl/issues"
)


def _size(list_of_ints):
    # ensures that np int64 elements don't slip through Size
    # see https://github.com/pytorch/pytorch/issues/127194
    return torch.Size([int(i) for i in list_of_ints])


# Akin to TD's NO_DEFAULT but won't raise a KeyError when found in a TD or used as default
class _NoDefault(enum.IntEnum):
    ZERO = 0
    ONE = 1


NO_DEFAULT_RL = _NoDefault.ONE


def _default_dtype_and_device(
    dtype: None | torch.dtype,
    device: None | str | int | torch.device,
    allow_none_device: bool = False,
) -> tuple[torch.dtype, torch.device | None]:
    if dtype is None:
        dtype = torch.get_default_dtype()
    if device is not None:
        device = _make_ordinal_device(torch.device(device))
    elif not allow_none_device:
        device = torch.zeros(()).device
    return dtype, device


def _validate_idx(shape: list[int], idx: int, axis: int = 0):
    """Raise an IndexError if idx is out of bounds for shape[axis].

    Args:
        shape (list[int]): Input shape
        idx (int): Index, may be negative
        axis (int): Shape axis to check
    """
    if shape[axis] >= 0 and (idx >= shape[axis] or idx < 0 and -idx > shape[axis]):
        raise IndexError(
            f"index {idx} is out of bounds for axis {axis} with size {shape[axis]}"
        )


def _validate_iterable(
    idx: Iterable[Any], expected_type: type, iterable_classname: str
):
    """Raise an IndexError if the iterable contains a type different from the expected type or Iterable.

    Args:
        idx (Iterable[Any]): Iterable, may contain nested iterables
        expected_type (Type): Required item type in the Iterable (e.g. int)
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


def _slice_indexing(shape: list[int], idx: slice) -> list[int]:
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
    shape: list[int] | torch.Size | tuple[int], idx: SHAPE_INDEX_TYPING
) -> list[int]:
    """Given an input shape and an index, returns the size of the resulting indexed spec.

    This function includes indexing checks and may raise IndexErrors.

    Args:
        shape (list[int], torch.Size, Tuple[int): Input shape
        idx (SHAPE_INDEX_TYPING): Index
    Returns:
        Shape of the resulting spec
    Examples:
        >>> idx = (2, ..., None)
        >>> Categorical(2, shape=(3, 4))[idx].shape
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

    def to(self, dest: torch.dtype | DEVICE_TYPING) -> ContinuousBox:
        raise NotImplementedError

    def __repr__(self):
        return f"{self.__class__.__name__}()"

    def clone(self) -> CategoricalBox:
        return deepcopy(self)


@dataclass(repr=False)
class ContinuousBox(Box):
    """A continuous box of values, in between a minimum (self.low) and a maximum (self.high)."""

    _low: torch.Tensor
    _high: torch.Tensor
    device: torch.device | None = None

    # We store the tensors on CPU to avoid overloading CUDA with tensors that are rarely used.
    @property
    def low(self):
        low = self._low
        if self.device is not None and low.device != self.device:
            low = low.to(self.device)
        return low

    @property
    def high(self):
        high = self._high
        if self.device is not None and high.device != self.device:
            high = high.to(self.device)
        return high

    def unbind(self, dim: int = 0):
        return tuple(
            type(self)(low, high, self.device)
            for (low, high) in zip(self.low.unbind(dim), self.high.unbind(dim))
        )

    @low.setter
    def low(self, value):
        self.device = value.device
        self._low = value

    @high.setter
    def high(self, value):
        self.device = value.device
        self._high = value

    def __post_init__(self):
        self.low = self.low.clone()
        self.high = self.high.clone()

    def __iter__(self):
        yield self.low
        yield self.high

    def to(self, dest: torch.dtype | DEVICE_TYPING) -> ContinuousBox:
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
        if other is None:

            minval, maxval = _minmax_dtype(self.low.dtype)
            minval = torch.as_tensor(minval).to(self.low.device, self.low.dtype)
            maxval = torch.as_tensor(maxval).to(self.low.device, self.low.dtype)
            if (
                torch.isclose(self.low, minval).all()
                and torch.isclose(self.high, maxval).all()
            ):
                return True
            if (
                not torch.isfinite(self.low).any()
                and not torch.isfinite(self.high).any()
            ):
                return True
            return False
        return (
            type(self) == type(other)
            and self.low.dtype == other.low.dtype
            and self.high.dtype == other.high.dtype
            and self.device == other.device
            and torch.isclose(self.low, other.low).all()
            and torch.isclose(self.high, other.high).all()
        )


@dataclass(repr=False, frozen=True)
class CategoricalBox(Box):
    """A box of discrete, categorical values."""

    n: int
    register = invertible_dict()

    def __post_init__(self):
        # n could be a numpy array or a tensor, making compile go a bit crazy
        # We want to make sure we're working with a regular integer
        self.__dict__["n"] = int(self.n)

    def to(self, dest: torch.dtype | DEVICE_TYPING) -> CategoricalBox:
        return deepcopy(self)

    def __repr__(self):
        return f"{self.__class__.__name__}(n={self.n})"


class DiscreteBox(CategoricalBox):
    """Deprecated version of :class:`CategoricalBox`."""

    ...


@dataclass(repr=False)
class BoxList(Box):
    """A box of discrete values."""

    boxes: list

    def to(self, dest: torch.dtype | DEVICE_TYPING) -> BoxList:
        return BoxList([box.to(dest) for box in self.boxes])

    def __iter__(self):
        yield from self.boxes

    def __repr__(self):
        return f"{self.__class__.__name__}(boxes={self.boxes})"

    def __len__(self):
        return len(self.boxes)

    @staticmethod
    def from_nvec(nvec: torch.Tensor):
        if nvec.ndim == 0:
            return CategoricalBox(nvec.item())
        else:
            return BoxList([BoxList.from_nvec(n) for n in nvec.unbind(-1)])


@dataclass(repr=False, frozen=True)
class BinaryBox(Box):
    """A box of n binary values."""

    n: int

    def to(self, dest: torch.dtype | DEVICE_TYPING) -> ContinuousBox:
        return deepcopy(self)

    def __repr__(self):
        return f"{self.__class__.__name__}(n={self.n})"


@dataclass(repr=False)
class TensorSpec(metaclass=abc.ABCMeta):
    """Parent class of the tensor meta-data containers.

    TorchRL's TensorSpec are used to present what input/output is to be expected for a specific class,
    or sometimes to simulate simple behaviors by generating random data within a defined space.

    TensorSpecs are primarily used in environments to specify their input/output structure without needing to
    execute the environment (or starting it). They can also be used to instantiate shared buffers to pass
    data from worker to worker.

    TensorSpecs are dataclasses that always share the following fields: `shape`, `space, `dtype` and `device`.

    As such, TensorSpecs possess some common behavior with :class:`~torch.Tensor` and :class:`~tensordict.TensorDict`:
    they can be reshaped, indexed, squeezed, unsqueezed, moved to another device etc.

    Args:
        shape (torch.Size): size of the tensor. The shape includes the batch dimensions as well as the feature
            dimension. A negative shape (``-1``) means that the dimension has a variable number of elements.
        space (Box): Box instance describing what kind of values can be expected.
        device (torch.device): device of the tensor.
        dtype (torch.dtype): dtype of the tensor.

    .. note:: A spec can be constructed from a :class:`~tensordict.TensorDict` using the :func:`~torchrl.envs.utils.make_composite_from_td`
        function. This function makes a low-assumption educated guess on the specs that may correspond to the input
        tensordict and can help to build specs automatically without an in-depth knowledge of the `TensorSpec` API.

    """

    shape: torch.Size
    space: None | Box
    device: torch.device | None = None
    dtype: torch.dtype = torch.float
    domain: str = ""
    _encode_memo_dict: dict[Any, Callable[[Any], Any]] = field(
        default_factory=dict,
    )

    SPEC_HANDLED_FUNCTIONS = {}

    def memoize_encode(self, mode: bool = True) -> None:
        """Creates a cached sequence of callables for the `encode` method that speeds up its execution.

        This should only be used whenever the input type, shape etc. are expected to be consistent across calls
        for a given spec.

        Args:
            mode (bool, optional): Whether the cache should be used. Defaults to `True`.

        .. seealso:: the cache can be erased via :meth:`~torchrl.data.TensorSpec.erase_memoize_cache`.
        """
        warnings.warn(
            "memoized encoding is an experimental feature. Use at your own risks."
        )
        if mode:
            self.encode = self._encode_memo
        else:
            self.encode = self._encode_eager

    def erase_memoize_cache(self) -> None:
        """Clears the memoized cache for cached encode execution.

        .. seealso:: :meth:`~torchrl.data.TensorSpec.memoize_encode`.
        """
        self._encode_memo_dict.clear()

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_encode"] = {}
        return state

    @classmethod
    def implements_for_spec(cls, torch_function: Callable) -> Callable:
        """Register a torch function override for TensorSpec."""

        @wraps(torch_function)
        def decorator(func):
            cls.SPEC_HANDLED_FUNCTIONS[torch_function] = func
            return func

        return decorator

    @property
    def device(self) -> torch.device:
        """The device of the spec.

        Only :class:`Composite` specs can have a ``None`` device. All leaves must have a non-null device.
        """
        return self._device

    @device.setter
    def device(self, device: torch.device | None) -> None:
        self._device = _make_ordinal_device(device)

    def clear_device_(self) -> T:
        """A no-op for all leaf specs (which must have a device).

        For :class:`Composite` specs, this method will erase the device.
        """
        return self

    @abc.abstractmethod
    def cardinality(self) -> int:
        """The cardinality of the spec.

        This refers to the number of possible outcomes in a spec. It is assumed that the cardinality of a composite
        spec is the cartesian product of all possible outcomes.

        """
        ...

    def encode(
        self,
        val: np.ndarray | list | torch.Tensor | TensorDictBase,
        *,
        ignore_device: bool = False,
    ) -> torch.Tensor | TensorDictBase:
        """Encodes a value given the specified spec, and return the corresponding tensor.

        This method is to be used in environments that return a value (eg, a numpy array) that can be
        easily mapped to the TorchRL required domain.
        If the value is already a tensor, the spec will not change its value and return it as-is.

        Args:
            val (np.ndarray or torch.Tensor): value to be encoded as tensor.

        Keyword Args:
            ignore_device (bool, optional): if ``True``, the spec device will
                be ignored. This is used to group tensor casting within a call
                to ``TensorDict(..., device="cuda")`` which is faster.

        Returns:
            torch.Tensor matching the required tensor specs.

        """
        raise NotImplementedError(
            "This is a placeholder that needs to be set during construction"
        )

    def _encode_eager(
        self,
        val: np.ndarray | list | torch.Tensor | TensorDictBase,
        *,
        ignore_device: bool = False,
    ) -> torch.Tensor | TensorDictBase:
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
                val = torch.as_tensor(val, device=self.device, dtype=self.dtype)
            else:
                val = torch.as_tensor(val, dtype=self.dtype)
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

    def _encode_memo(
        self,
        val: np.ndarray | list | torch.Tensor | TensorDictBase,
        *,
        ignore_device: bool = False,
    ) -> torch.Tensor | TensorDictBase:
        funcs = self._encode_memo_dict.get(ignore_device)
        if funcs is not None:
            return funcs(val)

        funcs = []
        val_orig = val
        if not isinstance(val, torch.Tensor):
            if isinstance(val, list):
                if len(val) == 1:
                    # gym used to return lists of images since 0.26.0
                    # We convert these lists in np.array or take the first element
                    # if there is just one.
                    # See https://github.com/pytorch/rl/pull/403/commits/73d77d033152c61d96126ccd10a2817fecd285a1
                    funcs.append(lambda val: val[0])
                else:
                    funcs.append(lambda val: np.array(val))
            val = _reduce_funcs(funcs)(val_orig)
            if isinstance(val, np.ndarray) and not all(
                stride > 0 for stride in val.strides
            ):
                funcs.append(lambda val: val.copy())
            val = _reduce_funcs(funcs)(val_orig)
            if not ignore_device:
                funcs.append(
                    lambda val: torch.as_tensor(
                        val, device=self.device, dtype=self.dtype
                    )
                )
            else:
                funcs.append(lambda val: torch.as_tensor(val, dtype=self.dtype))
            val = _reduce_funcs(funcs)(val_orig)
        if val.shape != self.shape:
            # if val.shape[-len(self.shape) :] != self.shape:
            # option 1: add a singleton dim at the end
            if val.shape == self.shape and self.shape[-1] == 1:
                funcs.append(lambda val: val.unsqueeze(-1))
            else:

                def reshape(val):
                    try:
                        return val.reshape(self.shape)
                    except Exception as err:
                        raise RuntimeError(
                            f"Shape mismatch: the value has shape {val.shape} which "
                            f"is incompatible with the spec shape {self.shape}."
                        ) from err

                funcs.append(reshape)
            val = _reduce_funcs(funcs)(val_orig)
        if _CHECK_SPEC_ENCODE:

            def check(val):
                self.assert_is_in(val)
                return val

            funcs.append(check)
        if len(funcs) == 0:
            self._encode_memo_dict[ignore_device] = lambda x: x
        elif len(funcs) == 1:
            self._encode_memo_dict[ignore_device] = funcs[0]
        else:
            self._encode_memo_dict[ignore_device] = _reduce_funcs(funcs)
        return self._encode_memo_dict[ignore_device](val_orig)

    @abc.abstractmethod
    def __eq__(self, other: Any) -> bool:
        # Implement minimal version if super() is called
        return type(self) is type(other)

    def __ne__(self, other):
        return not (self == other)

    def __setattr__(self, key, value):
        if key == "shape":
            value = _size(value)
        super().__setattr__(key, value)

    def to_numpy(
        self, val: torch.Tensor | TensorDictBase, safe: bool = None
    ) -> np.ndarray | dict:
        """Returns the ``np.ndarray`` correspondent of an input tensor.

        This is intended to be the inverse operation of :meth:`.encode`.

        Args:
            val (torch.Tensor): tensor to be transformed_in to numpy.
            safe (bool): boolean value indicating whether a check should be
                performed on the value against the domain of the spec.
                Defaults to the value of the ``CHECK_SPEC_ENCODE`` environment variable.

        Returns:
            a np.ndarray.

        """
        if safe is None:
            safe = _CHECK_SPEC_ENCODE
        if safe:
            self.assert_is_in(val)
        return val.detach().cpu().numpy()

    @property
    def ndim(self) -> int:
        """Number of dimensions of the spec shape.

        Shortcut for ``len(spec.shape)``.

        """
        return self.ndimension()

    def ndimension(self) -> int:
        """Number of dimensions of the spec shape.

        Shortcut for ``len(spec.shape)``.

        """
        return len(self.shape)

    @property
    def _safe_shape(self) -> torch.Size:
        """Returns a shape where all heterogeneous values are replaced by one (to be expandable)."""
        return _size([int(v) if v >= 0 else 1 for v in self.shape])

    @abc.abstractmethod
    def index(
        self, index: INDEX_TYPING, tensor_to_index: torch.Tensor | TensorDictBase
    ) -> torch.Tensor | TensorDictBase:
        """Indexes the input tensor.

        This method is to be used with specs that encode one or more categorical variables (e.g.,
        :class:`~torchrl.data.OneHot` or :class:`~torchrl.data.Categorical`), such that indexing of a tensor
        with a sample can be done without caring about the actual representation of the index.

        Args:
            index (int, torch.Tensor, slice or list): index of the tensor
            tensor_to_index: tensor to be indexed

        Returns:
            indexed tensor

        Exanples:
            >>> from torchrl.data import OneHot
            >>> import torch
            >>>
            >>> one_hot = OneHot(n=100)
            >>> categ = one_hot.to_categorical_spec()
            >>> idx_one_hot = torch.zeros((100,), dtype=torch.bool)
            >>> idx_one_hot[50] = 1
            >>> print(one_hot.index(idx_one_hot, torch.arange(100)))
            tensor(50)
            >>> idx_categ = one_hot.to_categorical(idx_one_hot)
            >>> print(categ.index(idx_categ, torch.arange(100)))
            tensor(50)

        """
        ...

    @overload
    def expand(self, shape: torch.Size):
        ...

    @abc.abstractmethod
    def expand(self, *shape: int) -> T:
        """Returns a new Spec with the expanded shape.

        Args:
            *shape (tuple or iterable of int): the new shape of the Spec.
                Must be broadcastable with the current shape:
                its length must be at least as long as the current shape length,
                and its last values must be compliant too; ie they can only differ
                from it if the current dimension is a singleton.

        """
        ...

    def squeeze(self, dim: int | None = None) -> T:
        """Returns a new Spec with all the dimensions of size ``1`` removed.

        When ``dim`` is given, a squeeze operation is done only in that dimension.

        Args:
            dim (int or None): the dimension to apply the squeeze operation to

        """
        shape = _squeezed_shape(self.shape, dim)
        if shape is None:
            return self
        return self.__class__(shape=shape, device=self.device, dtype=self.dtype)

    def unsqueeze(self, dim: int) -> T:
        """Returns a new Spec with one more singleton dimension (at the position indicated by ``dim``).

        Args:
            dim (int or None): the dimension to apply the unsqueeze operation to.

        """
        shape = _unsqueezed_shape(self.shape, dim)
        return self.__class__(shape=shape, device=self.device, dtype=self.dtype)

    def make_neg_dim(self, dim: int) -> T:
        """Converts a specific dimension to ``-1``."""
        if dim < 0:
            dim = self.ndim + dim
        if dim < 0 or dim > self.ndim - 1:
            raise ValueError(f"dim={dim} is out of bound for ndim={self.ndim}")
        self.shape = _size([s if i != dim else -1 for i, s in enumerate(self.shape)])

    @overload
    def reshape(self, shape) -> T:
        ...

    def reshape(self, *shape) -> T:
        """Reshapes a ``TensorSpec``.

        Check :func:`~torch.reshape` for more information on this method.

        """
        if len(shape) == 1 and not isinstance(shape[0], int):
            return self.reshape(*shape[0])
        return self._reshape(shape)

    view = reshape

    @abc.abstractmethod
    def _reshape(self, shape: torch.Size) -> T:
        ...

    def unflatten(self, dim: int, sizes: tuple[int]) -> T:
        """Unflattens a ``TensorSpec``.

        Check :func:`~torch.unflatten` for more information on this method.

        """
        return self._unflatten(dim, sizes)

    def _unflatten(self, dim: int, sizes: tuple[int]) -> T:
        shape = torch.zeros(self.shape, device="meta").unflatten(dim, sizes).shape
        return self._reshape(shape)

    def flatten(self, start_dim: int, end_dim: int) -> T:
        """Flattens a ``TensorSpec``.

        Check :func:`~torch.flatten` for more information on this method.

        """
        return self._flatten(start_dim, end_dim)

    def _flatten(self, start_dim, end_dim):
        shape = torch.zeros(self.shape, device="meta").flatten(start_dim, end_dim).shape
        return self._reshape(shape)

    @abc.abstractmethod
    def _project(
        self, val: torch.Tensor | TensorDictBase
    ) -> torch.Tensor | TensorDictBase:
        raise NotImplementedError(type(self))

    @abc.abstractmethod
    def is_in(self, val: torch.Tensor | TensorDictBase) -> bool:
        """If the value ``val`` could have been generated by the ``TensorSpec``, returns ``True``, otherwise ``False``.

        More precisely, the ``is_in`` methods checks that the value ``val`` is within the limits defined by the ``space``
        attribute (the box), and that the ``dtype``, ``device``, ``shape`` potentially other metadata match those
        of the spec. If any of these checks fails, the ``is_in`` method will return ``False``.

        Args:
            val (torch.Tensor): value to be checked.

        Returns:
            boolean indicating if values belongs to the TensorSpec box.

        """
        ...

    def contains(self, item: torch.Tensor | TensorDictBase) -> bool:
        """If the value ``val`` could have been generated by the ``TensorSpec``, returns ``True``, otherwise ``False``.

        See :meth:`is_in` for more information.
        """
        return self.is_in(item)

    @abc.abstractmethod
    def enumerate(self, use_mask: bool = False) -> Any:
        """Returns all the samples that can be obtained from the TensorSpec.

        The samples will be stacked along the first dimension.

        This method is only implemented for discrete specs.

        Args:
            use_mask (bool, optional): If ``True`` and the spec has a mask,
                samples that are masked are excluded. Default is ``False``.
        """
        ...

    def project(
        self, val: torch.Tensor | TensorDictBase
    ) -> torch.Tensor | TensorDictBase:
        """If the input tensor is not in the TensorSpec box, it maps it back to it given some defined heuristic.

        Args:
            val (torch.Tensor): tensor to be mapped to the box.

        Returns:
            a torch.Tensor belonging to the TensorSpec box.

        """
        if is_compiling() or not self.is_in(val):
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

    def type_check(self, value: torch.Tensor, key: NestedKey = None) -> None:
        """Checks the input value ``dtype`` against the ``TensorSpec`` ``dtype`` and raises an exception if they don't match.

        Args:
            value (torch.Tensor): tensor whose dtype has to be checked.
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
    def rand(self, shape: torch.Size = None) -> torch.Tensor | TensorDictBase:
        """Returns a random tensor in the space defined by the spec.

        The sampling will be done uniformly over the space, unless the box is unbounded in which case normal values
        will be drawn.

        Args:
            shape (torch.Size): shape of the random tensor

        Returns:
            a random tensor sampled in the TensorSpec box.

        """
        ...

    def sample(self, shape: torch.Size = None) -> torch.Tensor | TensorDictBase:
        """Returns a random tensor in the space defined by the spec.

        See :meth:`rand` for details.
        """
        return self.rand(shape=shape)

    def zero(self, shape: torch.Size = None) -> torch.Tensor | TensorDictBase:
        """Returns a zero-filled tensor in the box.

        .. note:: Even though there is no guarantee that ``0`` belongs to the spec domain,
            this method will not raise an exception when this condition is violated.
            The primary use case of ``zero`` is to generate empty data buffers, not meaningful data.

        Args:
            shape (torch.Size): shape of the zero-tensor

        Returns:
            a zero-filled tensor sampled in the TensorSpec box.

        """
        if shape is None:
            shape = _size([])
        return torch.zeros(
            (*shape, *self._safe_shape), dtype=self.dtype, device=self.device
        )

    def zeros(self, shape: torch.Size = None) -> torch.Tensor | TensorDictBase:
        """Proxy to :meth:`zero`."""
        return self.zero(shape=shape)

    def one(self, shape: torch.Size = None) -> torch.Tensor | TensorDictBase:
        """Returns a one-filled tensor in the box.

        .. note:: Even though there is no guarantee that ``1`` belongs to the spec domain,
            this method will not raise an exception when this condition is violated.
            The primary use case of ``one`` is to generate empty data buffers, not meaningful data.

        Args:
            shape (torch.Size): shape of the one-tensor

        Returns:
            a one-filled tensor sampled in the TensorSpec box.

        """
        if self.dtype == torch.bool:
            return ~self.zero(shape=shape)
        return self.zero(shape) + 1

    def ones(self, shape: torch.Size = None) -> torch.Tensor | TensorDictBase:
        """Proxy to :meth:`one`."""
        return self.one(shape=shape)

    @abc.abstractmethod
    def to(self, dest: torch.dtype | DEVICE_TYPING) -> TensorSpec:
        """Casts a TensorSpec to a device or a dtype.

        Returns the same spec if no change is made.
        """
        ...

    def cpu(self):
        """Casts the TensorSpec to 'cpu' device."""
        return self.to("cpu")

    def cuda(self, device=None):
        """Casts the TensorSpec to 'cuda' device."""
        if device is None:
            return self.to("cuda")
        return self.to(f"cuda:{device}")

    @abc.abstractmethod
    def clone(self) -> TensorSpec:
        """Creates a copy of the TensorSpec."""
        ...

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
        args: tuple = (),
        kwargs: dict | None = None,
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

    def unbind(self, dim: int = 0):
        raise NotImplementedError


T = TypeVar("T")


class _LazyStackedMixin(Generic[T]):
    def __init__(self, *specs: tuple[T, ...], dim: int) -> None:
        self._specs = list(specs)
        self.dim = dim
        if self.dim < 0:
            self.dim = len(self.shape) + self.dim

    def clear_device_(self):
        """Clears the device of the Composite."""
        for spec in self._specs:
            spec.clear_device_()
        return self

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

    def zero(self, shape: torch.Size = None) -> TensorDictBase:
        if shape is not None:
            dim = self.dim + len(shape)
        else:
            dim = self.dim
        if dim != 0:
            raise RuntimeError(
                f"Cannot create a nested tensor with a stack dimension other than 0. Got dim={0}"
            )
        return torch.nested.nested_tensor([spec.zero(shape) for spec in self._specs])

    def one(self, shape: torch.Size = None) -> TensorDictBase:
        if shape is not None:
            dim = self.dim + len(shape)
        else:
            dim = self.dim
        if dim != 0:
            raise RuntimeError(
                f"Cannot create a nested tensor with a stack dimension other than 0. Got dim={0}"
            )
        return torch.nested.nested_tensor([spec.one(shape) for spec in self._specs])

    def rand(self, shape: torch.Size = None) -> TensorDictBase:
        if shape is not None:
            dim = self.dim + len(shape)
        else:
            dim = self.dim
        samples = [spec.rand(shape) for spec in self._specs]
        if dim != 0:
            raise RuntimeError(
                f"Cannot create a nested tensor with a stack dimension other than 0. Got self.dim={self.dim}."
            )
        return torch.nested.nested_tensor(samples)

    def to(self, dest: torch.dtype | DEVICE_TYPING) -> T:
        if dest is None:
            return self
        return torch.stack([spec.to(dest) for spec in self._specs], self.dim)

    def unbind(self, dim: int = 0):
        if dim < 0:
            dim = self.ndim + dim
        shape = self.shape
        if dim < 0 or dim > self.ndim - 1 or shape[dim] == -1:
            raise ValueError(
                f"Provided dim {dim} is not valid for unbinding shape {shape}"
            )
        if dim == self.stack_dim:
            return self._specs
        elif dim > self.dim:
            dim = dim - 1
            return type(self)(*[spec.unbind(dim) for spec in self._specs], dim=self.dim)
        else:
            return type(self)(
                *[spec.unbind(dim) for spec in self._specs], dim=self.dim - 1
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

    def make_neg_dim(self, dim: int):
        if dim < 0:
            dim = self.ndim + dim
        if dim < 0 or dim > self.ndim - 1:
            raise ValueError(f"dim={dim} is out of bound for ndim={self.ndim}")
        if dim == self.dim:
            raise ValueError("Cannot make dim=self.dim negative")
        if dim < self.dim:
            for spec in self._specs:
                spec.make_neg_dim(dim)
        else:
            for spec in self._specs:
                spec.make_neg_dim(dim - 1)

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


class Stacked(_LazyStackedMixin[TensorSpec], TensorSpec):
    """A lazy representation of a stack of tensor specs.

    Stacks tensor-specs together along one dimension.
    When random samples are drawn, a stack of samples is returned if possible.
    If not, an error is thrown.

    Indexing is allowed but only along the stack dimension.

    This class aims at being used in multi-tasks and multi-agent settings, where
    heterogeneous specs may occur (same semantic but different shape).

    """

    def _reshape(
        self,
        *args,
        **kwargs,
    ) -> Any:
        raise NotImplementedError(
            f"`reshape` is not implemented for {type(self).__name__} specs."
        )

    def cardinality(
        self,
        *args,
        **kwargs,
    ) -> Any:
        raise NotImplementedError(
            f"`cardinality` is not implemented for {type(self).__name__} specs."
        )

    def index(
        self, index: INDEX_TYPING, tensor_to_index: torch.Tensor | TensorDictBase
    ) -> torch.Tensor | TensorDictBase:
        raise NotImplementedError(
            f"`index` is not implemented for {type(self).__name__} specs."
        )

    def __eq__(self, other):
        if not isinstance(other, Stacked):
            return False
        if self.device != other.device:
            raise RuntimeError((self, other))
            return False
        if len(self._specs) != len(other._specs):
            return False
        for _spec1, _spec2 in zip(self._specs, other._specs):
            if _spec1 != _spec2:
                return False
        return True

    def enumerate(self, use_mask: bool = False) -> torch.Tensor | TensorDictBase:
        return torch.stack(
            [spec.enumerate(use_mask) for spec in self._specs], dim=self.stack_dim + 1
        )

    def __len__(self):
        return self.shape[0]

    def to_numpy(self, val: torch.Tensor, safe: bool = None) -> dict:
        if safe is None:
            safe = _CHECK_SPEC_ENCODE
        if safe:
            if val.shape[self.dim] != len(self._specs):
                raise ValueError(
                    "Size of Stacked and val differ along the stacking " "dimension"
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
        string = f"Stacked{self._specs[0].__class__.__name__}(\n    {sub_string})"
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
        return _size(shape)

    @shape.setter
    def shape(self, shape):
        if len(shape) != len(self.shape):
            raise RuntimeError(
                f"Cannot set shape of different length from self. shape={shape}, self.shape={self.shape}"
            )
        if shape[self.dim] != self.shape[self.dim]:
            raise RuntimeError(
                f"The shape attribute mismatches between the input {shape} and self.shape={self.shape}."
            )
        shape_strip = _size([s for i, s in enumerate(self.shape) if i != self.dim])
        for spec in self._specs:
            spec.shape = shape_strip

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

    def type_check(self, value: torch.Tensor, key: NestedKey | None = None) -> None:
        for (val, spec) in zip(value.unbind(self.dim), self._specs):
            spec.type_check(val)

    def is_in(self, value) -> bool:
        if self.dim == 0 and not hasattr(value, "unbind"):
            # We don't use unbind because value could be a tuple or a nested tensor
            return all(
                spec.contains(value) for (value, spec) in zip(value, self._specs)
            )
        return all(
            spec.contains(value)
            for (value, spec) in zip(value.unbind(self.dim), self._specs)
        )

    @property
    def space(self):
        raise NOT_IMPLEMENTED_ERROR

    def _project(self, val: TensorDictBase) -> TensorDictBase:
        raise NOT_IMPLEMENTED_ERROR

    def encode(
        self, val: np.ndarray | torch.Tensor, *, ignore_device=False
    ) -> torch.Tensor:
        if self.dim != 0 and not isinstance(val, tuple):
            val = val.unbind(self.dim)
        samples = [spec.encode(_val) for _val, spec in zip(val, self._specs)]
        if is_tensor_collection(samples[0]):
            return LazyStackedTensorDict.maybe_dense_stack(samples, dim=self.dim)
        if isinstance(samples[0], torch.Tensor):
            if any(t.is_nested for t in samples):
                raise RuntimeError("Cannot stack nested tensors together.")
            if len(samples) > 1 and not all(
                t.shape == samples[0].shape for t in samples[1:]
            ):
                return torch.nested.nested_tensor(samples)
            return torch.stack(samples, dim=self.dim)


@dataclass(repr=False)
class OneHot(TensorSpec):
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
            If provided, the last dimension must match ``n``.
        device (str, int or torch.device, optional): device of the tensors.
        dtype (str or torch.dtype, optional): dtype of the tensors.
        use_register (bool): experimental feature. If ``True``, every integer
            will be mapped onto a binary vector in the order in which they
            appear. This feature is designed for environment with no
            a-priori definition of the number of possible outcomes (e.g.
            discrete outcomes are sampled from an arbitrary set, whose
            elements will be mapped in a register to a series of unique
            one-hot binary vectors).
        mask (torch.Tensor or None): mask some of the possible outcomes when a
            sample is taken. See :meth:`update_mask` for more information.

    Examples:
        >>> from torchrl.data.tensor_specs import OneHot
        >>> spec = OneHot(5, shape=(2, 5))
        >>> spec.rand()
        tensor([[False,  True, False, False, False],
                [False,  True, False, False, False]])
        >>> mask = torch.tensor([
        ... [False, False, False, False, True],
        ... [False, False, False, False, True]
        ... ])
        >>> spec.update_mask(mask)
        >>> spec.rand()
        tensor([[False, False, False, False,  True],
                [False, False, False, False,  True]])

    """

    shape: torch.Size
    space: CategoricalBox
    device: torch.device | None = None
    dtype: torch.dtype = torch.float
    domain: str = ""
    _encode_memo_dict: dict[Any, Callable[[Any], Any]] = field(
        default_factory=dict,
    )

    def __init__(
        self,
        n: int,
        shape: torch.Size | None = None,
        device: DEVICE_TYPING | None = None,
        dtype: str | torch.dtype | None = torch.bool,
        use_register: bool = False,
        mask: torch.Tensor | None = None,
    ):
        dtype, device = _default_dtype_and_device(
            dtype, device, allow_none_device=False
        )
        self.use_register = use_register
        space = CategoricalBox(n)
        if shape is None:
            shape = _size((space.n,))
        else:
            shape = _size(shape)
            if not len(shape) or shape[-1] != space.n:
                raise ValueError(
                    f"The last value of the shape must match n for transform of type {self.__class__}. "
                    f"Got n={space.n} and shape={shape}."
                )
        super().__init__(
            shape=shape, space=space, device=device, dtype=dtype, domain="discrete"
        )
        self.update_mask(mask)
        self.encode = self._encode_eager

    @property
    def n(self):
        return self.space.n

    def cardinality(self) -> int:
        return self.n

    def update_mask(self, mask):
        """Sets a mask to prevent some of the possible outcomes when a sample is taken.

        The mask can also be set during initialization of the spec.

        Args:
            mask (torch.Tensor or None): boolean mask. If None, the mask is
                disabled. Otherwise, the shape of the mask must be expandable to
                the shape of the spec. ``False`` masks an outcome and ``True``
                leaves the outcome unmasked. If all the possible outcomes are
                masked, then an error is raised when a sample is taken.

        Examples:
            >>> mask = torch.tensor([True, False, False])
            >>> ts = OneHot(3, (2, 3,), dtype=torch.int64, mask=mask)
            >>> # All but one of the three possible outcomes are masked
            >>> ts.rand()
            tensor([[1, 0, 0],
                    [1, 0, 0]])
        """
        if mask is not None:
            try:
                mask = mask.expand(self._safe_shape)
            except RuntimeError as err:
                raise RuntimeError("Cannot expand mask to the desired shape.") from err
            if mask.dtype != torch.bool:
                raise ValueError("Only boolean masks are accepted.")
        self.mask = mask

    def to(self, dest: torch.dtype | DEVICE_TYPING) -> OneHot:
        if dest is None:
            return self
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

    def clone(self) -> OneHot:
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
        if any(s1 != s2 and s2 != 1 for s1, s2 in zip(shape[-self.ndim :], self.shape)):
            raise ValueError(
                f"The last {self.ndim} of the expanded shape {shape} must match the"
                f"shape of the {self.__class__.__name__} spec in expand()."
            )
        mask = self.mask
        if mask is not None:
            mask = mask.expand(_remove_neg_shapes(shape))
        return self.__class__(
            n=shape[-1],
            shape=shape,
            device=self.device,
            dtype=self.dtype,
            mask=mask,
        )

    def _reshape(self, shape):
        mask = self.mask
        if mask is not None:
            mask = mask.reshape(shape)
        return self.__class__(
            n=shape[-1],
            shape=shape,
            device=self.device,
            dtype=self.dtype,
            mask=mask,
        )

    def _unflatten(self, dim, sizes):
        mask = self.mask
        if mask is not None:
            mask = mask.unflatten(dim, sizes)
        shape = torch.zeros(self.shape, device="meta").unflatten(dim, sizes).shape
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

    def unbind(self, dim: int = 0):
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

    @implement_for("torch", None, "2.1", compilable=True)
    def rand(self, shape: torch.Size = None) -> torch.Tensor:
        if shape is None:
            shape = self.shape[:-1]
        else:
            shape = _size([*shape, *self.shape[:-1]])
        mask = self.mask
        n = int(self.space.n)
        if mask is None:
            m = torch.randint(n, shape, device=self.device)
        else:
            mask = mask.expand(_remove_neg_shapes(*shape, mask.shape[-1]))
            if mask.ndim > 2:
                mask_flat = torch.flatten(mask, 0, -2)
            else:
                mask_flat = mask
            shape_out = mask.shape[:-1]
            m = torch.multinomial(mask_flat.float(), 1).reshape(shape_out)
        out = torch.nn.functional.one_hot(m, n).to(self.dtype)
        # torch.zeros((*shape, self.space.n), device=self.device, dtype=self.dtype)
        # out.scatter_(-1, m, 1)
        return out

    @implement_for("torch", "2.1", compilable=True)
    def rand(self, shape: torch.Size = None) -> torch.Tensor:  # noqa: F811
        if shape is None:
            shape = self.shape[:-1]
        else:
            shape = _size([*shape, *self.shape[:-1]])
        mask = self.mask
        if mask is None:
            n = self.space.n
            m = torch.randint(n, shape, device=self.device)
        else:
            mask = mask.expand(_remove_neg_shapes(*shape, mask.shape[-1]))
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

    def _encode_eager(
        self,
        val: np.ndarray | torch.Tensor,
        space: CategoricalBox | None = None,
        *,
        ignore_device: bool = False,
    ) -> torch.Tensor:
        if not isinstance(val, torch.Tensor):
            if ignore_device:
                val = torch.as_tensor(val)
            else:
                val = torch.as_tensor(val, device=self.device)

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

    def _encode_memo(
        self,
        val: np.ndarray | torch.Tensor,
        space: CategoricalBox | None = None,
        *,
        ignore_device: bool = False,
    ) -> torch.Tensor:

        funcs = self._encode_memo_dict.get(ignore_device)
        if funcs is not None:
            return funcs(val)
        funcs = []
        val_orig = val
        if not isinstance(val, torch.Tensor):
            if ignore_device:
                funcs.append(torch.as_tensor)
            else:
                funcs.append(lambda val: torch.as_tensor(val, device=self.device))
        val = _reduce_funcs(funcs)(val_orig)
        if space is None:
            # TODO: make sure this is the case when the encoding is cached
            space = self.space

        if self.use_register:

            def from_register(val):
                if val not in space.register:
                    space.register[val] = len(space.register)
                return space.register[val]

            funcs.append(from_register)

        val = _reduce_funcs(funcs)(val_orig)

        def check_and_one_hot(val):
            if (val >= space.n).any():
                raise AssertionError("Value must be less than action space.")
            val = torch.nn.functional.one_hot(val.long(), space.n).to(self.dtype)
            return val

        funcs.append(check_and_one_hot)
        if len(funcs) == 0:
            self._encode_memo_dict[ignore_device] = lambda x: x
        elif len(funcs) == 1:
            self._encode_memo_dict[ignore_device] = funcs[0]
        else:
            self._encode_memo_dict[ignore_device] = functools.partial(
                functools.reduce, lambda x, f: f(x), funcs
            )
        return self._encode_memo_dict[ignore_device](val_orig)

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

    def enumerate(self, use_mask: bool = False) -> torch.Tensor:
        if use_mask:
            raise NotImplementedError
        return (
            torch.eye(self.n, dtype=self.dtype, device=self.device)
            .expand(*self.shape, self.n)
            .permute(-2, *range(self.ndimension() - 1), -1)
        )

    def index(self, index: INDEX_TYPING, tensor_to_index: torch.Tensor) -> torch.Tensor:
        if not isinstance(index, torch.Tensor):
            raise ValueError(
                f"Only tensors are allowed for indexing using "
                f"{self.__class__.__name__}.index(...)"
            )
        index = index.nonzero(as_tuple=True)[-1]
        index = index.expand((*tensor_to_index.shape[:-1], index.shape[-1]))
        return tensor_to_index.gather(-1, index)

    def __getitem__(self, idx: SHAPE_INDEX_TYPING):
        """Indexes the current TensorSpec based on the provided index.

        The last dimension of the spec corresponding to the variable domain cannot be indexed.
        """
        indexed_shape = _shape_indexing(self.shape[:-1], idx)
        return self.__class__(
            n=self.space.n,
            shape=_size(indexed_shape + [self.shape[-1]]),
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
        new_val = torch.scatter(torch.zeros_like(val[oob]), -1, new_val, 1)
        val = val.masked_scatter(expand_as_right(oob, val), new_val)
        return val

    def is_in(self, val: torch.Tensor) -> bool:
        if self.mask is None:
            shape = torch.broadcast_shapes(self._safe_shape, val.shape)
            shape_match = val.shape == shape
            if not shape_match:
                return False
            dtype_match = val.dtype == self.dtype
            if not dtype_match:
                return False
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

        Examples:
            >>> one_hot = OneHot(3, shape=(2, 3))
            >>> one_hot_sample = one_hot.rand()
            >>> one_hot_sample
            tensor([[False,  True, False],
                    [False,  True, False]])
            >>> categ_sample = one_hot.to_categorical(one_hot_sample)
            >>> categ_sample
            tensor([1, 1])
        """
        if safe is None:
            safe = _CHECK_SPEC_ENCODE
        if safe:
            self.assert_is_in(val)
        return val.long().argmax(-1)

    def to_categorical_spec(self) -> Categorical:
        """Converts the spec to the equivalent categorical spec.

        Examples:
            >>> one_hot = OneHot(3, shape=(2, 3))
            >>> one_hot.to_categorical_spec()
            Categorical(
                shape=torch.Size([2]),
                space=CategoricalBox(n=3),
                device=cpu,
                dtype=torch.int64,
                domain=discrete)

        """
        return Categorical(
            self.space.n,
            device=self.device,
            shape=self.shape[:-1],
            mask=self.mask,
        )

    def to_one_hot(self, val: torch.Tensor, safe: bool = None) -> torch.Tensor:
        """No-op for OneHot."""
        return val

    def to_one_hot_spec(self) -> OneHot:
        """No-op for OneHot."""
        return self


class _BoundedMeta(abc.ABCMeta):
    def __call__(cls, *args, **kwargs):
        instance = super().__call__(*args, **kwargs)
        if instance.domain == "continuous":
            instance.__class__ = BoundedContinuous
        else:
            instance.__class__ = BoundedDiscrete
        return instance


@dataclass(repr=False)
class Bounded(TensorSpec, metaclass=_BoundedMeta):
    """A bounded tensor spec.

    ``Bounded`` specs will never appear as such and always be subclassed as :class:`BoundedContinuous`
    or :class:`BoundedDiscrete` depending on their dtype (floating points dtypes will result in
    :class:`BoundedContinuous` instances, all others in :class:`BoundedDiscrete` instances).

    Args:
        low (np.ndarray, torch.Tensor or number): lower bound of the box.
        high (np.ndarray, torch.Tensor or number): upper bound of the box.
        shape (torch.Size): the shape of the ``Bounded`` spec. The shape must be specified.
            Inputs ``low``, ``high`` and ``shape`` must be broadcastable.
        device (str, int or torch.device, optional): device of the tensors.
        dtype (str or torch.dtype, optional): dtype of the tensors.
        domain (str): `"continuous"` or `"discrete"`. Can be used to override the automatic type assignment.

    Examples:
        >>> spec = Bounded(low=-1, high=1, shape=(), dtype=torch.float)
        >>> spec
        BoundedContinuous(
            shape=torch.Size([]),
            space=ContinuousBox(
                low=Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, contiguous=True),
                high=Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, contiguous=True)),
            device=cpu,
            dtype=torch.float32,
            domain=continuous)
        >>> spec = Bounded(low=-1, high=1, shape=(), dtype=torch.int)
        >>> spec
        BoundedDiscrete(
            shape=torch.Size([]),
            space=ContinuousBox(
                low=Tensor(shape=torch.Size([]), device=cpu, dtype=torch.int32, contiguous=True),
                high=Tensor(shape=torch.Size([]), device=cpu, dtype=torch.int32, contiguous=True)),
            device=cpu,
            dtype=torch.int32,
            domain=discrete)
        >>> spec.to(torch.float)
        BoundedContinuous(
            shape=torch.Size([]),
            space=ContinuousBox(
                low=Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, contiguous=True),
                high=Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, contiguous=True)),
            device=cpu,
            dtype=torch.float32,
            domain=continuous)
        >>> spec = Bounded(low=-1, high=1, shape=(), dtype=torch.int, domain="continuous")
        >>> spec
        BoundedContinuous(
            shape=torch.Size([]),
            space=ContinuousBox(
                low=Tensor(shape=torch.Size([]), device=cpu, dtype=torch.int32, contiguous=True),
                high=Tensor(shape=torch.Size([]), device=cpu, dtype=torch.int32, contiguous=True)),
            device=cpu,
            dtype=torch.int32,
            domain=continuous)

    """

    # SPEC_HANDLED_FUNCTIONS = {}
    CONFLICTING_KWARGS = (
        "The keyword arguments {} and {} conflict. Only one of these can be passed."
    )

    def __init__(
        self,
        low: float | torch.Tensor | np.ndarray = None,
        high: float | torch.Tensor | np.ndarray = None,
        shape: torch.Size | int | None = None,
        device: DEVICE_TYPING | None = None,
        dtype: torch.dtype | str | None = None,
        **kwargs,
    ):
        if "maximum" in kwargs:
            if high is not None:
                raise TypeError(self.CONFLICTING_KWARGS.format("high", "maximum"))
            high = kwargs.pop("maximum")
            warnings.warn(
                "Maximum is deprecated since v0.4.0, using high instead.",
                category=DeprecationWarning,
            )
        if "minimum" in kwargs:
            if low is not None:
                raise TypeError(self.CONFLICTING_KWARGS.format("low", "minimum"))
            low = kwargs.pop("minimum")
            warnings.warn(
                "Minimum is deprecated since v0.4.0, using low instead.",
                category=DeprecationWarning,
            )
        domain = kwargs.pop("domain", None)
        if len(kwargs):
            raise TypeError(f"Got unrecognised kwargs {tuple(kwargs.keys())}.")

        dtype, device = _default_dtype_and_device(
            dtype, device, allow_none_device=False
        )
        if dtype is None:
            dtype = torch.get_default_dtype()
        if domain is None:
            if dtype.is_floating_point:
                domain = "continuous"
            else:
                domain = "discrete"

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
            "Bounded requires the shape to be explicitly (via "
            "the shape argument) or implicitly defined (via either the "
            "minimum or the maximum or both). If the maximum and/or the "
            "minimum have a non-singleton shape, they must match the "
            "provided shape if this one is set explicitly."
        )
        if shape is not None and not isinstance(shape, torch.Size):
            if isinstance(shape, int):
                shape = _size([shape])
            else:
                shape = _size(list(shape))
        if shape is not None:
            shape_corr = _remove_neg_shapes(shape)
        else:
            shape_corr = None
        if high.ndimension():
            if shape_corr is not None and shape_corr != high.shape:
                raise RuntimeError(err_msg)
            if shape is None:
                shape = high.shape
            if shape_corr is not None:
                low = low.expand(shape_corr).clone()
        elif low.ndimension():
            if shape_corr is not None and shape_corr != low.shape:
                raise RuntimeError(err_msg)
            if shape is None:
                shape = low.shape
            if shape_corr is not None:
                high = high.expand(shape_corr).clone()
        elif shape_corr is None:
            raise RuntimeError(err_msg)
        else:
            low = low.expand(shape_corr).clone()
            high = high.expand(shape_corr).clone()

        if low.numel() > high.numel():
            high = high.expand_as(low).clone()
        elif high.numel() > low.numel():
            low = low.expand_as(high).clone()
        if shape_corr is None:
            shape = low.shape
        else:
            if isinstance(shape_corr, float):
                shape_corr = _size([shape_corr])
            elif not isinstance(shape_corr, torch.Size):
                shape_corr = _size(shape_corr)
            shape_corr_err_msg = (
                f"low and shape_corr mismatch, got {low.shape} and {shape_corr}"
            )
            if len(low.shape) != len(shape_corr):
                raise RuntimeError(shape_corr_err_msg)
            if not all(_s == _sa for _s, _sa in zip(shape_corr, low.shape)):
                raise RuntimeError(shape_corr_err_msg)
        self.shape = shape

        super().__init__(
            shape=shape,
            space=ContinuousBox(low, high, device=device),
            device=device,
            dtype=dtype,
            domain=domain,
        )
        self.encode = self._encode_eager

    def index(
        self, index: INDEX_TYPING, tensor_to_index: torch.Tensor | TensorDictBase
    ) -> torch.Tensor | TensorDictBase:
        raise NotImplementedError("Indexing not implemented for Bounded.")

    def enumerate(self, use_mask: bool = False) -> Any:
        raise NotImplementedError(
            f"enumerate is not implemented for spec of class {type(self).__name__}."
        )

    def cardinality(self) -> int:
        return float("inf")

    def __eq__(self, other):
        return (
            type(other) == type(self)
            and self.device == other.device
            and self.shape == other.shape
            and self.space == other.space
            and self.dtype == other.dtype
        )

    @property
    def low(self):
        return self.space.low

    @property
    def high(self):
        return self.space.high

    def expand(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list, torch.Size)):
            shape = shape[0]
        if any(s1 != s2 and s2 != 1 for s1, s2 in zip(shape[-self.ndim :], self.shape)):
            raise ValueError(
                f"The last {self.ndim} of the expanded shape {shape} must match the"
                f"shape of the {self.__class__.__name__} spec in expand()."
            )
        return self.__class__(
            low=self.space.low.expand(_remove_neg_shapes(shape)).clone(),
            high=self.space.high.expand(_remove_neg_shapes(shape)).clone(),
            shape=shape,
            device=self.device,
            dtype=self.dtype,
        )

    def _reshape(self, shape):
        return self.__class__(
            low=self.space.low.reshape(shape).clone(),
            high=self.space.high.reshape(shape).clone(),
            shape=shape,
            device=self.device,
            dtype=self.dtype,
        )

    def _unflatten(self, dim, sizes):
        shape = torch.zeros(self.shape, device="meta").unflatten(dim, sizes).shape
        return self.__class__(
            low=self.space.low.unflatten(dim, sizes).clone(),
            high=self.space.high.unflatten(dim, sizes).clone(),
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

    def unbind(self, dim: int = 0):
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
                low=low,
                high=high,
                shape=shape,
                device=self.device,
                dtype=self.dtype,
            )
            for low, high in zip(low, high)
        )

    def rand(self, shape: torch.Size = None) -> torch.Tensor:
        if shape is None:
            shape = _size([])
        a, b = self.space
        if self.dtype in (torch.float, torch.double, torch.half):
            shape = [*shape, *self._safe_shape]
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
            r = torch.rand(_size([*shape, *self._safe_shape]), device=interval.device)
            r = interval * r
            r = self.space.low + r
            if r.dtype != self.dtype:
                r = r.to(self.dtype)
            if self.dtype is not None and r.device != self.device:
                r = r.to(self.device)
            return r

    def _project(self, val: torch.Tensor) -> torch.Tensor:
        low = self.space.low
        high = self.space.high
        if self.device != val.device:
            low = low.to(val.device)
            high = high.to(val.device)
        low = low.expand_as(val)
        high = high.expand_as(val)
        val = torch.clamp(val, low, high)
        return val

    def is_in(self, val: torch.Tensor) -> bool:
        val_shape = _remove_neg_shapes(tensordict.utils._shape(val))
        shape = torch.broadcast_shapes(self._safe_shape, val_shape)
        shape = list(shape)
        shape[-len(self.shape) :] = [
            s_prev if s_prev >= 0 else s
            for (s_prev, s) in zip(self.shape, shape[-len(self.shape) :])
        ]
        shape_match = all(s1 == s2 or s1 == -1 for s1, s2 in zip(shape, val_shape))
        if not shape_match:
            return False
        dtype_match = val.dtype == self.dtype
        if not dtype_match:
            return False
        try:
            within_bounds = (val >= self.space.low.to(val.device)).all() and (
                val <= self.space.high.to(val.device)
            ).all()
            return within_bounds
        except NotImplementedError:
            within_bounds = all(
                (_val >= space.low.to(val.device)).all()
                and (_val <= space.high.to(val.device)).all()
                for (_val, space) in zip(val, self.space.unbind(0))
            )
            return within_bounds
        except RuntimeError as err:
            if "The size of tensor a" in str(err):
                warnings.warn(f"Got a shape mismatch: {str(err)}")
                return False
            raise err

    def to(self, dest: torch.dtype | DEVICE_TYPING) -> Bounded:
        if isinstance(dest, torch.dtype):
            dest_dtype = dest
            dest_device = self.device
        elif dest is None:
            return self
        else:
            dest_dtype = self.dtype
            dest_device = torch.device(dest)
        if dest_device == self.device and dest_dtype == self.dtype:
            return self
        space = self.space.to(dest_device)
        return Bounded(
            low=space.low,
            high=space.high,
            shape=self.shape,
            device=dest_device,
            dtype=dest_dtype,
        )

    def clone(self) -> Bounded:
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

        indexed_shape = _size(_shape_indexing(self.shape, idx))
        # Expand is required as pytorch.tensor indexing
        return self.__class__(
            low=self.space.low[idx].clone().expand(indexed_shape),
            high=self.space.high[idx].clone().expand(indexed_shape),
            shape=indexed_shape,
            device=self.device,
            dtype=self.dtype,
        )


class BoundedContinuous(Bounded, metaclass=_BoundedMeta):
    """A specialized version of :class:`torchrl.data.Bounded` with continuous space."""

    def __init__(
        self,
        low: float | torch.Tensor | np.ndarray = None,
        high: float | torch.Tensor | np.ndarray = None,
        shape: torch.Size | int | None = None,
        device: DEVICE_TYPING | None = None,
        dtype: torch.dtype | str | None = None,
        domain: str = "continuous",
    ):
        super().__init__(
            low=low, high=high, shape=shape, device=device, dtype=dtype, domain=domain
        )
        self.encode = self._encode_eager


class BoundedDiscrete(Bounded, metaclass=_BoundedMeta):
    """A specialized version of :class:`torchrl.data.Bounded` with discrete space."""

    def __init__(
        self,
        low: float | torch.Tensor | np.ndarray = None,
        high: float | torch.Tensor | np.ndarray = None,
        shape: torch.Size | int | None = None,
        device: DEVICE_TYPING | None = None,
        dtype: torch.dtype | str | None = None,
        domain: str = "discrete",
    ):
        super().__init__(
            low=low,
            high=high,
            shape=shape,
            device=device,
            dtype=dtype,
            domain=domain,
        )
        self.encode = self._encode_eager


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


class NonTensor(TensorSpec):
    """A spec for non-tensor data.

    The `NonTensor` class is designed to handle specifications for data that do not conform to standard tensor
    structures.
    It maintains attributes such as shape, and device similar to the `NonTensorData` class.
    The dtype is optional and should in practice be left to `None` in most cases.
    Methods like `rand`, `zero`, and `one` will return a `NonTensorData` object with a `None` data value.

    .. warning:: The default shape of `NonTensor` is `(1,)`.

    Args:
        shape (Union[torch.Size, int], optional): The shape of the non-tensor data. Defaults to `(1,)`.
        device (Optional[DEVICE_TYPING], optional): The device on which the data is stored. Defaults to `None`.
        dtype (torch.dtype | None, optional): The data type of the non-tensor data. Defaults to `None`.
        example_data (Any, optional): An example of the data that this spec represents. This example is used as a
            template when generating new data with the `rand`, `zero`, and `one` methods.
        batched (bool, optional): Indicates whether the data is batched. If `True`, the `rand`, `zero`, and `one` methods
            will generate data with an additional batch dimension, stacking copies of the `example_data` across this dimension.
            Defaults to `False`.
        **kwargs: Additional keyword arguments passed to the parent class.

    .. seealso:: :class:`~torchrl.data.Choice` which allows to randomly choose among different specs when calling
      `rand`.

    Examples:
        >>> from torchrl.data import NonTensor
        >>> spec = NonTensor(example_data="a string", batched=False, shape=(3,))
        >>> spec.rand()
        NonTensorData(data=a string, batch_size=torch.Size([3]), device=None)
        >>> spec = NonTensor(example_data="a string", batched=True, shape=(3,))
        >>> spec.rand()
        NonTensorStack(
            ['a string', 'a string', 'a string'],
            batch_size=torch.Size([3]),
            device=None)
    """

    example_data: Any = None

    def __init__(
        self,
        shape: torch.Size | int = _DEFAULT_SHAPE,
        device: DEVICE_TYPING | None = None,
        dtype: torch.dtype | None = None,
        example_data: Any = None,
        batched: bool = False,
        **kwargs,
    ):
        if isinstance(shape, int):
            shape = _size([shape])

        domain = None
        super().__init__(
            shape=shape, space=None, device=device, dtype=dtype, domain=domain, **kwargs
        )
        self.example_data = example_data
        self.batched = batched
        self.encode = self._encode_eager

    def __repr__(self):
        shape_str = indent("shape=" + str(self.shape), " " * 4)
        space_str = indent("space=" + str(self.space), " " * 4)
        device_str = indent("device=" + str(self.device), " " * 4)
        dtype_str = indent("dtype=" + str(self.dtype), " " * 4)
        domain_str = indent("domain=" + str(self.domain), " " * 4)
        example_str = indent("example_data=" + str(self.example_data), " " * 4)
        sub_string = ",\n".join(
            [shape_str, space_str, device_str, dtype_str, domain_str, example_str]
        )
        string = f"{self.__class__.__name__}(\n{sub_string})"
        return string

    def __eq__(self, other):
        eq = super().__eq__(other)
        eq = eq & (self.example_data == getattr(other, "example_data", None))
        return eq

    def _project(self) -> Any:
        raise NotImplementedError("Cannot project a NonTensorSpec.")

    def index(
        self, index: INDEX_TYPING, tensor_to_index: torch.Tensor | TensorDictBase
    ) -> torch.Tensor | TensorDictBase:
        raise NotImplementedError("Cannot use index with a NonTensorSpec.")

    def cardinality(self) -> Any:
        raise NotImplementedError("Cannot enumerate a NonTensor spec.")

    def enumerate(self, use_mask: bool = False) -> Any:
        raise NotImplementedError("Cannot enumerate a NonTensor spec.")

    def to(self, dest: torch.dtype | DEVICE_TYPING) -> NonTensor:
        if isinstance(dest, torch.dtype):
            dest_dtype = dest
            dest_device = self.device
        elif dest is None:
            return self
        else:
            dest_dtype = self.dtype
            dest_device = torch.device(dest)
        if dest_device == self.device and dest_dtype == self.dtype:
            return self
        return self.__class__(
            shape=self.shape,
            device=dest_device,
            dtype=None,
            example_data=self.example_data,
            batched=self.batched,
        )

    def clone(self) -> NonTensor:
        return self.__class__(
            shape=self.shape,
            device=self.device,
            dtype=self.dtype,
            example_data=self.example_data,
            batched=self.batched,
        )

    def rand(self, shape=None):
        if shape is None:
            shape = ()
        if self.batched:
            with set_capture_non_tensor_stack(False):
                val = NonTensorData(
                    data=self.example_data,
                    batch_size=(),
                    device=self.device,
                )
                shape = (*shape, *self._safe_shape)
                if shape:
                    for i in shape:
                        val = torch.stack([val.copy() for _ in range(i)], -1)
                return val
        return NonTensorData(
            data=self.example_data,
            batch_size=(*shape, *self._safe_shape),
            device=self.device,
        )

    def zero(self, shape=None):
        return self.rand(shape=shape)

    def one(self, shape=None):
        return self.rand(shape=shape)

    def is_in(self, val: Any) -> bool:
        if not isinstance(val, torch.Tensor) and not is_tensor_collection(val):
            return True
        shape = torch.broadcast_shapes(self._safe_shape, val.shape)
        return (
            is_non_tensor(val)
            and val.shape == shape
            # We relax constrains on device as they're hard to enforce for non-tensor
            #  tensordicts and pointless
            # and val.device == self.device
            # TODO: do we want this?
            and val.dtype == self.dtype
        )

    def expand(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list, torch.Size)):
            shape = shape[0]
        shape = _size(shape)
        if not all(
            (old == 1) or (old == new)
            for old, new in zip(self.shape, shape[-len(self.shape) :])
        ):
            raise ValueError(
                f"The last elements of the expanded shape must match the current one. Got shape={shape} while self.shape={self.shape}."
            )
        return self.__class__(
            shape=shape,
            device=self.device,
            dtype=None,
            example_data=self.example_data,
            batched=self.batched,
        )

    def unsqueeze(self, dim: int) -> NonTensor:
        unsq = super().unsqueeze(dim=dim)
        unsq.example_data = self.example_data
        unsq.batched = self.batched
        return unsq

    def squeeze(self, dim: int | None = None) -> NonTensor:
        sq = super().squeeze(dim=dim)
        sq.example_data = self.example_data
        sq.batched = self.batched
        return sq

    def _reshape(self, shape):
        return self.__class__(
            shape=shape,
            device=self.device,
            dtype=self.dtype,
            example_data=self.example_data,
            batched=self.batched,
        )

    def _unflatten(self, dim, sizes):
        shape = torch.zeros(self.shape, device="meta").unflatten(dim, sizes).shape
        return self.__class__(
            shape=shape,
            device=self.device,
            dtype=self.dtype,
            example_data=self.example_data,
            batched=self.batched,
        )

    def __getitem__(self, idx: SHAPE_INDEX_TYPING):
        """Indexes the current TensorSpec based on the provided index."""
        indexed_shape = _size(_shape_indexing(self.shape, idx))
        return self.__class__(
            shape=indexed_shape,
            device=self.device,
            dtype=self.dtype,
            example_data=self.example_data,
            batched=self.batched,
        )

    def unbind(self, dim: int = 0):
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
                example_data=self.example_data,
                batched=self.batched,
            )
            for i in range(self.shape[dim])
        )

    def to_numpy(
        self, val: torch.Tensor | TensorDictBase, safe: bool = None
    ) -> np.ndarray | dict:
        return val

    def encode(
        self,
        val: np.ndarray | torch.Tensor | TensorDictBase,
        *,
        ignore_device: bool = False,
    ) -> torch.Tensor | TensorDictBase:
        return val


class _UnboundedMeta(abc.ABCMeta):
    def __call__(cls, *args, **kwargs):
        instance = super().__call__(*args, **kwargs)
        if instance.domain == "continuous":
            instance.__class__ = UnboundedContinuous
        else:
            instance.__class__ = UnboundedDiscrete
        return instance


@dataclass(repr=False)
class Unbounded(TensorSpec, metaclass=_UnboundedMeta):
    """An unbounded tensor spec.

    ``Unbounded`` specs will never appear as such and always be subclassed as :class:`UnboundedContinuous`
    or :class:`UnboundedDiscrete` depending on their dtype (floating points dtypes will result in
    :class:`UnboundedContinuous` instances, all others in :class:`UnboundedDiscrete` instances).

    Although it is not properly limited above and below, this class still has a :attr:`Box` space that encodes
    the maximum and minimum value that the dtype accepts.

    Args:
        shape (torch.Size): the shape of the ``Bounded`` spec. The shape must be specified.
            Inputs ``low``, ``high`` and ``shape`` must be broadcastable.
        device (str, int or torch.device, optional): device of the tensors.
        dtype (str or torch.dtype, optional): dtype of the tensors.
        domain (str): `"continuous"` or `"discrete"`. Can be used to override the automatic type assignment.

    Examples:
        >>> spec = Unbounded(shape=(), dtype=torch.float)
        >>> spec
        UnboundedContinuous(
            shape=torch.Size([]),
            space=ContinuousBox(
                low=Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, contiguous=True),
                high=Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, contiguous=True)),
            device=cpu,
            dtype=torch.float32,
            domain=continuous)
        >>> spec = Unbounded(shape=(), dtype=torch.int)
        >>> spec
        UnboundedDiscrete(
            shape=torch.Size([]),
            space=ContinuousBox(
                low=Tensor(shape=torch.Size([]), device=cpu, dtype=torch.int32, contiguous=True),
                high=Tensor(shape=torch.Size([]), device=cpu, dtype=torch.int32, contiguous=True)),
            device=cpu,
            dtype=torch.int32,
            domain=discrete)
        >>> spec.to(torch.float)
        UnboundedContinuous(
            shape=torch.Size([]),
            space=ContinuousBox(
                low=Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, contiguous=True),
                high=Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, contiguous=True)),
            device=cpu,
            dtype=torch.float32,
            domain=continuous)
        >>> spec = Unbounded(shape=(), dtype=torch.int, domain="continuous")
        >>> spec
        UnboundedContinuous(
            shape=torch.Size([]),
            space=ContinuousBox(
                low=Tensor(shape=torch.Size([]), device=cpu, dtype=torch.int32, contiguous=True),
                high=Tensor(shape=torch.Size([]), device=cpu, dtype=torch.int32, contiguous=True)),
            device=cpu,
            dtype=torch.int32,
            domain=continuous)

    """

    def __init__(
        self,
        shape: torch.Size | int = _DEFAULT_SHAPE,
        device: DEVICE_TYPING | None = None,
        dtype: str | torch.dtype | None = None,
        **kwargs,
    ):
        if isinstance(shape, int):
            shape = _size([shape])

        dtype, device = _default_dtype_and_device(
            dtype, device, allow_none_device=False
        )
        if dtype == torch.bool:
            min_value = False
            max_value = True
            default_domain = "discrete"
        else:
            if dtype.is_floating_point:
                min_value = torch.finfo(dtype).min
                max_value = torch.finfo(dtype).max
                default_domain = "continuous"
            else:
                min_value = torch.iinfo(dtype).min
                max_value = torch.iinfo(dtype).max
                default_domain = "discrete"
        box = ContinuousBox(
            torch.full(
                _remove_neg_shapes(shape), min_value, device=device, dtype=dtype
            ),
            torch.full(
                _remove_neg_shapes(shape), max_value, device=device, dtype=dtype
            ),
        )

        domain = kwargs.pop("domain", default_domain)
        super().__init__(
            shape=shape, space=box, device=device, dtype=dtype, domain=domain, **kwargs
        )
        self.encode = self._encode_eager

    def cardinality(self) -> int:
        raise NotImplementedError(
            "`cardinality` is not implemented for Unbounded specs."
        )

    def index(
        self, index: INDEX_TYPING, tensor_to_index: torch.Tensor | TensorDictBase
    ) -> torch.Tensor | TensorDictBase:
        raise NotImplementedError("`index` is not implemented for Unbounded specs.")

    def to(self, dest: torch.dtype | DEVICE_TYPING) -> Unbounded:
        if isinstance(dest, torch.dtype):
            dest_dtype = dest
            dest_device = self.device
        elif dest is None:
            return self
        else:
            dest_dtype = self.dtype
            dest_device = torch.device(dest)
        if dest_device == self.device and dest_dtype == self.dtype:
            return self
        return Unbounded(shape=self.shape, device=dest_device, dtype=dest_dtype)

    def clone(self) -> Unbounded:
        return self.__class__(shape=self.shape, device=self.device, dtype=self.dtype)

    def rand(self, shape: torch.Size = None) -> torch.Tensor:
        if shape is None:
            shape = _size([])
        shape = [*shape, *self._safe_shape]
        if self.dtype.is_floating_point:
            return torch.randn(shape, device=self.device, dtype=self.dtype)
        return torch.empty(shape, device=self.device, dtype=self.dtype).random_()

    def is_in(self, val: torch.Tensor) -> bool:
        shape = torch.broadcast_shapes(self._safe_shape, val.shape)
        return val.shape == shape and val.dtype == self.dtype

    def _project(self, val: torch.Tensor) -> torch.Tensor:
        return torch.as_tensor(val, dtype=self.dtype).reshape(
            val.shape[: -self.ndim] + self.shape
        )

    def enumerate(self, use_mask: bool = False) -> Any:
        raise NotImplementedError("enumerate cannot be called with continuous specs.")

    def expand(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list, torch.Size)):
            shape = shape[0]
        # TODO: this blocks batched envs which expand shapes
        # if any(val < 0 for val in shape):
        #     raise ValueError(
        #         f"{self.__class__.__name__}.expand does not support negative shapes."
        #     )
        if any(s1 != s2 and s2 != 1 for s1, s2 in zip(shape[-self.ndim :], self.shape)):
            raise ValueError(
                f"The last {self.ndim} of the expanded shape {shape} must match the"
                f"shape of the {self.__class__.__name__} spec in expand()."
            )
        return self.__class__(shape=shape, device=self.device, dtype=self.dtype)

    def _reshape(self, shape):
        return self.__class__(shape=shape, device=self.device, dtype=self.dtype)

    def _unflatten(self, dim, sizes):
        shape = torch.zeros(self.shape, device="meta").unflatten(dim, sizes).shape
        return self.__class__(
            shape=shape,
            device=self.device,
            dtype=self.dtype,
        )

    def __getitem__(self, idx: SHAPE_INDEX_TYPING):
        """Indexes the current TensorSpec based on the provided index."""
        indexed_shape = _size(_shape_indexing(self.shape, idx))
        return self.__class__(shape=indexed_shape, device=self.device, dtype=self.dtype)

    def unbind(self, dim: int = 0):
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

    def __eq__(self, other):
        # those specs are equivalent to a discrete spec
        if isinstance(other, Bounded):
            minval, maxval = _minmax_dtype(self.dtype)
            minval = torch.as_tensor(minval, device=self.device, dtype=self.dtype)
            maxval = torch.as_tensor(maxval, device=self.device, dtype=self.dtype)
            return (
                Bounded(
                    shape=self.shape,
                    high=maxval,
                    low=minval,
                    dtype=self.dtype,
                    device=self.device,
                    domain=self.domain,
                )
                == other
            )
        elif isinstance(other, Unbounded):
            if self.dtype != other.dtype:
                return False
            if self.shape != other.shape:
                return False
            if self.device != other.device:
                return False
            return True
        return super().__eq__(other)


class UnboundedContinuous(Unbounded):
    """A specialized version of :class:`torchrl.data.Unbounded` with continuous space."""

    ...


class UnboundedDiscrete(Unbounded):
    """A specialized version of :class:`torchrl.data.Unbounded` with discrete space."""

    def __init__(
        self,
        shape: torch.Size | int = _DEFAULT_SHAPE,
        device: DEVICE_TYPING | None = None,
        dtype: str | torch.dtype | None = torch.int64,
        **kwargs,
    ):
        super().__init__(shape=shape, device=device, dtype=dtype, **kwargs)
        self.encode = self._encode_eager


@dataclass(repr=False)
class MultiOneHot(OneHot):
    """A concatenation of one-hot discrete tensor spec.

    This class can be used when a single tensor must carry information about multiple one-hot encoded
    values.

    The last dimension of the shape (domain of the tensor elements) cannot be indexed.

    Args:
        nvec (iterable of integers): cardinality of each of the elements of
            the tensor.
        shape (torch.Size, optional): total shape of the sampled tensors.
            If provided, the last dimension must match sum(nvec).
        device (str, int or torch.device, optional): device of
            the tensors.
        dtype (str or torch.dtype, optional): dtype of the tensors.
        mask (torch.Tensor or None): mask some of the possible outcomes when a
            sample is taken. See :meth:`update_mask` for more information.

    Examples:
        >>> ts = MultiOneHot((3,2,3))
        >>> ts.rand()
        tensor([ True, False, False,  True, False, False, False,  True])
        >>> ts.is_in(torch.tensor([
        ...     0, 0, 1,
        ...     0, 1,
        ...     1, 0, 0], dtype=torch.bool))
        True
        >>> ts.is_in(torch.tensor([
        ...     1, 0, 1,
        ...     0, 1,
        ...     1, 0, 0], dtype=torch.bool))
        False

    """

    def __init__(
        self,
        nvec: Sequence[int],
        shape: torch.Size | None = None,
        device=None,
        dtype=torch.bool,
        use_register=False,
        mask: torch.Tensor | None = None,
    ):
        self.nvec = nvec
        dtype, device = _default_dtype_and_device(
            dtype, device, allow_none_device=False
        )
        if shape is None:
            shape = _size((sum(nvec),))
        else:
            shape = _size(shape)
            if shape[-1] != sum(nvec):
                raise ValueError(
                    f"The last value of the shape must match sum(nvec) for transform of type {self.__class__}. "
                    f"Got sum(nvec)={sum(nvec)} and shape={shape}."
                )
        space = BoxList([CategoricalBox(n) for n in nvec])
        self.use_register = use_register
        super(OneHot, self).__init__(
            shape,
            space,
            device,
            dtype,
            domain="discrete",
        )
        self.update_mask(mask)
        self.encode = self._encode_eager

    def cardinality(self) -> int:
        return torch.as_tensor(self.nvec).prod()

    def enumerate(self, use_mask: bool = False) -> torch.Tensor:
        nvec = self.nvec
        enum_disc = self.to_categorical_spec().enumerate(use_mask)
        enums = torch.cat(
            [
                torch.nn.functional.one_hot(enum_unb, nv).to(self.dtype)
                for nv, enum_unb in zip(nvec, enum_disc.unbind(-1))
            ],
            -1,
        )
        return enums

    def update_mask(self, mask):
        """Sets a mask to prevent some of the possible outcomes when a sample is taken.

        The mask can also be set during initialization of the spec.

        Args:
            mask (torch.Tensor or None): boolean mask. If None, the mask is
                disabled. Otherwise, the shape of the mask must be expandable to
                the shape of the spec. ``False`` masks an outcome and ``True``
                leaves the outcome unmasked. If all of the possible outcomes are
                masked, then an error is raised when a sample is taken.

        Examples:
            >>> mask = torch.tensor([True, False, False,
            ...                      True, True])
            >>> ts = MultiOneHot((3, 2), (2, 5), dtype=torch.int64, mask=mask)
            >>> # All but one of the three possible outcomes for the first
            >>> # one-hot group are masked, but neither of the two possible
            >>> # outcomes for the second one-hot group are masked.
            >>> ts.rand()
            tensor([[1, 0, 0, 0, 1],
                    [1, 0, 0, 1, 0]])
        """
        if mask is not None:
            try:
                mask = mask.expand(*self._safe_shape)
            except RuntimeError as err:
                raise RuntimeError("Cannot expand mask to the desired shape.") from err
            if mask.dtype != torch.bool:
                raise ValueError("Only boolean masks are accepted.")
        self.mask = mask

    def to(self, dest: torch.dtype | DEVICE_TYPING) -> MultiOneHot:
        if isinstance(dest, torch.dtype):
            dest_dtype = dest
            dest_device = self.device
        elif dest is None:
            return self
        else:
            dest_dtype = self.dtype
            dest_device = torch.device(dest)
        if dest_device == self.device and dest_dtype == self.dtype:
            return self
        return MultiOneHot(
            nvec=deepcopy(self.nvec),
            shape=self.shape,
            device=dest_device,
            dtype=dest_dtype,
            mask=self.mask.to(dest) if self.mask is not None else None,
        )

    def clone(self) -> MultiOneHot:
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

    def rand(self, shape: torch.Size | None = None) -> torch.Tensor:
        if shape is None:
            shape = self.shape[:-1]
        else:
            shape = _size([*shape, *self.shape[:-1]])
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
        mask = mask.expand(_remove_neg_shapes(*shape, mask.shape[-1]))
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

    def _encode_eager(
        self, val: np.ndarray | torch.Tensor, *, ignore_device: bool = False
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
                super(type(self), self)._encode_eager(
                    v, space, ignore_device=ignore_device
                )
            )
        return torch.cat(x, -1).reshape(self.shape)

    def _encode_memo(
        self, val: np.ndarray | torch.Tensor, *, ignore_device: bool = False
    ) -> torch.Tensor:
        funcs = self._encode_memo_dict.get(ignore_device)
        if funcs is not None:
            return funcs(val)
        funcs = []
        val_orig = val
        if not isinstance(val, torch.Tensor):
            if not ignore_device:

                def as_tensor(val):
                    val = torch.tensor(val, device=self.device)

            else:
                as_tensor = torch.as_tensor
            funcs.append(as_tensor)
        val = _reduce_funcs(funcs)(val_orig)

        def cat(val):
            x = []
            for v, space in zip(val.unbind(-1), self.space):
                if not (v < space.n).all():
                    raise RuntimeError(
                        f"value {v} is greater than the allowed max {space.n}"
                    )
                x.append(
                    super(type(self), self)._encode_eager(
                        v, space, ignore_device=ignore_device
                    )
                )
            return torch.cat(x, -1).reshape(self.shape)

        funcs.append(cat)
        val = _reduce_funcs(funcs)(val_orig)

        if len(funcs) == 0:
            self._encode_memo_dict[ignore_device] = lambda x: x
        elif len(funcs) == 1:
            self._encode_memo_dict[ignore_device] = funcs[0]
        else:
            self._encode_memo_dict[ignore_device] = functools.partial(
                functools.reduce, lambda x, f: f(x), funcs
            )
        return self._encode_memo_dict[ignore_device](val_orig)

    def _split(self, val: torch.Tensor) -> torch.Tensor | None:
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
                OneHot(
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

        Examples:
            >>> mone_hot = MultiOneHot((2, 3, 4))
            >>> onehot_sample = mone_hot.rand()
            >>> onehot_sample
            tensor([False,  True, False, False,  True, False,  True, False, False])
            >>> categ_sample = mone_hot.to_categorical(onehot_sample)
            >>> categ_sample
            tensor([1, 2, 1])

        """
        if safe is None:
            safe = _CHECK_SPEC_ENCODE
        if safe:
            self.assert_is_in(val)
        vals = self._split(val)
        return torch.stack([val.long().argmax(-1) for val in vals], -1)

    def to_categorical_spec(self) -> MultiCategorical:
        """Converts the spec to the equivalent categorical spec.

        Examples:
            >>> mone_hot = MultiOneHot((2, 3, 4))
            >>> categ = mone_hot.to_categorical_spec()
            >>> categ
            MultiCategorical(
                shape=torch.Size([3]),
                space=BoxList(boxes=[CategoricalBox(n=2), CategoricalBox(n=3), CategoricalBox(n=4)]),
                device=cpu,
                dtype=torch.int64,
                domain=discrete)

        """
        return MultiCategorical(
            [_space.n for _space in self.space],
            device=self.device,
            shape=[*self.shape[:-1], len(self.space)],
            mask=self.mask,
        )

    def to_one_hot(self, val: torch.Tensor, safe: bool = None) -> torch.Tensor:
        """No-op for MultiOneHot."""
        return val

    def to_one_hot_spec(self) -> OneHot:
        """No-op for MultiOneHot."""
        return self

    def expand(self, *shape):
        nvecs = [space.n for space in self.space]
        if len(shape) == 1 and isinstance(shape[0], (tuple, list, torch.Size)):
            shape = shape[0]
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

    def _reshape(self, shape):
        nvecs = [space.n for space in self.space]
        mask = self.mask.reshape(shape) if self.mask is not None else None
        return self.__class__(
            nvec=nvecs,
            shape=shape,
            device=self.device,
            dtype=self.dtype,
            mask=mask,
        )

    def _unflatten(self, dim, sizes):
        nvecs = [space.n for space in self.space]
        shape = torch.zeros(self.shape, device="meta").unflatten(dim, sizes).shape
        mask = self.mask.reshape(shape) if self.mask is not None else None
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

    def unbind(self, dim: int = 0):
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
            shape=_size(indexed_shape + [self.shape[-1]]),
            device=self.device,
            dtype=self.dtype,
        )


class Categorical(TensorSpec):
    """A discrete tensor spec.

    An alternative to :class:`OneHot` for categorical variables in TorchRL.
    Categorical variables perform indexing instead of masking, which can speed-up
    computation and reduce memory cost for large categorical variables.

    The spec will have the shape defined by the ``shape`` argument: if a singleton dimension is
    desired for the training dimension, one should specify it explicitly.

    Attributes:
        n (int): The number of possible outcomes.
        shape (torch.Size): The shape of the variable.
        device (torch.device): The device of the tensors.
        dtype (torch.dtype): The dtype of the tensors.

    Args:
        n (int): number of possible outcomes. If set to -1, the cardinality of the categorical spec is undefined,
            and `set_provisional_n` must be called before sampling from this spec.
        shape: (torch.Size, optional): shape of the variable, default is "torch.Size([])".
        device (str, int or torch.device, optional): the device of the tensors.
        dtype (str or torch.dtype, optional): the dtype of the tensors.
        mask (torch.Tensor or None): A boolean mask to prevent some of the possible outcomes when a sample is taken.
            See :meth:`update_mask` for more information.

    Examples:
        >>> categ = Categorical(3)
        >>> categ
        Categorical(
            shape=torch.Size([]),
            space=CategoricalBox(n=3),
            device=cpu,
            dtype=torch.int64,
            domain=discrete)
        >>> categ.rand()
        tensor(2)
        >>> categ = Categorical(3, shape=(1,))
        >>> categ
        Categorical(
            shape=torch.Size([1]),
            space=CategoricalBox(n=3),
            device=cpu,
            dtype=torch.int64,
            domain=discrete)
        >>> categ.rand()
        tensor([1])
        >>> categ = Categorical(-1)
        >>> categ.set_provisional_n(5)
        >>> categ.rand()
        tensor(3)

    .. note:: When n is set to -1, calling `rand` without first setting a provisional n using `set_provisional_n`
        will raise a ``RuntimeError``.

    """

    shape: torch.Size
    space: CategoricalBox
    device: torch.device | None = None
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
            shape = _size([])
        dtype, device = _default_dtype_and_device(
            dtype, device, allow_none_device=False
        )
        space = CategoricalBox(n)
        super().__init__(
            shape=shape, space=space, device=device, dtype=dtype, domain="discrete"
        )
        self.update_mask(mask)
        self._provisional_n = None
        self.encode = self._encode_eager

    @property
    def _undefined_n(self):
        return self.space.n < 0

    def enumerate(self, use_mask: bool = False) -> torch.Tensor:
        dtype = self.dtype
        if dtype is torch.bool:
            dtype = torch.uint8
        n = self.n
        arange = torch.arange(n, dtype=dtype, device=self.device)
        if use_mask and self.mask is not None:
            arange = arange[self.mask]
            n = arange.shape[0]
        if self.ndim:
            arange = arange.view(-1, *(1,) * self.ndim)
        return arange.expand(n, *self.shape)

    @property
    def n(self):
        n = self.space.n
        if n == -1:
            n = self._provisional_n
            if n is None:
                raise RuntimeError(
                    f"Undefined cardinality for {type(self)}. Please call "
                    f"spec.set_provisional_n(int)."
                )
        return n

    def cardinality(self) -> int:
        return self.n

    def update_mask(self, mask):
        """Sets a mask to prevent some of the possible outcomes when a sample is taken.

        The mask can also be set during initialization of the spec.

        Args:
            mask (torch.Tensor or None): boolean mask. If None, the mask is
                disabled. Otherwise, the shape of the mask must be expandable to
                the shape of the equivalent one-hot spec. ``False`` masks an
                outcome and ``True`` leaves the outcome unmasked. If all of the
                possible outcomes are masked, then an error is raised when a
                sample is taken.

        Examples:
            >>> mask = torch.tensor([True, False, True])
            >>> ts = Categorical(3, (10,), dtype=torch.int64, mask=mask)
            >>> # One of the three possible outcomes is masked
            >>> ts.rand()
            tensor([0, 2, 2, 0, 2, 0, 2, 2, 0, 2])
        """
        if mask is not None:
            try:
                mask = mask.expand(_remove_neg_shapes(*self.shape, self.space.n))
            except RuntimeError as err:
                raise RuntimeError("Cannot expand mask to the desired shape.") from err
            if mask.dtype != torch.bool:
                raise ValueError("Only boolean masks are accepted.")
        self.mask = mask

    def set_provisional_n(self, n: int):
        """Set the cardinality of the Categorical spec temporarily.

        This method is required to be called before sampling from the spec when n is -1.

        Args:
            n (int): The cardinality of the Categorical spec.

        """
        self._provisional_n = n

    def rand(self, shape: torch.Size = None) -> torch.Tensor:
        if self._undefined_n:
            if self._provisional_n is None:
                raise RuntimeError(
                    "Cannot generate random categorical samples for undefined cardinality (n=-1). "
                    "To sample from this class, first call Categorical.set_provisional_n(n) before calling rand()."
                )
            n = self._provisional_n
        else:
            n = self.space.n
        if shape is None:
            shape = _size([])
        if self.mask is None:
            return torch.randint(
                0,
                n,
                _size([*shape, *_remove_neg_shapes(self.shape)]),
                device=self.device,
                dtype=self.dtype,
            )
        mask = self.mask
        mask = mask.expand(_remove_neg_shapes(*shape, *mask.shape))
        if mask.ndim > 2:
            mask_flat = torch.flatten(mask, 0, -2)
        else:
            mask_flat = mask
        shape_out = mask.shape[:-1]
        # Check that the mask has the right size
        if mask_flat.shape[-1] != n:
            raise ValueError(
                "The last dimension of the mask must match the number of action allowed by the "
                f"Categorical spec. Got mask.shape={self.mask.shape} and n={n}."
            )
        out = torch.multinomial(mask_flat.float(), 1).reshape(shape_out)
        return out

    def index(
        self, index: INDEX_TYPING, tensor_to_index: torch.Tensor | TensorDictBase
    ) -> torch.Tensor | TensorDictBase:
        idx = index.expand(
            tensor_to_index.shape[: -self.ndim] + torch.Size([-1] * self.ndim)
        )
        return tensor_to_index.gather(-1, idx)

    def _project(self, val: torch.Tensor) -> torch.Tensor:
        if val.dtype not in (torch.int, torch.long):
            val = torch.round(val)
        if self.mask is None:
            return val.clamp_(min=0, max=self.space.n - 1)
        shape = self.mask.shape
        shape = _size([*torch.broadcast_shapes(shape[:-1], val.shape), shape[-1]])
        mask_expand = self.mask.expand(shape)
        gathered = mask_expand.gather(-1, val.unsqueeze(-1))
        oob = ~gathered.all(-1)
        new_val = torch.multinomial(mask_expand[oob].float(), 1).squeeze(-1)
        val = torch.masked_scatter(val, oob, new_val)
        return val

    def is_in(self, val: torch.Tensor) -> bool:
        if self.mask is None:
            shape = torch.broadcast_shapes(self._safe_shape, val.shape)
            shape_match = val.shape == shape
            if not shape_match:
                return False
            dtype_match = val.dtype == self.dtype
            if not dtype_match:
                return False
            if self.space.n == -1:
                return True
            return (0 <= val).all() and (val < self.space.n).all()
        shape = self.mask.shape
        shape = _size([*torch.broadcast_shapes(shape[:-1], val.shape), shape[-1]])
        mask_expand = self.mask.expand(shape)
        gathered = mask_expand.gather(-1, val.unsqueeze(-1))
        return gathered.all()

    def __getitem__(self, idx: SHAPE_INDEX_TYPING):
        """Indexes the current TensorSpec based on the provided index."""
        indexed_shape = _size(_shape_indexing(self.shape, idx))
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

        Examples:
            >>> categ = Categorical(3)
            >>> categ_sample = categ.zero()
            >>> categ_sample
            tensor(0)
            >>> onehot_sample = categ.to_one_hot(categ_sample)
            >>> onehot_sample
            tensor([ True, False, False])
        """
        if safe is None:
            safe = _CHECK_SPEC_ENCODE
        if safe:
            self.assert_is_in(val)
        return torch.nn.functional.one_hot(val, self.space.n).bool()

    def to_categorical(self, val: torch.Tensor, safe: bool = None) -> torch.Tensor:
        """No-op for categorical."""
        return val

    def to_one_hot_spec(self) -> OneHot:
        """Converts the spec to the equivalent one-hot spec.

        Examples:
            >>> categ = Categorical(3)
            >>> categ.to_one_hot_spec()
            OneHot(
                shape=torch.Size([3]),
                space=CategoricalBox(n=3),
                device=cpu,
                dtype=torch.bool,
                domain=discrete)

        """
        shape = [*self.shape, self.space.n]
        return OneHot(
            n=self.space.n,
            shape=shape,
            device=self.device,
        )

    def to_categorical_spec(self) -> Categorical:
        """No-op for categorical."""
        return self

    def expand(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list, torch.Size)):
            shape = shape[0]
        if any(s1 != s2 and s2 != 1 for s1, s2 in zip(shape[-self.ndim :], self.shape)):
            raise ValueError(
                f"The last {self.ndim} of the expanded shape {shape} must match the"
                f"shape of the {self.__class__.__name__} spec in expand()."
            )
        return self.__class__(
            n=self.space.n, shape=shape, device=self.device, dtype=self.dtype
        )

    def _reshape(self, shape):
        return self.__class__(
            n=self.space.n, shape=shape, device=self.device, dtype=self.dtype
        )

    def _unflatten(self, dim, sizes):
        shape = torch.zeros(self.shape, device="meta").unflatten(dim, sizes).shape
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

    def unbind(self, dim: int = 0):
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

    def to(self, dest: torch.dtype | DEVICE_TYPING) -> Categorical:
        if isinstance(dest, torch.dtype):
            dest_dtype = dest
            dest_device = self.device
        elif dest is None:
            return self
        else:
            dest_dtype = self.dtype
            dest_device = torch.device(dest)
        if dest_device == self.device and dest_dtype == self.dtype:
            return self
        return self.__class__(
            n=self.space.n, shape=self.shape, device=dest_device, dtype=dest_dtype
        )

    def clone(self) -> Categorical:
        return self.__class__(
            n=self.space.n,
            shape=self.shape,
            device=self.device,
            dtype=self.dtype,
            mask=self.mask.clone() if self.mask is not None else None,
        )


class Choice(TensorSpec):
    """A discrete choice spec for either tensor or non-tensor data.

    Args:
        choices (list[:class:`~TensorSpec`, :class:`~tensordict.NonTensorData`, :class:`~tensordict.NonTensorStack`]):
            List of specs or non-tensor data from which to choose during
            sampling. All elements must have the same type, shape, dtype, and
            device.

    Examples:
        >>> import torch
        >>> _ = torch.manual_seed(0)
        >>> from torchrl.data import Choice, Bounded
        >>> spec = Choice([
        ...     Bounded(0, 1, shape=(1,)),
        ...     Bounded(10, 11, shape=(1,))])
        >>> spec.rand()
        tensor([0.7682])
        >>> spec.rand()
        tensor([10.1320])
        >>> from tensordict import NonTensorData
        >>> _ = torch.manual_seed(0)
        >>> spec = Choice([NonTensorData(s) for s in ["a", "b", "c", "d"]])
        >>> spec.rand().data
        'a'
        >>> spec.rand().data
        'd'
    """

    def __init__(
        self,
        choices: list[TensorSpec | NonTensorData | NonTensorStack],
    ):
        if not isinstance(choices, list):
            raise TypeError("'choices' must be a list")

        if not isinstance(choices[0], (TensorSpec, NonTensorData, NonTensorStack)):
            raise TypeError(
                "Each choice must be either a TensorSpec, NonTensorData, or "
                f"NonTensorStack, but got {type(choices[0])}"
            )

        if not all([isinstance(choice, type(choices[0])) for choice in choices[1:]]):
            raise TypeError("All choices must be the same type")

        if not all([choice.shape == choices[0].shape for choice in choices[1:]]):
            raise ValueError("All choices must have the same shape")

        if not all([choice.dtype == choices[0].dtype for choice in choices[1:]]):
            raise ValueError("All choices must have the same dtype")

        if not all([choice.device == choices[0].device for choice in choices[1:]]):
            raise ValueError("All choices must have the same device")

        shape = choices[0].shape
        device = choices[0].device
        dtype = choices[0].dtype

        super().__init__(
            shape=shape, space=None, device=device, dtype=dtype, domain=None
        )

        self._choices = [choice.clone() for choice in choices]
        self.encode = self._encode_eager

    def _rand_idx(self):
        return torch.randint(0, len(self._choices), ()).item()

    def _sample(self, idx, spec_sample_fn) -> TensorDictBase:
        res = self._choices[idx]
        if isinstance(res, TensorSpec):
            return spec_sample_fn(res)
        else:
            return res

    def zero(self, shape: torch.Size = None) -> TensorDictBase:
        return self._sample(0, lambda x: x.zero(shape))

    def one(self, shape: torch.Size = None) -> TensorDictBase:
        return self._sample(min(1, len(self - 1)), lambda x: x.one(shape))

    def rand(self, shape: torch.Size = None) -> TensorDictBase:
        return self._sample(self._rand_idx(), lambda x: x.rand(shape))

    def is_in(self, val: torch.Tensor | TensorDictBase) -> bool:
        if isinstance(self._choices[0], TensorSpec):
            return any([choice.is_in(val) for choice in self._choices])
        else:
            return any([(choice == val).all() for choice in self._choices])

    def expand(self, *shape):
        return self.__class__([choice.expand(*shape) for choice in self._choices])

    def unsqueeze(self, dim: int):
        return self.__class__([choice.unsqueeze(dim) for choice in self._choices])

    def clone(self) -> Choice:
        return self.__class__([choice.clone() for choice in self._choices])

    def cardinality(self) -> int:
        if isinstance(self._choices[0], (NonTensorData, NonTensorStack)):
            return len(self._choices)
        else:
            return (
                torch.tensor([choice.cardinality() for choice in self._choices])
                .sum()
                .item()
            )

    def enumerate(self, use_mask: bool = False) -> list[Any]:
        return [s for choice in self._choices for s in choice.enumerate()]

    def _project(
        self, val: torch.Tensor | TensorDictBase
    ) -> torch.Tensor | TensorDictBase:
        raise NotImplementedError(
            "_project is not implemented for Choice. If this feature is required, please raise "
            "an issue on TorchRL github repo."
        )

    def _reshape(self, shape: torch.Size) -> T:
        return self.__class__(
            [choice.reshape(shape) for choice in self._choices],
        )

    def index(
        self, index: INDEX_TYPING, tensor_to_index: torch.Tensor | TensorDictBase
    ) -> torch.Tensor | TensorDictBase:
        raise NotImplementedError(
            "index is not implemented for Choice. If this feature is required, please raise "
            "an issue on TorchRL github repo."
        )

    @property
    def num_choices(self):
        """Number of choices for the spec."""
        return len(self._choices)

    def to(self, dest: torch.dtype | DEVICE_TYPING) -> Choice:
        return self.__class__([choice.to(dest) for choice in self._choices])

    def __eq__(self, other):
        if not isinstance(other, Choice):
            return False
        if self.num_choices != other.num_choices:
            return False
        return all(
            (s0 == s1).all()
            if isinstance(s0, torch.Tensor) or is_tensor_collection(s0)
            else s0 == s1
            for s0, s1 in zip(self._choices, other._choices)
        )


@dataclass(repr=False)
class Binary(Categorical):
    """A binary discrete tensor spec.

    A binary tensor spec encodes tensors of arbitrary size where the values are either 0 or 1 (or ``True`` or ``False``
    if the dtype it ``torch.bool``).

    Unlike :class:`OneHot`, `Binary` can have more than one non-null element along the last dimension.

    Args:
        n (int): length of the binary vector. If provided along with ``shape``, ``shape[-1]`` must match ``n``.
            If not provided, ``shape`` must be passed.

            .. warning:: the ``n`` argument from ``Binary`` must not be confused with the ``n`` argument from :class:`Categorical`
                or :class:`OneHot` which denotes the maximum number of elements that can be sampled.
                For clarity, use ``shape`` instead.

        shape (torch.Size, optional): total shape of the sampled tensors.
            If provided, the last dimension must match ``n``.
        device (str, int or torch.device, optional): device of the tensors.
        dtype (str or torch.dtype, optional): dtype of the tensors.
            Defaults to ``torch.int8``.

    Examples:
        >>> torch.manual_seed(0)
        >>> spec = Binary(n=4, shape=(2, 4))
        >>> print(spec.rand())
        tensor([[0, 1, 1, 0],
                [1, 1, 1, 1]], dtype=torch.int8)
        >>> spec = Binary(shape=(2, 4))
        >>> print(spec.rand())
        tensor([[1, 1, 1, 0],
                [0, 1, 0, 0]], dtype=torch.int8)
        >>> spec = Binary(n=4)
        >>> print(spec.rand())
        tensor([0, 0, 0, 1], dtype=torch.int8)

    """

    def __init__(
        self,
        n: int | None = None,
        shape: torch.Size | None = None,
        device: DEVICE_TYPING | None = None,
        dtype: str | torch.dtype = torch.int8,
    ):
        if n is None and shape is None:
            raise TypeError("Must provide either n or shape.")
        if n is None:
            n = shape[-1]
        if shape is None or not len(shape):
            shape = _size((n,))
        else:
            shape = _size(shape)
            if shape[-1] != n:
                raise ValueError(
                    f"The last value of the shape must match n for spec {self.__class__}. "
                    f"Got n={n} and shape={shape}."
                )
        super().__init__(n=2, shape=shape, device=device, dtype=dtype)
        self.encode = self._encode_eager

    def expand(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list, torch.Size)):
            shape = shape[0]
        if any(s1 != s2 and s2 != 1 for s1, s2 in zip(shape[-self.ndim :], self.shape)):
            raise ValueError(
                f"The last {self.ndim} of the expanded shape {shape} must match the"
                f"shape of the {self.__class__.__name__} spec in expand()."
            )
        return self.__class__(
            n=self.shape[-1], shape=shape, device=self.device, dtype=self.dtype
        )

    def _reshape(self, shape):
        return self.__class__(
            n=self.shape[-1], shape=shape, device=self.device, dtype=self.dtype
        )

    def _unflatten(self, dim, sizes):
        shape = (
            torch.zeros(_remove_neg_shapes(self.shape), device="meta")
            .unflatten(dim, sizes)
            .shape
        )
        return self.__class__(
            n=self.shape[-1], shape=shape, device=self.device, dtype=self.dtype
        )

    def squeeze(self, dim=None):
        shape = _squeezed_shape(_remove_neg_shapes(self.shape), dim)
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

    def unbind(self, dim: int = 0):
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

    def to(self, dest: torch.dtype | DEVICE_TYPING) -> Binary:
        if isinstance(dest, torch.dtype):
            dest_dtype = dest
            dest_device = self.device
        elif dest is None:
            return self
        else:
            dest_dtype = self.dtype
            dest_device = torch.device(dest)
        if dest_device == self.device and dest_dtype == self.dtype:
            return self
        return self.__class__(
            n=self.shape[-1], shape=self.shape, device=dest_device, dtype=dest_dtype
        )

    def clone(self) -> Binary:
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
            shape=_size(indexed_shape + [self.shape[-1]]),
            device=self.device,
            dtype=self.dtype,
        )

    def __eq__(self, other):
        if not isinstance(other, Binary):
            if isinstance(other, Categorical):
                return (
                    other.n == 2
                    and other.device == self.device
                    and other.shape == self.shape
                    and other.dtype == self.dtype
                )
            return False
        return super().__eq__(other)


@dataclass(repr=False)
class MultiCategorical(Categorical):
    """A concatenation of discrete tensor spec.

    Args:
        nvec (iterable of integers or torch.Tensor): cardinality of each of the elements of
            the tensor. Can have several axes.
        shape (torch.Size, optional): total shape of the sampled tensors.
            If provided, the last m dimensions must match nvec.shape.
        device (str, int or torch.device, optional): device of
            the tensors.
        dtype (str or torch.dtype, optional): dtype of the tensors.
        remove_singleton (bool, optional): if ``True``, singleton samples (of size [1])
            will be squeezed. Defaults to ``True``.
        mask (torch.Tensor or None): mask some of the possible outcomes when a
            sample is taken. See :meth:`update_mask` for more information.

    Examples:
        >>> ts = MultiCategorical((3, 2, 3))
        >>> ts.is_in(torch.tensor([2, 0, 1]))
        True
        >>> ts.is_in(torch.tensor([2, 10, 1]))
        False
    """

    def __init__(
        self,
        nvec: Sequence[int] | torch.Tensor | int,
        shape: torch.Size | None = None,
        device: DEVICE_TYPING | None = None,
        dtype: str | torch.dtype | None = torch.int64,
        mask: torch.Tensor | None = None,
        remove_singleton: bool = True,
    ):
        if not isinstance(nvec, torch.Tensor):
            nvec = torch.tensor(nvec)
        if nvec.ndim < 1:
            nvec = nvec.unsqueeze(0)
        self.nvec = nvec
        dtype, device = _default_dtype_and_device(
            dtype, device, allow_none_device=False
        )
        if shape is None:
            shape = nvec.shape
        else:
            shape = _size(shape)
            if shape[-1] != nvec.shape[-1]:
                raise ValueError(
                    f"The last value of the shape must match nvec.shape[-1] for transform of type {self.__class__}. "
                    f"Got nvec.shape[-1]={sum(nvec)} and shape={shape}."
                )

        self.nvec = self.nvec.expand(_remove_neg_shapes(shape))

        space = BoxList.from_nvec(self.nvec)
        super(Categorical, self).__init__(
            shape, space, device, dtype, domain="discrete"
        )
        self.update_mask(mask)
        self.remove_singleton = remove_singleton
        self.encode = self._encode_eager

    def enumerate(self, use_mask: bool = False) -> torch.Tensor:
        if use_mask:
            raise NotImplementedError()
        if self.mask is not None:
            raise RuntimeError(
                "Cannot enumerate a masked TensorSpec. Submit an issue on github if this feature is requested."
            )
        if self.nvec._base.ndim == 1:
            nvec = self.nvec._base
        else:
            # we have to use unique() to isolate the nvec
            nvec = self.nvec.view(-1, self.nvec.shape[-1]).unique(dim=0).squeeze(0)
            if nvec.ndim > 1:
                raise ValueError(
                    f"Cannot call enumerate on heterogeneous nvecs: unique nvecs={nvec}."
                )
        arange = torch.meshgrid(
            *[torch.arange(n, device=self.device, dtype=self.dtype) for n in nvec],
            indexing="ij",
        )
        arange = torch.stack([arange_.reshape(-1) for arange_ in arange], dim=-1)
        arange = arange.view(arange.shape[0], *(1,) * (self.ndim - 1), self.shape[-1])
        arange = arange.expand(arange.shape[0], *self.shape)
        return arange

    def cardinality(self) -> int:
        return self.nvec._base.prod()

    def update_mask(self, mask):
        """Sets a mask to prevent some of the possible outcomes when a sample is taken.

        The mask can also be set during initialization of the spec.

        Args:
            mask (torch.Tensor or None): boolean mask. If None, the mask is
                disabled. Otherwise, the shape of the mask must be expandable to
                the shape of the equivalent one-hot spec. ``False`` masks an
                outcome and ``True`` leaves the outcome unmasked. If all of the
                possible outcomes are masked, then an error is raised when a
                sample is taken.

        Examples:
            >>> torch.manual_seed(0)
            >>> mask = torch.tensor([False, False, True,
            ...                      True, True])
            >>> ts = MultiCategorical((3, 2), (5, 2,), dtype=torch.int64, mask=mask)
            >>> # All but one of the three possible outcomes for the first
            >>> # group are masked, but neither of the two possible
            >>> # outcomes for the second group are masked.
            >>> ts.rand()
            tensor([[2, 1],
                    [2, 0],
                    [2, 1],
                    [2, 1],
                    [2, 1]])
        """
        if mask is not None:
            try:
                mask = mask.expand(_remove_neg_shapes(*self.shape[:-1], mask.shape[-1]))
            except RuntimeError as err:
                raise RuntimeError("Cannot expand mask to the desired shape.") from err
            if mask.dtype != torch.bool:
                raise ValueError("Only boolean masks are accepted.")
        self.mask = mask

    def to(self, dest: torch.dtype | DEVICE_TYPING) -> MultiCategorical:
        if isinstance(dest, torch.dtype):
            dest_dtype = dest
            dest_device = self.device
        elif dest is None:
            return self
        else:
            dest_dtype = self.dtype
            dest_device = torch.device(dest)
        if dest_device == self.device and dest_dtype == self.dtype:
            return self
        mask = self.mask.to(dest) if self.mask is not None else None
        return self.__class__(
            nvec=self.nvec.to(dest),
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

    def clone(self) -> MultiCategorical:
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

    def rand(self, shape: torch.Size | None = None) -> torch.Tensor:
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
        if self.remove_singleton and self.shape == _size([1]):
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
                Categorical(n=n, shape=shape, device=device, dtype=dtype, mask=_mask)
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
            vals = val.unbind(-1)
            splits = self._split_self()
            if not len(vals) == len(splits):
                return False
            return all(spec.is_in(val) for (val, spec) in zip(vals, splits))

        if val.ndim < 1:
            val = val.unsqueeze(0)
        shape = _remove_neg_shapes(self.shape)
        shape = torch.broadcast_shapes(shape, val.shape)
        if shape != val.shape:
            return False
        if self.dtype != val.dtype:
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
    ) -> MultiOneHot | torch.Tensor:
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
                torch.nn.functional.one_hot(val[..., i], n).bool()
                for i, n in enumerate(self.nvec)
            ],
            -1,
        ).to(self.device)

    def to_one_hot_spec(self) -> MultiOneHot:
        """Converts the spec to the equivalent one-hot spec."""
        if self.ndim > 1:
            return torch.stack([spec.to_one_hot_spec() for spec in self.unbind(0)])
        nvec = [_space.n for _space in self.space]
        return MultiOneHot(
            nvec,
            device=self.device,
            shape=[*self.shape[:-1], sum(nvec)],
            mask=self.mask,
        )

    def to_categorical(self, val: torch.Tensor, safe: bool = None) -> MultiCategorical:
        """Not op for MultiCategorical."""
        return val

    def to_categorical_spec(self) -> MultiCategorical:
        """Not op for MultiCategorical."""
        return self

    def expand(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list, torch.Size)):
            shape = shape[0]
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

    def _reshape(self, shape):
        mask = (
            self.mask.reshape(*shape, self.mask.shape[-1])
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

    def _unflatten(self, dim, sizes):
        shape = torch.zeros(self.shape, device="meta").unflatten(dim, sizes).shape
        return self._reshape(shape)

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

    def unbind(self, dim: int = 0):
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


class Composite(TensorSpec):
    """A composition of TensorSpecs.

    If a ``TensorSpec`` is the set-description of Tensor category, the ``Composite`` class is akin to
    the :class:`~tensordict.TensorDict` class. Like :class:`~tensordict.TensorDict`, it has a ``shape`` (akin to the
    ``TensorDict``'s ``batch_size``) and an optional ``device``.

    Args:
        *args: if an unnamed argument is passed, it must be a dictionary with keys
            matching the expected keys to be found in the :obj:`Composite` object.
            This is useful to build nested CompositeSpecs with tuple indices.
        **kwargs (key (str): value (TensorSpec)): dictionary of tensorspecs
            to be stored. Values can be None, in which case is_in will be assumed
            to be ``True`` for the corresponding tensors, and :obj:`project()` will have no
            effect. `spec.encode` cannot be used with missing values.

    Attributes:
        device (torch.device or None): if not specified, the device of the composite
            spec is ``None`` (as it is the case for TensorDicts). A non-none device
            constraints all leaves to be of the same device. On the other hand,
            a ``None`` device allows leaves to have different devices. Defaults
            to ``None``.
        shape (torch.Size): the leading shape of all the leaves. Equivalent
            to the batch-size of the corresponding tensordicts.
        data_cls (type, optional): the tensordict subclass (TensorDict, TensorClass, tensorclass...) that should be
            enforced in the env. Defaults to ``None``.

    Examples:
        >>> pixels_spec = Bounded(
        ...     low=torch.zeros(4, 3, 32, 32),
        ...     high=torch.ones(4, 3, 32, 32),
        ...     dtype=torch.uint8
        ... )
        >>> observation_vector_spec = Bounded(
        ...     low=torch.zeros(4, 33),
        ...     high=torch.ones(4, 33),
        ...     dtype=torch.float)
        >>> composite_spec = Composite(
        ...     pixels=pixels_spec,
        ...     observation_vector=observation_vector_spec,
        ...     shape=(4,)
        ... )
        >>> composite_spec
        Composite(
            pixels: BoundedDiscrete(
                shape=torch.Size([4, 3, 32, 32]),
                space=ContinuousBox(
                    low=Tensor(shape=torch.Size([4, 3, 32, 32]), device=cpu, dtype=torch.uint8, contiguous=True),
                    high=Tensor(shape=torch.Size([4, 3, 32, 32]), device=cpu, dtype=torch.uint8, contiguous=True)),
                device=cpu,
                dtype=torch.uint8,
                domain=discrete),
            observation_vector: BoundedContinuous(
                shape=torch.Size([4, 33]),
                space=ContinuousBox(
                    low=Tensor(shape=torch.Size([4, 33]), device=cpu, dtype=torch.float32, contiguous=True),
                    high=Tensor(shape=torch.Size([4, 33]), device=cpu, dtype=torch.float32, contiguous=True)),
                device=cpu,
                dtype=torch.float32,
                domain=continuous),
            device=None,
            shape=torch.Size([4]))
        >>> td = composite_spec.rand()
        >>> td
        TensorDict(
            fields={
                observation_vector: Tensor(shape=torch.Size([4, 33]), device=cpu, dtype=torch.float32, is_shared=False),
                pixels: Tensor(shape=torch.Size([4, 3, 32, 32]), device=cpu, dtype=torch.uint8, is_shared=False)},
            batch_size=torch.Size([4]),
            device=None,
            is_shared=False)
        >>> # we can build a nested composite spec using unnamed arguments
        >>> print(Composite({("a", "b"): None, ("a", "c"): None}))
        Composite(
            a: Composite(
                b: None,
                c: None,
                device=None,
                shape=torch.Size([])),
            device=None,
            shape=torch.Size([]))

    """

    shape: torch.Size
    domain: str = "composite"

    SPEC_HANDLED_FUNCTIONS = {}

    @classmethod
    def __new__(cls, *args, **kwargs):
        cls._device = None
        cls._is_locked = False
        return super().__new__(cls)

    def __init__(
        self,
        *args,
        shape: torch.Size = None,
        device: torch.device = None,
        data_cls: type | None = None,
        **kwargs,
    ):
        # For compatibility with TensorDict
        batch_size = kwargs.pop("batch_size", None)
        if batch_size is not None:
            if shape is not None:
                raise TypeError("Cannot specify both batch_size and shape.")
            shape = batch_size

        if shape is None:
            shape = _size(())
        self._shape = _size(shape)
        self._specs = {}

        _device = (
            _make_ordinal_device(torch.device(device)) if device is not None else device
        )
        self._device = _device
        if len(args):
            if len(args) > 1:
                raise RuntimeError(
                    "Got multiple arguments, when at most one is expected for Composite."
                )
            argdict = args[0]
            if not isinstance(argdict, (dict, Composite)):
                raise RuntimeError(
                    f"Expected a dictionary of specs, but got an argument of type {type(argdict)}."
                )
            for k, item in argdict.items():
                if isinstance(item, dict):
                    item = Composite(item, shape=shape, device=_device)
                self[k] = item
        for k, item in kwargs.items():
            self[k] = item
        self.data_cls = data_cls
        self.encode = self._encode_eager
        self._encode_memo_dict = {}

    def memoize_encode(self, mode: bool = True) -> None:
        super().memoize_encode(mode=mode)
        for spec in self._specs.values():
            if spec is None:
                continue
            spec.memoize_encode(mode=mode)

    def erase_memoize_cache(self) -> None:
        self._encode_memo_dict.clear()
        for spec in self._specs.values():
            if spec is None:
                continue
            spec.erase_memoize_cache()

    @property
    def batch_size(self):
        return self._shape

    @batch_size.setter
    def batch_size(self, value: torch.Size):
        self._shape = value

    @property
    def shape(self):
        return self._shape

    @shape.setter
    def shape(self, value: torch.Size):
        if self.locked:
            raise RuntimeError("Cannot modify shape of locked composite spec.")
        for key, spec in self.items():
            if isinstance(spec, Composite):
                if spec.shape[: len(value)] != value:
                    spec.shape = value
            elif spec is not None:
                if spec.shape[: len(value)] != value:
                    raise ValueError(
                        f"The shape of the spec and the Composite mismatch during shape resetting: the "
                        f"{self.ndim} first dimensions should match but got self['{key}'].shape={spec.shape} and "
                        f"Composite.shape={self.shape}."
                    )
        self._shape = _size(value)

    def _project(
        self, val: torch.Tensor | TensorDictBase
    ) -> torch.Tensor | TensorDictBase:
        if self.data_cls is None:
            cls = TensorDict
        else:
            cls = self.data_cls
        return cls.from_dict(
            {k: item._project(val[k]) for k, item in self.items()},
            batch_size=self.shape,
            device=self.device,
        )

    def index(
        self, index: INDEX_TYPING, tensor_to_index: torch.Tensor | TensorDictBase
    ) -> torch.Tensor | TensorDictBase:
        raise NotImplementedError("`index` is not implemented for Composite specs.")

    def is_empty(self, recurse: bool = False):
        """Whether the composite spec contains specs or not.

        Args:
            recurse (bool): whether to recursively assess if the spec is empty.
                If ``True``, will return ``True`` if there are no leaves. If ``False``
                (default) will return whether there is any spec defined at the root level.

        """
        if recurse:
            for spec in self._specs.values():
                if spec is None:
                    continue
                if isinstance(spec, Composite) and spec.is_empty(recurse=True):
                    continue
                return False
        return len(self._specs) == 0

    @property
    def ndim(self):
        return self.ndimension()

    def ndimension(self):
        return len(self.shape)

    def pop(self, key: NestedKey, default: Any = NO_DEFAULT) -> Any:
        """Removes and returns the value associated with the specified key from the composite spec.

        This method searches for the given key in the composite spec, removes it, and returns its associated value.
        If the key is not found, it returns the provided default value if specified, otherwise raises a `KeyError`.

        Args:
            key (NestedKey):
                The key to be removed from the composite spec. It can be a single key or a nested key.
            default (Any, optional):
                The value to return if the specified key is not found in the composite spec.
                If not provided and the key is not found, a `KeyError` is raised.

        Returns:
            Any: The value associated with the specified key that was removed from the composite spec.

        Raises:
            KeyError: If the specified key is not found in the composite spec and no default value is provided.
        """
        key = unravel_key(key)
        if key in self.keys(True, True):
            result = self[key]
            del self[key]
            return result
        elif default is not NO_DEFAULT:
            return default
        raise KeyError(f"{key} not found in composite spec.")

    def separates(self, *keys: NestedKey, default: Any = None) -> Composite:
        """Splits the composite spec by extracting specified keys and their associated values into a new composite spec.

        This method iterates over the provided keys, removes them from the current composite spec, and adds them to a new
        composite spec. If a key is not found, the specified default value is used. The new composite spec is returned.

        Args:
            *keys (NestedKey):
                One or more keys to be extracted from the composite spec. Each key can be a single key or a nested key.
            default (Any, optional):
                The value to use if a specified key is not found in the composite spec. Defaults to `None`.

        Returns:
            Composite: A new composite spec containing the extracted keys and their associated values.

        Note:
            If none of the specified keys are found, the method returns `None`.
        """
        out = None
        for key in keys:
            result = self.pop(key, default=default)
            if result is not None:
                if out is None:
                    out = Composite(batch_size=self.batch_size, device=self.device)
                out[key] = result
        return out

    def set(self, name: str, spec: TensorSpec) -> Composite:
        """Sets a spec in the Composite spec."""
        if not isinstance(name, str):
            self[name] = spec
            return self
        if self.locked:
            raise RuntimeError("Cannot modify a locked Composite.")
        if spec is not None and self.device is not None and spec.device != self.device:
            if spec.device is None:
                # We make a clone not to mess up the spec that was provided.
                # in set() we do the same for shape - these two ops should be grouped.
                # we don't care about the overhead of cloning twice though because in theory
                # we don't set specs often.
                spec = spec.clone().to(self._device)
            else:
                raise RuntimeError(
                    f"Setting a new attribute ({name}) with spec type {type(spec).__name__} on another device ({spec.device} against {self.device}). "
                    f"All devices of Composite must match."
                )
        if spec is not None:
            shape = spec.shape
            if shape[: self.ndim] != self.shape:
                if (
                    isinstance(spec, (Composite, NonTensor))
                    and spec.ndim < self.ndim
                    and self.shape[: spec.ndim] == spec.shape
                ):
                    # Try to set the composite shape
                    spec = spec.clone()
                    spec.shape = self.shape
                else:
                    raise ValueError(
                        f"The shapes of the spec {type(spec).__name__} and the {type(self).__name__} mismatch: the first "
                        f"{self.ndim} dimensions should match but got spec.shape={spec.shape} and "
                        f"Composite.shape={self.shape}."
                    )
        self._specs[name] = spec
        return self

    @property
    def device(self) -> DEVICE_TYPING:
        return self._device

    @device.setter
    def device(self, device: DEVICE_TYPING):
        if device is None and self._device is not None:
            raise RuntimeError(
                "To erase the device of a composite spec, call " "spec.clear_device_()."
            )
        device = _make_ordinal_device(torch.device(device))
        self.to(device)

    def clear_device_(self):
        """Clears the device of the Composite."""
        self._device = None
        for spec in self._specs.values():
            if spec is None:
                continue
            spec.clear_device_()
        return self

    def __getitem__(self, idx):
        """Indexes the current Composite based on the provided index."""
        if isinstance(idx, (str, tuple)):
            idx_unravel = unravel_key(idx)
        else:
            idx_unravel = ()
        if idx_unravel:
            if isinstance(idx_unravel, tuple):
                return self[idx[0]][idx[1:]]
            if idx_unravel in {"shape", "device", "dtype", "space"}:
                raise AttributeError(f"Composite has no key {idx_unravel}")
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
                        Binary,
                        MultiCategorical,
                        OneHot,
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

    def get(self, item, default=NO_DEFAULT):
        """Gets an item from the Composite.

        If the item is absent, a default value can be passed.

        """
        try:
            return self[item]
        except KeyError:
            if item is not NO_DEFAULT:
                return default
            raise

    def __setitem__(self, key, value):
        dest = self
        if isinstance(key, tuple) and len(key) > 1:
            while key[0] not in dest.keys():
                dest[key[0]] = dest = Composite(shape=self.shape, device=self.device)
                if len(key) > 2:
                    key = key[1:]
                else:
                    break
            else:
                dest = self[key[0]]
            dest[key[1:]] = value
            return
        elif isinstance(key, tuple):
            self[key[0]] = value
            return
        elif not isinstance(key, str):
            raise TypeError(f"Got key of type {type(key)} when a string was expected.")
        if key in {"shape", "device", "dtype", "space"}:
            raise AttributeError(f"Composite[{key}] cannot be set")
        if isinstance(value, dict):
            value = Composite(value, device=self._device, shape=self.shape)

        self.set(key, value)

    def __iter__(self):
        yield from self._specs

    def __delitem__(self, key: NestedKey) -> None:
        key = unravel_key(key)
        if isinstance(key, tuple):
            spec = self[key[:-1]]
            del spec[key[-1]]
            return
        elif not isinstance(key, str):
            raise TypeError(
                f"Got key of type {type(key)} when a string or a tuple of strings was expected."
            )

        if key in {"shape", "device", "dtype", "space"}:
            raise ValueError(f"Key name {key} is prohibited.")
        del self._specs[key]

    def _encode_eager(
        self, vals: dict[str, Any], *, ignore_device: bool = False
    ) -> dict[str, torch.Tensor]:
        if isinstance(vals, TensorDict):
            out = vals.empty()  # create and empty tensordict similar to vals
        elif self.data_cls is not None:
            out = {}
        else:
            out = TensorDict._new_unsafe({}, _size([]))
        for key, item in vals.items():
            if item is None:
                raise RuntimeError(
                    "Composite.encode cannot be used with missing values."
                )
            try:
                out[key] = self[key].encode(item, ignore_device=ignore_device)
            except KeyError:
                raise KeyError(
                    f"The Composite instance with keys {self.keys()} does not have a '{key}' key."
                )
            except RuntimeError as err:
                raise RuntimeError(
                    f"Encoding key {key} raised a RuntimeError. Scroll up to know more."
                ) from err
        if self.data_cls is not None:
            return self.data_cls.from_dict(out)
        return out

    def _encode_memo(
        self, vals: dict[str, Any], *, ignore_device: bool = False
    ) -> dict[str, torch.Tensor]:
        funcs = self._encode_memo_dict.get(ignore_device)
        if funcs is not None:
            return funcs(vals)
        funcs = []
        vals_orig = vals
        if isinstance(vals, TensorDictBase):

            def empty(vals):
                out = vals.empty()  # create and empty tensordict similar to vals
                return vals, out

        elif self.data_cls is not None:

            def empty(vals):
                out = {}
                return vals, out

        else:

            def empty(vals):
                out = TensorDict._new_unsafe({}, _size([]))
                return vals, out

        funcs.append(empty)

        def populate(vals_out):
            vals, out = vals_out
            for key, item in vals.items():
                if item is None:
                    raise RuntimeError(
                        "Composite.encode cannot be used with missing values."
                    )
                try:
                    out[key] = self[key].encode(item, ignore_device=ignore_device)
                except KeyError:
                    raise KeyError(
                        f"The Composite instance with keys {self.keys()} does not have a '{key}' key."
                    )
                except RuntimeError as err:
                    raise RuntimeError(
                        f"Encoding key {key} raised a RuntimeError. Scroll up to know more."
                    ) from err
            return out

        funcs.append(populate)
        if self.data_cls is not None:
            funcs.append(self.data_cls.from_dict)
        if len(funcs) == 0:
            self._encode_memo_dict[ignore_device] = lambda x: x
        elif len(funcs) == 1:
            self._encode_memo_dict[ignore_device] = funcs[0]
        else:
            self._encode_memo_dict[ignore_device] = functools.partial(
                functools.reduce, lambda x, f: f(x), funcs
            )
        return self._encode_memo_dict[ignore_device](vals_orig)

    def __repr__(self) -> str:
        sub_str = [
            indent(f"{k}: {str(item)}", 4 * " ") for k, item in self._specs.items()
        ]
        sub_str = ",\n".join(sub_str)
        return f"Composite(\n{sub_str},\n    device={self._device},\n    shape={self.shape})"

    def type_check(
        self,
        value: torch.Tensor | TensorDictBase,
        selected_keys: str | Sequence[str] | None = None,
    ):
        if isinstance(value, torch.Tensor) and isinstance(selected_keys, str):
            value = {selected_keys: value}
            selected_keys = [selected_keys]

        for _key in self.keys():
            if self[_key] is not None and (
                selected_keys is None or _key in selected_keys
            ):
                self._specs[_key].type_check(value[_key], _key)

    def is_in(self, val: dict | TensorDictBase) -> bool:
        # TODO: make warnings for these
        # if val.device != self.device:
        #     print(val.device, self.device)
        #     return False
        # if val.shape[-self.ndim:] != self.shape:
        #     return False
        if self.data_cls is not None and type(val) != self.data_cls:
            return False
        for key, item in self._specs.items():
            if item is None or (isinstance(item, Composite) and item.is_empty()):
                continue
            val_item = val.get(key, NO_DEFAULT)
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

    def rand(self, shape: torch.Size = None) -> TensorDictBase:
        if shape is None:
            shape = _size([])
        _dict = {}
        for key, item in self.items():
            if item is not None:
                _dict[key] = item.rand(shape)
        if self.data_cls is None:
            cls = TensorDict
        else:
            cls = self.data_cls
        # No need to run checks since we know Composite is compliant with
        # TensorDict requirements
        return cls.from_dict(
            _dict,
            batch_size=_size([*shape, *_remove_neg_shapes(self.shape)]),
            device=self.device,
        )

    def keys(
        self,
        include_nested: bool = False,
        leaves_only: bool = False,
        *,
        is_leaf: Callable[[type], bool] | None = None,
    ) -> _CompositeSpecKeysView:  # noqa: D417
        """Keys of the Composite.

        The keys argument reflect those of :class:`tensordict.TensorDict`.

        Args:
            include_nested (bool, optional): if ``False``, the returned keys will not be nested. They will
                represent only the immediate children of the root, and not the whole nested sequence, i.e. a
                :obj:`Composite(next=Composite(obs=None))` will lead to the keys
                :obj:`["next"]. Default is ``False``, i.e. nested keys will not
                be returned.
            leaves_only (bool, optional): if ``False``, the values returned
                will contain every level of nesting, i.e. a :obj:`Composite(next=Composite(obs=None))`
                will lead to the keys :obj:`["next", ("next", "obs")]`.
                Default is ``False``.

        Keyword Args:
            is_leaf (callable, optional): reads a type and returns a boolean indicating if that type
                should be seen as a leaf. By default, all non-Composite nodes are considered as
                leaves.

        """
        return _CompositeSpecItemsView(
            self,
            include_nested=include_nested,
            leaves_only=leaves_only,
            is_leaf=is_leaf,
        )._keys()

    def items(
        self,
        include_nested: bool = False,
        leaves_only: bool = False,
        *,
        is_leaf: Callable[[type], bool] | None = None,
    ) -> _CompositeSpecItemsView:  # noqa: D417
        """Items of the Composite.

        Args:
            include_nested (bool, optional): if ``False``, the returned keys will not be nested. They will
                represent only the immediate children of the root, and not the whole nested sequence, i.e. a
                :obj:`Composite(next=Composite(obs=None))` will lead to the keys
                :obj:`["next"]. Default is ``False``, i.e. nested keys will not
                be returned.
            leaves_only (bool, optional): if ``False``, the values returned
                will contain every level of nesting, i.e. a :obj:`Composite(next=Composite(obs=None))`
                will lead to the keys :obj:`["next", ("next", "obs")]`.
                Default is ``False``.

        Keyword Args:
            is_leaf (callable, optional): reads a type and returns a boolean indicating if that type
                should be seen as a leaf. By default, all non-Composite nodes are considered as
                leaves.
        """
        return _CompositeSpecItemsView(
            self,
            include_nested=include_nested,
            leaves_only=leaves_only,
            is_leaf=is_leaf,
        )

    def values(
        self,
        include_nested: bool = False,
        leaves_only: bool = False,
        *,
        is_leaf: Callable[[type], bool] | None = None,
    ) -> _CompositeSpecValuesView:  # noqa: D417
        """Values of the Composite.

        Args:
            include_nested (bool, optional): if ``False``, the returned keys will not be nested. They will
                represent only the immediate children of the root, and not the whole nested sequence, i.e. a
                :obj:`Composite(next=Composite(obs=None))` will lead to the keys
                :obj:`["next"]. Default is ``False``, i.e. nested keys will not
                be returned.
            leaves_only (bool, optional): if ``False``, the values returned
                will contain every level of nesting, i.e. a :obj:`Composite(next=Composite(obs=None))`
                will lead to the keys :obj:`["next", ("next", "obs")]`.
                Default is ``False``.

        Keyword Args:
            is_leaf (callable, optional): reads a type and returns a boolean indicating if that type
                should be seen as a leaf. By default, all non-Composite nodes are considered as
                leaves.
        """
        return _CompositeSpecItemsView(
            self,
            include_nested=include_nested,
            leaves_only=leaves_only,
            is_leaf=is_leaf,
        )._values()

    def _reshape(self, shape):
        _specs = {
            key: val.reshape((*shape, *val.shape[self.ndimension() :]))
            for key, val in self._specs.items()
        }
        return self.__class__(
            _specs, shape=shape, device=self.device, data_cls=self.data_cls
        )

    def _unflatten(self, dim, sizes):
        shape = torch.zeros(self.shape, device="meta").unflatten(dim, sizes).shape
        return self._reshape(shape)

    def __len__(self):
        return len(self.keys())

    def to(self, dest: torch.dtype | DEVICE_TYPING) -> Composite:
        if dest is None:
            return self
        if isinstance(dest, torch.dtype):
            items = list(self.items())
            kwargs = {}
            for key, value in items:
                if value is None:
                    kwargs[key] = value
                    continue
                kwargs[key] = value.to(dest)
            return self.__class__(
                **kwargs, device=self.device, shape=self.shape, data_cls=self.data_cls
            )
        if not isinstance(dest, (str, int, torch.device)):
            raise ValueError(
                "Only device/dtype casting is allowed with specs of type Composite."
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
        return self.__class__(
            **kwargs, device=_device, shape=self.shape, data_cls=self.data_cls
        )

    def clone(self) -> Composite:
        """Clones the Composite spec.

        Locked specs will not produce locked clones.
        """
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
            data_cls=self.data_cls,
        )

    def cardinality(self) -> int:
        n = None
        for spec in self.values():
            if spec is None:
                continue
            if n is None:
                n = 1
            n = n * spec.cardinality()
        if n is None:
            n = 0
        return n

    def enumerate(self, use_mask: bool = False) -> TensorDictBase:
        # We are going to use meshgrid to create samples of all the subspecs in here
        #  but first let's get rid of the batch size, we'll put it back later
        self_without_batch = self
        while self_without_batch.ndim:
            self_without_batch = self_without_batch[0]
        samples = {
            key: spec.enumerate(use_mask) for key, spec in self_without_batch.items()
        }
        if self.data_cls is not None:
            cls = self.data_cls
        else:
            cls = TensorDict
        if samples:
            idx_rep = torch.meshgrid(
                *(torch.arange(s.shape[0]) for s in samples.values()), indexing="ij"
            )
            idx_rep = tuple(idx.reshape(-1) for idx in idx_rep)
            samples = {
                key: sample[idx]
                for ((key, sample), idx) in zip(samples.items(), idx_rep)
            }
            samples = cls.from_dict(
                samples, batch_size=idx_rep[0].shape[:1], device=self.device
            )
            # Expand
            if self.ndim:
                samples = samples.reshape(-1, *(1,) * self.ndim)
                samples = samples.expand(samples.shape[0], *self.shape)
        else:
            samples = cls.from_dict({}, batch_size=self.shape, device=self.device)
        return samples

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
            data_cls=self.data_cls,
        )

    def to_numpy(self, val: TensorDict, safe: bool = None) -> dict:
        return {key: self[key].to_numpy(val) for key, val in val.items()}

    def zero(self, shape: torch.Size = None) -> TensorDictBase:
        if shape is None:
            shape = _size([])
        try:
            device = self.device
        except RuntimeError:
            device = self._device

        if self.data_cls is not None:
            cls = self.data_cls
        else:
            cls = TensorDict

        return cls.from_dict(
            {
                key: self[key].zero(shape)
                for key in self.keys(True)
                if isinstance(key, str) and self[key] is not None
            },
            batch_size=_size([*shape, *self._safe_shape]),
            device=device,
        )

    def __eq__(self, other):
        return (
            type(self) == type(other)
            and self.shape == other.shape
            and self._device == other._device
            and set(self._specs.keys()) == set(other._specs.keys())
            and all((self._specs[key] == spec) for (key, spec) in other._specs.items())
            and other.data_cls == self.data_cls
        )

    def update(self, dict_or_spec: Composite | dict[str, TensorSpec]) -> None:
        for key, item in dict_or_spec.items():
            if key in self.keys(True) and isinstance(self[key], Composite):
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
        if any(s1 != s2 and s2 != 1 for s1, s2 in zip(shape[-self.ndim :], self.shape)):
            raise ValueError(
                f"The last {self.ndim} of the expanded shape {shape} must match the "
                f"shape of the {self.__class__.__name__} spec in expand()."
            )
        try:
            device = self.device
        except RuntimeError:
            device = self._device
        specs = {
            key: value.expand((*shape, *value.shape[self.ndim :]))
            if value is not None
            else None
            for key, value in tuple(self.items())
        }
        out = Composite(
            specs,
            shape=shape,
            device=device,
            data_cls=self.data_cls,
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

            return self.__class__(
                {key: value.squeeze(dim) for key, value in self.items()},
                shape=shape,
                device=device,
                data_cls=self.data_cls,
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

        return self.__class__(
            {
                key: value.unsqueeze(dim) if value is not None else None
                for key, value in self.items()
            },
            shape=shape,
            device=device,
            data_cls=self.data_cls,
        )

    def unbind(self, dim: int = 0):
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
                data_cls=self.data_cls,
            )
            for i in range(self.shape[dim])
        )

    # Locking functionality
    @property
    def is_locked(self) -> bool:
        return self._is_locked

    @is_locked.setter
    def is_locked(self, value: bool) -> None:
        if value:
            self.lock_()
        else:
            self.unlock_()

    def __getstate__(self):
        result = self.__dict__.copy()
        __lock_parents_weakrefs = result.pop("__lock_parents_weakrefs", None)
        if __lock_parents_weakrefs is not None:
            result["_lock_recurse"] = True
        return result

    def __setstate__(self, state):
        _lock_recurse = state.pop("_lock_recurse", False)
        for key, value in state.items():
            setattr(self, key, value)
        if self._is_locked:
            self._is_locked = False
            self.lock_(recurse=_lock_recurse)

    def _propagate_lock(
        self, *, recurse: bool, lock_parents_weakrefs=None, is_compiling
    ):
        """Registers the parent composite that handles the lock."""
        self._is_locked = True
        if lock_parents_weakrefs is not None:
            lock_parents_weakrefs = [
                ref
                for ref in lock_parents_weakrefs
                if not any(refref is ref for refref in self._lock_parents_weakrefs)
            ]
        if not is_compiling:
            is_root = lock_parents_weakrefs is None
            if is_root:
                lock_parents_weakrefs = []
            else:
                self._lock_parents_weakrefs = (
                    self._lock_parents_weakrefs + lock_parents_weakrefs
                )
            lock_parents_weakrefs = list(lock_parents_weakrefs)
            lock_parents_weakrefs.append(weakref.ref(self))

        if recurse:
            for value in self.values():
                if isinstance(value, Composite):
                    value._propagate_lock(
                        recurse=True,
                        lock_parents_weakrefs=lock_parents_weakrefs,
                        is_compiling=is_compiling,
                    )

    @property
    def _lock_parents_weakrefs(self):
        _lock_parents_weakrefs = self.__dict__.get("__lock_parents_weakrefs")
        if _lock_parents_weakrefs is None:
            self.__dict__["__lock_parents_weakrefs"] = []
            _lock_parents_weakrefs = self.__dict__["__lock_parents_weakrefs"]
        return _lock_parents_weakrefs

    @_lock_parents_weakrefs.setter
    def _lock_parents_weakrefs(self, value: list):
        self.__dict__["__lock_parents_weakrefs"] = value

    def lock_(self, recurse: bool | None = None) -> T:
        """Locks the Composite and prevents modification of its content.

        The recurse argument control whether the lock will be propagated to sub-specs.
        The current default is ``False`` but it will be turned to ``True`` for consistency
        with the TensorDict API in v0.8.

        Examples:
            >>> shape = [3, 4, 5]
            >>> spec = Composite(
            ...         a=Composite(
            ...         b=Composite(shape=shape[:3], device="cpu"), shape=shape[:2]
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
        if self.is_locked:
            return self
        is_comp = is_compiling()
        if is_comp:
            # TODO: See what to do when compiling
            pass
        if recurse is None:
            recurse = True
        self._propagate_lock(recurse=recurse, is_compiling=is_comp)
        return self

    def _propagate_unlock(self, recurse: bool):
        # if we end up here, we can clear the graph associated with this td
        self._is_locked = False

        self._is_shared = False
        self._is_memmap = False

        if recurse:
            sub_specs = []
            for value in self.values():
                if isinstance(value, Composite):
                    sub_specs.extend(value._propagate_unlock(recurse=recurse))
                    sub_specs.append(value)
            return sub_specs
        return []

    def _check_unlock(self, first_attempt=True):
        if not first_attempt:
            gc.collect()
        obj = None
        for ref in self._lock_parents_weakrefs:
            obj = ref()
            # check if the locked parent exists and if it's locked
            # we check _is_locked because it can be False or None in the case of Lazy stacks,
            # but if we check obj.is_locked it will be True for this class.
            if obj is not None and obj._is_locked:
                break

        else:
            try:
                self._lock_parents_weakrefs = []
            except AttributeError:
                # Some tds (eg, LazyStack) have an automated way of creating the _lock_parents_weakref
                pass
            return

        if first_attempt:
            del obj
            return self._check_unlock(False)
        raise RuntimeError(
            "Cannot unlock a Composite that is part of a locked graph. "
            "Graphs are locked when a Composite is locked with recurse=True. "
            "Unlock the root Composite first. If the Composite is part of multiple graphs, "
            "group the graphs under a common Composite an unlock this root. "
            f"self: {self}, obj: {obj}"
        )

    def unlock_(self, recurse: bool | None = None) -> T:
        """Unlocks the Composite and allows modification of its content.

        This is only a first-level lock modification, unless specified
        otherwise through the ``recurse`` arg.

        """
        try:
            if recurse is None:
                recurse = True
            sub_specs = self._propagate_unlock(recurse=recurse)
            if recurse:
                for sub_spec in sub_specs:
                    sub_spec._check_unlock()
            self._check_unlock()
        except RuntimeError as err:
            self.lock_(recurse=recurse)
            raise err
        return self

    @property
    def locked(self):
        return self._is_locked


class StackedComposite(_LazyStackedMixin[Composite], Composite):
    """A lazy representation of a stack of composite specs.

    Stacks composite specs together along one dimension.
    When random samples are drawn, a LazyStackedTensorDict is returned.

    Indexing is allowed but only along the stack dimension.

    This class is aimed to be used in multi-task and multi-agent settings, where
    heterogeneous specs may occur (same semantic but different shape).

    """

    def _reshape(
        self,
        *args,
        **kwargs,
    ) -> Any:
        raise NotImplementedError(
            f"`reshape` is not implemented for {type(self).__name__} specs."
        )

    def cardinality(
        self,
        *args,
        **kwargs,
    ) -> Any:
        raise NotImplementedError(
            f"`cardinality` is not implemented for {type(self).__name__} specs."
        )

    def index(
        self, index: INDEX_TYPING, tensor_to_index: torch.Tensor | TensorDictBase
    ) -> torch.Tensor | TensorDictBase:
        raise NotImplementedError(
            f"`index` is not implemented for {type(self).__name__} specs."
        )

    def update(self, dict) -> None:
        for key, item in dict.items():
            if key in self.keys() and isinstance(
                item, (Dict, Composite, StackedComposite)
            ):
                for spec, sub_item in zip(self._specs, item.unbind(self.dim)):
                    spec[key].update(sub_item)
                continue
            self[key] = item
        return self

    def enumerate(self, use_mask: bool = False) -> TensorDictBase:
        dim = self.stack_dim
        return LazyStackedTensorDict.maybe_dense_stack(
            [spec.enumerate(use_mask) for spec in self._specs], dim + 1
        )

    def __eq__(self, other):
        if not isinstance(other, StackedComposite):
            return False
        if len(self._specs) != len(other._specs):
            return False
        if self.stack_dim != other.stack_dim:
            return False
        if self.device != other.device:
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
                    "Size of StackedComposite and val differ along the "
                    "stacking dimension"
                )
            for spec, v in zip(self._specs, torch.unbind(val, dim=self.dim)):
                spec.assert_is_in(v)
        return {key: self[key].to_numpy(val) for key, val in val.items()}

    def __len__(self):
        return self.shape[0]

    def keys(
        self,
        include_nested: bool = False,
        leaves_only: bool = False,
        *,
        is_leaf: Callable[[type], bool] | None = None,
    ) -> _CompositeSpecKeysView:
        return _CompositeSpecItemsView(
            self,
            include_nested=include_nested,
            leaves_only=leaves_only,
            is_leaf=is_leaf,
        )._keys()

    def items(
        self,
        include_nested: bool = False,
        leaves_only: bool = False,
        *,
        is_leaf: Callable[[type], bool] | None = None,
    ) -> _CompositeSpecItemsView:
        return list(
            _CompositeSpecItemsView(
                self,
                include_nested=include_nested,
                leaves_only=leaves_only,
                is_leaf=is_leaf,
            )
        )

    def values(
        self,
        include_nested: bool = False,
        leaves_only: bool = False,
        *,
        is_leaf: Callable[[type], bool] | None = None,
    ) -> _CompositeSpecValuesView:
        return _CompositeSpecItemsView(
            self,
            include_nested=include_nested,
            leaves_only=leaves_only,
            is_leaf=is_leaf,
        )._values()

    def project(self, val: TensorDictBase) -> TensorDictBase:
        vals = []
        for spec, subval in zip(self._specs, val.unbind(self.dim)):
            if not spec.is_in(subval):
                vals.append(spec.project(subval))
            else:
                vals.append(subval)
        res = LazyStackedTensorDict.maybe_dense_stack(vals, dim=self.dim)
        if not isinstance(val, LazyStackedTensorDict):
            res = res.to_tensordict()
        return res

    def type_check(
        self,
        value: torch.Tensor | TensorDictBase,
        selected_keys: NestedKey | Sequence[NestedKey] | None = None,
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
        return f"StackedComposite(\n{string})"

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

    def is_in(self, value) -> bool:
        for spec, subval in zip(self._specs, value.unbind(self.dim)):
            if not spec.contains(subval):
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
                f"Key {key} must be present in at least one of the stacked specs."
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
        device = self.__dict__.get("_device", NO_DEFAULT)
        if device is NO_DEFAULT:
            devices = {spec.device for spec in self._specs}
            if len(devices) == 1:
                device = list(devices)[0]
            elif len(devices) == 2:
                device0, device1 = devices
                if device0 is None:
                    device = device1
                elif device1 is None:
                    device = device0
                else:
                    device = None
            else:
                device = None
            self.__dict__["_device"] = device
        return device

    @property
    def ndim(self):
        return self.ndimension()

    def ndimension(self):
        return len(self.shape)

    def set(self, name: str, spec: TensorSpec) -> StackedComposite:
        for sub_spec, sub_item in zip(self._specs, spec.unbind(self.dim)):
            sub_spec[name] = sub_item
        return self

    @property
    def shape(self):
        shape = list(self._specs[0].shape)
        dim = self.dim
        if dim < 0:
            dim = len(shape) + dim + 1
        shape.insert(dim, len(self._specs))
        return _size(shape)

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
        return StackedComposite.maybe_dense_stack(
            [spec.empty() for spec in self._specs], dim=self.stack_dim
        )

    def encode(
        self, vals: dict[str, Any], ignore_device: bool = False
    ) -> dict[str, torch.Tensor]:
        raise NOT_IMPLEMENTED_ERROR

    def zero(self, shape: torch.Size = None) -> TensorDictBase:
        if shape is not None:
            dim = self.dim + len(shape)
        else:
            dim = self.dim
        return LazyStackedTensorDict.maybe_dense_stack(
            [spec.zero(shape) for spec in self._specs], dim
        )

    def one(self, shape: torch.Size = None) -> TensorDictBase:
        if shape is not None:
            dim = self.dim + len(shape)
        else:
            dim = self.dim
        return LazyStackedTensorDict.maybe_dense_stack(
            [spec.one(shape) for spec in self._specs], dim
        )

    def rand(self, shape: torch.Size = None) -> TensorDictBase:
        if shape is not None:
            dim = self.dim + len(shape)
        else:
            dim = self.dim
        return LazyStackedTensorDict.maybe_dense_stack(
            [spec.rand(shape) for spec in self._specs], dim
        )


@TensorSpec.implements_for_spec(torch.stack)
def _stack_specs(list_of_spec, dim=0, out=None):
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
        return Stacked(*list_of_spec, dim=dim)
    else:
        raise NotImplementedError


@Composite.implements_for_spec(torch.stack)
def _stack_composite_specs(list_of_spec, dim=0, out=None):
    if out is not None:
        raise NotImplementedError(
            "In-place spec modification is not a feature of torchrl, hence "
            "torch.stack(list_of_specs, dim, out=spec) is not implemented."
        )
    if not len(list_of_spec):
        raise ValueError("Cannot stack an empty list of specs.")
    spec0 = list_of_spec[0]
    if isinstance(spec0, Composite):
        devices = {spec.device for spec in list_of_spec}
        if len(devices) == 1:
            device = list(devices)[0]
        elif len(devices) == 2:
            device0, device1 = devices
            if device0 is None:
                device = device1
            elif device1 is None:
                device = device0
            else:
                device = None

        all_equal = True
        for spec in list_of_spec[1:]:
            if not isinstance(spec, Composite):
                raise RuntimeError(
                    "Stacking specs cannot occur: Found more than one type of spec in "
                    "the list."
                )
            if device != spec.device and device is not None:
                # spec.device must be None
                spec = spec.to(device)
            if spec.shape != spec0.shape:
                raise RuntimeError(f"Shapes differ, got {spec.shape} and {spec0.shape}")
            all_equal = all_equal and spec == spec0
        if all_equal:
            shape = list(spec0.shape)
            if dim < 0:
                dim += len(shape) + 1
            shape.insert(dim, len(list_of_spec))
            return spec0.clone().unsqueeze(dim).expand(shape)
        return StackedComposite(*list_of_spec, dim=dim)
    else:
        raise NotImplementedError


@TensorSpec.implements_for_spec(torch.squeeze)
def _squeeze_spec(spec: TensorSpec, *args, **kwargs) -> TensorSpec:
    return spec.squeeze(*args, **kwargs)


@Composite.implements_for_spec(torch.squeeze)
def _squeeze_composite_spec(spec: Composite, *args, **kwargs) -> Composite:
    return spec.squeeze(*args, **kwargs)


@TensorSpec.implements_for_spec(torch.unsqueeze)
def _unsqueeze_spec(spec: TensorSpec, *args, **kwargs) -> TensorSpec:
    return spec.unsqueeze(*args, **kwargs)


@Composite.implements_for_spec(torch.unsqueeze)
def _unsqueeze_composite_spec(spec: Composite, *args, **kwargs) -> Composite:
    return spec.unsqueeze(*args, **kwargs)


def _keys_to_empty_composite_spec(keys):
    """Given a list of keys, creates a Composite tree where each leaf is assigned a None value."""
    if not len(keys):
        return
    c = Composite()
    for key in keys:
        if isinstance(key, str):
            c[key] = None
        elif key[0] in c.keys():
            if c[key[0]] is None:
                # if the value is None we just replace it
                c[key[0]] = _keys_to_empty_composite_spec([key[1:]])
            elif isinstance(c[key[0]], Composite):
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
        new_shape = _size([s for s in shape if s != 1])
    else:
        if dim < 0:
            dim += len(shape)

        if shape[dim] != 1:
            return None

        new_shape = _size([s for i, s in enumerate(shape) if i != dim])
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
    return _size(new_shape)


class _CompositeSpecItemsView:
    """Wrapper class that enables richer behavior of `items` for Composite."""

    def __init__(
        self,
        composite: Composite,
        include_nested,
        leaves_only,
        *,
        is_leaf,
    ):
        self.composite = composite
        self.leaves_only = leaves_only
        self.include_nested = include_nested
        self.is_leaf = is_leaf

    def __iter__(self):
        from tensordict.base import _NESTED_TENSORS_AS_LISTS

        is_leaf = self.is_leaf
        if is_leaf in (None, _NESTED_TENSORS_AS_LISTS):

            def _is_leaf(cls):
                return not issubclass(cls, Composite)

        else:
            _is_leaf = is_leaf

        def _iter_from_item(key, item):
            if self.include_nested and isinstance(item, Composite):
                for subkey, subitem in item.items(
                    include_nested=True,
                    leaves_only=self.leaves_only,
                    is_leaf=is_leaf,
                ):
                    if not isinstance(subkey, tuple):
                        subkey = (subkey,)
                    yield (key, *subkey), subitem
            if not self.leaves_only and not _is_leaf(type(item)):
                yield (key, item)
            elif not self.leaves_only or _is_leaf(type(item)):
                yield key, item

        for key, item in self._get_composite_items(is_leaf):
            if is_leaf is _NESTED_TENSORS_AS_LISTS and isinstance(
                item, _LazyStackedMixin
            ):
                for (i, spec) in enumerate(item._specs):
                    yield from _iter_from_item(unravel_key((key, str(i))), spec)
            else:
                yield from _iter_from_item(key, item)

    def _get_composite_items(self, is_leaf):

        if isinstance(self.composite, StackedComposite):
            from tensordict.base import _NESTED_TENSORS_AS_LISTS

            if is_leaf is _NESTED_TENSORS_AS_LISTS:
                for i, spec in enumerate(self.composite._specs):
                    for key, item in spec.items():
                        yield ((str(i), key), item)
            else:
                keys = self.composite._specs[0].keys()
                keys = set(keys)
                for spec in self.composite._specs[1:]:
                    keys = keys.intersection(spec.keys())
                yield from ((key, self.composite[key]) for key in sorted(keys, key=str))
        else:
            yield from self.composite._specs.items()

    def __len__(self):
        i = 0
        for _ in self:
            i += 1
        return i

    def __repr__(self):
        return f"{type(self).__name__}(keys={list(self)})"

    def __contains__(self, item):
        item = unravel_key(item)

        if len(item) == 1:
            item = item[0]
        for key in self.__iter__():
            if key == item:
                return True
        else:
            return False

    def _keys(self):
        return _CompositeSpecKeysView(self)

    def _values(self):
        return _CompositeSpecValuesView(self)


class _CompositeSpecKeysView:
    def __init__(self, items: _CompositeSpecItemsView):
        self.items = items

    def __iter__(self):
        yield from (key for (key, _) in self.items)

    def __contains__(self, item):
        item = unravel_key(item)
        return any(key == item for key in self)

    def __len__(self):
        return len(self.items)

    def __repr__(self):
        return f"{type(self).__name__}(keys={list(self)})"


class _CompositeSpecValuesView:
    def __init__(self, items: _CompositeSpecItemsView):
        self.items = items

    def __iter__(self):
        yield from (val for (_, val) in self.items)

    def __len__(self):
        return len(self.items)

    def __repr__(self):
        return f"{type(self).__name__}(values={list(self)})"


def _minmax_dtype(dtype):
    if dtype is torch.bool:
        return False, True
    if dtype.is_floating_point:
        info = torch.finfo(dtype)
    else:
        info = torch.iinfo(dtype)
    return info.min, info.max


def _remove_neg_shapes(*shape):
    if len(shape) == 1 and not isinstance(shape[0], int):
        shape = shape[0]
        if isinstance(shape, np.integer):
            shape = (int(shape),)
        return _remove_neg_shapes(*shape)
    return _size([int(d) if d >= 0 else 1 for d in shape])


##############
# Legacy
#
class _LegacySpecMeta(abc.ABCMeta):
    def __call__(cls, *args, **kwargs):
        warnings.warn(
            f"The {cls.__name__} has been deprecated and will be removed in v0.8. Please use "
            f"{cls.__bases__[-1].__name__} instead.",
            category=DeprecationWarning,
        )
        instance = super().__call__(*args, **kwargs)
        if (
            type(instance) in (UnboundedDiscreteTensorSpec, UnboundedDiscrete)
            and instance.domain == "continuous"
        ):
            instance.__class__ = UnboundedContinuous
        elif (
            type(instance) in (UnboundedContinuousTensorSpec, UnboundedContinuous)
            and instance.domain == "discrete"
        ):
            instance.__class__ = UnboundedDiscrete
        return instance

    def __instancecheck__(cls, instance):
        check0 = super().__instancecheck__(instance)
        if check0:
            return True
        parent_cls = cls.__bases__[-1]
        return isinstance(instance, parent_cls)


class CompositeSpec(Composite, metaclass=_LegacySpecMeta):
    """Deprecated version of :class:`torchrl.data.Composite`."""

    ...


class OneHotDiscreteTensorSpec(OneHot, metaclass=_LegacySpecMeta):
    """Deprecated version of :class:`torchrl.data.OneHot`."""

    ...


class MultiOneHotDiscreteTensorSpec(MultiOneHot, metaclass=_LegacySpecMeta):
    """Deprecated version of :class:`torchrl.data.MultiOneHot`."""

    ...


class NonTensorSpec(NonTensor, metaclass=_LegacySpecMeta):
    """Deprecated version of :class:`torchrl.data.NonTensor`."""

    ...


class MultiDiscreteTensorSpec(MultiCategorical, metaclass=_LegacySpecMeta):
    """Deprecated version of :class:`torchrl.data.MultiCategorical`."""

    ...


class LazyStackedTensorSpec(Stacked, metaclass=_LegacySpecMeta):
    """Deprecated version of :class:`torchrl.data.Stacked`."""

    ...


class LazyStackedCompositeSpec(StackedComposite, metaclass=_LegacySpecMeta):
    """Deprecated version of :class:`torchrl.data.StackedComposite`."""

    ...


class DiscreteTensorSpec(Categorical, metaclass=_LegacySpecMeta):
    """Deprecated version of :class:`torchrl.data.Categorical`."""

    ...


class BinaryDiscreteTensorSpec(Binary, metaclass=_LegacySpecMeta):
    """Deprecated version of :class:`torchrl.data.Binary`."""

    ...


_BoundedLegacyMeta = type("_BoundedLegacyMeta", (_LegacySpecMeta, _BoundedMeta), {})


class BoundedTensorSpec(Bounded, metaclass=_BoundedLegacyMeta):
    """Deprecated version of :class:`torchrl.data.Bounded`."""

    ...


class _UnboundedContinuousMetaclass(_UnboundedMeta):
    def __instancecheck__(cls, instance):
        return isinstance(instance, Unbounded) and instance.domain == "continuous"


_LegacyUnboundedContinuousMetaclass = type(
    "_LegacyUnboundedDiscreteMetaclass",
    (_UnboundedContinuousMetaclass, _LegacySpecMeta),
    {},
)


class UnboundedContinuousTensorSpec(
    Unbounded, metaclass=_LegacyUnboundedContinuousMetaclass
):
    """Deprecated version of :class:`torchrl.data.Unbounded` with continuous space."""

    ...


class _UnboundedDiscreteMetaclass(_UnboundedMeta):
    def __instancecheck__(cls, instance):
        return isinstance(instance, Unbounded) and instance.domain == "discrete"


_LegacyUnboundedDiscreteMetaclass = type(
    "_LegacyUnboundedDiscreteMetaclass",
    (_UnboundedDiscreteMetaclass, _LegacySpecMeta),
    {},
)


class UnboundedDiscreteTensorSpec(
    Unbounded, metaclass=_LegacyUnboundedDiscreteMetaclass
):
    """Deprecated version of :class:`torchrl.data.Unbounded` with discrete space."""

    def __init__(
        self,
        shape: torch.Size | int = _DEFAULT_SHAPE,
        device: DEVICE_TYPING | None = None,
        dtype: str | torch.dtype | None = torch.int64,
        **kwargs,
    ):
        super().__init__(shape=shape, device=device, dtype=dtype, **kwargs)


def _reduce_funcs(funcs):
    return functools.partial(functools.reduce, lambda x, f: f(x), funcs)
