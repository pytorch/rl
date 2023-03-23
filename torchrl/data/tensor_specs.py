# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import abc
import warnings
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
from tensordict.tensordict import TensorDict, TensorDictBase

from torchrl._utils import get_binary_env_var

DEVICE_TYPING = Union[torch.device, str, int]

INDEX_TYPING = Union[int, torch.Tensor, np.ndarray, slice, List]

# By default, we do not check that an obs is in the domain. THis should be done when validating the env beforehand
_CHECK_SPEC_ENCODE = get_binary_env_var("CHECK_SPEC_ENCODE")


_DEFAULT_SHAPE = torch.Size((1,))

DEVICE_ERR_MSG = "device of empty CompositeSpec is not defined."


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
    """A continuous box of values, in between a minimum and a maximum."""

    _minimum: torch.Tensor
    _maximum: torch.Tensor
    device: torch.device = None

    # We store the tensors on CPU to avoid overloading CUDA with tensors that are rarely used.
    @property
    def minimum(self):
        return self._minimum.to(self.device)

    @property
    def maximum(self):
        return self._maximum.to(self.device)

    @minimum.setter
    def minimum(self, value):
        self.device = value.device
        self._minimum = value.cpu()

    @maximum.setter
    def maximum(self, value):
        self.device = value.device
        self._maximum = value.cpu()

    def __post_init__(self):
        self.minimum = self.minimum.clone()
        self.maximum = self.maximum.clone()

    def __iter__(self):
        yield self.minimum
        yield self.maximum

    def to(self, dest: Union[torch.dtype, DEVICE_TYPING]) -> ContinuousBox:
        return self.__class__(self.minimum.to(dest), self.maximum.to(dest))

    def clone(self) -> ContinuousBox:
        return self.__class__(self.minimum.clone(), self.maximum.clone())

    def __repr__(self):
        min_str = f"minimum=Tensor(shape={self.minimum.shape}, device={self.minimum.device}, dtype={self.minimum.dtype}, contiguous={self.maximum.is_contiguous()})"
        max_str = f"maximum=Tensor(shape={self.maximum.shape}, device={self.maximum.device}, dtype={self.maximum.dtype}, contiguous={self.maximum.is_contiguous()})"
        return f"{self.__class__.__name__}({min_str}, {max_str})"

    def __eq__(self, other):
        return (
            type(self) == type(other)
            and self.minimum.dtype == other.minimum.dtype
            and self.maximum.dtype == other.maximum.dtype
            and torch.equal(self.minimum, other.minimum)
            and torch.equal(self.maximum, other.maximum)
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

    def encode(self, val: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """Encodes a value given the specified spec, and return the corresponding tensor.

        Args:
            val (np.ndarray or torch.Tensor): value to be encoded as tensor.

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
            val = torch.tensor(val, device=self.device, dtype=self.dtype)
            if val.shape[-len(self.shape) :] != self.shape:
                # option 1: add a singleton dim at the end
                if (
                    val.shape[-len(self.shape) :] == self.shape[:-1]
                    and self.shape[-1] == 1
                ):
                    val = val.unsqueeze(-1)
                else:
                    raise RuntimeError(
                        f"Shape mismatch: the value has shape {val.shape} which "
                        f"is incompatible with the spec shape {self.shape}."
                    )
        if _CHECK_SPEC_ENCODE:
            self.assert_is_in(val)
        return val

    def __setattr__(self, key, value):
        if key == "shape":
            value = torch.Size(value)
        super().__setattr__(key, value)

    def to_numpy(self, val: torch.Tensor, safe: bool = True) -> np.ndarray:
        """Returns the np.ndarray correspondent of an input tensor.

        Args:
            val (torch.Tensor): tensor to be transformed_in to numpy
            safe (bool): boolean value indicating whether a check should be
                performed on the value against the domain of the spec.

        Returns:
            a np.ndarray

        """
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
        shape_str = "shape=" + str(self.shape)
        space_str = "space=" + str(self.space)
        device_str = "device=" + str(self.device)
        dtype_str = "dtype=" + str(self.dtype)
        domain_str = "domain=" + str(self.domain)
        sub_string = ", ".join(
            [shape_str, space_str, device_str, dtype_str, domain_str]
        )
        string = f"{self.__class__.__name__}(\n     {sub_string})"
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


T = TypeVar("T")


class _LazyStackedMixin(Generic[T]):
    def __init__(self, *specs: tuple[T, ...], dim: int) -> None:
        self._specs = specs
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

    @property
    def shape(self):
        shape = list(self._specs[0].shape)
        dim = self.dim
        if dim < 0:
            dim = len(shape) + dim + 1
        shape.insert(dim, len(self._specs))
        return torch.Size(shape)

    def clone(self) -> T:
        return torch.stack([spec.clone() for spec in self._specs], 0)

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


class LazyStackedTensorSpec(_LazyStackedMixin[TensorSpec], TensorSpec):
    """A lazy representation of a stack of tensor specs.

    Stacks tensor-specs together along one dimension.
    When random samples are drawn, a stack of samples is returned if possible.
    If not, an error is thrown.

    Indexing is allowed but only along the stack dimension.

    This class is aimed to be used in multi-task and multi-agent settings, where
    heterogeneous specs may occur (same semantic but different shape).

    """

    @property
    def space(self):
        return self._specs[0].space

    def __eq__(self, other):
        # requires unbind to be implemented
        pass

    def to_numpy(self, val: torch.Tensor, safe: bool = True) -> dict:
        if safe:
            if val.shape[self.dim] != len(self._specs):
                raise ValueError(
                    "Size of LazyStackedTensorSpec and val differ along the stacking "
                    "dimension"
                )
            for spec, v in zip(self._specs, torch.unbind(val, dim=self.dim)):
                spec.assert_is_in(v)
        return val.detach().cpu().numpy()

    def __len__(self):
        pass

    def project(self, val: TensorDictBase) -> TensorDictBase:
        pass

    def __repr__(self):
        shape_str = "shape=" + str(self.shape)
        space_str = "space=" + str(self._specs[0].space)
        device_str = "device=" + str(self.device)
        dtype_str = "dtype=" + str(self.dtype)
        domain_str = "domain=" + str(self._specs[0].domain)
        sub_string = ", ".join(
            [shape_str, space_str, device_str, dtype_str, domain_str]
        )
        string = f"{self.__class__.__name__}(\n     {sub_string})"
        return string

    def __iter__(self):
        pass

    def __setitem__(self, key, value):
        pass

    @property
    def device(self) -> DEVICE_TYPING:
        return self._specs[0].device

    @property
    def ndim(self):
        return self.ndimension()

    def ndimension(self):
        return len(self.shape)

    def set(self, name, spec):
        if spec is not None:
            shape = spec.shape
            if shape[: self.ndim] != self.shape:
                raise ValueError(
                    "The shape of the spec and the CompositeSpec mismatch: the first "
                    f"{self.ndim} dimensions should match but got spec.shape={spec.shape} and "
                    f"CompositeSpec.shape={self.shape}."
                )
        self._specs[name] = spec


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
        dtype: Optional[Union[str, torch.dtype]] = torch.long,
        use_register: bool = False,
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
        )

    def clone(self) -> CompositeSpec:
        return self.__class__(
            n=self.space.n,
            shape=self.shape,
            device=self.device,
            dtype=self.dtype,
            use_register=self.use_register,
        )

    def expand(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list, torch.Size)):
            shape = shape[0]
        if any(val < 0 for val in shape):
            raise ValueError(
                f"{self.__class__.__name__}.extend does not support negative shapes."
            )
        if any(s1 != s2 and s2 != 1 for s1, s2 in zip(shape[-self.ndim :], self.shape)):
            raise ValueError(
                f"The last {self.ndim} of the extended shape must match the"
                f"shape of the CompositeSpec in CompositeSpec.extend."
            )
        return self.__class__(
            n=shape[-1], shape=shape, device=self.device, dtype=self.dtype
        )

    def squeeze(self, dim=None):
        if self.shape[-1] == 1 and dim in (len(self.shape), -1, None):
            raise ValueError(
                "Final dimension of OneHotDiscreteTensorSpec must remain unchanged"
            )

        shape = _squeezed_shape(self.shape, dim)
        if shape is None:
            return self

        return self.__class__(
            n=shape[-1],
            shape=shape,
            device=self.device,
            dtype=self.dtype,
            use_register=self.use_register,
        )

    def unsqueeze(self, dim: int):
        if dim in (len(self.shape), -1):
            raise ValueError(
                "Final dimension of OneHotDiscreteTensorSpec must remain unchanged"
            )

        shape = _unsqueezed_shape(self.shape, dim)
        return self.__class__(
            n=shape[-1],
            shape=shape,
            device=self.device,
            dtype=self.dtype,
            use_register=self.use_register,
        )

    def rand(self, shape=None) -> torch.Tensor:
        if shape is None:
            shape = self.shape[:-1]
        else:
            shape = torch.Size([*shape, *self.shape[:-1]])
        return torch.nn.functional.gumbel_softmax(
            torch.rand(torch.Size([*shape, self.space.n]), device=self.device),
            hard=True,
            dim=-1,
        ).to(torch.long)

    def encode(
        self,
        val: Union[np.ndarray, torch.Tensor],
        space: Optional[DiscreteBox] = None,
    ) -> torch.Tensor:
        if not isinstance(val, torch.Tensor):
            val = torch.tensor(val, dtype=self.dtype, device=self.device)

        if space is None:
            space = self.space

        if self.use_register:
            if val not in space.register:
                space.register[val] = len(space.register)
            val = space.register[val]

        if (val >= space.n).any():
            raise AssertionError("Value must be less than action space.")

        val = torch.nn.functional.one_hot(val.long(), space.n)
        return val

    def to_numpy(self, val: torch.Tensor, safe: bool = True) -> np.ndarray:
        if safe:
            if not isinstance(val, torch.Tensor):
                raise NotImplementedError
            self.assert_is_in(val)
        val = val.argmax(-1).cpu().numpy()
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

    def _project(self, val: torch.Tensor) -> torch.Tensor:
        # idx = val.sum(-1) != 1
        out = torch.nn.functional.gumbel_softmax(val.to(torch.float))
        out = (out == out.max(dim=-1, keepdim=True)[0]).to(torch.long)
        return out

    def is_in(self, val: torch.Tensor) -> bool:
        return (val.sum(-1) == 1).all()

    def __eq__(self, other):
        return (
            type(self) == type(other)
            and self.shape == other.shape
            and self.space == other.space
            and self.device == other.device
            and self.dtype == other.dtype
            and self.domain == other.domain
            and self.use_register == other.use_register
        )

    def to_categorical(self, val: torch.Tensor, safe: bool = True) -> torch.Tensor:
        """Converts a given one-hot tensor in categorical format.

        Args:
            val (torch.Tensor, optional): One-hot tensor to convert in categorical format.
            safe (bool): boolean value indicating whether a check should be
                performed on the value against the domain of the spec.

        Returns:
            The categorical tensor.
        """
        if safe:
            self.assert_is_in(val)
        return val.argmax(-1)

    def to_categorical_spec(self) -> DiscreteTensorSpec:
        """Converts the spec to the equivalent categorical spec."""
        return DiscreteTensorSpec(
            self.space.n,
            device=self.device,
            dtype=self.dtype,
            shape=self.shape[:-1],
        )


@dataclass(repr=False)
class BoundedTensorSpec(TensorSpec):
    """A bounded continuous tensor spec.

    Args:
        minimum (np.ndarray, torch.Tensor or number): lower bound of the box.
        maximum (np.ndarray, torch.Tensor or number): upper bound of the box.
        device (str, int or torch.device, optional): device of the tensors.
        dtype (str or torch.dtype, optional): dtype of the tensors.

    """

    # SPEC_HANDLED_FUNCTIONS = {}

    def __init__(
        self,
        minimum: Union[float, torch.Tensor, np.ndarray],
        maximum: Union[float, torch.Tensor, np.ndarray],
        shape: Optional[Union[torch.Size, int]] = None,
        device: Optional[DEVICE_TYPING] = None,
        dtype: Optional[Union[torch.dtype, str]] = None,
    ):
        dtype, device = _default_dtype_and_device(dtype, device)
        if dtype is None:
            dtype = torch.get_default_dtype()
        if device is None:
            device = torch._get_default_device()

        if not isinstance(minimum, torch.Tensor):
            minimum = torch.tensor(minimum, dtype=dtype, device=device)
        if not isinstance(maximum, torch.Tensor):
            maximum = torch.tensor(maximum, dtype=dtype, device=device)
        if maximum.device != device:
            maximum = maximum.to(device)
        if minimum.device != device:
            minimum = minimum.to(device)
        if dtype is not None and minimum.dtype is not dtype:
            minimum = minimum.to(dtype)
        if dtype is not None and maximum.dtype is not dtype:
            maximum = maximum.to(dtype)
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

        if maximum.ndimension():
            if shape is not None and shape != maximum.shape:
                raise RuntimeError(err_msg)
            shape = maximum.shape
            minimum = minimum.expand(shape).clone()
        elif minimum.ndimension():
            if shape is not None and shape != minimum.shape:
                raise RuntimeError(err_msg)
            shape = minimum.shape
            maximum = maximum.expand(shape).clone()
        elif shape is None:
            raise RuntimeError(err_msg)
        else:
            minimum = minimum.expand(shape).clone()
            maximum = maximum.expand(shape).clone()

        if minimum.numel() > maximum.numel():
            maximum = maximum.expand_as(minimum).clone()
        elif maximum.numel() > minimum.numel():
            minimum = minimum.expand_as(maximum).clone()
        if shape is None:
            shape = minimum.shape
        else:
            if isinstance(shape, float):
                shape = torch.Size([shape])
            elif not isinstance(shape, torch.Size):
                shape = torch.Size(shape)
            shape_err_msg = (
                f"minimum and shape mismatch, got {minimum.shape} and {shape}"
            )
            if len(minimum.shape) != len(shape):
                raise RuntimeError(shape_err_msg)
            if not all(_s == _sa for _s, _sa in zip(shape, minimum.shape)):
                raise RuntimeError(shape_err_msg)
        self.shape = shape

        super().__init__(
            shape, ContinuousBox(minimum, maximum), device, dtype, "continuous"
        )

    def expand(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list, torch.Size)):
            shape = shape[0]
        if any(val < 0 for val in shape):
            raise ValueError(
                f"{self.__class__.__name__}.extend does not support negative shapes."
            )
        if any(s1 != s2 and s2 != 1 for s1, s2 in zip(shape[-self.ndim :], self.shape)):
            raise ValueError(
                f"The last {self.ndim} of the extended shape must match the"
                f"shape of the CompositeSpec in CompositeSpec.extend."
            )
        return self.__class__(
            minimum=self.space.minimum.expand(shape).clone(),
            maximum=self.space.maximum.expand(shape).clone(),
            shape=shape,
            device=self.device,
            dtype=self.dtype,
        )

    def squeeze(self, dim: int | None = None):
        shape = _squeezed_shape(self.shape, dim)
        if shape is None:
            return self

        if dim is None:
            minimum = self.space.minimum.squeeze().clone()
            maximum = self.space.maximum.squeeze().clone()
        else:
            minimum = self.space.minimum.squeeze(dim).clone()
            maximum = self.space.maximum.squeeze(dim).clone()

        return self.__class__(
            minimum=minimum,
            maximum=maximum,
            shape=shape,
            device=self.device,
            dtype=self.dtype,
        )

    def unsqueeze(self, dim: int):
        shape = _unsqueezed_shape(self.shape, dim)
        return self.__class__(
            minimum=self.space.minimum.unsqueeze(dim).clone(),
            maximum=self.space.maximum.unsqueeze(dim).clone(),
            shape=shape,
            device=self.device,
            dtype=self.dtype,
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
            if self.space.maximum.dtype == torch.bool:
                maxi = self.space.maximum.int()
            else:
                maxi = self.space.maximum
            if self.space.minimum.dtype == torch.bool:
                mini = self.space.minimum.int()
            else:
                mini = self.space.minimum
            interval = maxi - mini
            r = torch.rand(torch.Size([*shape, *self.shape]), device=interval.device)
            r = interval * r
            r = self.space.minimum + r
            r = r.to(self.dtype).to(self.device)
            return r

    def _project(self, val: torch.Tensor) -> torch.Tensor:
        minimum = self.space.minimum.to(val.device)
        maximum = self.space.maximum.to(val.device)
        try:
            val = val.clamp_(minimum.item(), maximum.item())
        except ValueError:
            minimum = minimum.expand_as(val)
            maximum = maximum.expand_as(val)
            val[val < minimum] = minimum[val < minimum]
            val[val > maximum] = maximum[val > maximum]
        except RuntimeError:
            minimum = minimum.expand_as(val)
            maximum = maximum.expand_as(val)
            val[val < minimum] = minimum[val < minimum]
            val[val > maximum] = maximum[val > maximum]
        return val

    def is_in(self, val: torch.Tensor) -> bool:
        try:
            return (val >= self.space.minimum.to(val.device)).all() and (
                val <= self.space.maximum.to(val.device)
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
            minimum=self.space.minimum.to(dest),
            maximum=self.space.maximum.to(dest),
            shape=self.shape,
            device=dest_device,
            dtype=dest_dtype,
        )

    def clone(self) -> CompositeSpec:
        return self.__class__(
            minimum=self.space.minimum.clone(),
            maximum=self.space.maximum.clone(),
            shape=self.shape,
            device=self.device,
            dtype=self.dtype,
        )


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
            ContinuousBox(torch.tensor(-np.inf), torch.tensor(np.inf))
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

    def clone(self) -> CompositeSpec:
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
                f"{self.__class__.__name__}.extend does not support negative shapes."
            )
        if any(s1 != s2 and s2 != 1 for s1, s2 in zip(shape[-self.ndim :], self.shape)):
            raise ValueError(
                f"The last {self.ndim} of the extended shape must match the"
                f"shape of the CompositeSpec in CompositeSpec.extend."
            )
        return self.__class__(shape=shape, device=self.device, dtype=self.dtype)


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

    def clone(self) -> CompositeSpec:
        return self.__class__(shape=self.shape, device=self.device, dtype=self.dtype)

    def rand(self, shape=None) -> torch.Tensor:
        if shape is None:
            shape = torch.Size([])
        interval = self.space.maximum - self.space.minimum
        r = torch.rand(torch.Size([*shape, *interval.shape]), device=interval.device)
        r = r * interval
        r = self.space.minimum + r
        r = r.to(self.dtype)
        return r.to(self.device)

    def is_in(self, val: torch.Tensor) -> bool:
        return True

    def expand(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list, torch.Size)):
            shape = shape[0]
        if any(val < 0 for val in shape):
            raise ValueError(
                f"{self.__class__.__name__}.extend does not support negative shapes."
            )
        if any(s1 != s2 and s2 != 1 for s1, s2 in zip(shape[-self.ndim :], self.shape)):
            raise ValueError(
                f"The last {self.ndim} of the extended shape must match the"
                f"shape of the CompositeSpec in CompositeSpec.extend."
            )
        return self.__class__(shape=shape, device=self.device, dtype=self.dtype)


@dataclass(repr=False)
class MultiOneHotDiscreteTensorSpec(OneHotDiscreteTensorSpec):
    """A concatenation of one-hot discrete tensor spec.

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
        dtype=torch.long,
        use_register=False,
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
            shape, space, device, dtype, domain="discrete"
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
            nvec=deepcopy(self.nvec),
            shape=self.shape,
            device=dest_device,
            dtype=dest_dtype,
        )

    def clone(self) -> CompositeSpec:
        return self.__class__(
            nvec=deepcopy(self.nvec),
            shape=self.shape,
            device=self.device,
            dtype=self.dtype,
        )

    def rand(self, shape: Optional[torch.Size] = None) -> torch.Tensor:
        if shape is None:
            shape = self.shape[:-1]
        else:
            shape = torch.Size([*shape, *self.shape[:-1]])

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
                ).to(torch.long)
                for space in self.space
            ],
            -1,
        ).squeeze(-2)
        return x

    def encode(self, val: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        if not isinstance(val, torch.Tensor):
            val = torch.tensor(val, device=self.device)

        x = []
        for v, space in zip(val.unbind(-1), self.space):
            if not (v < space.n).all():
                raise RuntimeError(
                    f"value {v} is greater than the allowed max {space.n}"
                )
            x.append(super(MultiOneHotDiscreteTensorSpec, self).encode(v, space))
        return torch.cat(x, -1)

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
        return all(
            super(MultiOneHotDiscreteTensorSpec, self).is_in(_val) for _val in vals
        )

    def _project(self, val: torch.Tensor) -> torch.Tensor:
        vals = self._split(val)
        return torch.cat([super()._project(_val) for _val in vals], -1)

    def to_categorical(self, val: torch.Tensor, safe: bool = True) -> torch.Tensor:
        """Converts a given one-hot tensor in categorical format.

        Args:
            val (torch.Tensor, optional): One-hot tensor to convert in categorical format.
            safe (bool): boolean value indicating whether a check should be
                performed on the value against the domain of the spec.

        Returns:
            The categorical tensor.
        """
        if safe:
            self.assert_is_in(val)
        vals = self._split(val)
        return torch.stack([val.argmax(-1) for val in vals], -1)

    def to_categorical_spec(self) -> MultiDiscreteTensorSpec:
        """Converts the spec to the equivalent categorical spec."""
        return MultiDiscreteTensorSpec(
            [_space.n for _space in self.space],
            device=self.device,
            dtype=self.dtype,
            shape=[*self.shape[:-1], len(self.space)],
        )

    def expand(self, *shape):
        nvecs = [space.n for space in self.space]
        if len(shape) == 1 and isinstance(shape[0], (tuple, list, torch.Size)):
            shape = shape[0]
        if any(val < 0 for val in shape):
            raise ValueError(
                f"{self.__class__.__name__}.extend does not support negative shapes."
            )
        if any(s1 != s2 and s2 != 1 for s1, s2 in zip(shape[-self.ndim :], self.shape)):
            raise ValueError(
                f"The last {self.ndim} of the extended shape must match the"
                f"shape of the CompositeSpec in CompositeSpec.extend."
            )
        return self.__class__(
            nvec=nvecs, shape=shape, device=self.device, dtype=self.dtype
        )

    def squeeze(self, dim=None):
        if self.shape[-1] == 1 and dim in (len(self.shape), -1, None):
            raise ValueError(
                "Final dimension of MultiOneHotDiscreteTensorSpec must remain unchanged"
            )

        shape = _squeezed_shape(self.shape, dim)
        if shape is None:
            return self
        return self.__class__(
            nvec=self.nvec, shape=shape, device=self.device, dtype=self.dtype
        )

    def unsqueeze(self, dim: int):
        if dim in (len(self.shape), -1):
            raise ValueError(
                "Final dimension of MultiOneHotDiscreteTensorSpec must remain unchanged"
            )
        shape = _unsqueezed_shape(self.shape, dim)
        return self.__class__(
            nvec=self.nvec, shape=shape, device=self.device, dtype=self.dtype
        )


class DiscreteTensorSpec(TensorSpec):
    """A discrete tensor spec.

    An alternative to OneHotTensorSpec for categorical variables in TorchRL. Instead of
    using multiplication, categorical variables perform indexing which can speed up
    computation and reduce memory cost for large categorical variables.

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
        shape: Optional[torch.Size] = None,
        device: Optional[DEVICE_TYPING] = None,
        dtype: Optional[Union[str, torch.dtype]] = torch.long,
    ):
        if shape is None:
            shape = torch.Size([])
        dtype, device = _default_dtype_and_device(dtype, device)
        space = DiscreteBox(n)
        super().__init__(shape, space, device, dtype, domain="discrete")

    def rand(self, shape=None) -> torch.Tensor:
        if shape is None:
            shape = torch.Size([])
        return torch.randint(
            0,
            self.space.n,
            torch.Size([*shape, *self.shape]),
            device=self.device,
            dtype=self.dtype,
        )

    def _project(self, val: torch.Tensor) -> torch.Tensor:
        if val.dtype not in (torch.int, torch.long):
            val = torch.round(val)
        return val.clamp_(min=0, max=self.space.n - 1)

    def is_in(self, val: torch.Tensor) -> bool:
        return (0 <= val).all() and (val < self.space.n).all()

    def __eq__(self, other):
        return (
            type(self) == type(other)
            and self.shape == other.shape
            and self.space == other.space
            and self.device == other.device
            and self.dtype == other.dtype
            and self.domain == other.domain
        )

    def to_numpy(self, val: TensorDict, safe: bool = True) -> dict:
        return super().to_numpy(val, safe)

    def to_one_hot(self, val: torch.Tensor, safe: bool = True) -> torch.Tensor:
        """Encodes a discrete tensor from the spec domain into its one-hot correspondent.

        Args:
            val (torch.Tensor, optional): Tensor to one-hot encode.
            safe (bool): boolean value indicating whether a check should be
                performed on the value against the domain of the spec.

        Returns:
            The one-hot encoded tensor.
        """
        if safe:
            self.assert_is_in(val)
        return torch.nn.functional.one_hot(val, self.space.n)

    def to_one_hot_spec(self) -> OneHotDiscreteTensorSpec:
        """Converts the spec to the equivalent one-hot spec."""
        shape = [*self.shape, self.space.n]
        return OneHotDiscreteTensorSpec(
            n=self.space.n, shape=shape, device=self.device, dtype=self.dtype
        )

    def expand(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list, torch.Size)):
            shape = shape[0]
        if any(val < 0 for val in shape):
            raise ValueError(
                f"{self.__class__.__name__}.extend does not support negative shapes."
            )
        if any(s1 != s2 and s2 != 1 for s1, s2 in zip(shape[-self.ndim :], self.shape)):
            raise ValueError(
                f"The last {self.ndim} of the extended shape must match the"
                f"shape of the CompositeSpec in CompositeSpec.extend."
            )
        return self.__class__(
            n=self.space.n, shape=shape, device=self.device, dtype=self.dtype
        )

    def squeeze(self, dim=None):
        shape = _squeezed_shape(self.shape, dim)
        if shape is None:
            return self
        return self.__class__(
            n=self.space.n, shape=shape, device=self.device, dtype=self.dtype
        )

    def unsqueeze(self, dim: int):
        shape = _unsqueezed_shape(self.shape, dim)
        return self.__class__(
            n=self.space.n, shape=shape, device=self.device, dtype=self.dtype
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

    def clone(self) -> CompositeSpec:
        return self.__class__(
            n=self.space.n,
            shape=self.shape,
            device=self.device,
            dtype=self.dtype,
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
                f"{self.__class__.__name__}.extend does not support negative shapes."
            )
        if any(s1 != s2 and s2 != 1 for s1, s2 in zip(shape[-self.ndim :], self.shape)):
            raise ValueError(
                f"The last {self.ndim} of the extended shape must match the"
                f"shape of the CompositeSpec in CompositeSpec.extend."
            )
        return self.__class__(
            n=shape[-1], shape=shape, device=self.device, dtype=self.dtype
        )

    def squeeze(self, dim=None):
        shape = _squeezed_shape(self.shape, dim)
        if shape is None:
            return self
        return self.__class__(
            n=shape[-1], shape=shape, device=self.device, dtype=self.dtype
        )

    def unsqueeze(self, dim: int):
        shape = _unsqueezed_shape(self.shape, dim)
        return self.__class__(
            n=shape[-1], shape=shape, device=self.device, dtype=self.dtype
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

    def clone(self) -> CompositeSpec:
        return self.__class__(
            n=self.shape[-1],
            shape=self.shape,
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
            If provided, the last dimension must match nvec.shape[-1].
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
            n=self.nvec.to(dest), shape=None, device=dest_device, dtype=dest_dtype
        )

    def clone(self) -> CompositeSpec:
        return self.__class__(
            nvec=self.nvec.clone(),
            shape=None,
            device=self.device,
            dtype=self.dtype,
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

    def _project(self, val: torch.Tensor) -> torch.Tensor:
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
        self, val: torch.Tensor, safe: bool = True
    ) -> Union[MultiOneHotDiscreteTensorSpec, torch.Tensor]:
        """Encodes a discrete tensor from the spec domain into its one-hot correspondent.

        Args:
            val (torch.Tensor, optional): Tensor to one-hot encode.
            safe (bool): boolean value indicating whether a check should be
                performed on the value against the domain of the spec.

        Returns:
            The one-hot encoded tensor.
        """
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
            dtype=self.dtype,
            shape=[*self.shape[:-1], sum(nvec)],
        )

    def expand(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list, torch.Size)):
            shape = shape[0]
        if any(val < 0 for val in shape):
            raise ValueError(
                f"{self.__class__.__name__}.extend does not support negative shapes."
            )
        if any(s1 != s2 and s2 != 1 for s1, s2 in zip(shape[-self.ndim :], self.shape)):
            raise ValueError(
                f"The last {self.ndim} of the extended shape must match the"
                f"shape of the CompositeSpec in CompositeSpec.extend."
            )
        return self.__class__(
            nvec=self.nvec, shape=shape, device=self.device, dtype=self.dtype
        )

    def squeeze(self, dim: int | None = None):
        if self.shape[-1] == 1 and dim in (len(self.shape), -1, None):
            raise ValueError(
                "Final dimension of MultiDiscreteTensorSpec must remain unchanged"
            )

        shape = _squeezed_shape(self.shape, dim)
        if shape is None:
            return self

        if dim is None:
            nvec = self.nvec.squeeze()
        else:
            nvec = self.nvec.squeeze(dim)

        return self.__class__(
            nvec=nvec, shape=shape, device=self.device, dtype=self.dtype
        )

    def unsqueeze(self, dim: int):
        if dim in (len(self.shape), -1):
            raise ValueError(
                "Final dimension of MultiDiscreteTensorSpec must remain unchanged"
            )
        shape = _unsqueezed_shape(self.shape, dim)
        nvec = self.nvec.unsqueeze(dim)
        return self.__class__(
            nvec=nvec, shape=shape, device=self.device, dtype=self.dtype
        )


class CompositeSpec(TensorSpec):
    """A composition of TensorSpecs.

    Args:
        *args: if an unnamed argument is passed, it must be a dictionary with keys
            matching the expected keys to be found in the :obj:`CompositeSpec` object.
            This is useful to build nested CompositeSpecs with tuple indices.
        **kwargs (key (str): value (TensorSpec)): dictionary of tensorspecs
            to be stored. Values can be None, in which case is_in will be assumed
            to be :obj:`True` for the corresponding tensors, and :obj:`project()` will have no
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
        return super().__new__(cls)

    @property
    def shape(self):
        return self._shape

    @shape.setter
    def shape(self, value: torch.Size):
        for key, spec in self.items():
            if spec.shape[: self.ndim] != self.shape:
                raise ValueError(
                    f"The shape of the spec and the CompositeSpec mismatch during shape resetting: the "
                    f"{self.ndim} first dimensions should match but got self['{key}'].shape={spec.shape} and "
                    f"CompositeSpec.shape={self.shape}."
                )
        self._shape = torch.Size(value)

    @property
    def ndim(self):
        return self.ndimension()

    def ndimension(self):
        return len(self.shape)

    def set(self, name, spec):
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
                if item is not None:
                    if self._device is None:
                        self._device = item.device
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

    def __getitem__(self, item):
        if isinstance(item, tuple) and len(item) > 1:
            return self[item[0]][item[1:]]
        elif isinstance(item, tuple):
            return self[item[0]]

        if item in {"shape", "device", "dtype", "space"}:
            raise AttributeError(f"CompositeSpec has no key {item}")
        return self._specs[item]

    def __setitem__(self, key, value):
        if isinstance(key, tuple) and len(key) > 1:
            if key[0] not in self.keys(True):
                self[key[0]] = CompositeSpec()
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
        del self._specs[key]

    def encode(self, vals: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        if isinstance(vals, TensorDict):
            out = vals.select()  # create and empty tensordict similar to vals
        else:
            out = TensorDict({}, [], _run_checks=False)
        for key, item in vals.items():
            if item is None:
                raise RuntimeError(
                    "CompositeSpec.encode cannot be used with missing values."
                )
            try:
                out[key] = self[key].encode(item)
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

        for _key in self:
            if self[_key] is not None and (
                selected_keys is None or _key in selected_keys
            ):
                self._specs[_key].type_check(value[_key], _key)

    def is_in(self, val: Union[dict, TensorDictBase]) -> bool:
        for (key, item) in self._specs.items():
            if item is None:
                continue
            if not item.is_in(val.get(key)):
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
            key: self[key].rand(shape)
            for key in self.keys(True)
            if isinstance(key, str) and self[key] is not None
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
            leaves_only (bool, optional): if :obj:`False`, the values returned
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
            leaves_only (bool, optional): if :obj:`False`, the values returned
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
            leaves_only (bool, optional): if :obj:`False`, the values returned
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
            {key: item.clone() for key, item in self.items()},
            device=device,
            shape=self.shape,
        )

    def to_numpy(self, val: TensorDict, safe: bool = True) -> dict:
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
            raise ValueError("CompositeSpec.extend does not support negative shapes.")
        if any(s1 != s2 and s2 != 1 for s1, s2 in zip(shape[-self.ndim :], self.shape)):
            raise ValueError(
                f"The last {self.ndim} of the extended shape must match the"
                f"shape of the CompositeSpec in CompositeSpec.extend."
            )
        try:
            device = self.device
        except RuntimeError:
            device = self._device
        out = CompositeSpec(
            {
                key: value.expand((*shape, *value.shape[self.ndim :]))
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
            dim += len(self.shape)

        shape = _unsqueezed_shape(self.shape, dim)

        try:
            device = self.device
        except RuntimeError:
            device = self._device

        return CompositeSpec(
            {key: value.unsqueeze(dim) for key, value in self.items()},
            shape=shape,
            device=device,
        )


class LazyStackedCompositeSpec(_LazyStackedMixin[CompositeSpec], CompositeSpec):
    """A lazy representation of a stack of composite specs.

    Stacks composite specs together along one dimension.
    When random samples are drawn, a LazyStackedTensorDict is returned.

    Indexing is allowed but only along the stack dimension.

    This class is aimed to be used in multi-task and multi-agent settings, where
    heterogeneous specs may occur (same semantic but different shape).

    """

    def update(self, dict_or_spec: Union[CompositeSpec, Dict[str, TensorSpec]]) -> None:
        pass

    def __eq__(self, other):
        pass

    def to_numpy(self, val: TensorDict, safe: bool = True) -> dict:
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
        pass

    def values(self):
        for key in self.keys():
            yield self[key]

    def items(self):
        for key in self.keys():
            yield key, self[key]

    def keys(
        self,
        include_nested: bool = False,
        leaves_only: bool = False,
    ) -> KeysView:
        return self._specs[0].keys(
            include_nested=include_nested, leaves_only=leaves_only
        )

    def project(self, val: TensorDictBase) -> TensorDictBase:
        pass

    def is_in(self, val: Union[dict, TensorDictBase]) -> bool:
        pass

    def type_check(
        self,
        value: Union[torch.Tensor, TensorDictBase],
        selected_keys: Union[str, Optional[Sequence[str]]] = None,
    ):
        pass

    def __repr__(self) -> str:
        sub_str = ",\n".join(
            [indent(f"{k}: {repr(item)}", 4 * " ") for k, item in self.items()]
        )
        device_str = f"device={self._specs[0].device}"
        shape_str = f"shape={self.shape}"
        sub_str = ", ".join([sub_str, device_str, shape_str])
        return (
            f"LazyStackedCompositeSpec(\n{', '.join([sub_str, device_str, shape_str])})"
        )

    def encode(self, vals: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        pass

    def __delitem__(self, key):
        pass

    def __iter__(self):
        pass

    def __setitem__(self, key, value):
        pass

    @property
    def device(self) -> DEVICE_TYPING:
        return self._specs[0].device

    @property
    def ndim(self):
        return self.ndimension()

    def ndimension(self):
        return len(self.shape)

    def set(self, name, spec):
        if spec is not None:
            shape = spec.shape
            if shape[: self.ndim] != self.shape:
                raise ValueError(
                    "The shape of the spec and the CompositeSpec mismatch: the first "
                    f"{self.ndim} dimensions should match but got spec.shape={spec.shape} and "
                    f"CompositeSpec.shape={self.shape}."
                )
        self._specs[name] = spec


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
            if not isinstance(spec, TensorSpec):
                raise RuntimeError(
                    "Stacking specs cannot occur: Found more than one type of specs in the list."
                )
            if device != spec.device:
                raise RuntimeError(f"Devices differ, got {device} and {spec.device}")
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
