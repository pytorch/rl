from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from numbers import Number
from typing import Tuple, Union, Optional, Iterable, List

import numpy as np
import torch

__all__ = [
    "TensorSpec",
    "BoundedTensorSpec",
    "OneHotDiscreteTensorSpec",
    "UnboundedContinuousTensorSpec",
    "NdBoundedTensorSpec",
    "NdUnboundedContinuousTensorSpec",
    "BinaryDiscreteTensorSpec",
    "MultOneHotDiscreteTensorSpec",
    "CompositeSpec",
]

from torchrl.data.tensordict.tensordict import _TensorDict, TensorDict

DEVICE_TYPING = Union[torch.device, str, int]

INDEX_TYPING = Union[int, torch.Tensor, np.ndarray, slice, List]


def _default_dtype_and_device(
    dtype: Union[None, str, torch.dtype],
    device: Union[None, str, int, torch.device]
) -> Tuple[torch.dtype, torch.device]:
    if dtype is None:
        dtype = torch.get_default_dtype()
    if device is None:
        device = torch.device("cpu")
    return dtype, device


class invertible_dict(dict):
    def __init__(self, *args, inv_dict=dict(), **kwargs):
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
    """
    A box of values
    """

    def __iter__(self):
        raise NotImplementedError


@dataclass
class Values:
    values: Tuple


@dataclass
class ContinuousBox(Box):
    """
    A continuous box of values, in between a minimum and a maximum.

    """

    minimum: torch.Tensor
    maximum: torch.Tensor

    def __iter__(self):
        yield self.minimum
        yield self.maximum


@dataclass
class DiscreteBox(Box):
    """
    A box of discrete values

    """

    n: int
    register = invertible_dict()


@dataclass
class BinaryBox(Box):
    """
    A box of n binary values

    """

    n: int


@dataclass
class TensorSpec:
    """
    Parent class of the tensor meta-data containers for observation, actions and rewards.

    Properties:
        shape (torch.Size): size of the tensor
        space (Box): Box instance describing what kind of values can be expected
        device (torch.device): device of the tensor
        dtype (torch.dtype): dtype of the tensor
    """

    shape: torch.Size
    space: Box
    device: torch.device = torch.device("cpu")
    dtype: torch.dtype = torch.float
    domain: str = ""

    def encode(self, val: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """
        Encodes a value given the specified spec, and return the corresponding tensor.
        Args:
            val (np.ndarray or torch.Tensor): value to be encoded as tensor.

        Returns: torch.Tensor matching the required tensor specs.

        """
        if not isinstance(val, torch.Tensor):
            try:
                val = torch.tensor(val, dtype=self.dtype)
            except ValueError:
                val = torch.tensor(deepcopy(val), dtype=self.dtype)
        self.assert_is_in(val)
        return val

    def to_numpy(self, val: torch.Tensor) -> np.ndarray:
        """
        Returns the np.ndarray correspondent of an input tensor.

        Args:
            val (torch.Tensor): tensor to be transformed to numpy

        Returns: a np.ndarray

        """
        self.assert_is_in(val)
        return val.detach().cpu().numpy()

    def index(self, index: INDEX_TYPING, tensor_to_index: torch.Tensor) -> torch.Tensor:
        """
        Indexes the input tensor

        Args:
            index (int, torch.Tensor, slice or list): index of the tensor
            tensor_to_index: tensor to be indexed

        Returns: indexed tensor

        """
        raise NotImplementedError

    def _project(self, val: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def is_in(self, val: torch.Tensor) -> bool:
        """
        If the value `val` is in the box defined by the TensorSpec, returns True, otherwise False.

        Args:
            val (torch.Tensor): value to be checked

        Returns: boolean indicating if values belongs to the TensorSpec box

        """
        raise NotImplementedError

    def project(self, val: torch.Tensor) -> torch.Tensor:
        """
        If the input tensor is not in the TensorSpec box, it maps it back to it given some heuristic.

        Args:
            val (torch.Tensor): tensor to be mapped to the box.

        Returns: a torch.Tensor belonging to the TensorSpec box.

        """
        if not self.is_in(val):
            return self._project(val)
        return val

    def assert_is_in(self, value: torch.Tensor) -> None:
        """
        Asserts whether a tensor belongs to the box, and raises an exception otherwise.

        Args:
            value (torch.Tensor): value to be checked.

        """
        if not self.is_in(value):
            raise AssertionError(
                f"Encoding failed because value is not in space. "
                f"Consider calling project(val) first. value was = {value}"
            )

    def type_check(self, value: torch.Tensor, key=None) -> None:
        """
        Checks the input value dtype against the TensorSpec dtype and raises an exception if they don't match.

        Args:
            value (torch.Tensor): tensor whose dtype has to be checked

        """
        if not value.dtype is self.dtype:
            raise TypeError(
                f"value.dtype={value.dtype} but {self.__class__.__name__}.dtype={self.dtype}"
            )

    def rand(self, shape=torch.Size([])) -> torch.Tensor:
        """
        Returns a random tensor in the box. The sampling will be uniform unless the box is unbounded.

        Args:
            shape (torch.Size): shape of the random tensor

        Returns: a random tensor sampled in the TensorSpec box.

        """
        raise NotImplementedError

    def to(self, dest: Union[torch.dtype, DEVICE_TYPING]) -> "TensorSpec":
        if isinstance(dest, (torch.device, str, int)):
            self.device = torch.device(dest)
        else:
            self.dtype = dest
        return self


@dataclass
class BoundedTensorSpec(TensorSpec):
    """
    A bounded, unidimensional, continuous tensor spec.

    Args:
        minimum (np.ndarray, torch.Tensor or number): lower bound of the box.
        maximum (np.ndarray, torch.Tensor or number): upper bound of the box.
        device (str, int or torch.device, optional): device of the tensors.
        dtype (str or torch.dtype, optional): dtype of the tensors.
    """

    def __init__(
        self,
        minimum: Union[np.ndarray, torch.Tensor, float],
        maximum: Union[np.ndarray, torch.Tensor, float],
        device: Optional[DEVICE_TYPING] = None,
        dtype: Optional[Union[str, torch.dtype]] = None,
    ):
        dtype, device = _default_dtype_and_device(dtype, device)
        if not isinstance(minimum, torch.Tensor) or minimum.dtype is not dtype:
            minimum = torch.tensor(minimum, dtype=dtype, device=device)
        if not isinstance(maximum, torch.Tensor) or maximum.dtype is not dtype:
            maximum = torch.tensor(maximum, dtype=dtype, device=device)
        super().__init__(
            torch.Size([1, ]),
            ContinuousBox(minimum, maximum),
            device,
            dtype,
            "continuous"
        )

    def rand(self, shape=torch.Size([])) -> torch.Tensor:
        a, b = self.space
        out = (
            torch.zeros(
                *shape, *self.shape, dtype=self.dtype, device=self.device
            ).uniform_()
            * (b - a)
            + a
        )
        if (out > b).any():
            out[out > b] = b.expand_as(out)[out > b]
        if (out < a).any():
            out[out < a] = a.expand_as(out)[out < a]
        return out

    def _project(self, val: torch.Tensor) -> torch.Tensor:
        minimum = self.space.minimum.to(val.device)  # type: ignore
        maximum = self.space.maximum.to(val.device)  # type: ignore
        try:
            val = val.clamp_(minimum.item(), maximum.item())
        except:
            minimum = minimum.expand_as(val)
            maximum = maximum.expand_as(val)
            val[val < minimum] = minimum[val < minimum]
            val[val > maximum] = maximum[val > maximum]
        return val

    def is_in(self, val: torch.Tensor) -> bool:
        return (val >= self.space.minimum.to(val.device)).all() and (
            val <= self.space.maximum.to(val.device)
        ).all()  # type: ignore


@dataclass
class OneHotDiscreteTensorSpec(TensorSpec):
    """
    A unidimensional, one-hot discrete tensor spec.

    Args:
        n (int): number of possible outcomes.
        device (str, int or torch.device, optional): device of the tensors.
        dtype (str or torch.dtype, optional): dtype of the tensors.
        user_register (bool): experimental feature. If True, every integer will be mapped onto a binary vector
            in the order in which they appear. This feature is designed for environment with no a-priori definition of
            the number of possible outcomes (e.g. discrete outcomes are sampled from an arbitrary set, whose elements
            will be mapped in a register to a series of unique one-hot binary vectors).
    """

    def __init__(
        self,
        n: int,
        device: Optional[DEVICE_TYPING] = None,
        dtype: Optional[Union[str, torch.dtype]] = torch.long,
        use_register: bool = False,
    ):

        dtype, device = _default_dtype_and_device(dtype, device)
        self.use_register = use_register
        space = DiscreteBox(
            n,
        )
        shape = torch.Size((space.n,))
        super().__init__(shape, space, device, dtype, "discrete")

    def rand(self, shape=torch.Size([])) -> torch.Tensor:
        return torch.nn.functional.gumbel_softmax(
            torch.rand(*shape, self.space.n, device=self.device), hard=True, dim=-1
        ).to(torch.long)

    def encode(self, val: torch.Tensor, space: Optional[Box] = None) -> torch.Tensor:
        val = torch.tensor(val, dtype=torch.long)
        if space is None:
            space = self.space

        if self.use_register:
            if val not in space.register:
                space.register[val] = len(space.register)
            val = space.register[val]

        val = torch.nn.functional.one_hot(val, space.n).to(torch.long)
        return val

    def to_numpy(self, val: torch.Tensor) -> np.ndarray:
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

    def index(self, index: torch.Tensor, tensor_to_index: torch.Tensor) -> torch.Tensor:
        index = index.nonzero().squeeze()
        index = index.expand(*tensor_to_index.shape[:-1], index.shape[-1])
        return tensor_to_index.gather(-1, index)

    def _project(self, val: torch.Tensor) -> torch.Tensor:
        idx = val.sum(-1) != 1
        out = torch.nn.functional.gumbel_softmax(val.to(torch.float))
        out = (out == out.max(dim=-1, keepdim=True)[0]).to(torch.long)
        return out

    def is_in(self, val: torch.Tensor) -> bool:
        return (val.sum(-1) == 1).all()


@dataclass
class UnboundedContinuousTensorSpec(TensorSpec):
    """
    An unbounded, unidimensional, continuous tensor spec.

    Args:
        device (str, int or torch.device, optional): device of the tensors.
        dtype (str or torch.dtype, optional): dtype of the tensors.

    """

    def __init__(self, device=None, dtype=None):
        dtype, device = _default_dtype_and_device(dtype, device)
        box = ContinuousBox(torch.tensor(-np.inf), torch.tensor(np.inf))
        super().__init__(torch.Size((1,)), box, device, dtype, "composite")

    def rand(self, shape=torch.Size([])) -> torch.Tensor:
        return torch.randn(*shape, *self.shape, device=self.device, dtype=self.dtype)

    def is_in(self, val: torch.Tensor) -> bool:
        return True


@dataclass
class NdBoundedTensorSpec(BoundedTensorSpec):
    """
    A bounded, multi-dimensional, continuous tensor spec.

    Args:
        minimum (np.ndarray, torch.Tensor or number): lower bound of the box.
        maximum (np.ndarray, torch.Tensor or number): upper bound of the box.
        device (str, int or torch.device, optional): device of the tensors.
        dtype (str or torch.dtype, optional): dtype of the tensors.
    """

    def __init__(
        self,
        minimum: Union[float, torch.Tensor, np.ndarray],
        maximum: Union[float, torch.Tensor, np.ndarray],
        shape: Optional[torch.Size] = None,
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
        if dtype is not None and minimum.dtype is not dtype:
            minimum = minimum.to(dtype)
        if dtype is not None and maximum.dtype is not dtype:
            maximum = maximum.to(dtype)
        err_msg = (
            "NdBoundedTensorSpec requires the shape to be explicitely (via the shape argument) or "
            "implicitely defined (via either the minimum or the maximum or both). If the maximum and/or the "
            "minimum have a non-singleton shape, they must match the provided shape if this one is set "
            "explicitely."
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
            minimum = minimum.expand(*shape)
        elif minimum.ndimension():
            if shape is not None and shape != minimum.shape:
                raise RuntimeError(err_msg)
            shape = minimum.shape
            maximum = maximum.expand(*shape)
        elif shape is None:
            raise RuntimeError(err_msg)
        else:
            mimimum = minimum.expand(*shape)
            maximum = maximum.expand(*shape)

        if minimum.numel() > maximum.numel():
            maximum = maximum.expand_as(minimum)
        elif maximum.numel() > minimum.numel():
            minimum = minimum.expand_as(maximum)
        if shape is None:
            shape = minimum.shape
        else:
            if isinstance(shape, Number):
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

        super(BoundedTensorSpec, self).__init__(
            shape, ContinuousBox(minimum, maximum), device, dtype, "continuous"
        )


@dataclass
class NdUnboundedContinuousTensorSpec(UnboundedContinuousTensorSpec):
    """
    An unbounded, multi-dimensional, continuous tensor spec.

    Args:
        device (str, int or torch.device, optional): device of the tensors.
        dtype (str or torch.dtype, optional): dtype of the tensors.
    """

    def __init__(
        self,
        shape: Union[torch.Size, int],
        device: Optional[DEVICE_TYPING] = None,
        dtype: Optional[Union[str, torch.dtype]] = None,
    ):
        if isinstance(shape, int):
            shape = torch.Size([shape])

        dtype, device = _default_dtype_and_device(dtype, device)
        super(UnboundedContinuousTensorSpec, self).__init__(
            shape=shape, space=None, device=device, dtype=dtype, domain="continuous"
        )


@dataclass
class BinaryDiscreteTensorSpec(TensorSpec):
    """
    A binary discrete tensor spec.

    Args:
        n (int): length of the binary vector.
        device (str, int or torch.device, optional): device of the tensors.
        dtype (str or torch.dtype, optional): dtype of the tensors.

    """

    def __init__(
        self,
        n: int,
        device: Optional[DEVICE_TYPING] = None,
        dtype: Union[str, torch.dtype] = torch.long,
    ):
        dtype, device = _default_dtype_and_device(dtype, device)
        shape = torch.Size((n,))
        box = BinaryBox(n)
        super().__init__(shape, box, device, dtype, domain="discrete")

    def rand(self, shape=torch.Size([])) -> torch.Tensor:
        return torch.zeros(
            *shape, *self.shape, device=self.device, dtype=self.dtype
        ).bernoulli_()

    def index(self, index: torch.Tensor, tensor_to_index: torch.Tensor) -> torch.Tensor:
        index = index.nonzero().squeeze()
        index = index.expand(*tensor_to_index.shape[:-1], index.shape[-1])
        return tensor_to_index.gather(-1, index)

    def is_in(self, val: torch.Tensor) -> bool:
        return ((val == 0) | (val == 1)).all()


@dataclass
class MultOneHotDiscreteTensorSpec(OneHotDiscreteTensorSpec):
    """
    A concatenation of one-hot discrete tensor spec.

    Args:
        nvec (iterable of integers): cardinality of each of the elements of the tensor.
        device (str, int or torch.device, optional): device of the tensors.
        dtype (str or torch.dtype, optional): dtype of the tensors.

    Examples:
        >>> ts = MultOneHotDiscreteTensorSpec((3,2,3))
        >>> ts.is_in(torch.tensor([0,0,1,
        >>>                        0,1,
        >>>                        1,0,0])) # True
        >>> ts.is_in(torch.tensor([1,0,1,
        >>>                        0,1,
        >>>                        1,0,0])) # False

    """

    def __init__(
        self, nvec: Iterable[int], device=None, dtype=torch.long, use_register=False
    ):
        dtype, device = _default_dtype_and_device(dtype, device)
        shape = torch.Size((sum(nvec),))
        space = [DiscreteBox(n) for n in nvec]
        self.use_register = use_register
        super(OneHotDiscreteTensorSpec, self).__init__(
            shape, space, device, dtype, domain="discrete"
        )

    def rand(self, shape: torch.Size = torch.Size([])) -> torch.Tensor:
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
            val = torch.tensor(val)

        x = []
        for v, space in zip(val.unbind(-1), self.space):
            if not (v < space.n).all():
                raise RuntimeError(
                    f"value {v} is greater than the allowed max {space.n}"
                )
            x.append(super(MultOneHotDiscreteTensorSpec, self).encode(v, space))
        return torch.cat(x, -1)

    def _split(self, val: torch.Tensor) -> torch.Tensor:
        vals = val.split([space.n for space in self.space], dim=-1)
        return vals

    def to_numpy(self, val: torch.Tensor) -> np.ndarray:
        vals = self._split(val)
        out = torch.stack([val.argmax(-1) for val in vals], -1).numpy()
        return out

    def index(self, index: torch.Tensor, tensor_to_index: torch.Tensor) -> torch.Tensor:
        indices = self._split(index)
        tensor_to_index = self._split(tensor_to_index)

        out = []
        for _index, _tensor_to_index in zip(indices, tensor_to_index):
            _index = _index.nonzero().squeeze()
            _index = _index.expand(*_tensor_to_index.shape[:-1], _index.shape[-1])
            out.append(_tensor_to_index.gather(-1, _index))
        return torch.cat(out, -1)

    def is_in(self, val: torch.Tensor) -> bool:
        vals = self._split(val)
        return all(
            [super(MultOneHotDiscreteTensorSpec, self).is_in(_val) for _val in vals]
        )

    def _project(self, val: torch.Tensor) -> torch.Tensor:
        vals = self._split(val)
        return torch.cat([super()._project(_val) for _val in vals], -1)


class CompositeSpec(TensorSpec):
    """
    A composition of TensorSpecs.

    Args:
        **kwargs (key (str): value (TensorSpec)): dictionary of tensorspecs to be stored

    Examples:
        >>> observation_pixels_spec = NdBoundedTensorSpec(torch.zeros(3,32,32), torch.ones(3, 32, 32))
        >>> observation_vector_spec = NdBoundedTensorSpec(torch.zeros(33), torch.ones(33))
        >>> composite_spec = CompositeSpec(
        >>>     observation_pixels=observation_pixels_spec,
        >>>     observation_vector=observation_vector_spec)
        >>> td = TensorDict({"observation_pixels": torch.rand(10,3,32,32), "observation_vector": torch.rand(10,33)}, batch_size=[10])
        >>> print("td (rand) is within bounds: ", composite_spec.is_in(td))
        >>>
        >>> td = TensorDict({"observation_pixels": torch.randn(10,3,32,32), "observation_vector": torch.randn(10,33)}, batch_size=[10])
        >>> print("td (randn) is within bounds: ", composite_spec.is_in(td))
        >>> td_project = composite_spec.project(td)
        >>> print("td modification done in place: ", td_project is td)
        >>> print("check td is within bounds after projection: ", composite_spec.is_in(td_project))
        >>> print("random td: ", composite_spec.rand([3,]))
    """

    domain: str = "composite"

    def __init__(self, **kwargs):
        self._specs = kwargs

    def __getitem__(self, item):
        if item in {"shape", "device", "dtype", "space"}:
            raise AttributeError(f"CompositeSpec has no key {item}")
        return self._specs[item]

    def __setitem__(self, key, value):
        if key in {"shape", "device", "dtype", "space"}:
            raise AttributeError(f"CompositeSpec[{key}] cannot be set")
        self._specs[key] = value

    def __iter__(self):
        for k in self._specs:
            yield k

    def del_(self, key: str) -> None:
        del self._specs[key]

    def encode(self, vals: dict) -> dict:
        out = {}
        for key, item in vals.items():
            out[key] = self[key].encode(item)
        return out

    def __repr__(self) -> str:
        sub_str = [f"{k}: {item.__repr__()}" for k, item in self._specs.items()]
        sub_str = ",\n\t".join(sub_str)
        return f"CompositeSpec(\n\t{sub_str})"

    def type_check(self, value, key):
        for _key in self:
            if _key in key:
                self._specs[_key].type_check(value, _key)

    def is_in(self, val: Union[dict, _TensorDict]) -> bool:
        return all([self[key].is_in(val.get(key)) for key in self._specs])

    def project(self, val: _TensorDict) -> _TensorDict:
        for key in self._specs:
            _val = val.get(key)
            if not self._specs[key].is_in(_val):
                val.set(key, self._specs[key].project(_val))
        return val

    def rand(self, shape=torch.Size([])):
        return TensorDict(
            {key: value.rand(shape) for key, value in self._specs.items()},
            batch_size=shape,
        )
