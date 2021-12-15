from copy import deepcopy
from dataclasses import dataclass
from numbers import Number
from typing import Tuple

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


def _default_dtype_and_device(dtype, device):
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
    pass


@dataclass
class Values:
    values: Tuple


@dataclass
class ContinuousBox(Box):
    minimum: torch.Tensor
    maximum: torch.Tensor

    def __iter__(self):
        yield self.minimum
        yield self.maximum


@dataclass
class DiscreteBox(Box):
    n: int
    register = invertible_dict()


@dataclass
class BinaryBox(Box):
    n: int


@dataclass
class TensorSpec:
    shape: torch.Size
    space: Box
    device: str = "cpu"
    dtype: torch.dtype = torch.float

    def encode(self, val: torch.Tensor):
        if not isinstance(val, torch.Tensor):
            try:
                val = torch.tensor(val, dtype=self.dtype)
            except ValueError:
                val = torch.tensor(deepcopy(val), dtype=self.dtype)
        self.assert_is_in(val)
        return val

    def to_numpy(self, val):
        self.assert_is_in(val)
        return val.cpu().numpy()

    def index(self, val):
        raise NotImplementedError

    def _project(self, val):
        raise NotImplementedError

    def is_in(self, val) -> bool:
        raise NotImplementedError

    def project(self, val):
        if not self.is_in(val):
            return self._project(val)
        return val

    def assert_is_in(self, value):
        assert self.is_in(
            value), f"Encoding failed because value is not in space. " \
                    f"Consider calling project(val) first. value was = {value}"

    def type_check(self, value, key):
        assert value.dtype is self.dtype, f"value.dtype={value.dtype} but {self.__class__.__name__}.dtype={self.dtype}"

    def rand(self, shape=torch.Size([])):
        raise NotImplementedError


@dataclass
class BoundedTensorSpec(TensorSpec):
    def __init__(self, minimum, maximum, device=None, dtype=None):
        dtype, device = _default_dtype_and_device(dtype, device)
        if not isinstance(minimum, torch.Tensor):
            minimum = torch.tensor(minimum, dtype=dtype, device=device)
        if not isinstance(maximum, torch.Tensor):
            maximum = torch.tensor(maximum, dtype=dtype, device=device)
        super().__init__((1,), ContinuousBox(minimum, maximum), device, dtype)

    def rand(self, shape=torch.Size([])):
        a, b = self.space
        return torch.zeros(*shape, *self.shape, device=self.device).uniform_() * (b - a) + a

    def _project(self, val):
        minimum = self.space.minimum.to(val.device)
        maximum = self.space.maximum.to(val.device)
        try:
            val = val.clamp_(minimum.item(), maximum.item())
        except:
            minimum = minimum.expand_as(val)
            maximum = maximum.expand_as(val)
            val[val < minimum] = minimum[val < minimum]
            val[val > maximum] = maximum[val > maximum]
        return val

    def is_in(self, val):
        return (val >= self.space.minimum.to(val.device)).all() and (val <= self.space.maximum.to(val.device)).all()


@dataclass
class OneHotDiscreteTensorSpec(TensorSpec):
    def __init__(self, n, device=None, dtype=torch.long, use_register=False):
        dtype, device = _default_dtype_and_device(dtype, device)
        self.use_register = use_register
        space = DiscreteBox(n, )
        shape = torch.Size((space.n,))
        super().__init__(shape, space, device, dtype)

    def rand(self, shape=torch.Size([])):
        return torch.nn.functional.gumbel_softmax(
            torch.rand(*shape, self.space.n, device=self.device), hard=True, dim=-1
        ).to(torch.long)

    def encode(self, val, space=None):
        val = torch.tensor(val, dtype=torch.long)
        if space is None:
            space = self.space

        if self.use_register:
            if val not in space.register:
                space.register[val] = len(space.register)
            val = space.register[val]

        val = torch.nn.functional.one_hot(val, space.n).to(torch.long)
        return val

    def to_numpy(self, val: torch.Tensor):
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

    def index(self, index: torch.Tensor, tensor_to_index: torch.Tensor):
        index = index.nonzero().squeeze()
        index = index.expand(*tensor_to_index.shape[:-1], index.shape[-1])
        return tensor_to_index.gather(-1, index)

    def _project(self, val):
        idx = val.sum(-1) != 1
        out = torch.nn.functional.gumbel_softmax(val.to(torch.float))
        out = (out == out.max(dim=-1, keepdim=True)[0]).to(torch.long)
        return out

    def is_in(self, val):
        return (val.sum(-1) == 1).all()


@dataclass
class UnboundedContinuousTensorSpec(TensorSpec):
    def __init__(self, device=None, dtype=None):
        dtype, device = _default_dtype_and_device(dtype, device)
        box = ContinuousBox(torch.tensor(-np.inf), torch.tensor(np.inf))
        super().__init__(torch.Size((1,)), box, device, dtype)

    def rand(self, shape=torch.Size([])):
        return torch.randn(*shape, *self.shape, device=self.device, dtype=self.dtype)

    def is_in(self, val):
        return True


@dataclass
class NdBoundedTensorSpec(BoundedTensorSpec):
    def __init__(self, minimum, maximum, shape=None, device=None, dtype=None):
        dtype, device = _default_dtype_and_device(dtype, device)
        if dtype is None:
            dtype = torch.get_default_dtype()
        if device is None:
            device = torch._get_default_device()

        if not isinstance(minimum, torch.Tensor):
            minimum = torch.tensor(minimum, dtype=dtype, device=device)
        if not isinstance(maximum, torch.Tensor):
            maximum = torch.tensor(maximum, dtype=dtype, device=device)

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
            assert len(minimum.shape) == len(shape)
            assert all(_s == _sa for _s, _sa in zip(shape, minimum.shape))
        self.shape = shape

        super(BoundedTensorSpec, self).__init__(
            shape, ContinuousBox(minimum, maximum), device, dtype
        )


@dataclass
class NdUnboundedContinuousTensorSpec(UnboundedContinuousTensorSpec):
    def __init__(self, shape, device=None, dtype=None):
        dtype, device = _default_dtype_and_device(dtype, device)
        super(UnboundedContinuousTensorSpec, self).__init__(shape=shape, space=None, device=device, dtype=dtype)


@dataclass
class BinaryDiscreteTensorSpec(TensorSpec):
    def __init__(self, n, device=None, dtype=torch.long):
        dtype, device = _default_dtype_and_device(dtype, device)
        shape = torch.Size((n,))
        box = BinaryBox(n)
        super().__init__(shape, box, device, dtype)

    def rand(self, shape=torch.Size([])):
        return (
            torch.zeros(*shape, *self.shape, device=self.device, dtype=self.dtype).bernoulli_()
        )

    def index(self, index: torch.Tensor, tensor_to_index: torch.Tensor):
        index = index.nonzero().squeeze()
        index = index.expand(*tensor_to_index.shape[:-1], index.shape[-1])
        return tensor_to_index.gather(-1, index)


@dataclass
class MultOneHotDiscreteTensorSpec(OneHotDiscreteTensorSpec):
    def __init__(self, nvec, device=None, dtype=torch.long, use_register=False):
        dtype, device = _default_dtype_and_device(dtype, device)
        shape = torch.Size((sum(nvec),))
        space = [DiscreteBox(n) for n in nvec]
        self.use_register = use_register
        super(OneHotDiscreteTensorSpec, self).__init__(shape, space, device, dtype)

    def rand(self, shape=torch.Size([])):
        x = torch.cat(
            [
                torch.nn.functional.one_hot(
                    torch.randint(space.n, (*shape, 1,), device=self.device), space.n,
                ).to(torch.long)
                for space in self.space
            ],
            -1,
        ).squeeze(-2)
        return x

    def encode(self, val):
        x = torch.cat(
            [
                super(MultOneHotDiscreteTensorSpec, self).encode(v, space)
                for v, space in zip(val, self.space)
            ],
            0,
        )
        return x

    def _split(self, val):
        vals = val.split([space.n for space in self.space], -1)
        return vals

    def to_numpy(self, val: torch.Tensor):
        vals = self._split(val)
        try:
            out = np.concatenate(tuple(val.argmax(-1).numpy() for val in vals), -1)
        except ValueError:
            out = np.array(tuple(val.argmax(-1).numpy() for val in vals))
        return out

    def index(self, index: torch.Tensor, tensor_to_index: torch.Tensor):
        indices = self._split(index)
        tensor_to_index = self._split(tensor_to_index)

        out = []
        for _index, _tensor_to_index in zip(indices, tensor_to_index):
            _index = _index.nonzero().squeeze()
            _index = _index.expand(*_tensor_to_index.shape[:-1], _index.shape[-1])
            out.append(_tensor_to_index.gather(-1, _index))
        return torch.cat(out, -1)

    def is_in(self, val):
        vals = self._split(val)
        return all([super().is_in(_val) for _val in vals])

    def _project(self, val):
        vals = self._split(val)
        return torch.cat([super()._project(_val) for _val in vals], -1)


class CompositeSpec(TensorSpec):

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

    def del_(self, key):
        del self._specs[key]

    def rand(self, shape=torch.Size([])):
        return {k: item.rand(shape) for k, item in self._specs.items()}

    def encode(self, vals: dict):
        out = {}
        for key, item in vals.items():
            out[key] = self[key].encode(item)
        return out

    def __repr__(self):
        return f"CompositeSpec({', '.join([item.__repr__() for k, item in self._specs.items()])})"

    def type_check(self, value, key):
        for _key in self:
            if _key in key:
                self._specs[_key].type_check(value, _key)
