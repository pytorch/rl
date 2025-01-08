# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

from typing import Callable, List

import torch
from torch.nn import Module


class BinaryToDecimal(Module):
    """A Module to convert binaries encoded tensors to decimals.

    This is a utility class that allow to convert a binary encoding tensor (e.g. `1001`) to
    its decimal value (e.g. `9`)

    Args:
        num_bits (int): the number of bits to use for the bases table.
            The number of bits must be lower or equal to the input length and the input length
            must be divisible by ``num_bits``. If ``num_bits`` is lower than the number of
            bits in the input, the end result will be aggregated on the last dimension using
            :func:`~torch.sum`.
        device (torch.device): the device where inputs and outputs are to be expected.
        dtype (torch.dtype): the output dtype.
        convert_to_binary (bool, optional): if ``True``, the input to the ``forward``
            method will be cast to a binary input using :func:`~torch.heavyside`.
            Defaults to ``False``.

    Examples:
        >>> binary_to_decimal = BinaryToDecimal(
        ...    num_bits=4, device="cpu", dtype=torch.int32, convert_to_binary=True
        ... )
        >>> binary = torch.Tensor([[0, 0, 1, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, 10, 0]])
        >>> decimal = binary_to_decimal(binary)
        >>> assert decimal.shape == (2,)
        >>> assert (decimal == torch.Tensor([3, 2])).all()
    """

    def __init__(
        self,
        num_bits: int,
        device: torch.device,
        dtype: torch.dtype,
        convert_to_binary: bool = False,
    ):
        super().__init__()
        self.convert_to_binary = convert_to_binary
        self.bases = 2 ** torch.arange(num_bits - 1, -1, -1, device=device, dtype=dtype)
        self.num_bits = num_bits
        self.zero_tensor = torch.zeros((1,), device=device)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        num_features = features.shape[-1]
        if self.num_bits > num_features:
            raise ValueError(f"{num_features=} is less than {self.num_bits=}")
        elif num_features % self.num_bits != 0:
            raise ValueError(f"{num_features=} is not divisible by {self.num_bits=}")

        binary_features = (
            torch.heaviside(features, self.zero_tensor)
            if self.convert_to_binary
            else features
        )
        feature_parts = binary_features.reshape(shape=(-1, self.num_bits))
        digits = torch.vmap(torch.dot, (None, 0))(
            self.bases, feature_parts.to(self.bases.dtype)
        )
        digits = digits.reshape(shape=(-1, features.shape[-1] // self.num_bits))
        aggregated_digits = torch.sum(digits, dim=-1)
        return aggregated_digits


class SipHash(Module):
    """A Module to Compute SipHash values for given tensors.

    A hash function module based on SipHash implementation in python. Input tensors should have shape ``[batch_size, num_features]``
    and the output shape will be ``[batch_size]``.

    Args:
        as_tensor (bool, optional): if ``True``, the bytes will be turned into integers
            through the builtin ``hash`` function and mapped to a tensor. Default: ``True``.

        .. warning:: This module relies on the builtin ``hash`` function.
            To get reproducible results across runs, the ``PYTHONHASHSEED`` environment
            variable must be set before the code is run (changing this value during code
            execution is without effect).

    Examples:
        >>> # Assuming we set PYTHONHASHSEED=0 prior to running this code
        >>> a = torch.tensor([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
        >>> b = a.clone()
        >>> hash_module = SipHash(as_tensor=True)
        >>> hash_a = hash_module(a)
        >>> hash_a
        tensor([-4669941682990263259, -3778166555168484291, -9122128731510687521])
        >>> hash_b = hash_module(b)
        >>> assert (hash_a == hash_b).all()
    """

    def __init__(self, as_tensor: bool = True):
        super().__init__()
        self.as_tensor = as_tensor

    def forward(self, x: torch.Tensor) -> torch.Tensor | List[bytes]:
        hash_values = []
        if x.dtype in (torch.bfloat16,):
            x = x.to(torch.float16)
        for x_i in x.detach().cpu().numpy():
            hash_value = x_i.tobytes()
            hash_values.append(hash_value)
        if not self.as_tensor:
            return hash_values
        result = torch.tensor([hash(x) for x in hash_values], dtype=torch.int64)
        return result


class RandomProjectionHash(SipHash):
    """A module that combines random projections with SipHash to get a low-dimensional tensor, easier to embed through :class:`~.SipHash`.

    This module requires sklearn to be installed.

    Keyword Args:
        n_components (int, optional): the low-dimensional number of components of the projections.
            Defaults to 16.
        dtype_cast (torch.dtype, optional): the dtype to cast the projection to.
            Defaults to ``torch.bfloat16``.
        as_tensor (bool, optional): if ``True``, the bytes will be turned into integers
            through the builtin ``hash`` function and mapped to a tensor. Default: ``True``.

        .. warning:: This module relies on the builtin ``hash`` function.
            To get reproducible results across runs, the ``PYTHONHASHSEED`` environment
            variable must be set before the code is run (changing this value during code
            execution is without effect).

        init_method: TODO
    """

    _N_COMPONENTS_DEFAULT = 16

    def __init__(
        self,
        *,
        n_components: int | None = None,
        dtype_cast=torch.bfloat16,
        as_tensor: bool = True,
        init_method: Callable[[torch.Tensor], torch.Tensor | None] | None = None,
        **kwargs,
    ):
        if n_components is None:
            n_components = self._N_COMPONENTS_DEFAULT

        super().__init__(as_tensor=as_tensor)
        self.register_buffer("_n_components", torch.as_tensor(n_components))

        self._init = False
        if init_method is None:
            init_method = torch.nn.init.normal_
        self.init_method = init_method

        self.dtype_cast = dtype_cast
        self.register_buffer("transform", torch.nn.UninitializedBuffer())

    @property
    def n_components(self):
        return self._n_components.item()

    def fit(self, x):
        """Fits the random projection to the input data."""
        self.transform.materialize(
            (x.shape[-1], self.n_components), dtype=self.dtype_cast, device=x.device
        )
        self.init_method(self.transform)
        self._init = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self._init:
            self.fit(x)
        elif not self._init:
            raise RuntimeError(
                f"The {type(self).__name__} has not been initialized. Call fit before calling this method."
            )
        x = x.to(self.dtype_cast) @ self.transform
        return super().forward(x)
