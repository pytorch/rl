# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Callable, Optional, Tuple

import torch
from torch.utils._pytree import tree_map


class FiniteTensor(torch.Tensor):
    """A finite tensor.

    If the data contained in this tensor contain non-finite values (nan or inf)
    a :obj:`RuntimeError` will be thrown.

    """

    @staticmethod
    def __new__(cls, elem: torch.Tensor, *args, **kwargs):
        if not torch.isfinite(elem).all():
            raise RuntimeError("FiniteTensor encountered a non-finite tensor.")
        return torch.Tensor._make_subclass(cls, elem, elem.requires_grad)

    def __repr__(self) -> str:
        return f"FiniteTensor({super().__repr__()})"

    @classmethod
    def __torch_dispatch__(
        cls,
        func: Callable,
        types,
        args: Tuple = (),
        kwargs: Optional[dict] = None,
    ):
        # TODO: also explicitly recheck invariants on inplace/out mutation
        if kwargs:
            raise Exception("Expected empty kwargs")
        rs = func(*args)
        return tree_map(
            lambda e: FiniteTensor(e) if isinstance(e, torch.Tensor) else e, rs
        )
