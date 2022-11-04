# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# import tree
import typing
from functools import wraps
from typing import Union

import numpy as np
import torch
from torch import Tensor

INT_CLASSES_TYPING = Union[int, np.integer]
if hasattr(typing, "get_args"):
    INT_CLASSES = typing.get_args(INT_CLASSES_TYPING)
else:
    # python 3.7
    INT_CLASSES = (int, np.integer)


def _to_numpy(data: Tensor) -> np.ndarray:
    return data.detach().cpu().numpy() if isinstance(data, torch.Tensor) else data


def _to_torch(
    data: Tensor, device, pin_memory: bool = False, non_blocking: bool = False
) -> torch.Tensor:
    if isinstance(data, np.generic):
        return torch.tensor(data, device=device)

    if isinstance(data, np.ndarray):
        data = torch.from_numpy(data)

    if pin_memory:
        data = data.pin_memory()
    if device is not None:
        data = data.to(device, non_blocking=non_blocking)

    return data


def accept_remote_rref_invocation(func):
    """Object method decorator that allows a method to be invoked remotely by passing the `rpc.RRef` associated with the remote object construction as first argument in place of the object reference."""

    @wraps(func)
    def unpack_rref_and_invoke_function(self, *args, **kwargs):
        if isinstance(self, torch._C._distributed_rpc.PyRRef):
            self = self.local_value()
        return func(self, *args, **kwargs)

    return unpack_rref_and_invoke_function


def accept_remote_rref_udf_invocation(decorated_class):
    """Class decorator that applies `accept_remote_rref_invocation` to all public methods."""
    # ignores private methods
    for name in dir(decorated_class):
        method = getattr(decorated_class, name)
        if callable(method) and not name.startswith("_"):
            setattr(decorated_class, name, accept_remote_rref_invocation(method))
    return decorated_class
