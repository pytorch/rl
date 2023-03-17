# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# import tree
import typing
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
