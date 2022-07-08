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


def fields_pin_memory(input):
    raise NotImplementedError
    # return tree.map_structure(lambda x: pin_memory(x), input)


def pin_memory(data: Tensor) -> Tensor:
    if isinstance(data, torch.Tensor):
        return data.pin_memory()
    else:
        return data


def to_numpy(data: Tensor) -> np.ndarray:
    return data.detach().cpu().numpy() if isinstance(data, torch.Tensor) else data


def fast_map(func, *inputs):
    raise NotImplementedError
    # flat_inputs = (tree.flatten(x) for x in inputs)
    # entries = zip(*flat_inputs)
    # return tree.unflatten_as(inputs[-1], [func(*x) for x in entries])


def stack_tensors(input):
    if not len(input):
        raise RuntimeError("input length must be non-null")
    if isinstance(input[0], torch.Tensor):
        return torch.stack(input)
    else:
        return np.stack(input)


def stack_fields(input):
    if not len(input):
        raise RuntimeError("stack_fields requires non-empty list if tensors")
    return fast_map(lambda *x: stack_tensors(x), *input)


def first_field(data) -> Tensor:
    raise NotImplementedError


def to_torch(
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


def cat_fields_to_device(
    input, device, pin_memory: bool = False, non_blocking: bool = False
):
    input_on_device = fields_to_device(input, device, pin_memory, non_blocking)
    return cat_fields(input_on_device)


def cat_fields(input):
    if not input:
        raise RuntimeError("cat_fields requires a non-empty input collection.")
    return fast_map(lambda *x: torch.cat(x), *input)


def fields_to_device(
    input, device, pin_memory: bool = False, non_blocking: bool = False
):  # type:ignore
    raise NotImplementedError
