# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from numbers import Number
from typing import Tuple, List, Union

import numpy as np
import torch

try:
    try:
        from functorch._C import is_batchedtensor, get_unwrapped
    except ImportError:
        from torch._C._functorch import is_batchedtensor, get_unwrapped

    _has_functorch = True
except ImportError:
    _has_functorch = False

from torchrl.data.utils import INDEX_TYPING


def _sub_index(tensor: torch.Tensor, idx: INDEX_TYPING) -> torch.Tensor:
    """Allows indexing of tensors with nested tuples.

     >>> sub_tensor1 = tensor[tuple1][tuple2]
     >>> sub_tensor2 = _sub_index(tensor, (tuple1, tuple2))
     >>> assert torch.allclose(sub_tensor1, sub_tensor2)

    Args:
        tensor (torch.Tensor): tensor to be indexed.
        idx (tuple of indices): indices sequence to be used.

    """
    if isinstance(idx, tuple) and len(idx) and isinstance(idx[0], tuple):
        idx0 = idx[0]
        idx1 = idx[1:]
        return _sub_index(_sub_index(tensor, idx0), idx1)
    return tensor[idx]


def _getitem_batch_size(
    shape: torch.Size,
    items: INDEX_TYPING,
) -> torch.Size:
    """Given an input shape and an index, returns the size of the resulting indexed tensor.

    This function is aimed to be used when indexing is an
    expensive operation.
    Args:
        shape (torch.Size): Input shape
        items (index): Index of the hypothetical tensor

    Returns:
        Size of the resulting object (tensor or tensordict)
    """
    # let's start with simple cases
    if isinstance(items, tuple) and len(items) == 1:
        items = items[0]
    if isinstance(items, int):
        return shape[1:]
    if isinstance(items, torch.Tensor) and items.dtype is torch.bool:
        return torch.Size([items.sum(), *shape[items.ndimension() :]])
    if (
        isinstance(items, (torch.Tensor, np.ndarray)) and len(items.shape) <= 1
    ) or isinstance(items, list):
        if len(items):
            return torch.Size([len(items), *shape[1:]])
        else:
            return shape[1:]

    if not isinstance(items, tuple):
        items = (items,)
    bs = []
    iter_bs = iter(shape)
    if all(isinstance(_item, torch.Tensor) for _item in items) and len(items) == len(
        shape
    ):
        shape0 = items[0].shape
        for _item in items[1:]:
            if _item.shape != shape0:
                raise RuntimeError(
                    f"all tensor indices must have the same shape, "
                    f"got {_item.shape} and {shape0}"
                )
        return shape0

    for _item in items:
        if isinstance(_item, slice):
            batch = next(iter_bs)
            v = len(range(*_item.indices(batch)))
        elif isinstance(_item, (list, torch.Tensor, np.ndarray)):
            batch = next(iter_bs)
            if isinstance(_item, torch.Tensor) and _item.dtype is torch.bool:
                v = _item.sum()
            else:
                v = len(_item)
        elif _item is None:
            v = 1
        elif isinstance(_item, Number):
            try:
                batch = next(iter_bs)
            except StopIteration:
                raise RuntimeError(
                    f"The shape {shape} is incompatible with " f"the index {items}."
                )
            continue
        else:
            raise NotImplementedError(
                f"batch dim cannot be computed for type {type(_item)}"
            )
        bs.append(v)
    list_iter_bs = list(iter_bs)
    bs += list_iter_bs
    return torch.Size(bs)


def convert_ellipsis_to_idx(idx: Union[Tuple, Ellipsis], batch_size: List[int]):
    """Given an index containing an ellipsis or just an ellipsis, converts any ellipsis to slice(None).

    Example:
        >>> idx = (..., 0)
        >>> batch_size = [1,2,3]
        >>> new_index = convert_ellipsis_to_idx(idx, batch_size)
        >>> print(new_index)
        (slice(None, None, None), slice(None, None, None), 0)

    Args:
        idx (tuple, Ellipsis): Input index
        batch_size (list): Shape of tensor to be indexed

    Returns:
        new_index (tuple): Output index
    """
    new_index = ()
    num_dims = len(batch_size)

    if idx is Ellipsis:
        idx = (...,)
    num_ellipsis = sum(_idx is Ellipsis for _idx in idx)
    if num_dims < (len(idx) - num_ellipsis):
        raise RuntimeError("Not enough dimensions in TensorDict for index provided.")

    start_pos, after_ellipsis_length = None, 0
    for i, item in enumerate(idx):
        if item is Ellipsis:
            if start_pos is not None:
                raise RuntimeError("An index can only have one ellipsis at most.")
            else:
                start_pos = i
        if item is not Ellipsis and start_pos is not None:
            after_ellipsis_length += 1

    before_ellipsis_length = start_pos
    ellipsis_length = num_dims - after_ellipsis_length - before_ellipsis_length

    new_index += idx[:start_pos]

    ellipsis_start = start_pos
    ellipsis_end = start_pos + ellipsis_length
    new_index += (slice(None),) * (ellipsis_end - ellipsis_start)

    new_index += idx[start_pos + 1 : start_pos + 1 + after_ellipsis_length]

    if len(new_index) != num_dims:
        raise RuntimeError(
            f"The new index {new_index} is incompatible with the dimensions of the batch size {num_dims}."
        )

    return new_index


def _get_shape(value):
    # we call it "legacy code"
    return value.shape


def _unwrap_value(value):
    # batch_dims = value.ndimension()
    if not isinstance(value, torch.Tensor):
        out = value
    elif is_batchedtensor(value):
        out = get_unwrapped(value)
    else:
        out = value
    return out
    # batch_dims = out.ndimension() - batch_dims
    # batch_size = out.shape[:batch_dims]
    # return out, batch_size
