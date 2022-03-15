from __future__ import annotations

from numbers import Number

import torch

from torchrl.data.utils import INDEX_TYPING


def _sub_index(tensor: torch.Tensor, idx: INDEX_TYPING) -> torch.Tensor:
    """Allows indexing of tensors with nested tuples, i.e.
    tensor[tuple1][tuple2] can be indexed via _sub_index(tensor, (tuple1,
    tuple2))
    """
    if isinstance(idx, tuple) and len(idx) and isinstance(idx[0], tuple):
        idx0 = idx[0]
        idx1 = idx[1:]
        return _sub_index(_sub_index(tensor, idx0), idx1)
    return tensor[idx]


def _getitem_batch_size(
    shape: torch.Size,
    items: INDEX_TYPING,
):
    """
    Given an input shape and an index, returns the size of the resulting
    indexed tensor.

    This function is aimed to be used when indexing is an
    expensive operation.
    Args:
        shape: Input shape
        items: Index of the hypothetical tensor

    Returns:

    """
    if not isinstance(items, tuple):
        items = (items,)
    bs = []
    iter_bs = iter(shape)
    if all(isinstance(_item, torch.Tensor) for _item in items) and len(
        items
    ) == len(shape):
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
        elif isinstance(_item, (list, torch.Tensor)):
            batch = next(iter_bs)
            v = len(_item)
        elif _item is None:
            v = 1
        elif isinstance(_item, Number):
            batch = next(iter_bs)
            continue
        else:
            raise NotImplementedError(
                f"batch dim cannot be computed for type {type(_item)}"
            )
        bs.append(v)
    list_iter_bs = list(iter_bs)
    bs += list_iter_bs
    return torch.Size(bs)
