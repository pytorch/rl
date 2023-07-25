# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import typing
from typing import Any, Callable, List, Tuple, Union

import numpy as np
import torch
from tensordict import (
    is_tensor_collection,
    LazyStackedTensorDict,
    TensorDict,
    TensorDictBase,
)
from torch import Tensor


numpy_to_torch_dtype_dict = {
    np.dtype("bool"): torch.bool,
    np.dtype("uint8"): torch.uint8,
    np.dtype("int8"): torch.int8,
    np.dtype("int16"): torch.int16,
    np.dtype("int32"): torch.int32,
    np.dtype("int64"): torch.int64,
    np.dtype("float16"): torch.float16,
    np.dtype("float32"): torch.float32,
    np.dtype("float64"): torch.float64,
    np.dtype("complex64"): torch.complex64,
    np.dtype("complex128"): torch.complex128,
}
torch_to_numpy_dtype_dict = {
    value: key for key, value in numpy_to_torch_dtype_dict.items()
}
DEVICE_TYPING = Union[torch.device, str, int]
if hasattr(typing, "get_args"):
    DEVICE_TYPING_ARGS = typing.get_args(DEVICE_TYPING)
else:
    DEVICE_TYPING_ARGS = (torch.device, str, int)

INDEX_TYPING = Union[None, int, slice, str, Tensor, List[Any], Tuple[Any, ...]]


def dense_stack_tds(
    td_list: typing.Sequence[TensorDictBase], stack_dim: int = 0
) -> TensorDictBase:
    """Dnesely stack a list of TensorDictBase objects given that they have the same structure."""
    shape = list(td_list[0].shape)
    shape.insert(stack_dim, len(td_list))

    out = td_list[0].unsqueeze(stack_dim).expand(shape).clone()

    return torch.stack(td_list, dim=stack_dim, out=out)


def _unlazyfy_td(
    td,
    recurse_through_entries: bool = True,
    recurse_through_stack: bool = True,
):
    """Given a TensorDictBase, removes lazy keys by adding 0 shaped tensors."""
    td = td.clone()

    if not is_tensor_collection(td):
        return td

    if isinstance(td, LazyStackedTensorDict):
        keys = set(td.keys())  # shared keys
        lazy_keys_per_td = [
            set() for _ in range(len(td.tensordicts))
        ]  # list of lazy keys per td
        lazy_keys_examples = {}  # set of all lazy keys with an example for each
        for td_index in range(len(td.tensordicts)):  # gather all lazy keys
            sub_td = td.tensordicts[td_index]
            if recurse_through_stack:
                sub_td = _unlazyfy_td(
                    sub_td, recurse_through_entries, recurse_through_stack
                )
                td.tensordicts[td_index] = sub_td
            for sub_td_key in sub_td.keys():
                if sub_td_key not in keys:  # lazy key
                    lazy_keys_per_td[td_index].add(sub_td_key)
                    if sub_td_key not in lazy_keys_examples:
                        shape = sub_td.get_item_shape(sub_td_key)
                        if -1 not in shape:
                            value = sub_td.get(sub_td_key)
                        else:
                            # sub_td_key is het leaf so lets recurse to a dense version of it to get the example
                            temp_td = sub_td
                            while isinstance(temp_td, LazyStackedTensorDict):
                                # we need to grab the het tensor from the inner nesting level
                                temp_td = temp_td.tensordicts[0]
                            value = temp_td.get(sub_td_key)

                        lazy_keys_examples.update({sub_td_key: value})

        for td_index in range(len(td.tensordicts)):  # add missing lazy entries
            sub_td = td.tensordicts[td_index]
            for lazy_key in set(lazy_keys_examples.keys()).difference(
                lazy_keys_per_td[td_index]
            ):
                lazy_key_example = lazy_keys_examples[lazy_key]
                sub_td.set(
                    lazy_key,
                    _empty_like(lazy_key_example, sub_td.batch_size),
                )
            td.tensordicts[td_index] = sub_td

    if recurse_through_entries:
        for key in td.keys():
            shape = td.get_item_shape(key)
            if -1 not in shape:
                td.set(
                    key,
                    _unlazyfy_td(
                        td.get(key), recurse_through_entries, recurse_through_stack
                    ),
                )

    return td


def _relazyfy_td(
    td,
):
    """Given a TensorDictBase, restores lazy keys by removing 0 shaped tensors and related orphan tensordicts."""
    if not is_tensor_collection(td):
        return None if td.numel() == 0 else td.clone()

    td = td.clone()

    if isinstance(td, LazyStackedTensorDict):
        for td_index in range(len(td.tensordicts)):
            sub_td = td.tensordicts[td_index]
            sub_td = _relazyfy_td(sub_td)
            td.tensordicts[td_index] = sub_td

    for key in list(td.keys()):
        shape = td.get_item_shape(key)
        if -1 not in shape:
            value = _relazyfy_td(td.get(key))
            if value is None:
                del td[key]
            else:
                td.set(
                    key,
                    value,
                )

    if isinstance(td, TensorDict) and not len(td.keys()):
        return None
    return td


def _empty_like(td, batch_size):
    if is_tensor_collection(td):
        return td.empty()
    else:
        shape = [dim if i < len(batch_size) else 0 for i, dim in enumerate(td.shape)]

        return torch.empty(
            shape,
            dtype=td.dtype,
            device=td.device,
        )


def _check_no_lazy_keys(td, recurse: bool = True):
    """Given a TensorDictBase, returns true if there are no lazy keys."""
    if isinstance(td, LazyStackedTensorDict):
        keys = set(td.keys())
        for inner_td in td.tensordicts:
            if recurse and not _check_no_lazy_keys(inner_td):
                return False
            if set(inner_td.keys()) != keys:
                return False
    elif isinstance(td, TensorDict) and recurse:
        for i in td.values():
            if not _check_no_lazy_keys(i):
                return False
    elif isinstance(td, torch.Tensor):
        return True
    else:
        return False

    return True


def _all_eq(
    td: Union[TensorDictBase, torch.Tensor],
    other: Union[TensorDictBase, torch.Tensor],
):
    """Returns true if the two classes match all entries in the keys and stack dimensions."""
    if td.__class__ != other.__class__:
        return False

    if td.shape != other.shape or td.device != other.device:
        return False

    if isinstance(td, LazyStackedTensorDict):
        if td.stack_dim != other.stack_dim:
            return False
        for stacked_td, stacked_other in zip(td.tensordicts, other.tensordicts):
            if not _all_eq(stacked_td, stacked_other):
                return False
    elif isinstance(td, TensorDictBase):
        td_keys = set(td.keys())
        other_keys = set(other.keys())
        if td_keys != other_keys:
            return False
        for key in td_keys:
            if not _all_eq(td[key], other[key]):
                return False
    elif isinstance(td, torch.Tensor):
        return torch.equal(td, other)
    else:
        raise ValueError("_all_eq was provided arguments from the wrong class")

    return True


class CloudpickleWrapper(object):
    """A wrapper for functions that allow for serialization in multiprocessed settings."""

    def __init__(self, fn: Callable, **kwargs):
        if fn.__class__.__name__ == "EnvCreator":
            raise RuntimeError(
                "CloudpickleWrapper usage with EnvCreator class is "
                "prohibited as it breaks the transmission of shared tensors."
            )
        self.fn = fn
        self.kwargs = kwargs

    def __getstate__(self):
        import cloudpickle

        return cloudpickle.dumps((self.fn, self.kwargs))

    def __setstate__(self, ob: bytes):
        import pickle

        self.fn, self.kwargs = pickle.loads(ob)

    def __call__(self, *args, **kwargs) -> Any:
        kwargs = {k: item for k, item in kwargs.items()}
        kwargs.update(self.kwargs)
        return self.fn(**kwargs)
