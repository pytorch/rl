# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import typing
from typing import Any, Callable, List, Tuple, Union

import numpy as np
import torch
from tensordict import is_tensor_collection, LazyStackedTensorDict
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


def _consolidate_entries_td(
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
        ]  # list of exclusive keys per td
        lazy_keys_examples = {}  # set of all exclusive keys with an example for each
        for td_index in range(len(td.tensordicts)):  # gather all exclusive keys
            sub_td = td.tensordicts[td_index]
            if recurse_through_stack:
                sub_td = _consolidate_entries_td(
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

        for td_index in range(len(td.tensordicts)):  # add missing exclusive entries
            sub_td = td.tensordicts[td_index]
            for lazy_key in set(lazy_keys_examples.keys()).difference(
                lazy_keys_per_td[td_index]
            ):
                lazy_key_example = lazy_keys_examples[lazy_key]
                sub_td.set(
                    lazy_key,
                    _empty_like_td(lazy_key_example, sub_td.batch_size),
                )
            td.tensordicts[td_index] = sub_td

    if recurse_through_entries:
        for key in td.keys():
            shape = td.get_item_shape(key)
            if -1 not in shape:
                td.set(
                    key,
                    _consolidate_entries_td(
                        td.get(key), recurse_through_entries, recurse_through_stack
                    ),
                )

    return td


def _empty_like_td(td, batch_size):
    if is_tensor_collection(td):
        return td.empty()
    else:
        shape = [dim if i < len(batch_size) else 0 for i, dim in enumerate(td.shape)]

        return torch.empty(
            shape,
            dtype=td.dtype,
            device=td.device,
        )


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
