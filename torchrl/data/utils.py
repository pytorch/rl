# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import typing
from typing import Any, Callable, List, Tuple, Union

import numpy as np
import torch
from torch import Tensor

from torchrl.data.tensor_specs import (
    CompositeSpec,
    LazyStackedCompositeSpec,
    LazyStackedTensorSpec,
    TensorSpec,
)

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


def consolidate_spec(
    spec: CompositeSpec,
    recurse_through_entries: bool = True,
    recurse_through_stack: bool = True,
):
    """Given a TensorSpec, removes exclusive keys by adding 0 shaped specs.

    Args:
        spec (CompositeSpec): the spec to be consolidated.
        recurse_through_entries (bool): if True, call the function recursively on all entries of the spec.
            Default is True.
        recurse_through_stack (bool): if True, if the provided spec is lazy, the function recursively
            on all specs in its list. Default is True.

    """
    spec = spec.clone()

    if not isinstance(spec, (CompositeSpec, LazyStackedCompositeSpec)):
        return spec

    if isinstance(spec, LazyStackedCompositeSpec):
        keys = set(spec.keys())  # shared keys
        lazy_keys_per_spec = [
            set() for _ in range(len(spec._specs))
        ]  # list of exclusive keys per td
        lazy_keys_examples = {}  # set of all exclusive keys with an example for each
        for spec_index in range(len(spec._specs)):  # gather all lazy keys
            sub_spec = spec._specs[spec_index]
            if recurse_through_stack:
                sub_spec = consolidate_spec(
                    sub_spec, recurse_through_entries, recurse_through_stack
                )
                spec._specs[spec_index] = sub_spec
            for sub_spec_key in sub_spec.keys():
                if sub_spec_key not in keys:  # lazy key
                    lazy_keys_per_spec[spec_index].add(sub_spec_key)
                    if sub_spec_key not in lazy_keys_examples:
                        value = sub_spec[sub_spec_key]
                        lazy_keys_examples.update({sub_spec_key: value})

        for sub_spec, lazy_keys in zip(
            spec._specs, lazy_keys_per_spec
        ):  # add missing exclusive entries
            for lazy_key in set(lazy_keys_examples.keys()).difference(lazy_keys):
                lazy_key_example = lazy_keys_examples[lazy_key]
                sub_spec.set(
                    lazy_key,
                    _empty_like_spec(lazy_key_example, sub_spec.shape),
                )

    if recurse_through_entries:
        for key, value in spec.items():
            if isinstance(value, (CompositeSpec, LazyStackedCompositeSpec)):
                spec.set(
                    key,
                    consolidate_spec(
                        value, recurse_through_entries, recurse_through_stack
                    ),
                )
    return spec


def _empty_like_spec(spec, shape):
    if isinstance(spec, (CompositeSpec, LazyStackedCompositeSpec)):
        return spec.empty()
    elif isinstance(spec, LazyStackedTensorSpec):
        shape = list(shape[: spec.stack_dim]) + list(shape[spec.stack_dim + 1 :])
        return torch.stack(
            [_empty_like_spec(sub_spec, shape) for sub_spec in spec._specs],
            spec.stack_dim,
        )
    else:
        spec_shape = spec.shape
        shape = [dim if i < len(shape) else 0 for i, dim in enumerate(spec_shape)]
        spec = spec[(0,) * len(spec_shape)]
        spec = spec.expand(shape)

        return spec


def check_no_exclusive_keys(spec: TensorSpec, recurse: bool = True):
    """Given a TensorSpec, returns true if there are no exclusive keys.

    Args:
        spec (TensorSpec): the spec to check
        recurse (bool): if True, check recursively in nested specs. Default is True.
    """
    if isinstance(spec, LazyStackedCompositeSpec):
        keys = set(spec.keys())
        for inner_td in spec._specs:
            if recurse and not check_no_exclusive_keys(inner_td):
                return False
            if set(inner_td.keys()) != keys:
                return False
    elif isinstance(spec, CompositeSpec) and recurse:
        for value in spec.values():
            if not check_no_exclusive_keys(value):
                return False
    else:
        return True
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
