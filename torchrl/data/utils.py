# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import functools
import typing
from typing import Any, Callable, List, Tuple, Union

import numpy as np
import torch
from torch import Tensor
from torchrl.data.tensor_specs import (
    Binary,
    Categorical,
    Composite,
    MultiCategorical,
    MultiOneHot,
    OneHot,
    Stacked,
    StackedComposite,
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


ACTION_SPACE_MAP = {
    OneHot: "one_hot",
    MultiOneHot: "mult_one_hot",
    Binary: "binary",
    Categorical: "categorical",
    "one_hot": "one_hot",
    "one-hot": "one_hot",
    "mult_one_hot": "mult_one_hot",
    "mult-one-hot": "mult_one_hot",
    "multi_one_hot": "mult_one_hot",
    "multi-one-hot": "mult_one_hot",
    "binary": "binary",
    "categorical": "categorical",
    MultiCategorical: "multi_categorical",
    "multi_categorical": "multi_categorical",
    "multi-categorical": "multi_categorical",
    "multi_discrete": "multi_categorical",
    "multi-discrete": "multi_categorical",
}


def consolidate_spec(
    spec: Composite,
    recurse_through_entries: bool = True,
    recurse_through_stack: bool = True,
):
    """Given a TensorSpec, removes exclusive keys by adding 0 shaped specs.

    Args:
        spec (Composite): the spec to be consolidated.
        recurse_through_entries (bool): if True, call the function recursively on all entries of the spec.
            Default is True.
        recurse_through_stack (bool): if True, if the provided spec is lazy, the function recursively
            on all specs in its list. Default is True.

    """
    spec = spec.clone()

    if not isinstance(spec, (Composite, StackedComposite)):
        return spec

    if isinstance(spec, StackedComposite):
        keys = set(spec.keys())  # shared keys
        exclusive_keys_per_spec = [
            set() for _ in range(len(spec._specs))
        ]  # list of exclusive keys per td
        exclusive_keys_examples = (
            {}
        )  # map of all exclusive keys to a list of their values
        for spec_index in range(len(spec._specs)):  # gather all exclusive keys
            sub_spec = spec._specs[spec_index]
            if recurse_through_stack:
                sub_spec = consolidate_spec(
                    sub_spec, recurse_through_entries, recurse_through_stack
                )
                spec._specs[spec_index] = sub_spec
            for sub_spec_key in sub_spec.keys():
                if sub_spec_key not in keys:  # exclusive key
                    exclusive_keys_per_spec[spec_index].add(sub_spec_key)
                    value = sub_spec[sub_spec_key]
                    if sub_spec_key in exclusive_keys_examples:
                        exclusive_keys_examples[sub_spec_key].append(value)
                    else:
                        exclusive_keys_examples.update({sub_spec_key: [value]})

        for sub_spec, exclusive_keys in zip(
            spec._specs, exclusive_keys_per_spec
        ):  # add missing exclusive entries
            for exclusive_key in set(exclusive_keys_examples.keys()).difference(
                exclusive_keys
            ):
                exclusive_keys_example_list = exclusive_keys_examples[exclusive_key]
                sub_spec.set(
                    exclusive_key,
                    _empty_like_spec(exclusive_keys_example_list, sub_spec.shape),
                )

    if recurse_through_entries:
        for key, value in spec.items():
            if isinstance(value, (Composite, StackedComposite)):
                spec.set(
                    key,
                    consolidate_spec(
                        value, recurse_through_entries, recurse_through_stack
                    ),
                )
    return spec


def _empty_like_spec(specs: list[TensorSpec], shape):
    for spec in specs[1:]:
        if spec.__class__ != specs[0].__class__:
            raise ValueError(
                "Found same key in lazy specs corresponding to entries with different classes"
            )
    spec = specs[0]
    if isinstance(spec, (Composite, StackedComposite)):
        # the exclusive key has values which are CompositeSpecs ->
        # we create an empty composite spec with same batch size
        return spec.empty()
    elif isinstance(spec, Stacked):
        # the exclusive key has values which are LazyStackedTensorSpecs ->
        # we create a LazyStackedTensorSpec with the same shape (aka same -1s) as the first in the list.
        # this will not add any new -1s when they are stacked
        shape = list(shape[: spec.stack_dim]) + list(shape[spec.stack_dim + 1 :])
        return Stacked(
            *[_empty_like_spec(spec._specs, shape) for _ in spec._specs],
            dim=spec.stack_dim,
        )
    else:
        # the exclusive key has values which are TensorSpecs ->
        # if the shapes of the values are all the same, we create a TensorSpec with leading shape `shape` and following dims 0 (having the same ndims as the values)
        # if the shapes of the values differ,  we create a TensorSpec with 0 size in the differing dims
        spec_shape = list(spec.shape)

        for dim_index in range(len(spec_shape)):
            hetero_dim = False
            for sub_spec in specs:
                if sub_spec.shape[dim_index] != spec.shape[dim_index]:
                    hetero_dim = True
                    break
            if hetero_dim:
                spec_shape[dim_index] = 0

        if 0 not in spec_shape:  # the values have all same shape
            spec_shape = [
                dim if i < len(shape) else 0 for i, dim in enumerate(spec_shape)
            ]

        spec = spec[(0,) * len(spec.shape)]
        spec = spec.expand(spec_shape)

        return spec


def check_no_exclusive_keys(spec: TensorSpec, recurse: bool = True):
    """Given a TensorSpec, returns true if there are no exclusive keys.

    Args:
        spec (TensorSpec): the spec to check
        recurse (bool): if True, check recursively in nested specs. Default is True.
    """
    if isinstance(spec, StackedComposite):
        keys = set(spec.keys())
        for inner_td in spec._specs:
            if recurse and not check_no_exclusive_keys(inner_td):
                return False
            if set(inner_td.keys()) != keys:
                return False
    elif isinstance(spec, Composite) and recurse:
        for value in spec.values():
            if not check_no_exclusive_keys(value):
                return False
    else:
        return True
    return True


def contains_lazy_spec(spec: TensorSpec) -> bool:
    """Returns true if a spec contains lazy stacked specs.

    Args:
        spec (TensorSpec): the spec to check

    """
    if isinstance(spec, (Stacked, StackedComposite)):
        return True
    elif isinstance(spec, Composite):
        for inner_spec in spec.values():
            if contains_lazy_spec(inner_spec):
                return True
    return False


class _CloudpickleWrapperMeta(type):
    def __call__(cls, obj):
        if isinstance(obj, cls):
            return obj
        else:
            return super().__call__(obj)


class CloudpickleWrapper(metaclass=_CloudpickleWrapperMeta):
    """A wrapper for functions that allow for serialization in multiprocessed settings."""

    def __init__(self, fn: Callable, **kwargs):
        if fn.__class__.__name__ == "EnvCreator":
            raise RuntimeError(
                "CloudpickleWrapper usage with EnvCreator class is "
                "prohibited as it breaks the transmission of shared tensors."
            )
        self.fn = fn
        self.kwargs = kwargs

        functools.update_wrapper(self, getattr(fn, "forward", fn))

    def __getstate__(self):
        import cloudpickle

        return cloudpickle.dumps((self.fn, self.kwargs))

    def __setstate__(self, ob: bytes):
        import pickle

        self.fn, self.kwargs = pickle.loads(ob)
        functools.update_wrapper(self, self.fn)

    def __call__(self, *args, **kwargs) -> Any:
        kwargs.update(self.kwargs)
        return self.fn(*args, **kwargs)


def _process_action_space_spec(action_space, spec):
    original_spec = spec
    composite_spec = False
    if isinstance(spec, Composite):
        # this will break whenever our action is more complex than a single tensor
        try:
            if "action" in spec.keys():
                _key = "action"
            else:
                # the first key is the action
                for _key in spec.keys(True, True):
                    if isinstance(_key, tuple) and _key[-1] == "action":
                        break
                else:
                    raise KeyError
            spec = spec[_key]
            composite_spec = True
        except KeyError:
            raise KeyError(
                "action could not be found in the spec. Make sure "
                "you pass a spec that is either a native action spec or a composite action spec "
                "with a leaf 'action' entry. Otherwise, simply remove the spec and use the action_space only."
            )
    if action_space is not None:
        if isinstance(action_space, Composite):
            raise ValueError("action_space cannot be of type Composite.")
        if (
            spec is not None
            and isinstance(action_space, TensorSpec)
            and action_space is not spec
        ):
            raise ValueError(
                "Passing an action_space as a TensorSpec and a spec isn't allowed, unless they match."
            )
        if isinstance(action_space, TensorSpec):
            spec = action_space
        action_space = _find_action_space(action_space)
        # check that the spec and action_space match
        if spec is not None and _find_action_space(spec) != action_space:
            raise ValueError(
                f"The action spec and the action space do not match: got action_space={action_space} and spec={spec}."
            )
    elif spec is not None:
        action_space = _find_action_space(spec)
    else:
        raise ValueError(
            "Neither action_space nor spec was defined. The action space cannot be inferred."
        )
    if composite_spec:
        spec = original_spec
    return action_space, spec


def _find_action_space(action_space) -> str:
    if isinstance(action_space, TensorSpec):
        if isinstance(action_space, Composite):
            if "action" in action_space.keys():
                _key = "action"
            else:
                # the first key is the action
                for _key in action_space.keys(True, True):
                    if isinstance(_key, tuple) and _key[-1] == "action":
                        break
                else:
                    raise KeyError
            action_space = action_space[_key]
        action_space = type(action_space)
    try:
        action_space = ACTION_SPACE_MAP[action_space]
    except KeyError:
        raise ValueError(
            f"action_space was not specified/not compatible and could not be retrieved from the value network. Got action_space={action_space}."
        )
    return action_space
