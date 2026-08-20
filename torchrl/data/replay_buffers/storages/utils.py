# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import torch
from tensordict import is_tensor_collection, lazy_stack, LazyStackedTensorDict
from torch.utils._pytree import tree_flatten, tree_map, tree_unflatten

from torchrl._utils import implement_for

from .ensemble import StorageEnsemble
from .list import CompressedListStorage, LazyStackStorage, ListStorage
from .store import StoreStorage
from .tensor import TensorStorage


def _collate_list_tensordict(x):
    out = torch.stack(x, 0)
    return out


@implement_for("torch", "2.4")
def _stack_anything(data):
    if is_tensor_collection(data[0]):
        return LazyStackedTensorDict.maybe_dense_stack(data)
    return tree_map(
        lambda *x: torch.stack(x),
        *data,
        is_leaf=lambda x: isinstance(x, torch.Tensor) or is_tensor_collection(x),
    )


@implement_for("torch", None, "2.4")
def _stack_anything(data):  # noqa: F811
    from tensordict import _pytree

    if not _pytree.PYTREE_REGISTERED_TDS:
        raise RuntimeError(
            "TensorDict is not registered within PyTree. "
            "If you see this error, it means tensordicts instances cannot be natively stacked using tree_map. "
            "To solve this issue, (a) upgrade pytorch to a version > 2.4, or (b) make sure TensorDict is registered in PyTree. "
            "If this error persists, open an issue on https://github.com/pytorch/rl/issues"
        )
    if is_tensor_collection(data[0]):
        return LazyStackedTensorDict.maybe_dense_stack(data)
    flat_trees = []
    spec = None
    for d in data:
        flat_tree, spec = tree_flatten(d)
        flat_trees.append(flat_tree)

    leaves = []
    for leaf in zip(*flat_trees):
        leaf = torch.stack(leaf)
        leaves.append(leaf)

    return tree_unflatten(leaves, spec)


def _collate_id(x):
    return x


def _get_default_collate(storage, _is_tensordict=False):
    if isinstance(storage, (LazyStackStorage, TensorStorage, StoreStorage)):
        return _collate_id
    elif isinstance(storage, CompressedListStorage):
        return lazy_stack
    elif isinstance(storage, (ListStorage, StorageEnsemble)):
        return _stack_anything
    else:
        raise NotImplementedError(
            f"Could not find a default collate_fn for storage {type(storage)}."
        )
