# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from .base import Storage
from .ensemble import StorageEnsemble
from .list import CompressedListStorage, LazyStackStorage, ListStorage
from .store import StoreStorage
from .tensor import (
    _cleanup_all_memmap_storages as _cleanup_all_memmap_storages,
    _ensure_cleanup_handlers as _ensure_cleanup_handlers,
    _flip_list as _flip_list,
    _make_empty_memmap as _make_empty_memmap,
    _make_memmap as _make_memmap,
    _mem_map_tensor_as_tensor as _mem_map_tensor_as_tensor,
    _MEMMAP_STORAGE_REGISTRY as _MEMMAP_STORAGE_REGISTRY,
    _register_cleanup_handlers as _register_cleanup_handlers,
    _signal_cleanup_handler as _signal_cleanup_handler,
    LazyMemmapStorage,
    LazyTensorStorage,
    TensorStorage,
)
from .utils import (
    _collate_id as _collate_id,
    _collate_list_tensordict as _collate_list_tensordict,
    _get_default_collate as _get_default_collate,
    _stack_anything as _stack_anything,
)

__all__ = [
    "CompressedListStorage",
    "LazyMemmapStorage",
    "LazyStackStorage",
    "LazyTensorStorage",
    "ListStorage",
    "Storage",
    "StorageEnsemble",
    "StoreStorage",
    "TensorStorage",
]

for _export in (
    _cleanup_all_memmap_storages,
    _signal_cleanup_handler,
    _register_cleanup_handlers,
    _ensure_cleanup_handlers,
    Storage,
    ListStorage,
    LazyStackStorage,
    TensorStorage,
    LazyTensorStorage,
    LazyMemmapStorage,
    CompressedListStorage,
    StorageEnsemble,
    StoreStorage,
    _mem_map_tensor_as_tensor,
    _collate_list_tensordict,
    _stack_anything,
    _collate_id,
    _get_default_collate,
    _make_memmap,
    _make_empty_memmap,
    _flip_list,
):
    _export.__module__ = __name__
del _export
