# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import atexit
import logging
import multiprocessing as mp
import os
import shutil
import signal
import tempfile
import textwrap
import warnings
import weakref
from collections import OrderedDict
from collections.abc import Sequence
from copy import copy
from multiprocessing.context import get_spawning_popen
from typing import Any

import torch
from tensordict import (
    is_tensor_collection,
    LazyStackedTensorDict,
    NestedKey,
    TensorDict,
    TensorDictBase,
)
from tensordict.base import _NESTED_TENSORS_AS_LISTS
from tensordict.memmap import MemoryMappedTensor
from torch.utils._pytree import tree_flatten, tree_map, tree_unflatten

from torchrl._utils import _make_ordinal_device, implement_for, logger as torchrl_logger
from torchrl.data.replay_buffers.checkpointers import TensorStorageCheckpointer
from torchrl.data.replay_buffers.utils import (
    _init_pytree,
    _is_int,
    INT_CLASSES,
    tree_iter,
)

try:
    from torch.compiler import disable as compile_disable, is_compiling
except ImportError:
    from torch._dynamo import disable as compile_disable, is_compiling


# =============================================================================
# Memmap Storage Cleanup Infrastructure
# =============================================================================
# This module-level infrastructure ensures that memmap files created by
# LazyMemmapStorage are cleaned up even when scripts are interrupted with
# Ctrl+C (SIGINT) or killed with SIGTERM.

# Registry of storages to clean up (weak references to avoid preventing GC)
_MEMMAP_STORAGE_REGISTRY: weakref.WeakSet = weakref.WeakSet()

# Track if cleanup has already run (to avoid double cleanup)
_CLEANUP_DONE = False

# Store original signal handlers to restore after cleanup
_ORIGINAL_SIGINT_HANDLER = None
_ORIGINAL_SIGTERM_HANDLER = None


def _cleanup_all_memmap_storages():
    """Clean up all registered memmap storages.

    This function is called on exit (via atexit) and on signal interrupts.
    It removes all temporary memmap directories that were created with
    auto_cleanup=True.
    """
    global _CLEANUP_DONE
    if _CLEANUP_DONE:
        return
    _CLEANUP_DONE = True

    for storage in list(_MEMMAP_STORAGE_REGISTRY):
        try:
            storage.cleanup()
        except Exception:
            # Ignore errors during cleanup - the storage might already be gone
            pass


def _signal_cleanup_handler(signum, frame):
    """Signal handler that cleans up memmap storages before exiting.

    This handler is robust to cleanup failures - it will always re-raise the
    signal to ensure proper process termination.
    """
    # Always ensure we re-raise the signal, even if cleanup fails
    try:
        _cleanup_all_memmap_storages()
    except Exception:
        # Ignore any cleanup errors - we must re-raise the signal
        pass

    # Re-raise the signal with the original handler (or default behavior)
    if signum == signal.SIGINT:
        original = _ORIGINAL_SIGINT_HANDLER
    elif signum == signal.SIGTERM:
        original = _ORIGINAL_SIGTERM_HANDLER
    else:
        original = signal.SIG_DFL

    # Restore original handler and re-raise
    signal.signal(signum, original if original else signal.SIG_DFL)
    os.kill(os.getpid(), signum)


def _register_cleanup_handlers():
    """Register atexit and signal handlers for memmap cleanup.

    This is called once when the first storage with auto_cleanup=True is created.
    """
    global _ORIGINAL_SIGINT_HANDLER, _ORIGINAL_SIGTERM_HANDLER

    # Register atexit handler (for normal exits)
    atexit.register(_cleanup_all_memmap_storages)

    # Register signal handlers (for Ctrl+C and kill)
    # Only register if we're in the main thread (signals can only be handled in main thread)
    try:
        import threading

        if threading.current_thread() is threading.main_thread():
            _ORIGINAL_SIGINT_HANDLER = signal.signal(
                signal.SIGINT, _signal_cleanup_handler
            )
            _ORIGINAL_SIGTERM_HANDLER = signal.signal(
                signal.SIGTERM, _signal_cleanup_handler
            )
    except (ValueError, RuntimeError):
        # Signal handling not available (e.g., not main thread)
        pass


# Flag to track if handlers have been registered
_CLEANUP_HANDLERS_REGISTERED = False


def _ensure_cleanup_handlers():
    """Ensure cleanup handlers are registered (called once per process)."""
    global _CLEANUP_HANDLERS_REGISTERED
    if not _CLEANUP_HANDLERS_REGISTERED:
        _register_cleanup_handlers()
        _CLEANUP_HANDLERS_REGISTERED = True


from .base import Storage


class TensorStorage(Storage):
    """A storage for tensors and tensordicts.

    Args:
        storage (tensor or TensorDict): the data buffer to be used.
        max_size (int): size of the storage, i.e. maximum number of elements stored
            in the buffer.

    Keyword Args:
        device (torch.device, optional): device where the sampled tensors will be
            stored and sent. Default is :obj:`torch.device("cpu")`.
            If "auto" is passed, the device is automatically gathered from the
            first batch of data passed. This is not enabled by default to avoid
            data placed on GPU by mistake, causing OOM issues.
        ndim (int, optional): the number of dimensions to be accounted for when
            measuring the storage size. For instance, a storage of shape ``[3, 4]``
            has capacity ``3`` if ``ndim=1`` and ``12`` if ``ndim=2``.
            Defaults to ``1``.

            .. important:: When using a collector with ``trajs_per_batch``,
                keep the default ``ndim=1``.  ``trajs_per_batch`` writes
                variable-length trajectories as flat 1-D sequences, which is
                incompatible with a storage that expects a fixed second
                dimension (``ndim >= 2``).

        compilable (bool, optional): whether the storage is compilable.
            If ``True``, the writer cannot be shared between multiple processes.
            Defaults to ``False``.

    Examples:
        >>> data = TensorDict({
        ...     "some data": torch.randn(10, 11),
        ...     ("some", "nested", "data"): torch.randn(10, 11, 12),
        ... }, batch_size=[10, 11])
        >>> storage = TensorStorage(data)
        >>> len(storage)  # only the first dimension is considered as indexable
        10
        >>> storage.get(0)
        TensorDict(
            fields={
                some data: Tensor(shape=torch.Size([11]), device=cpu, dtype=torch.float32, is_shared=False),
                some: TensorDict(
                    fields={
                        nested: TensorDict(
                            fields={
                                data: Tensor(shape=torch.Size([11, 12]), device=cpu, dtype=torch.float32, is_shared=False)},
                            batch_size=torch.Size([11]),
                            device=None,
                            is_shared=False)},
                    batch_size=torch.Size([11]),
                    device=None,
                    is_shared=False)},
            batch_size=torch.Size([11]),
            device=None,
            is_shared=False)
        >>> storage.set(0, storage.get(0).zero_()) # zeros the data along index ``0``

    This class also supports tensorclass data.

    Examples:
        >>> from tensordict import tensorclass
        >>> @tensorclass
        ... class MyClass:
        ...     foo: torch.Tensor
        ...     bar: torch.Tensor
        >>> data = MyClass(foo=torch.randn(10, 11), bar=torch.randn(10, 11, 12), batch_size=[10, 11])
        >>> storage = TensorStorage(data)
        >>> storage.get(0)
        MyClass(
            bar=Tensor(shape=torch.Size([11, 12]), device=cpu, dtype=torch.float32, is_shared=False),
            foo=Tensor(shape=torch.Size([11]), device=cpu, dtype=torch.float32, is_shared=False),
            batch_size=torch.Size([11]),
            device=None,
            is_shared=False)

    """

    _storage = None
    _default_checkpointer = TensorStorageCheckpointer
    supports_conditional_update = True

    def __init__(
        self,
        storage,
        max_size=None,
        *,
        device: torch.device | str = "cpu",
        ndim: int = 1,
        compilable: bool = False,
    ):
        if not ((storage is None) ^ (max_size is None)):
            if storage is None:
                raise ValueError("Expected storage to be non-null.")
            if max_size != storage.shape[0]:
                raise ValueError(
                    "The max-size and the storage shape mismatch: got "
                    f"max_size={max_size} for a storage of shape {storage.shape}."
                )
        elif storage is not None:
            if is_tensor_collection(storage):
                max_size = storage.shape[0]
            else:
                max_size = tree_flatten(storage)[0][0].shape[0]
        self.ndim = ndim
        super().__init__(max_size, compilable=compilable)
        self.initialized = storage is not None
        if self.initialized:
            self._len = max_size
        else:
            self._len = 0
        self.device = (
            _make_ordinal_device(torch.device(device))
            if device != "auto"
            else storage.device
            if storage is not None
            else "auto"
        )
        self._storage = storage
        self._last_cursor = None
        self.__dict__["_storage_keys"] = None

    @property
    def _storage_keys(self) -> list | None:
        """Cached list of storage keys for filtering incoming data.

        Returns None if storage is not locked, not a tensor collection, or not initialized.
        Only locked storage (shared memory) needs key filtering to prevent adding
        keys that won't propagate in multiprocessing pipelines.
        """
        keys = self.__dict__.get("_storage_keys")
        if keys is None and self.initialized and is_tensor_collection(self._storage):
            # Only cache keys if storage is locked - unlocked storage can accept new keys
            if self._storage.is_locked:
                keys = list(
                    self._storage.keys(
                        include_nested=True,
                        leaves_only=True,
                        is_leaf=_NESTED_TENSORS_AS_LISTS,
                    )
                )
                self.__dict__["_storage_keys"] = keys
        return keys

    @_storage_keys.setter
    def _storage_keys(self, value):
        self.__dict__["_storage_keys"] = value

    @property
    def _len(self):
        _len_value = getattr(self, "_len_value", None)
        if not self._compilable:
            if _len_value is None:
                _len_value = self._len_value = mp.Value("i", 0)
            return _len_value.value
        else:
            if _len_value is None:
                _len_value = self._len_value = 0
            return _len_value

    @_len.setter
    def _len(self, value):
        if not is_compiling() and not self._compilable:
            _len_value = getattr(self, "_len_value", None)
            if _len_value is None:
                _len_value = self._len_value = mp.Value("i", 0)
            _len_value.value = value
        else:
            self._len_value = value

    @property
    def _total_shape(self):
        # Total shape, irrespective of how full the storage is
        _total_shape = getattr(self, "_total_shape_value", None)
        if _total_shape is None and self.initialized:
            if is_tensor_collection(self._storage):
                _total_shape = self._storage.shape[: self.ndim]
            else:
                leaf = next(tree_iter(self._storage))
                _total_shape = leaf.shape[: self.ndim]
            self.__dict__["_total_shape_value"] = _total_shape
            self._len = torch.Size([self._len_along_dim0, *_total_shape[1:]]).numel()
        return _total_shape

    @property
    def _is_full(self):
        # whether the storage is full
        return len(self) == self.max_size

    @property
    def _len_along_dim0(self):
        # returns the length of the buffer along dim0
        len_along_dim = len(self)
        if self.ndim > 1:
            _total_shape = self._total_shape
            if _total_shape is not None:
                len_along_dim = -(len_along_dim // -_total_shape[1:].numel())
            else:
                return None
        return len_along_dim

    def _max_size_along_dim0(self, *, single_data=None, batched_data=None):
        # returns the max_size of the buffer along dim0
        max_size = self.max_size
        if self.ndim > 1:
            shape = self.shape
            if shape is None:
                if single_data is not None:
                    data = single_data
                elif batched_data is not None:
                    data = batched_data
                else:
                    raise ValueError("single_data or batched_data must be passed.")
                if is_tensor_collection(data):
                    datashape = data.shape[: self.ndim]
                else:
                    for leaf in tree_iter(data):
                        datashape = leaf.shape[: self.ndim]
                        break
                if batched_data is not None:
                    datashape = datashape[1:]
                max_size = -(max_size // -datashape.numel())
            else:
                max_size = -(max_size // -self._total_shape[1:].numel())
        return max_size

    @property
    def shape(self):
        # Shape, truncated where needed to accommodate for the length of the storage
        if self._is_full:
            return self._total_shape
        _total_shape = self._total_shape
        if _total_shape is not None:
            return torch.Size([self._len_along_dim0] + list(_total_shape[1:]))

    # TODO: Without this disable, compiler recompiles for back-to-back calls.
    # Figuring out a way to avoid this disable would give better performance.
    @compile_disable()
    def _rand_given_ndim(self, batch_size):
        return self._rand_given_ndim_impl(batch_size)

    # At the moment, this is separated into its own function so that we can test
    # it without the `disable` and detect if future updates to the
    # compiler fix the recompile issue.
    def _rand_given_ndim_impl(self, batch_size):
        if self.ndim == 1:
            return super()._rand_given_ndim(batch_size)
        shape = self.shape
        return tuple(
            torch.randint(_dim, (batch_size,), generator=self._rng, device=self.device)
            for _dim in shape
        )

    def flatten(self):
        if self.ndim == 1:
            return self
        if not self.initialized:
            raise RuntimeError("Cannot flatten a non-initialized storage.")
        if is_tensor_collection(self._storage):
            if self._is_full:
                return TensorStorage(self._storage.flatten(0, self.ndim - 1))
            return TensorStorage(
                self._storage[: self._len_along_dim0].flatten(0, self.ndim - 1)
            )
        if self._is_full:
            return TensorStorage(
                tree_map(lambda x: x.flatten(0, self.ndim - 1), self._storage)
            )
        return TensorStorage(
            tree_map(
                lambda x: x[: self._len_along_dim0].flatten(0, self.ndim - 1),
                self._storage,
            )
        )

    def _conditional_patch_leaf(self, key: NestedKey) -> torch.Tensor:
        storage = getattr(self, "_storage", None)
        if storage is None or not self.initialized:
            raise RuntimeError(
                "Conditional updates require an initialized storage. Write some "
                "data to the buffer before calling update_if_present."
            )
        leaf = None
        if is_tensor_collection(storage):
            leaf = storage.get(key, default=None)
        if leaf is None:
            raise KeyError(
                f"Key {key} does not exist in the storage. Conditional patches "
                "can only target existing tensor fields of a tensordict storage."
            )
        return leaf

    def _validate_conditional_patch(
        self, index: torch.Tensor, patch: dict[NestedKey, torch.Tensor]
    ) -> dict[NestedKey, torch.Tensor]:
        n_coords = index.shape[-1] if index.ndim > 1 else 1
        n_rows = index.shape[0] if index.ndim > 1 else index.numel()
        normalized = {}
        for key, value in patch.items():
            leaf = self._conditional_patch_leaf(key)
            value = torch.as_tensor(value)
            if value.dtype != leaf.dtype:
                raise ValueError(
                    f"dtype mismatch for patch key {key}: got {value.dtype}, "
                    f"the storage holds {leaf.dtype}."
                )
            feature_shape = leaf.shape[n_coords:]
            try:
                value = value.reshape((n_rows, *feature_shape))
            except RuntimeError:
                raise ValueError(
                    f"shape mismatch for patch key {key}: got {tuple(value.shape)}, "
                    f"expected {n_rows} records with feature shape {tuple(feature_shape)}."
                )
            normalized[key] = value.to(leaf.device)
        return normalized

    def _apply_conditional_patch(
        self, index: torch.Tensor, patch: dict[NestedKey, torch.Tensor]
    ) -> None:
        if index.ndim > 1:
            coords = tuple(index.unbind(-1))
        else:
            coords = (index,)
        for key, value in patch.items():
            leaf = self._conditional_patch_leaf(key)
            leaf[coords] = value

    def __getstate__(self):
        state = super().__getstate__()
        if get_spawning_popen() is None:
            length = self._len
            del state["_len_value"]
            state["len__context"] = length
        elif not self.initialized:
            if not self.shared_init:
                # check that the storage is initialized
                raise RuntimeError(
                    f"Cowardly refusing to share a storage of type {type(self)} between processes if "
                    f"it has not been initialized yet. You can either:\n"
                    f"- Populate the buffer with some data in the main process before passing it to the other processes (or create the buffer explicitly with a TensorStorage).\n"
                    f"- set shared_init=True when creating the storage such that it can be initialized by the remote processes."
                )
            return state
        else:
            # check that the content is shared, otherwise tell the user we can't help
            storage = self._storage
            STORAGE_ERR = "The storage must be place in shared memory or memmapped before being shared between processes."

            # If the content is on cpu, it will be placed in shared memory.
            # If it's on cuda it's already shared.
            # If it's memmaped no worry in this case either.
            # Only if the device is not "cpu" or "cuda" we may have a problem.
            def assert_is_sharable(tensor):
                if tensor.device is None or tensor.device.type in (
                    "cuda",
                    "cpu",
                    "meta",
                ):
                    return
                raise RuntimeError(STORAGE_ERR)

            if is_tensor_collection(storage):
                storage.apply(assert_is_sharable, filter_empty=True)
            else:
                tree_map(storage, assert_is_sharable)

        return state

    def __setstate__(self, state):
        len = state.pop("len__context", None)
        if len is not None:
            if not state["_compilable"]:
                _len_value = mp.Value("i", len)
                state["_len_value"] = _len_value
            else:
                state["_len_value"] = len
        Storage.__setstate__(self, state)

    def state_dict(self) -> dict[str, Any]:
        _storage = self._storage
        if isinstance(_storage, torch.Tensor):
            pass
        elif is_tensor_collection(_storage):
            _storage = _storage.state_dict()
        elif _storage is None:
            _storage = {}
        else:
            raise TypeError(
                f"Objects of type {type(_storage)} are not supported by {type(self)}.state_dict"
            )
        return {
            "_storage": _storage,
            "initialized": self.initialized,
            "_len": self._len,
        }

    def load_state_dict(self, state_dict):
        _storage = copy(state_dict["_storage"])
        if isinstance(_storage, torch.Tensor):
            if isinstance(self._storage, torch.Tensor):
                self._storage.copy_(_storage)
            elif self._storage is None:
                # clone to decouple the storage from the caller's tensor (which
                # may e.g. be mmap-backed by a checkpoint file)
                self._storage = _storage.clone()
            else:
                raise RuntimeError(
                    f"Cannot copy a storage of type {type(_storage)} onto another of type {type(self._storage)}"
                )
        elif isinstance(_storage, (dict, OrderedDict)):
            if is_tensor_collection(self._storage):
                self._storage.load_state_dict(_storage, strict=False)
            elif self._storage is None:
                # loading on an empty TensorDict assigns the state-dict tensors
                # by reference: clone to decouple from the caller's tensors
                self._storage = (
                    TensorDict().load_state_dict(_storage, strict=False).clone()
                )
            else:
                raise RuntimeError(
                    f"Cannot copy a storage of type {type(_storage)} onto another of type {type(self._storage)}. If your storage is pytree-based, use the dumps/load API instead."
                )
        else:
            raise TypeError(
                f"Objects of type {type(_storage)} are not supported by ListStorage.load_state_dict"
            )
        self.initialized = state_dict["initialized"]
        self._len = state_dict["_len"]
        self._bump_mutation_revision()

    @implement_for("torch", "2.3", compilable=True)
    def _set_tree_map(self, cursor, data, storage):
        def set_tensor(datum, store):
            store[cursor] = datum

        # this won't be available until v2.3
        tree_map(set_tensor, data, storage)

    @implement_for("torch", "2.0", "2.3", compilable=True)
    def _set_tree_map(self, cursor, data, storage):  # noqa: 534
        # flatten data and cursor
        data_flat = tree_flatten(data)[0]
        storage_flat = tree_flatten(storage)[0]
        for datum, store in zip(data_flat, storage_flat):
            store[cursor] = datum

    def _get_new_len(self, data, cursor):
        int_cursor = _is_int(cursor)
        ndim = self.ndim - int_cursor
        if is_tensor_collection(data) or isinstance(data, torch.Tensor):
            numel = data.shape[:ndim].numel()
        else:
            leaf = next(tree_iter(data))
            numel = leaf.shape[:ndim].numel()
        self._len = min(self._len + numel, self.max_size)

    @implement_for("torch", "2.0", None, compilable=True)
    def set(
        self,
        cursor: int | Sequence[int] | slice,
        data: TensorDictBase | torch.Tensor,
        *,
        set_cursor: bool = True,
    ):
        if set_cursor:
            self._set_last_cursor(cursor)

        if isinstance(data, list):
            # flip list
            try:
                data = _flip_list(data)
            except Exception:
                raise RuntimeError(
                    "Stacking the elements of the list resulted in "
                    "an error. "
                    f"Storages of type {type(self)} expect all elements of the list "
                    f"to have the same tree structure. If the list is compact (each "
                    f"leaf is itself a batch with the appropriate number of elements) "
                    f"consider using a tuple instead, as lists are used within `extend` "
                    f"for per-item addition."
                )

        if set_cursor:
            self._get_new_len(data, cursor)

        if not self.initialized:
            if not isinstance(cursor, INT_CLASSES):
                if is_tensor_collection(data):
                    self._init(data[0])
                else:
                    self._init(tree_map(lambda x: x[0], data))
            else:
                self._init(data)

        if is_tensor_collection(data):
            # Filter data to only include keys present in storage.
            # _storage_keys is only set when storage is locked (shared memory),
            # so this handles cases where policy outputs extra keys that can't
            # be added to locked shared memory.
            storage_keys = self._storage_keys
            if storage_keys is not None:
                data = data.select(*storage_keys, strict=False)
            try:
                # Optimize lazy stack writes: write each tensordict directly to
                # storage to avoid creating an intermediate contiguous copy.
                if isinstance(data, LazyStackedTensorDict):
                    stack_dim = data.stack_dim
                    if isinstance(cursor, slice):
                        # For slices, storage[slice] typically returns a view.
                        # Use _stack_onto_ to write directly without intermediate copy.
                        self._storage[cursor]._stack_onto_(
                            list(data.unbind(stack_dim)), dim=stack_dim
                        )
                    else:
                        # For tensor/sequence indices, use update_at_ which handles
                        # lazy stacks efficiently in a single call.
                        self._storage.update_at_(data, cursor)
                else:
                    self._storage[cursor] = data
            except RuntimeError as e:
                if "locked" in str(e).lower():
                    # Provide informative error about key differences
                    self._raise_informative_lock_error(data, e)
                raise
        else:
            self._set_tree_map(cursor, data, self._storage)
        self._bump_mutation_revision()

    @implement_for("torch", None, "2.0", compilable=True)
    def set(  # noqa: F811
        self,
        cursor: int | Sequence[int] | slice,
        data: TensorDictBase | torch.Tensor,
        *,
        set_cursor: bool = True,
    ):
        if set_cursor:
            self._set_last_cursor(cursor)

        if isinstance(data, list):
            # flip list
            try:
                data = _flip_list(data)
            except Exception:
                raise RuntimeError(
                    "Stacking the elements of the list resulted in "
                    "an error. "
                    f"Storages of type {type(self)} expect all elements of the list "
                    f"to have the same tree structure. If the list is compact (each "
                    f"leaf is itself a batch with the appropriate number of elements) "
                    f"consider using a tuple instead, as lists are used within `extend` "
                    f"for per-item addition."
                )
        if set_cursor:
            self._get_new_len(data, cursor)

        if not is_tensor_collection(data) and not isinstance(data, torch.Tensor):
            raise NotImplementedError(
                "storage extension with pytrees is only available with torch >= 2.0. If you need this "
                "feature, please open an issue on TorchRL's github repository."
            )
        if not self.initialized:
            if not isinstance(cursor, INT_CLASSES):
                self._init(data[0])
            else:
                self._init(data)

        if not isinstance(cursor, (*INT_CLASSES, slice)):
            if not isinstance(cursor, torch.Tensor):
                cursor = torch.tensor(cursor, dtype=torch.long)
            elif cursor.dtype != torch.long:
                cursor = cursor.to(dtype=torch.long)
            if len(cursor) > self._len_along_dim0:
                warnings.warn(
                    "A cursor of length superior to the storage capacity was provided. "
                    "To accommodate for this, the cursor will be truncated to its last "
                    "element such that its length matched the length of the storage. "
                    "This may **not** be the optimal behavior for your application! "
                    "Make sure that the storage capacity is big enough to support the "
                    "batch size provided."
                )
        # Filter data to only include keys present in storage.
        # _storage_keys is only set when storage is locked (shared memory),
        # so this handles cases where policy outputs extra keys that can't
        # be added to locked shared memory.
        if is_tensor_collection(data):
            storage_keys = self._storage_keys
            if storage_keys is not None:
                data = data.select(*storage_keys, strict=False)
        try:
            # Optimize lazy stack writes: write each tensordict directly to
            # storage to avoid creating an intermediate contiguous copy.
            if is_tensor_collection(data) and isinstance(data, LazyStackedTensorDict):
                stack_dim = data.stack_dim
                if isinstance(cursor, slice):
                    # For slices, storage[slice] typically returns a view.
                    # Use _stack_onto_ to write directly without intermediate copy.
                    self._storage[cursor]._stack_onto_(
                        list(data.unbind(stack_dim)), dim=stack_dim
                    )
                else:
                    # For tensor/sequence indices, use update_at_ which handles
                    # lazy stacks efficiently in a single call.
                    self._storage.update_at_(data, cursor)
            else:
                self._storage[cursor] = data
        except RuntimeError as e:
            if "locked" in str(e).lower():
                # Provide informative error about key differences
                self._raise_informative_lock_error(data, e)
            raise
        self._bump_mutation_revision()

    def _wait_for_init(self):
        pass

    def _raise_informative_lock_error(
        self, data: TensorDictBase | torch.Tensor, original_error: RuntimeError
    ) -> None:
        """Raise an informative error when storage is locked and data has different keys.

        This method is called when an assignment to the storage fails due to a lock error.
        It provides detailed information about which keys are new in the data vs what the
        storage expects.
        """
        if not is_tensor_collection(data) or not is_tensor_collection(self._storage):
            # Can only provide detailed info for tensor collections
            raise original_error

        # Get all keys from both storage and data
        storage_keys = set(
            self._storage.keys(
                include_nested=True, leaves_only=True, is_leaf=_NESTED_TENSORS_AS_LISTS
            )
        )
        data_keys = set(
            data.keys(
                include_nested=True, leaves_only=True, is_leaf=_NESTED_TENSORS_AS_LISTS
            )
        )

        new_keys = data_keys - storage_keys
        missing_keys = storage_keys - data_keys

        error_parts = [
            "Cannot write to locked storage due to key mismatch.",
            f"\nOriginal error: {original_error}",
        ]

        if new_keys:
            error_parts.append(
                f"\n\nNew keys in data (not in storage): {sorted(str(k) for k in new_keys)}"
            )
        if missing_keys:
            error_parts.append(
                f"\n\nMissing keys in data (present in storage): {sorted(str(k) for k in missing_keys)}"
            )

        if new_keys or missing_keys:
            error_parts.append(
                "\n\nThis typically happens when:"
                "\n  1. The policy is called on some steps but not others (e.g., during init_random_frames)"
                "\n  2. A transform conditionally adds keys based on data content"
                "\n  3. Different collectors/workers produce data with different keys"
                "\n\nTo fix this, ensure all data written to the buffer has consistent keys."
            )
        else:
            error_parts.append(
                "\n\nNo key differences detected. The lock error may be due to shape or dtype mismatches."
            )

        raise RuntimeError("".join(error_parts)) from original_error

    def get(self, index: int | Sequence[int] | slice) -> Any:
        _storage = self._storage
        is_tc = is_tensor_collection(_storage)
        if not self.initialized:
            if getattr(self, "shared_init", False):
                self._wait_for_init()
            raise RuntimeError("Cannot get elements out of a non-initialized storage.")
        if not self._is_full:
            if is_tc:
                storage = self._storage[: self._len_along_dim0]
            else:
                storage = tree_map(lambda x: x[: self._len_along_dim0], self._storage)
        else:
            storage = self._storage
        if not self.initialized:
            raise RuntimeError(
                "Cannot get an item from an uninitialized LazyMemmapStorage"
            )
        if is_tc:
            return storage[index]
        else:
            return tree_map(lambda x: x[index], storage)

    # TODO: Without this disable, compiler recompiles due to changing _len_value guards.
    @compile_disable()
    def __len__(self):
        return self._len

    def _empty(self):
        # assuming that the data structure is the same, we don't need to to
        # anything if the cursor is reset to 0
        self._len = 0
        self._bump_mutation_revision()

    def _init(self):
        raise NotImplementedError(
            f"{type(self)} must be initialized during construction."
        )

    def __repr__(self):
        if not self.initialized:
            storage_str = textwrap.indent("data=<empty>", 4 * " ")
        elif is_tensor_collection(self._storage):
            storage_str = textwrap.indent(f"data={self[:]}", 4 * " ")
        else:

            def repr_item(x):
                if isinstance(x, torch.Tensor):
                    return f"{x.__class__.__name__}(shape={x.shape}, dtype={x.dtype}, device={x.device})"
                return x.__class__.__name__

            storage_str = textwrap.indent(
                f"data={tree_map(repr_item, self[:])}", 4 * " "
            )
        shape_str = textwrap.indent(f"shape={self.shape}", 4 * " ")
        len_str = textwrap.indent(f"len={len(self)}", 4 * " ")
        maxsize_str = textwrap.indent(f"max_size={self.max_size}", 4 * " ")
        return f"{self.__class__.__name__}(\n{storage_str}, \n{shape_str}, \n{len_str}, \n{maxsize_str})"

    def contains(self, item):
        if isinstance(item, int):
            if item < 0:
                item += self._len_along_dim0

            return 0 <= item < self._len_along_dim0
        if isinstance(item, torch.Tensor):

            def _is_valid_index(idx):
                try:
                    torch.zeros(self.shape, device="meta")[idx]
                    return True
                except IndexError:
                    return False

            if item.ndim:
                return torch.tensor(
                    [_is_valid_index(idx) for idx in item],
                    dtype=torch.bool,
                    device=item.device,
                )
            return torch.tensor(_is_valid_index(item), device=item.device)
        raise NotImplementedError(f"type {type(item)} is not supported yet.")


class LazyTensorStorage(TensorStorage):
    """A pre-allocated tensor storage for tensors and tensordicts.

    Args:
        max_size (int): size of the storage, i.e. maximum number of elements stored
            in the buffer.

    Keyword Args:
        device (torch.device, optional): device where the sampled tensors will be
            stored and sent. Default is :obj:`torch.device("cpu")`.
            If "auto" is passed, the device is automatically gathered from the
            first batch of data passed. This is not enabled by default to avoid
            data placed on GPU by mistake, causing OOM issues.
        ndim (int, optional): the number of dimensions to be accounted for when
            measuring the storage size. For instance, a storage of shape ``[3, 4]``
            has capacity ``3`` if ``ndim=1`` and ``12`` if ``ndim=2``.
            Defaults to ``1``.

            .. important:: When using a collector with ``trajs_per_batch``,
                keep the default ``ndim=1``.  ``trajs_per_batch`` writes
                variable-length trajectories as flat 1-D sequences, which is
                incompatible with a storage that expects a fixed second
                dimension (``ndim >= 2``).
        compilable (bool, optional): whether the storage is compilable.
            If ``True``, the writer cannot be shared between multiple processes.
            Defaults to ``False``.
        consolidated (bool, optional): if ``True``, the storage will be consolidated after
            its first expansion. Defaults to ``False``.
        shared_init (bool, optional): if ``True``, enables multiprocess coordination
            during storage initialization. First process initializes with memmap,
            others wait and load from the shared memmap. Defaults to ``False``.
        cleanup_memmap (bool, optional): if ``True`` and ``shared_init=True``,
            the temporary memmap will be deleted after initialization and the
            storage will operate in RAM. Defaults to ``True``.

    Examples:
        >>> data = TensorDict({
        ...     "some data": torch.randn(10, 11),
        ...     ("some", "nested", "data"): torch.randn(10, 11, 12),
        ... }, batch_size=[10, 11])
        >>> storage = LazyTensorStorage(100)
        >>> storage.set(range(10), data)
        >>> len(storage)  # only the first dimension is considered as indexable
        10
        >>> storage.get(0)
        TensorDict(
            fields={
                some data: Tensor(shape=torch.Size([11]), device=cpu, dtype=torch.float32, is_shared=False),
                some: TensorDict(
                    fields={
                        nested: TensorDict(
                            fields={
                                data: Tensor(shape=torch.Size([11, 12]), device=cpu, dtype=torch.float32, is_shared=False)},
                            batch_size=torch.Size([11]),
                            device=cpu,
                            is_shared=False)},
                    batch_size=torch.Size([11]),
                    device=cpu,
                    is_shared=False)},
            batch_size=torch.Size([11]),
            device=cpu,
            is_shared=False)
        >>> storage.set(0, storage.get(0).zero_()) # zeros the data along index ``0``

    This class also supports tensorclass data.

    Examples:
        >>> from tensordict import tensorclass
        >>> @tensorclass
        ... class MyClass:
        ...     foo: torch.Tensor
        ...     bar: torch.Tensor
        >>> data = MyClass(foo=torch.randn(10, 11), bar=torch.randn(10, 11, 12), batch_size=[10, 11])
        >>> storage = LazyTensorStorage(10)
        >>> storage.set(range(10), data)
        >>> storage.get(0)
        MyClass(
            bar=Tensor(shape=torch.Size([11, 12]), device=cpu, dtype=torch.float32, is_shared=False),
            foo=Tensor(shape=torch.Size([11]), device=cpu, dtype=torch.float32, is_shared=False),
            batch_size=torch.Size([11]),
            device=cpu,
            is_shared=False)

    """

    _default_checkpointer = TensorStorageCheckpointer

    def __init__(
        self,
        max_size: int,
        *,
        device: torch.device | str = "cpu",
        ndim: int = 1,
        compilable: bool = False,
        consolidated: bool = False,
        shared_init: bool = False,
        cleanup_memmap: bool = True,
    ):
        super().__init__(
            storage=None,
            max_size=max_size,
            device=device,
            ndim=ndim,
            compilable=compilable,
        )
        self.consolidated = consolidated
        self.shared_init = shared_init
        self.cleanup_memmap = cleanup_memmap

        # Initialize multiprocess coordination objects if shared_init is enabled
        if self.shared_init:
            if self._compilable:
                raise RuntimeError(
                    "Cannot share a compilable storage between processes."
                )
            self._init_lock = mp.Lock()
            self._init_event = mp.Event()
            self._make_init_directory()

    def _make_init_directory(self):
        if getattr(self, "scratch_dir", None) is not None:
            self._init_directory = self.scratch_dir
            return
        # Create a shared directory
        self.scratch_dir = self._init_directory = tempfile.mkdtemp(
            prefix="torchrl_storage_init_"
        )
        return

    def _init(
        self,
        data: TensorDictBase | torch.Tensor | PyTree,  # noqa: F821
    ) -> None:
        if not self.shared_init:
            return self._init_standard(data)

        # Try to become coordinator
        is_coordinator = not self._init_event.is_set()
        is_coordinator = is_coordinator and self._init_lock.acquire(block=False)

        if is_coordinator:
            try:
                # We are the coordinator
                self._init_coordinator(data)
            finally:
                # Signal other processes that initialization is complete
                self._init_event.set()
                self._init_lock.release()
        else:
            # Failed to acquire lock, wait for coordinator
            self._wait_for_init()

        self.initialized = True

    def _init_standard(
        self,
        data: TensorDictBase | torch.Tensor | PyTree,  # noqa: F821
    ) -> None:
        """Standard initialization without multiprocess coordination."""
        if not self._compilable:
            # TODO: Investigate why this seems to have a performance impact with
            # the compiler
            torchrl_logger.debug("Creating a TensorStorage...")
        if self.device == "auto":
            self.device = data.device

        def max_size_along_dim0(data_shape):
            if self.ndim > 1:
                result = (
                    -(self.max_size // -data_shape[: self.ndim - 1].numel()),
                    *data_shape,
                )
                self.max_size = torch.Size(result).numel()
                return result
            return (self.max_size, *data_shape)

        if is_tensor_collection(data):
            out = data.to(self.device)
            out: TensorDictBase = torch.empty_like(
                out.expand(max_size_along_dim0(data.shape))
            )
            if self.consolidated:
                out = out.consolidate()
        else:
            # if Tensor, we just create a MemoryMappedTensor of the desired shape, device and dtype
            out = tree_map(
                lambda data: torch.empty(
                    max_size_along_dim0(data.shape),
                    device=self.device,
                    dtype=data.dtype,
                ),
                data,
            )
            if self.consolidated:
                raise ValueError("Cannot consolidate non-tensordict storages.")

        self._storage = out
        self.initialized = True
        if hasattr(self._storage, "shape"):
            torchrl_logger.info(
                f"Initialized LazyTensorStorage with {self._storage.shape} shape"
            )

    def _init_coordinator(
        self,
        data: TensorDictBase | torch.Tensor | PyTree,  # noqa: F821
    ) -> None:
        """Initialize storage as the coordinating process using temporary memmap."""
        # Use LazyMemmapStorage which does everything we want
        temp_memmap_storage = LazyMemmapStorage(
            max_size=self.max_size,
            scratch_dir=self._init_directory,
            ndim=self.ndim,
            existsok=False,
            shared_init=False,  # Don't recurse
        )
        temp_memmap_storage._init_standard(data)
        self._storage = temp_memmap_storage._storage
        self._reconcile_shared_init_device()
        return

    def _wait_for_init(self) -> None:
        # wait till coordinator has initialized
        self._init_event.wait()
        storage = TensorDict.load_memmap(self._init_directory)
        self._storage = storage
        self._reconcile_shared_init_device()
        self.initialized = True
        return

    def _reconcile_shared_init_device(self) -> None:
        # Shared init swaps the backing for a CPU memory-mapped tensordict;
        # a stale non-cpu self.device would make samplers build indices on
        # the wrong device (RuntimeError at the first sample).
        device = self.device
        if device not in (None, "auto") and torch.device(device).type != "cpu":
            warnings.warn(
                f"LazyTensorStorage(shared_init=True) stores data in a CPU "
                f"memory-mapped tensordict; the requested storage device "
                f"({device}) cannot be honored and is reset to 'cpu'."
            )
        self.device = torch.device("cpu")

    # Read blocks
    def get(self, indices: slice) -> TensorDictBase | torch.Tensor | Any:
        if not self.initialized and self.shared_init:
            # Trigger initialization with dummy data
            self._wait_for_init()
        idx = super().get(indices)
        return idx


class LazyMemmapStorage(LazyTensorStorage):
    """A memory-mapped storage for tensors and tensordicts.

    Args:
        max_size (int): size of the storage, i.e. maximum number of elements stored
            in the buffer.

    Keyword Args:
        scratch_dir (str or path): directory where memmap-tensors will be written.
            If ``shared_init=True`` and no ``scratch_dir`` is provided, a shared
            temporary directory will be created automatically.
        device (torch.device, optional): device where the sampled tensors will be
            stored and sent. Default is :obj:`torch.device("cpu")`.
            If ``None`` is provided, the device is automatically gathered from the
            first batch of data passed. This is not enabled by default to avoid
            data placed on GPU by mistake, causing OOM issues.
        ndim (int, optional): the number of dimensions to be accounted for when
            measuring the storage size. For instance, a storage of shape ``[3, 4]``
            has capacity ``3`` if ``ndim=1`` and ``12`` if ``ndim=2``.
            Defaults to ``1``.

            .. important:: When using a collector with ``trajs_per_batch``,
                keep the default ``ndim=1``.  ``trajs_per_batch`` writes
                variable-length trajectories as flat 1-D sequences, which is
                incompatible with a storage that expects a fixed second
                dimension (``ndim >= 2``).

        existsok (bool, optional): whether an error should be raised if any of the
            tensors already exists on disk. Defaults to ``True``. If ``False``, the
            tensor will be opened as is, not overewritten.
        shared_init (bool, optional): if ``True``, enables multiprocess coordination
            during storage initialization. First process initializes the memmap,
            others wait and load from the shared directory. Defaults to ``False``.
        auto_cleanup (bool, optional): if ``True``, automatically registers this
            storage for cleanup when the process exits (normally or via Ctrl+C/SIGTERM).
            This removes the memmap files from disk when no longer needed.
            Defaults to ``True`` when ``scratch_dir`` is ``None`` (using temp directory),
            and ``False`` when a custom ``scratch_dir`` is provided (preserving user data).

    .. note:: When checkpointing a ``LazyMemmapStorage``, one can provide a path identical to where the storage is
        already stored to avoid executing long copies of data that is already stored on disk.
        This will only work if the default :class:`~torchrl.data.TensorStorageCheckpointer` checkpointer is used.

        Example::

            >>> from tensordict import TensorDict
            >>> from torchrl.data import TensorStorage, LazyMemmapStorage, ReplayBuffer
            >>> import tempfile
            >>> from pathlib import Path
            >>> import time
            >>> td = TensorDict(a=0, b=1).expand(1000).clone()
            >>> # We pass a path that is <main_ckpt_dir>/storage to LazyMemmapStorage
            >>> rb_memmap = ReplayBuffer(storage=LazyMemmapStorage(10_000_000, scratch_dir="dump/storage"))
            >>> rb_memmap.extend(td);
            >>> # Checkpointing in `dump` is a zero-copy, as the data is already in `dump/storage`
            >>> rb_memmap.dumps(Path("./dump"))


    Examples:
        >>> data = TensorDict({
        ...     "some data": torch.randn(10, 11),
        ...     ("some", "nested", "data"): torch.randn(10, 11, 12),
        ... }, batch_size=[10, 11])
        >>> storage = LazyMemmapStorage(100)
        >>> storage.set(range(10), data)
        >>> len(storage)  # only the first dimension is considered as indexable
        10
        >>> storage.get(0)
        TensorDict(
            fields={
                some data: MemoryMappedTensor(shape=torch.Size([11]), device=cpu, dtype=torch.float32, is_shared=False),
                some: TensorDict(
                    fields={
                        nested: TensorDict(
                            fields={
                                data: MemoryMappedTensor(shape=torch.Size([11, 12]), device=cpu, dtype=torch.float32, is_shared=False)},
                            batch_size=torch.Size([11]),
                            device=cpu,
                            is_shared=False)},
                    batch_size=torch.Size([11]),
                    device=cpu,
                    is_shared=False)},
            batch_size=torch.Size([11]),
            device=cpu,
            is_shared=False)

    This class also supports tensorclass data.

    Examples:
        >>> from tensordict import tensorclass
        >>> @tensorclass
        ... class MyClass:
        ...     foo: torch.Tensor
        ...     bar: torch.Tensor
        >>> data = MyClass(foo=torch.randn(10, 11), bar=torch.randn(10, 11, 12), batch_size=[10, 11])
        >>> storage = LazyMemmapStorage(10)
        >>> storage.set(range(10), data)
        >>> storage.get(0)
        MyClass(
            bar=MemoryMappedTensor(shape=torch.Size([11, 12]), device=cpu, dtype=torch.float32, is_shared=False),
            foo=MemoryMappedTensor(shape=torch.Size([11]), device=cpu, dtype=torch.float32, is_shared=False),
            batch_size=torch.Size([11]),
            device=cpu,
            is_shared=False)

    """

    _default_checkpointer = TensorStorageCheckpointer

    def __init__(
        self,
        max_size: int,
        *,
        scratch_dir=None,
        device: torch.device | str = "cpu",
        ndim: int = 1,
        existsok: bool = False,
        compilable: bool = False,
        shared_init: bool = False,
        auto_cleanup: bool | None = None,
    ):
        self.initialized = False
        self.scratch_dir = None
        self._scratch_dir_is_temp = scratch_dir is None
        self.existsok = existsok
        if scratch_dir is not None:
            self.scratch_dir = str(scratch_dir)
            if self.scratch_dir[-1] != "/":
                self.scratch_dir += "/"
        super().__init__(
            max_size,
            ndim=ndim,
            compilable=compilable,
            shared_init=shared_init,
            cleanup_memmap=False,
        )
        self.device = (
            _make_ordinal_device(torch.device(device))
            if device != "auto"
            else torch.device("cpu")
        )
        if self.device.type != "cpu":
            raise ValueError(
                "Memory map device other than CPU isn't supported. To cast your data to the desired device, "
                "use `buffer.append_transform(lambda x: x.to(device))` or a similar transform."
            )
        self._len = 0

        # Auto cleanup: default to True for temp dirs, False for user-specified dirs
        if auto_cleanup is None:
            auto_cleanup = self._scratch_dir_is_temp
        self._auto_cleanup = auto_cleanup
        self._cleaned_up = False

        if self._auto_cleanup:
            _ensure_cleanup_handlers()
            _MEMMAP_STORAGE_REGISTRY.add(self)

    def state_dict(self) -> dict[str, Any]:
        _storage = self._storage
        if isinstance(_storage, torch.Tensor):
            _storage = _mem_map_tensor_as_tensor(_storage)
        elif isinstance(_storage, TensorDictBase):
            _storage = _storage.apply(_mem_map_tensor_as_tensor).state_dict()
        elif _storage is None:
            _storage = {}
        else:
            raise TypeError(
                f"Objects of type {type(_storage)} are not supported by LazyTensorStorage.state_dict. If you are trying to serialize a PyTree, the storage.dumps/loads is preferred."
            )
        return {
            "_storage": _storage,
            "initialized": self.initialized,
            "_len": self._len,
        }

    def load_state_dict(self, state_dict):
        _storage = copy(state_dict["_storage"])
        if isinstance(_storage, torch.Tensor):
            if isinstance(self._storage, torch.Tensor):
                _mem_map_tensor_as_tensor(self._storage).copy_(_storage)
            elif self._storage is None:
                self._storage = _make_memmap(
                    _storage,
                    path=self.scratch_dir + "/tensor.memmap"
                    if self.scratch_dir is not None
                    else None,
                )
            else:
                raise RuntimeError(
                    f"Cannot copy a storage of type {type(_storage)} onto another of type {type(self._storage)}"
                )
        elif isinstance(_storage, (dict, OrderedDict)):
            if is_tensor_collection(self._storage):
                self._storage.load_state_dict(_storage, strict=False)
                self._storage.memmap_()
            elif self._storage is None:
                warnings.warn(
                    "Loading the storage on an uninitialized TensorDict."
                    "It is preferable to load a storage onto a"
                    "pre-allocated one whenever possible."
                )
                self._storage = TensorDict().load_state_dict(_storage, strict=False)
                self._storage.memmap_()
            else:
                raise RuntimeError(
                    f"Cannot copy a storage of type {type(_storage)} onto another of type {type(self._storage)}"
                )
        else:
            raise TypeError(
                f"Objects of type {type(_storage)} are not supported by ListStorage.load_state_dict"
            )
        self.initialized = state_dict["initialized"]
        self._len = state_dict["_len"]
        self._bump_mutation_revision()

    def _init(
        self,
        data: TensorDictBase | torch.Tensor | PyTree,  # noqa: F821
    ) -> None:
        if not self.shared_init:
            return self._init_standard(data)
        is_coordinator = not self._init_event.is_set()
        is_coordinator = is_coordinator and self._init_lock.acquire(block=False)

        if is_coordinator:
            # coordinator init
            try:
                return self._init_coordinator(data)
            finally:
                self._init_event.set()
                self._init_lock.release()
        else:
            # Standard initialization
            self._wait_for_init()
        self.initialized = True

    def _init_coordinator(self, data: TensorDictBase | torch.Tensor | Any) -> None:
        return self._init_standard(data)

    def _init_standard(self, data: TensorDictBase | torch.Tensor) -> None:
        torchrl_logger.debug("Creating a MemmapStorage...")
        if self.device == "auto":
            self.device = data.device
        if self.device.type != "cpu":
            raise RuntimeError("Support for Memmap device other than CPU is deprecated")

        def max_size_along_dim0(data_shape):
            if self.ndim > 1:
                result = (
                    -(self.max_size // -data_shape[: self.ndim - 1].numel()),
                    *data_shape,
                )
                self.max_size = torch.Size(result).numel()
                return result
            return (self.max_size, *data_shape)

        if is_tensor_collection(data):
            out = data.clone().to(self.device)
            out = out.expand(max_size_along_dim0(data.shape))
            out = out.memmap_like(prefix=self.scratch_dir, existsok=self.existsok)
            if torchrl_logger.isEnabledFor(logging.DEBUG):
                for key, tensor in sorted(
                    out.items(
                        include_nested=True,
                        leaves_only=True,
                        is_leaf=_NESTED_TENSORS_AS_LISTS,
                    ),
                    key=str,
                ):
                    try:
                        filesize = os.path.getsize(tensor.filename) / 1024 / 1024
                        torchrl_logger.debug(
                            f"\t{key}: {tensor.filename}, {filesize} Mb of storage (size: {tensor.shape})."
                        )
                    except (AttributeError, RuntimeError):
                        pass
        else:
            out = _init_pytree(self.scratch_dir, max_size_along_dim0, data)
        self._storage = out
        if hasattr(self._storage, "shape"):
            torchrl_logger.info(
                f"Initialized LazyMemmapStorage with {self._storage.shape} shape"
            )
        self.initialized = True

    def get(self, index: int | Sequence[int] | slice) -> Any:
        if not self.initialized and self.shared_init:
            # Trigger initialization with dummy data
            self._wait_for_init()
        result = super().get(index)
        return result

    def cleanup(self) -> bool:
        """Clean up memmap files from disk.

        This method removes the memmap directory and all its contents from disk.
        It is automatically called on process exit if ``auto_cleanup=True``.

        Returns:
            bool: ``True`` if cleanup was performed, ``False`` if already cleaned up
                or no cleanup needed.

        Note:
            After cleanup, the storage is no longer usable. Any attempt to access
            the storage will result in undefined behavior.

        Example:
            >>> storage = LazyMemmapStorage(1000, auto_cleanup=True)
            >>> # ... use storage ...
            >>> storage.cleanup()  # Manually clean up when done
        """
        if getattr(self, "_cleaned_up", False):
            return False

        self._cleaned_up = True

        # Get the directory to clean up
        scratch_dir = getattr(self, "scratch_dir", None)
        if scratch_dir is None:
            # No scratch dir - check if storage has memmap tensors with temp paths
            storage = getattr(self, "_storage", None)
            if storage is not None and is_tensor_collection(storage):
                # Get all memmap file paths and find their common directory
                paths = set()
                try:
                    for tensor in storage.values(include_nested=True, leaves_only=True):
                        if hasattr(tensor, "filename") and tensor.filename:
                            paths.add(os.path.dirname(tensor.filename))
                except Exception:
                    # Storage might be in an invalid state during cleanup
                    pass
                for path in paths:
                    if (
                        path
                        and os.path.isdir(path)
                        and path.startswith(tempfile.gettempdir())
                    ):
                        try:
                            shutil.rmtree(path)
                            torchrl_logger.debug(f"Cleaned up memmap directory: {path}")
                        except Exception:
                            # Ignore errors - file might be in use or already deleted
                            pass
                return bool(paths)
            return False

        # Clean up the scratch directory
        scratch_dir = scratch_dir.rstrip("/")
        if os.path.isdir(scratch_dir):
            try:
                shutil.rmtree(scratch_dir)
                torchrl_logger.debug(f"Cleaned up memmap directory: {scratch_dir}")
                return True
            except Exception as e:
                torchrl_logger.warning(f"Failed to clean up memmap directory: {e}")
                return False
        return False

    def __del__(self):
        """Ensure cleanup on garbage collection if auto_cleanup is enabled."""
        if getattr(self, "_auto_cleanup", False) and not getattr(
            self, "_cleaned_up", True
        ):
            self.cleanup()


def _mem_map_tensor_as_tensor(mem_map_tensor) -> torch.Tensor:
    if isinstance(mem_map_tensor, torch.Tensor):
        # This will account for MemoryMappedTensors
        return mem_map_tensor


def _make_memmap(tensor, path):
    return MemoryMappedTensor.from_tensor(tensor, filename=path)


def _make_empty_memmap(shape, dtype, path):
    return MemoryMappedTensor.empty(shape=shape, dtype=dtype, filename=path)


def _flip_list(data):
    if all(is_tensor_collection(_data) for _data in data):
        return torch.stack(data)
    flat_data, flat_specs = zip(*[tree_flatten(item) for item in data])
    flat_data = zip(*flat_data)
    stacks = [torch.stack(item) for item in flat_data]
    return tree_unflatten(stacks, flat_specs[0])
