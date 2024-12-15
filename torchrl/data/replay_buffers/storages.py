# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import abc

import logging
import os
import textwrap
import warnings
from collections import OrderedDict
from copy import copy
from multiprocessing.context import get_spawning_popen
from typing import Any, Dict, List, Sequence, Union

import numpy as np
import tensordict
import torch
from tensordict import (
    is_tensor_collection,
    LazyStackedTensorDict,
    TensorDict,
    TensorDictBase,
)
from tensordict.base import _NESTED_TENSORS_AS_LISTS
from tensordict.memmap import MemoryMappedTensor
from tensordict.utils import _zip_strict
from torch import multiprocessing as mp
from torch.utils._pytree import tree_flatten, tree_map, tree_unflatten
from torchrl._utils import _make_ordinal_device, implement_for, logger as torchrl_logger
from torchrl.data.replay_buffers.checkpointers import (
    ListStorageCheckpointer,
    StorageCheckpointerBase,
    StorageEnsembleCheckpointer,
    TensorStorageCheckpointer,
)
from torchrl.data.replay_buffers.utils import (
    _init_pytree,
    _is_int,
    INT_CLASSES,
    tree_iter,
)


class Storage:
    """A Storage is the container of a replay buffer.

    Every storage must have a set, get and __len__ methods implemented.
    Get and set should support integers as well as list of integers.

    The storage does not need to have a definite size, but if it does one should
    make sure that it is compatible with the buffer size.

    """

    ndim = 1
    max_size: int
    _default_checkpointer: StorageCheckpointerBase = StorageCheckpointerBase
    _rng: torch.Generator | None = None

    def __init__(
        self,
        max_size: int,
        checkpointer: StorageCheckpointerBase | None = None,
        compilable: bool = False,
    ) -> None:
        self.max_size = int(max_size)
        self.checkpointer = checkpointer
        self._compilable = compilable
        self._attached_entities_list = []

    @property
    def checkpointer(self):
        return self._checkpointer

    @checkpointer.setter
    def checkpointer(self, value: StorageCheckpointerBase | None) -> None:
        if value is None:
            value = self._default_checkpointer()
        self._checkpointer = value

    @property
    def _is_full(self):
        return len(self) == self.max_size

    @property
    def _attached_entities(self) -> List:
        # RBs that use a given instance of Storage should add
        # themselves to this set.
        _attached_entities_list = getattr(self, "_attached_entities_list", None)
        if _attached_entities_list is None:
            self._attached_entities_list = _attached_entities_list = []
        return _attached_entities_list

    @torch._dynamo.assume_constant_result
    def _attached_entities_iter(self):
        return self._attached_entities

    @abc.abstractmethod
    def set(self, cursor: int, data: Any, *, set_cursor: bool = True):
        ...

    @abc.abstractmethod
    def get(self, index: int) -> Any:
        ...

    def dumps(self, path):
        self.checkpointer.dumps(self, path)

    def loads(self, path):
        self.checkpointer.loads(self, path)

    def attach(self, buffer: Any) -> None:
        """This function attaches a sampler to this storage.

        Buffers that read from this storage must be included as an attached
        entity by calling this method. This guarantees that when data
        in the storage changes, components are made aware of changes even if the storage
        is shared with other buffers (eg. Priority Samplers).

        Args:
            buffer: the object that reads from this storage.
        """
        if buffer not in self._attached_entities:
            self._attached_entities.append(buffer)

    def __getitem__(self, item):
        return self.get(item)

    def __setitem__(self, index, value):
        """Sets values in the storage without updating the cursor or length."""
        return self.set(index, value, set_cursor=False)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    @abc.abstractmethod
    def __len__(self):
        ...

    @abc.abstractmethod
    def state_dict(self) -> Dict[str, Any]:
        ...

    @abc.abstractmethod
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        ...

    @abc.abstractmethod
    def _empty(self):
        ...

    def _rand_given_ndim(self, batch_size):
        # a method to return random indices given the storage ndim
        if self.ndim == 1:
            return torch.randint(
                0,
                len(self),
                (batch_size,),
                generator=self._rng,
                device=getattr(self, "device", None),
            )
        raise RuntimeError(
            f"Random number generation is not implemented for storage of type {type(self)} with ndim {self.ndim}. "
            f"Please report this exception as well as the use case (incl. buffer construction) on github."
        )

    @property
    def shape(self):
        if self.ndim == 1:
            return torch.Size([self.max_size])
        raise RuntimeError(
            f"storage.shape is not supported for storages of type {type(self)} when ndim > 1."
            f"Please report this exception as well as the use case (incl. buffer construction) on github."
        )

    def _max_size_along_dim0(self, *, single_data=None, batched_data=None):
        if self.ndim == 1:
            return self.max_size
        raise RuntimeError(
            f"storage._max_size_along_dim0 is not supported for storages of type {type(self)} when ndim > 1."
            f"Please report this exception as well as the use case (incl. buffer construction) on github."
        )

    def flatten(self):
        if self.ndim == 1:
            return self
        raise RuntimeError(
            f"storage.flatten is not supported for storages of type {type(self)} when ndim > 1."
            f"Please report this exception as well as the use case (incl. buffer construction) on github."
        )

    def save(self, *args, **kwargs):
        """Alias for :meth:`~.dumps`."""
        return self.dumps(*args, **kwargs)

    def dump(self, *args, **kwargs):
        """Alias for :meth:`~.dumps`."""
        return self.dumps(*args, **kwargs)

    def load(self, *args, **kwargs):
        """Alias for :meth:`~.loads`."""
        return self.loads(*args, **kwargs)

    def __getstate__(self):
        state = copy(self.__dict__)
        state["_rng"] = None
        return state

    def __contains__(self, item):
        return self.contains(item)

    @abc.abstractmethod
    def contains(self, item):
        ...


class ListStorage(Storage):
    """A storage stored in a list.

    This class cannot be extended with PyTrees, the data provided during calls to
    :meth:`~torchrl.data.replay_buffers.ReplayBuffer.extend` should be iterables
    (like lists, tuples, tensors or tensordicts with non-empty batch-size).

    Args:
        max_size (int, optional): the maximum number of elements stored in the storage.
            If not provided, an unlimited storage is created.

    """

    _default_checkpointer = ListStorageCheckpointer

    def __init__(self, max_size: int | None = None, compilable: bool = False):
        if max_size is None:
            max_size = torch.iinfo(torch.int64).max
        super().__init__(max_size, compilable=compilable)
        self._storage = []

    def set(
        self,
        cursor: Union[int, Sequence[int], slice],
        data: Any,
        *,
        set_cursor: bool = True,
    ):
        if not isinstance(cursor, INT_CLASSES):
            if (isinstance(cursor, torch.Tensor) and cursor.ndim == 0) or (
                isinstance(cursor, np.ndarray) and cursor.ndim == 0
            ):
                self.set(int(cursor), data, set_cursor=set_cursor)
                return
            if isinstance(cursor, slice):
                self._storage[cursor] = data
                return
            if isinstance(
                data,
                (
                    list,
                    tuple,
                    torch.Tensor,
                    TensorDictBase,
                    *tensordict.base._ACCEPTED_CLASSES,
                    range,
                    set,
                    np.ndarray,
                ),
            ):
                for _cursor, _data in _zip_strict(cursor, data):
                    self.set(_cursor, _data, set_cursor=set_cursor)
            else:
                raise TypeError(
                    f"Cannot extend a {type(self)} with data of type {type(data)}. "
                    f"Provide a list, tuple, set, range, np.ndarray, tensor or tensordict subclass instead."
                )
            return
        else:
            if cursor > len(self._storage):
                raise RuntimeError(
                    "Cannot append data located more than one item away from "
                    f"the storage size: the storage size is {len(self)} "
                    f"and the index of the item to be set is {cursor}."
                )
            if cursor >= self.max_size:
                raise RuntimeError(
                    f"Cannot append data to the list storage: "
                    f"maximum capacity is {self.max_size} "
                    f"and the index of the item to be set is {cursor}."
                )
            if cursor == len(self._storage):
                self._storage.append(data)
            else:
                self._storage[cursor] = data

    def get(self, index: Union[int, Sequence[int], slice]) -> Any:
        if isinstance(index, (INT_CLASSES, slice)):
            return self._storage[index]
        else:
            return [self._storage[i] for i in index]

    def __len__(self):
        return len(self._storage)

    def state_dict(self) -> Dict[str, Any]:
        return {
            "_storage": [
                elt if not hasattr(elt, "state_dict") else elt.state_dict()
                for elt in self._storage
            ]
        }

    def load_state_dict(self, state_dict):
        _storage = state_dict["_storage"]
        self._storage = []
        for elt in _storage:
            if isinstance(elt, torch.Tensor):
                self._storage.append(elt)
            elif isinstance(elt, (dict, OrderedDict)):
                self._storage.append(TensorDict().load_state_dict(elt, strict=False))
            else:
                raise TypeError(
                    f"Objects of type {type(elt)} are not supported by ListStorage.load_state_dict"
                )

    def _empty(self):
        self._storage = []

    def __getstate__(self):
        if get_spawning_popen() is not None:
            raise RuntimeError(
                f"Cannot share a storage of type {type(self)} between processes."
            )
        state = super().__getstate__()
        return state

    def __repr__(self):
        return f"{self.__class__.__name__}(items=[{self._storage[0]}, ...])"

    def contains(self, item):
        if isinstance(item, int):
            if item < 0:
                item += len(self._storage)

            return 0 <= item < len(self._storage)
        if isinstance(item, torch.Tensor):
            return torch.tensor(
                [self.contains(elt) for elt in item.tolist()],
                dtype=torch.bool,
                device=item.device,
            ).reshape_as(item)
        raise NotImplementedError(f"type {type(item)} is not supported yet.")


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

    def __init__(
        self,
        storage,
        max_size=None,
        *,
        device: torch.device = "cpu",
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

    @property
    def _len(self):
        _len_value = self.__dict__.get("_len_value", None)
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
        if not self._compilable:
            _len_value = self.__dict__.get("_len_value", None)
            if _len_value is None:
                _len_value = self._len_value = mp.Value("i", 0)
            _len_value.value = value
        else:
            self._len_value = value

    @property
    def _total_shape(self):
        # Total shape, irrespective of how full the storage is
        _total_shape = self.__dict__.get("_total_shape_value", None)
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
    @torch._dynamo.disable()
    def _rand_given_ndim(self, batch_size):
        return self._rand_given_ndim_impl(batch_size)

    # At the moment, this is separated into its own function so that we can test
    # it without the `torch._dynamo.disable` and detect if future updates to the
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

    def __getstate__(self):
        state = super().__getstate__()
        if get_spawning_popen() is None:
            length = self._len
            del state["_len_value"]
            state["len__context"] = length
        elif not self.initialized:
            # check that the storage is initialized
            raise RuntimeError(
                f"Cannot share a storage of type {type(self)} between processes if "
                f"it has not been initialized yet. Populate the buffer with "
                f"some data in the main process before passing it to the other "
                f"subprocesses (or create the buffer explicitly with a TensorStorage)."
            )
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
                state["_len_value"] = len
            else:
                _len_value = mp.Value("i", len)
                state["_len_value"] = _len_value
        self.__dict__.update(state)

    def state_dict(self) -> Dict[str, Any]:
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
                self._storage = _storage
            else:
                raise RuntimeError(
                    f"Cannot copy a storage of type {type(_storage)} onto another of type {type(self._storage)}"
                )
        elif isinstance(_storage, (dict, OrderedDict)):
            if is_tensor_collection(self._storage):
                self._storage.load_state_dict(_storage, strict=False)
            elif self._storage is None:
                self._storage = TensorDict().load_state_dict(_storage, strict=False)
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
        cursor: Union[int, Sequence[int], slice],
        data: Union[TensorDictBase, torch.Tensor],
        *,
        set_cursor: bool = True,
    ):
        if set_cursor:
            self._last_cursor = cursor

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
            self._storage[cursor] = data
        else:
            self._set_tree_map(cursor, data, self._storage)

    @implement_for("torch", None, "2.0", compilable=True)
    def set(  # noqa: F811
        self,
        cursor: Union[int, Sequence[int], slice],
        data: Union[TensorDictBase, torch.Tensor],
        *,
        set_cursor: bool = True,
    ):

        if set_cursor:
            self._last_cursor = cursor

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
        self._storage[cursor] = data

    def get(self, index: Union[int, Sequence[int], slice]) -> Any:
        _storage = self._storage
        is_tc = is_tensor_collection(_storage)
        if not self.initialized:
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
                "Cannot get an item from an unitialized LazyMemmapStorage"
            )
        if is_tc:
            return storage[index]
        else:
            return tree_map(lambda x: x[index], storage)

    def __len__(self):
        return self._len

    def _empty(self):
        # assuming that the data structure is the same, we don't need to to
        # anything if the cursor is reset to 0
        self._len = 0

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
        compilable (bool, optional): whether the storage is compilable.
            If ``True``, the writer cannot be shared between multiple processes.
            Defaults to ``False``.
        consolidated (bool, optional): if ``True``, the storage will be consolidated after
            its first expansion. Defaults to ``False``.

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
        device: torch.device = "cpu",
        ndim: int = 1,
        compilable: bool = False,
        consolidated: bool = False,
    ):
        super().__init__(
            storage=None,
            max_size=max_size,
            device=device,
            ndim=ndim,
            compilable=compilable,
        )
        self.consolidated = consolidated

    def _init(
        self,
        data: Union[TensorDictBase, torch.Tensor, "PyTree"],  # noqa: F821
    ) -> None:
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


class LazyMemmapStorage(LazyTensorStorage):
    """A memory-mapped storage for tensors and tensordicts.

    Args:
        max_size (int): size of the storage, i.e. maximum number of elements stored
            in the buffer.

    Keyword Args:
        scratch_dir (str or path): directory where memmap-tensors will be written.
        device (torch.device, optional): device where the sampled tensors will be
            stored and sent. Default is :obj:`torch.device("cpu")`.
            If ``None`` is provided, the device is automatically gathered from the
            first batch of data passed. This is not enabled by default to avoid
            data placed on GPU by mistake, causing OOM issues.
        ndim (int, optional): the number of dimensions to be accounted for when
            measuring the storage size. For instance, a storage of shape ``[3, 4]``
            has capacity ``3`` if ``ndim=1`` and ``12`` if ``ndim=2``.
            Defaults to ``1``.
        existsok (bool, optional): whether an error should be raised if any of the
            tensors already exists on disk. Defaults to ``True``. If ``False``, the
            tensor will be opened as is, not overewritten.

    .. note:: When checkpointing a ``LazyMemmapStorage``, one can provide a path identical to where the storage is
        already stored to avoid executing long copies of data that is already stored on disk.
        This will only work if the default :class:`~torchrl.data.TensorStorageCheckpointer` checkpointer is used.
        Example:
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
        device: torch.device = "cpu",
        ndim: int = 1,
        existsok: bool = False,
        compilable: bool = False,
    ):
        super().__init__(max_size, ndim=ndim, compilable=compilable)
        self.initialized = False
        self.scratch_dir = None
        self.existsok = existsok
        if scratch_dir is not None:
            self.scratch_dir = str(scratch_dir)
            if self.scratch_dir[-1] != "/":
                self.scratch_dir += "/"
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

    def state_dict(self) -> Dict[str, Any]:
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

    def _init(self, data: Union[TensorDictBase, torch.Tensor]) -> None:
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
        self.initialized = True

    def get(self, index: Union[int, Sequence[int], slice]) -> Any:
        result = super().get(index)
        return result


class StorageEnsemble(Storage):
    """An ensemble of storages.

    This class is designed to work with :class:`~torchrl.data.replay_buffers.replay_buffers.ReplayBufferEnsemble`.

    Args:
        storages (sequence of Storage): the storages to make the composite storage.

    Keyword Args:
        transforms (list of :class:`~torchrl.envs.Transform`, optional): a list of
            transforms of the same length as storages.

    .. warning::
      This class signatures for :meth:`~.get` does not match other storages, as
      it will return a tuple ``(buffer_id, samples)`` rather than just the samples.

    .. warning::
       This class does not support writing (similarly to :class:`~torchrl.data.replay_buffers.writers.WriterEnsemble`).
       To extend one of the replay buffers, simply index the parent
       :class:`~torchrl.data.ReplayBufferEnsemble` object.

    """

    _default_checkpointer = StorageEnsembleCheckpointer

    def __init__(
        self,
        *storages: Storage,
        transforms: List["Transform"] = None,  # noqa: F821
    ):
        self._rng_private = None
        self._storages = storages
        self._transforms = transforms
        if transforms is not None and len(transforms) != len(storages):
            raise TypeError(
                "transforms must have the same length as the storages " "provided."
            )

    @property
    def _rng(self):
        return self._rng_private

    @_rng.setter
    def _rng(self, value):
        self._rng_private = value
        for storage in self._storages:
            storage._rng = value

    def extend(self, value):
        raise RuntimeError

    def add(self, value):
        raise RuntimeError

    def get(self, item):
        # we return the buffer id too to be able to track the appropriate collate_fn
        buffer_ids = item.get("buffer_ids")
        index = item.get("index")
        results = []
        for (buffer_id, sample) in zip(buffer_ids, index):
            buffer_id = self._convert_id(buffer_id)
            results.append((buffer_id, self._get_storage(buffer_id).get(sample)))
        if self._transforms is not None:
            results = [
                (buffer_id, self._transforms[buffer_id](result))
                if self._transforms[buffer_id] is not None
                else (buffer_id, result)
                for buffer_id, result in results
            ]
        return results

    def _convert_id(self, sub):
        if isinstance(sub, torch.Tensor):
            sub = sub.item()
        return sub

    def _get_storage(self, sub):
        return self._storages[sub]

    def state_dict(self) -> Dict[str, Any]:
        raise NotImplementedError

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        raise NotImplementedError

    _INDEX_ERROR = "Expected an index of type torch.Tensor, range, np.ndarray, int, slice or ellipsis, got {} instead."

    def __getitem__(self, index):
        if isinstance(index, tuple):
            if index[0] is Ellipsis:
                index = (slice(None), index[1:])
            result = self[index[0]]
            if len(index) > 1:
                if result is self:
                    # then index[0] is an ellipsis/slice(None)
                    sample = [storage[index[1:]] for storage in self._storages]
                    return sample
                if isinstance(result, StorageEnsemble):
                    new_index = (slice(None), *index[1:])
                    return result[new_index]
                return result[index[1:]]
            return result
        if isinstance(index, slice) and index == slice(None):
            return self
        if isinstance(index, (list, range, np.ndarray)):
            index = torch.as_tensor(index)
        if isinstance(index, torch.Tensor):
            if index.ndim > 1:
                raise RuntimeError(
                    f"Cannot index a {type(self)} with tensor indices that have more than one dimension."
                )
            if index.is_floating_point():
                raise TypeError(
                    "A floating point index was recieved when an integer dtype was expected."
                )
        if isinstance(index, int) or (not isinstance(index, slice) and len(index) == 0):
            try:
                index = int(index)
            except Exception:
                raise IndexError(self._INDEX_ERROR.format(type(index)))
            try:
                return self._storages[index]
            except IndexError:
                raise IndexError(self._INDEX_ERROR.format(type(index)))
        if isinstance(index, torch.Tensor):
            index = index.tolist()
            storages = [self._storages[i] for i in index]
            transforms = (
                [self._transforms[i] for i in index]
                if self._transforms is not None
                else [None] * len(index)
            )
        else:
            # slice
            storages = self._storages[index]
            transforms = (
                self._transforms[index]
                if self._transforms is not None
                else [None] * len(storages)
            )

        return StorageEnsemble(*storages, transforms=transforms)

    def __len__(self):
        return len(self._storages)

    def __repr__(self):
        storages = textwrap.indent(f"storages={self._storages}", " " * 4)
        transforms = textwrap.indent(f"transforms={self._transforms}", " " * 4)
        return f"StorageEnsemble(\n{storages}, \n{transforms})"


# Utils
def _mem_map_tensor_as_tensor(mem_map_tensor) -> torch.Tensor:
    if isinstance(mem_map_tensor, torch.Tensor):
        # This will account for MemoryMappedTensors
        return mem_map_tensor


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
    if isinstance(storage, ListStorage):
        return _stack_anything
    elif isinstance(storage, TensorStorage):
        return _collate_id
    else:
        raise NotImplementedError(
            f"Could not find a default collate_fn for storage {type(storage)}."
        )


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
