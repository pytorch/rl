# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import abc
import multiprocessing as mp
from copy import copy
from multiprocessing.context import get_spawning_popen
from typing import Any

import torch

from torchrl.data.replay_buffers.checkpointers import StorageCheckpointerBase

try:
    from torch.compiler import disable as compile_disable, is_compiling
except ImportError:
    from torch._dynamo import disable as compile_disable, is_compiling


class Storage:
    """A Storage is the container of a replay buffer.

    Every storage must have a set, get and __len__ methods implemented.
    Get and set should support integers as well as list of integers.

    The storage does not need to have a definite size, but if it does one should
    make sure that it is compatible with the buffer size.

    """

    ndim = 1
    max_size: int
    supports_conditional_update: bool = False
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
        self._mutation_revision_value = (
            torch.zeros((), dtype=torch.int64) if compilable else mp.Value("q", 0)
        )
        self._last_cursor_index_value = (
            torch.full((), -1, dtype=torch.int64) if compilable else mp.Value("q", -1)
        )

    @property
    def _mutation_revision(self) -> int | torch.Tensor:
        """Monotonic storage-content revision shared with spawned processes."""
        revision = getattr(self, "_mutation_revision_value", None)
        if not self._compilable:
            if revision is None:
                revision = self._mutation_revision_value = mp.Value("q", 0)
            return revision.value
        if revision is None:
            revision = self._mutation_revision_value = torch.zeros(
                (), dtype=torch.int64
            )
        return revision if is_compiling() else int(revision.item())

    def _bump_mutation_revision(self) -> None:
        """Invalidates process-local metadata derived from storage contents."""
        revision = getattr(self, "_mutation_revision_value", None)
        if not self._compilable:
            if revision is None:
                revision = self._mutation_revision_value = mp.Value("q", 0)
            with revision.get_lock():
                revision.value += 1
        else:
            if revision is None:
                revision = self._mutation_revision_value = torch.zeros(
                    (), dtype=torch.int64
                )
            revision.add_(1)

    @property
    def _last_cursor_index(self) -> int | torch.Tensor | None:
        """Last written time coordinate, shared with spawned processes."""
        cursor = getattr(self, "_last_cursor_index_value", None)
        if cursor is None:
            return None
        if self._compilable:
            if is_compiling():
                return cursor
            cursor = int(cursor.item())
        else:
            cursor = cursor.value
        return None if cursor < 0 else cursor

    def _set_last_cursor(self, cursor: Any) -> None:
        self._last_cursor = cursor
        if isinstance(cursor, torch.Tensor):
            cursor = cursor.reshape(-1)
            cursor = cursor[-1] if cursor.numel() else -1
        elif isinstance(cursor, range):
            cursor = int(cursor[-1]) if len(cursor) else -1
        elif isinstance(cursor, (tuple, list)):
            time_cursor = cursor[0]
            if isinstance(time_cursor, torch.Tensor):
                time_cursor = time_cursor.reshape(-1)
                cursor = time_cursor[-1] if time_cursor.numel() else -1
            else:
                cursor = int(time_cursor)
        elif cursor is None:
            cursor = -1
        else:
            cursor = int(cursor)
        if self._compilable:
            if isinstance(cursor, torch.Tensor):
                self._last_cursor_index_value = cursor
            else:
                self._last_cursor_index_value.fill_(cursor)
            return
        if isinstance(cursor, torch.Tensor):
            cursor = int(cursor.item())
        shared_cursor = self._last_cursor_index_value
        with shared_cursor.get_lock():
            shared_cursor.value = cursor

    @property
    def checkpointer(self):
        return self._checkpointer

    def register_save_hook(self, hook):
        """Register a save hook for this storage.

        The hook is forwarded to the checkpointer.
        """
        self._checkpointer.register_save_hook(hook)

    def register_load_hook(self, hook):
        """Register a load hook for this storage.

        The hook is forwarded to the checkpointer.
        """
        self._checkpointer.register_load_hook(hook)

    @checkpointer.setter
    def checkpointer(self, value: StorageCheckpointerBase | None) -> None:
        if value is None:
            value = self._default_checkpointer()
        self._checkpointer = value

    @property
    def _is_full(self):
        return len(self) == self.max_size

    @property
    def _attached_entities(self) -> list:
        # RBs that use a given instance of Storage should add
        # themselves to this set.
        _attached_entities_list = getattr(self, "_attached_entities_list", None)
        if _attached_entities_list is None:
            self._attached_entities_list = _attached_entities_list = []
        return _attached_entities_list

    # TODO: Check this
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
        self._bump_mutation_revision()

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
    def state_dict(self) -> dict[str, Any]:
        ...

    @abc.abstractmethod
    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        ...

    @abc.abstractmethod
    def _empty(self):
        ...

    # TODO: Without this disable, compiler recompiles due to changing len(self) guards.
    @compile_disable()
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
        """Alias for :meth:`dumps`."""
        return self.dumps(*args, **kwargs)

    def dump(self, *args, **kwargs):
        """Alias for :meth:`dumps`."""
        return self.dumps(*args, **kwargs)

    def load(self, *args, **kwargs):
        """Alias for :meth:`loads`."""
        return self.loads(*args, **kwargs)

    def __getstate__(self):
        state = copy(self.__dict__)
        state["_rng"] = None
        if get_spawning_popen() is None:
            revision = self._mutation_revision
            last_cursor = self._last_cursor_index
            state.pop("_mutation_revision_value", None)
            state.pop("_last_cursor_index_value", None)
            state["mutation_revision__context"] = revision
            state["last_cursor_index__context"] = last_cursor
        return state

    def __setstate__(self, state):
        revision = state.pop("mutation_revision__context", None)
        last_cursor = state.pop("last_cursor_index__context", None)
        compilable = state.get("_compilable", False)
        state.setdefault("_compilable", compilable)
        if revision is not None:
            if compilable:
                state["_mutation_revision_value"] = torch.tensor(
                    revision, dtype=torch.int64
                )
            else:
                state["_mutation_revision_value"] = mp.Value("q", revision)
        if last_cursor is not None:
            if compilable:
                state["_last_cursor_index_value"] = torch.tensor(
                    last_cursor, dtype=torch.int64
                )
            else:
                state["_last_cursor_index_value"] = mp.Value("q", last_cursor)
        elif "_last_cursor_index_value" not in state:
            state["_last_cursor_index_value"] = (
                torch.full((), -1, dtype=torch.int64)
                if compilable
                else mp.Value("q", -1)
            )
        if "_mutation_revision_value" not in state:
            state["_mutation_revision_value"] = (
                torch.zeros((), dtype=torch.int64) if compilable else mp.Value("q", 0)
            )
        self.__dict__.update(state)

    def __contains__(self, item):
        return self.contains(item)

    @abc.abstractmethod
    def contains(self, item):
        ...
