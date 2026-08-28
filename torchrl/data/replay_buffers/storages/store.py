# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import importlib
import multiprocessing as mp
import textwrap
from collections.abc import Sequence
from copy import copy
from multiprocessing.context import get_spawning_popen
from typing import Any

import torch
from tensordict import is_tensor_collection, TensorDict

from torchrl.data.replay_buffers.utils import INT_CLASSES

try:
    from torch.compiler import is_compiling
except ImportError:
    from torch._dynamo import is_compiling


_has_store = (
    importlib.util.find_spec("redis", None) is not None
    and importlib.util.find_spec("tensordict.store", None) is not None
)


from .base import Storage
from .tensor import _flip_list


class StoreStorage(Storage):
    """A replay buffer storage backed by a key-value store (Redis, Dragonfly, etc.).

    Uses :class:`~tensordict.store.TensorDictStore` for out-of-core storage of
    tensors, non-tensor data (strings, Python objects), TensorDicts, and
    TensorClasses. This enables replay buffers whose data lives in a
    Redis-compatible server rather than local RAM or disk.

    The storage is lazily initialized: the backing
    :class:`~tensordict.store.TensorDictStore` is created on the first call to
    :meth:`set`, using the structure of the incoming data to determine the key
    layout.

    Args:
        max_size (int): Maximum number of elements the storage can hold.

    Keyword Args:
        backend (str): Name of the store backend. Accepted values include
            ``"redis"`` (default), ``"dragonfly"``, ``"keydb"``, or any
            Redis-wire-compatible server name.
        host (str): Server hostname. Defaults to ``"localhost"``.
        port (int): Server port. Defaults to ``6379``.
        db (int): Database number. Defaults to ``0``.
        compilable (bool): Whether the storage is compilable. Defaults to ``False``.
        **store_kwargs: Additional keyword arguments forwarded to
            :class:`~tensordict.store.TensorDictStore`.

    .. note:: Requires ``redis`` package: ``pip install redis``.

    .. note:: Requires a tensordict version that includes the ``tensordict.store``
        module (with per-element non-tensor indexing support).

    Examples:
        >>> from torchrl.data import ReplayBuffer
        >>> from torchrl.data.replay_buffers import StoreStorage
        >>> storage = StoreStorage(max_size=1000, host="localhost", port=6379)
        >>> rb = ReplayBuffer(storage=storage, batch_size=32)
        >>> data = TensorDict({"obs": torch.randn(10, 4), "action": torch.randn(10, 2)}, [10])
        >>> rb.extend(data)
        >>> sample = rb.sample()
    """

    _storage = None

    def __init__(
        self,
        max_size: int,
        *,
        backend: str = "redis",
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        compilable: bool = False,
        **store_kwargs,
    ):
        if not _has_store:
            raise ModuleNotFoundError(
                "StoreStorage requires both the `redis` package and a version of "
                "tensordict that includes `tensordict.store`. "
                "Install redis with: pip install redis"
            )
        super().__init__(max_size=max_size, compilable=compilable)
        self._backend = backend
        self._host = host
        self._port = port
        self._db = db
        self._store_kwargs = store_kwargs
        self.initialized = False
        self._len = 0
        self._last_cursor = None
        self._is_tensor = False
        self._tensorclass_type = None

    @property
    def _len(self):
        _len_value = getattr(self, "_len_value", None)
        if not self._compilable:
            if _len_value is None:
                _len_value = self._len_value = mp.Value("i", 0)
            return _len_value.value
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

    def _init(self, data):
        """Initialize the TensorDictStore from a single data element."""
        from tensordict.store import TensorDictStore

        if isinstance(data, torch.Tensor) and not is_tensor_collection(data):
            self._is_tensor = True
            # Wrap raw tensors under a "_tensor" key so TensorDictStore can manage them.
            template = TensorDict(
                {"_tensor": torch.empty(self.max_size, *data.shape, dtype=data.dtype)},
                batch_size=[self.max_size],
            )
            self._storage = TensorDictStore.from_tensordict(
                template,
                backend=self._backend,
                host=self._host,
                port=self._port,
                db=self._db,
                **self._store_kwargs,
            )
            self.initialized = True
            return

        if not is_tensor_collection(data):
            raise TypeError(
                f"StoreStorage does not support data of type {type(data)}. "
                "Use TensorDict, tensorclass, or torch.Tensor."
            )

        from tensordict.utils import _is_tensorclass

        data_type = type(data)
        self._tensorclass_type = data_type if _is_tensorclass(data_type) else None

        # Create an empty TensorDictStore -- keys will be registered lazily on
        # the first write via __setitem__.
        self._storage = TensorDictStore(
            batch_size=[self.max_size],
            backend=self._backend,
            host=self._host,
            port=self._port,
            db=self._db,
            **self._store_kwargs,
        )
        self.initialized = True

    def set(
        self,
        cursor: int | Sequence[int] | slice,
        data: Any,
        *,
        set_cursor: bool = True,
    ):
        if set_cursor:
            self._set_last_cursor(cursor)

        if isinstance(data, list):
            data = _flip_list(data)

        if set_cursor:
            self._get_new_len(data, cursor)

        if not self.initialized:
            if not isinstance(cursor, INT_CLASSES):
                if is_tensor_collection(data):
                    self._init(data[0])
                elif isinstance(data, torch.Tensor):
                    self._init(data[0])
                else:
                    raise TypeError(
                        f"StoreStorage does not support data of type {type(data)}."
                    )
            else:
                self._init(data)

        if self._is_tensor:
            self._storage["_tensor"][cursor] = data
        else:
            self._storage[cursor] = data
        self._bump_mutation_revision()

    def _get_new_len(self, data, cursor):
        if is_tensor_collection(data) or isinstance(data, torch.Tensor):
            numel = data.shape[0] if data.shape else 1
        else:
            numel = 1
        self._len = min(self._len + numel, self.max_size)

    def get(self, index: int | Sequence[int] | slice) -> Any:
        if not self.initialized:
            raise RuntimeError("Cannot get elements out of a non-initialized storage.")
        if self._is_tensor:
            return self._storage["_tensor"][index]
        result = self._storage[index]
        if self._tensorclass_type is not None:
            result = self._tensorclass_type.from_tensordict(result)
        return result

    def __len__(self):
        return self._len

    def _empty(self):
        self._len = 0
        self._bump_mutation_revision()

    def state_dict(self) -> dict[str, Any]:
        return {
            "initialized": self.initialized,
            "_len": self._len,
            "_backend": self._backend,
            "_host": self._host,
            "_port": self._port,
            "_db": self._db,
            "_store_kwargs": self._store_kwargs,
            "_td_id": self._storage._td_id if self._storage is not None else None,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self._len = state_dict["_len"]
        self.initialized = state_dict["initialized"]
        td_id = state_dict.get("_td_id")
        if td_id is not None and self.initialized:
            from tensordict.store import TensorDictStore

            self._storage = TensorDictStore.from_store(
                backend=state_dict["_backend"],
                host=state_dict["_host"],
                port=state_dict["_port"],
                db=state_dict["_db"],
                td_id=td_id,
            )
        self._bump_mutation_revision()

    def contains(self, item):
        if isinstance(item, int):
            if item < 0:
                item += self._len
            return 0 <= item < self._len
        if isinstance(item, torch.Tensor):
            if item.dtype == torch.bool:
                return item.sum() <= self._len
            if item.ndim:
                return torch.tensor(
                    [self.contains(idx.item()) for idx in item],
                    dtype=torch.bool,
                    device=item.device,
                )
            return torch.tensor(self.contains(item.item()), device=item.device)
        raise NotImplementedError(f"type {type(item)} is not supported yet.")

    def __getstate__(self):
        state = copy(self.__dict__)
        state["_rng"] = None
        if get_spawning_popen() is None:
            length = self._len
            del state["_len_value"]
            state["len__context"] = length
        return state

    def __setstate__(self, state):
        length = state.pop("len__context", None)
        if length is not None:
            if not state.get("_compilable", False):
                _len_value = mp.Value("i", length)
                state["_len_value"] = _len_value
            else:
                state["_len_value"] = length
        self.__dict__.update(state)

    def __repr__(self):
        if not self.initialized:
            storage_str = textwrap.indent("data=<empty>", 4 * " ")
        else:
            storage_str = textwrap.indent(
                f"data=TensorDictStore(td_id={self._storage._td_id!r})", 4 * " "
            )
        len_str = textwrap.indent(f"len={len(self)}", 4 * " ")
        maxsize_str = textwrap.indent(f"max_size={self.max_size}", 4 * " ")
        backend_str = textwrap.indent(
            f"backend={self._backend!r}, host={self._host!r}, port={self._port}",
            4 * " ",
        )
        return f"{self.__class__.__name__}(\n{storage_str}, \n{len_str}, \n{maxsize_str}, \n{backend_str})"


# Utils
