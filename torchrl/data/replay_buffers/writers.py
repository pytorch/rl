# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC, abstractmethod
from typing import Any, Dict, Sequence

import numpy as np
import torch

from .storages import Storage


class Writer(ABC):
    """A ReplayBuffer base Writer class."""

    def __init__(self) -> None:
        self._storage = None

    def register_storage(self, storage: Storage) -> None:
        self._storage = storage

    @abstractmethod
    def add(self, data: Any) -> int:
        """Inserts one piece of data at an appropriate index, and returns that index."""
        raise NotImplementedError

    @abstractmethod
    def extend(self, data: Sequence) -> torch.Tensor:
        """Inserts a series of data points at appropriate indices, and returns a tensor containing the indices."""
        raise NotImplementedError

    def state_dict(self) -> Dict[str, Any]:
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        return


class RoundRobinWriter(Writer):
    """A RoundRobin Writer class for composable replay buffers."""

    def __init__(self, **kw) -> None:
        super().__init__(**kw)
        self._cursor = 0

    def add(self, data: Any) -> int:
        ret = self._cursor
        self._storage[self._cursor] = data
        self._cursor = (self._cursor + 1) % self._storage.max_size
        return ret

    def extend(self, data: Sequence) -> torch.Tensor:
        cur_size = self._cursor
        batch_size = len(data)
        if cur_size + batch_size <= self._storage.max_size:
            index = np.arange(cur_size, cur_size + batch_size)
            self._cursor = (self._cursor + batch_size) % self._storage.max_size
        else:
            d = self._storage.max_size - cur_size
            index = np.empty(batch_size, dtype=np.int64)
            index[:d] = np.arange(cur_size, self._storage.max_size)
            index[d:] = np.arange(batch_size - d)
            self._cursor = batch_size - d
        # storage must convert the data to the appropriate format if needed
        self._storage[index] = data
        return index

    def state_dict(self) -> Dict[str, Any]:
        return {"_cursor": self._cursor}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self._cursor = state_dict["_cursor"]
