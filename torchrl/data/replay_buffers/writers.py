# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import heapq
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
        ...

    @abstractmethod
    def extend(self, data: Sequence) -> torch.Tensor:
        """Inserts a series of data points at appropriate indices, and returns a tensor containing the indices."""
        ...

    @abstractmethod
    def _empty(self):
        ...

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
        index = np.arange(cur_size, batch_size + cur_size) % self._storage.max_size
        self._cursor = (batch_size + cur_size) % self._storage.max_size
        self._storage[index] = data
        return index

    def state_dict(self) -> Dict[str, Any]:
        return {"_cursor": self._cursor}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self._cursor = state_dict["_cursor"]

    def _empty(self):
        self._cursor = 0


class TensorDictRoundRobinWriter(RoundRobinWriter):
    """A RoundRobin Writer class for composable, tensordict-based replay buffers."""

    def add(self, data: Any) -> int:
        ret = self._cursor
        data["index"] = ret
        self._storage[self._cursor] = data
        self._cursor = (self._cursor + 1) % self._storage.max_size
        return ret

    def extend(self, data: Sequence) -> torch.Tensor:
        cur_size = self._cursor
        batch_size = len(data)
        index = np.arange(cur_size, batch_size + cur_size) % self._storage.max_size
        self._cursor = (batch_size + cur_size) % self._storage.max_size
        # storage must convert the data to the appropriate format if needed
        data["index"] = index
        self._storage[index] = data
        return index


class TensorDictMaxValueWriter(Writer):
    """A Writer class for composable replay buffers that keeps the top elements based on some ranking key.

    If rank_key is not provided, the key will be ("next", "reward").

    Examples:
    >>> import torch
    >>> from tensordict import TensorDict
    >>> from torchrl.data import LazyTensorStorage, TensorDictReplayBuffer, TensorDictMaxValueWriter
    >>> from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement

    >>> rb = TensorDictReplayBuffer(
    ...     storage=LazyTensorStorage(1),
    ...     sampler=SamplerWithoutReplacement(),
    ...     batch_size=1,
    ...     writer=TensorDictMaxValueWriter(rank_key="key"),
    ... )
    >>> td = TensorDict({
    ...     "key": torch.tensor(range(10)),
    ...     "obs": torch.tensor(range(10))
    ... }, batch_size=10)
    >>> rb.extend(td)
    >>> print(rb.sample().get("obs").item())
    9

    >>> td = TensorDict({
    ...     "key": torch.tensor(range(10, 20)),
    ...     "obs": torch.tensor(range(10, 20))
    ... }, batch_size=10)
    >>> rb.extend(td)
    >>> print(rb.sample().get("obs").item())
    19

    >>> td = TensorDict({
    ...     "key": torch.tensor(range(10)),
    ...     "obs": torch.tensor(range(10))
    ... }, batch_size=10)
    >>> rb.extend(td)
    >>> print(rb.sample().get("obs").item())
    19
    """

    def __init__(self, rank_key=None, **kw) -> None:
        super().__init__(**kw)
        self._cursor = 0
        self._current_top_values = []
        self._rank_key = rank_key
        if self._rank_key is None:
            self._rank_key = ("next", "reward")

    def get_insert_index(self, data: Any) -> int:
        """Returns the index where the data should be inserted, or None if it should not be inserted."""
        ret = None
        rank_data = data.get("_data").get(self._rank_key)

        # Sum the rank key, in case it is a whole trajectory
        rank_data = rank_data.sum().item()

        if rank_data is None:
            raise ValueError(f"Rank key {self._rank_key} not found in data.")

        # If the buffer is not full, add the data
        if len(self._current_top_values) < self._storage.max_size:

            ret = self._cursor
            self._cursor = (self._cursor + 1) % self._storage.max_size

            # Add new reward to the heap
            heapq.heappush(self._current_top_values, (rank_data, ret))

        # If the buffer is full, check if the new data is better than the worst data in the buffer
        elif rank_data > self._current_top_values[0][0]:

            # retrieve position of the smallest value
            min_sample = heapq.heappop(self._current_top_values)
            ret = min_sample[1]

            # Add new reward to the heap
            heapq.heappush(self._current_top_values, (rank_data, ret))

        return ret

    def add(self, data: Any) -> int:
        """Inserts a single element of data at an appropriate index, and returns that index."""
        index = self.get_insert_index(data)
        if index is not None:
            data.set("index", index)
            self._storage[index] = data
        return index

    def extend(self, data: Sequence) -> None:
        """Inserts a series of data points at appropriate indices."""
        data_to_replace = {}
        for i, sample in enumerate(data):
            index = self.get_insert_index(sample)
            if index is not None:
                data_to_replace[index] = i

        # Replace the data in the storage all at once
        keys = list(data_to_replace.keys())
        if len(keys) > 0:
            values = list(data_to_replace.values())
            data["index"][values].copy_(torch.tensor(keys))
            self._storage[keys] = data[values]

    def _empty(self) -> None:
        self._cursor = 0
        self._current_top_values = []
