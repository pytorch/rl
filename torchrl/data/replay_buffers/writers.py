# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import heapq
import json
import textwrap
from abc import ABC, abstractmethod
from copy import copy
from multiprocessing.context import get_spawning_popen
from pathlib import Path
from typing import Any, Dict, Sequence

import numpy as np
import torch

from tensordict import is_tensor_collection, MemoryMappedTensor
from tensordict.utils import _STRDTYPE2DTYPE
from torch import multiprocessing as mp
from torch.utils._pytree import tree_flatten

from torchrl.data.replay_buffers.storages import Storage
from torchrl.data.replay_buffers.utils import _reduce


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

    @abstractmethod
    def dumps(self, path):
        ...

    @abstractmethod
    def loads(self, path):
        ...

    @abstractmethod
    def state_dict(self) -> Dict[str, Any]:
        ...

    @abstractmethod
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        ...


class ImmutableDatasetWriter(Writer):
    """A blocking writer for immutable datasets."""

    WRITING_ERR = "This dataset doesn't allow writing."

    def add(self, data: Any) -> int:
        raise RuntimeError(self.WRITING_ERR)

    def extend(self, data: Sequence) -> torch.Tensor:
        raise RuntimeError(self.WRITING_ERR)

    def _empty(self):
        raise RuntimeError(self.WRITING_ERR)

    def dumps(self, path):
        ...

    def loads(self, path):
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

    def dumps(self, path):
        path = Path(path).absolute()
        path.mkdir(exist_ok=True)
        with open(path / "metadata.json", "w") as file:
            json.dump({"cursor": self._cursor}, file)

    def loads(self, path):
        path = Path(path).absolute()
        with open(path / "metadata.json", "r") as file:
            metadata = json.load(file)
            self._cursor = metadata["cursor"]

    def add(self, data: Any) -> int:
        ret = self._cursor
        _cursor = self._cursor
        # we need to update the cursor first to avoid race conditions between workers
        self._cursor = (self._cursor + 1) % self._storage.max_size
        self._storage[_cursor] = data
        return ret

    def extend(self, data: Sequence) -> torch.Tensor:
        cur_size = self._cursor
        if is_tensor_collection(data) or isinstance(data, torch.Tensor):
            batch_size = len(data)
        elif isinstance(data, list):
            batch_size = len(data)
        else:
            batch_size = len(tree_flatten(data)[0][0])
        if batch_size == 0:
            raise RuntimeError("Expected at least one element in extend.")
        device = data.device if hasattr(data, "device") else None
        index = (
            torch.arange(
                cur_size, batch_size + cur_size, dtype=torch.long, device=device
            )
            % self._storage.max_size
        )
        # we need to update the cursor first to avoid race conditions between workers
        self._cursor = (batch_size + cur_size) % self._storage.max_size
        self._storage[index] = data
        return index

    def state_dict(self) -> Dict[str, Any]:
        return {"_cursor": self._cursor}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self._cursor = state_dict["_cursor"]

    def _empty(self):
        self._cursor = 0

    @property
    def _cursor(self):
        _cursor_value = self.__dict__.get("_cursor_value", None)
        if _cursor_value is None:
            _cursor_value = self._cursor_value = mp.Value("i", 0)
        return _cursor_value.value

    @_cursor.setter
    def _cursor(self, value):
        _cursor_value = self.__dict__.get("_cursor_value", None)
        if _cursor_value is None:
            _cursor_value = self._cursor_value = mp.Value("i", 0)
        _cursor_value.value = value

    def __getstate__(self):
        state = copy(self.__dict__)
        if get_spawning_popen() is None:
            cursor = self._cursor
            del state["_cursor_value"]
            state["cursor__context"] = cursor
        return state

    def __setstate__(self, state):
        cursor = state.pop("cursor__context", None)
        if cursor is not None:
            _cursor_value = mp.Value("i", cursor)
            state["_cursor_value"] = _cursor_value
        self.__dict__.update(state)


class TensorDictRoundRobinWriter(RoundRobinWriter):
    """A RoundRobin Writer class for composable, tensordict-based replay buffers."""

    def add(self, data: Any) -> int:
        ret = self._cursor
        # we need to update the cursor first to avoid race conditions between workers
        self._cursor = (ret + 1) % self._storage.max_size
        data["index"] = ret
        self._storage[ret] = data
        return ret

    def extend(self, data: Sequence) -> torch.Tensor:
        cur_size = self._cursor
        batch_size = len(data)
        device = data.device if hasattr(data, "device") else None
        index = (
            torch.arange(
                cur_size, batch_size + cur_size, dtype=torch.long, device=device
            )
            % self._storage.max_size
        )
        # we need to update the cursor first to avoid race conditions between workers
        self._cursor = (batch_size + cur_size) % self._storage.max_size
        # storage must convert the data to the appropriate format if needed
        data["index"] = index
        self._storage[index] = data
        return index


class TensorDictMaxValueWriter(Writer):
    """A Writer class for composable replay buffers that keeps the top elements based on some ranking key.

    Args:
        rank_key (str or tuple of str): the key to rank the elements by. Defaults to ``("next", "reward")``.
        reduction (str): the reduction method to use if the rank key has more than one element.
            Can be ``"max"``, ``"min"``, ``"mean"``, ``"median"`` or ``"sum"``.

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

    def __init__(self, rank_key=None, reduction: str = "sum", **kwargs) -> None:
        super().__init__(**kwargs)
        self._cursor = 0
        self._current_top_values = []
        self._rank_key = rank_key
        self._reduction = reduction
        if self._rank_key is None:
            self._rank_key = ("next", "reward")

    def get_insert_index(self, data: Any) -> int:
        """Returns the index where the data should be inserted, or ``None`` if it should not be inserted."""
        if not is_tensor_collection(data):
            raise RuntimeError(
                f"{type(self)} expects data to be a tensor collection (tensordict or tensorclass). Found a {type(data)} instead."
            )
        if data.batch_dims > 1:
            raise RuntimeError(
                "Expected input tensordict to have no more than 1 dimension, got"
                f"tensordict.batch_size = {data.batch_size}"
            )

        ret = None
        rank_data = data.get("_data", default=data).get(self._rank_key)

        # If time dimension, sum along it.
        if rank_data.numel() > 1:
            rank_data = _reduce(rank_data.reshape(-1), self._reduction, dim=0)
        else:
            rank_data = rank_data.item()

        if rank_data is None:
            raise KeyError(f"Rank key {self._rank_key} not found in data.")

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
        """Inserts a single element of data at an appropriate index, and returns that index.

        The ``rank_key`` in the data passed to this module should be structured as [].
        If it has more dimensions, it will be reduced to a single value using the ``reduction`` method.
        """
        index = self.get_insert_index(data)
        if index is not None:
            data.set("index", index)
            self._storage[index] = data
        return index

    def extend(self, data: Sequence) -> None:
        """Inserts a series of data points at appropriate indices.

        The ``rank_key`` in the data passed to this module should be structured as [B].
        If it has more dimensions, it will be reduced to a single value using the ``reduction`` method.
        """
        data_to_replace = {}
        for i, sample in enumerate(data):
            index = self.get_insert_index(sample)
            if index is not None:
                data_to_replace[index] = i

        # Replace the data in the storage all at once
        if len(data_to_replace) > 0:
            keys, values = zip(*data_to_replace.items())
            index = data.get("index", None)
            dtype = index.dtype if index is not None else torch.long
            device = index.device if index is not None else data.device
            values = list(values)
            keys = torch.tensor(keys, dtype=dtype, device=device)
            if index is not None:
                index[values] = keys
                data.set("index", index)
            self._storage.set(keys, data[values])
            return keys.long()
        return None

    def _empty(self) -> None:
        self._cursor = 0
        self._current_top_values = []

    def __getstate__(self):
        if get_spawning_popen() is not None:
            raise RuntimeError(
                f"Writers of type {type(self)} cannot be shared between processes."
            )
        state = copy(self.__dict__)
        return state

    def dumps(self, path):
        path = Path(path).absolute()
        path.mkdir(exist_ok=True)
        t = torch.as_tensor(self._current_top_values)
        try:
            MemoryMappedTensor.from_filename(
                filename=path / "current_top_values.memmap",
                shape=t.shape,
                dtype=t.dtype,
            ).copy_(t)
        except FileNotFoundError:
            MemoryMappedTensor.from_tensor(
                t, filename=path / "current_top_values.memmap"
            )
        with open(path / "metadata.json", "w") as file:
            json.dump(
                {
                    "cursor": self._cursor,
                    "rank_key": self._rank_key,
                    "dtype": str(t.dtype),
                    "shape": list(t.shape),
                },
                file,
            )

    def loads(self, path):
        path = Path(path).absolute()
        with open(path / "metadata.json", "r") as file:
            metadata = json.load(file)
            self._cursor = metadata["cursor"]
            self._rank_key = metadata["rank_key"]
            shape = torch.Size(metadata["shape"])
            dtype = metadata["dtype"]
        self._current_top_values = MemoryMappedTensor.from_filename(
            filename=path / "current_top_values.memmap",
            dtype=_STRDTYPE2DTYPE[dtype],
            shape=shape,
        ).tolist()

    def state_dict(self) -> Dict[str, Any]:
        raise NotImplementedError

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        raise NotImplementedError


class WriterEnsemble(Writer):
    """An ensemble of writers.

    This class is designed to work with :class:`~torchrl.data.replay_buffers.replay_buffers.ReplayBufferEnsemble`.
    It contains the writers but blocks writing with any of them.

    Args:
        writers (sequence of Writer): the writers to make the composite writer.

    .. warning::
       This class does not support writing.
       To extend one of the replay buffers, simply index the parent
       :class:`~torchrl.data.ReplayBufferEnsemble` object.

    """

    def __init__(self, *writers):
        self._writers = writers

    def _empty(self):
        raise NotImplementedError

    def dumps(self, path: Path):
        path = Path(path).absolute()
        for i, writer in enumerate(self._writers):
            writer.dumps(path / str(i))

    def loads(self, path: Path):
        path = Path(path).absolute()
        for i, writer in enumerate(self._writers):
            writer.loads(path / str(i))

    def add(self):
        raise NotImplementedError

    def extend(self):
        raise NotImplementedError

    _INDEX_ERROR = "Expected an index of type torch.Tensor, range, np.ndarray, int, slice or ellipsis, got {} instead."

    def __getitem__(self, index):
        if isinstance(index, tuple):
            if index[0] is Ellipsis:
                index = (slice(None), index[1:])
            result = self[index[0]]
            if len(index) > 1:
                raise IndexError(
                    f"Tuple of length greater than 1 are not accepted to index writers of type {type(self)}."
                )
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
                return self._writers[index]
            except IndexError:
                raise IndexError(self._INDEX_ERROR.format(type(index)))
        if isinstance(index, torch.Tensor):
            index = index.tolist()
            writers = [self._writers[i] for i in index]
        else:
            # slice
            writers = self._writers[index]
        return WriterEnsemble(*writers)

    def __len__(self):
        return len(self._writers)

    def __repr__(self):
        writers = textwrap.indent(f"writers={self._writers}", " " * 4)
        return f"WriterEnsemble(\n{writers})"

    def state_dict(self) -> Dict[str, Any]:
        raise NotImplementedError

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        raise NotImplementedError
