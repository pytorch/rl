# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import heapq
import json
import textwrap
from abc import ABC, abstractmethod
from copy import copy
from multiprocessing.context import get_spawning_popen
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from tensordict import is_tensor_collection, MemoryMappedTensor, TensorDictBase
from tensordict.utils import expand_as_right, is_tensorclass
from torch import multiprocessing as mp
from torchrl._utils import _STRDTYPE2DTYPE

try:
    from torch.utils._pytree import tree_leaves
except ImportError:
    from torch.utils._pytree import tree_flatten

    def tree_leaves(data):  # noqa: D103
        tree_flat, _ = tree_flatten(data)
        return tree_flat


from torchrl.data.replay_buffers.storages import Storage
from torchrl.data.replay_buffers.utils import _is_int, _reduce


class Writer(ABC):
    """A ReplayBuffer base Writer class."""

    _storage: Storage
    _rng: torch.Generator | None = None

    def __init__(self, compilable: bool = False) -> None:
        self._storage = None
        self._compilable = compilable

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
    def state_dict(self) -> dict[str, Any]:
        ...

    @abstractmethod
    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        ...

    def _replicate_index(self, index):
        # replicates the index in a non-zero format to have as many indices as
        # elements truly written when the storage is multidim
        if self._storage.ndim == 1:
            return index
        device = (
            index.device if isinstance(index, torch.Tensor) else torch.device("cpu")
        )
        mesh = torch.stack(
            torch.meshgrid(
                *(torch.arange(dim, device=device) for dim in self._storage.shape[1:])
            ),
            -1,
        ).flatten(0, -2)
        if _is_int(index):
            index0 = torch.as_tensor(int(index)).expand(mesh.shape[0], 1)
            return torch.cat([index0, mesh], 1)
        return torch.cat(
            [
                index.repeat_interleave(mesh.shape[0]).unsqueeze(1),
                mesh.repeat(index.numel(), 1),
            ],
            1,
        )

    def __repr__(self):
        return f"{self.__class__.__name__}()"

    def __getstate__(self):
        state = copy(self.__dict__)
        state["_rng"] = None
        return state


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

    def state_dict(self) -> dict[str, Any]:
        return {}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        return


class RoundRobinWriter(Writer):
    """A RoundRobin Writer class for composable replay buffers.

    Args:
        compilable (bool, optional): whether the writer is compilable.
            If ``True``, the writer cannot be shared between multiple processes.
            Defaults to ``False``.

    """

    def __init__(self, compilable: bool = False) -> None:
        super().__init__(compilable=compilable)
        self._cursor = 0

    def dumps(self, path):
        path = Path(path).absolute()
        path.mkdir(exist_ok=True)
        with open(path / "metadata.json", "w") as file:
            json.dump({"cursor": self._cursor}, file)

    def loads(self, path):
        path = Path(path).absolute()
        with open(path / "metadata.json") as file:
            metadata = json.load(file)
            self._cursor = metadata["cursor"]

    def add(self, data: Any) -> int | torch.Tensor:
        index = self._cursor
        _cursor = self._cursor
        # we need to update the cursor first to avoid race conditions between workers
        self._cursor = (self._cursor + 1) % self._storage._max_size_along_dim0(
            single_data=data
        )
        self._write_count += 1
        # Replicate index requires the shape of the storage to be known
        # Other than that, a "flat" (1d) index is ok to write the data
        self._storage.set(_cursor, data)
        index = self._replicate_index(index)
        for ent in self._storage._attached_entities_iter():
            ent.mark_update(index)
        return index

    def extend(self, data: Sequence) -> torch.Tensor:
        cur_size = self._cursor
        if is_tensor_collection(data) or isinstance(data, torch.Tensor):
            batch_size = len(data)
        elif isinstance(data, list):
            batch_size = len(data)
        else:
            batch_size = len(tree_leaves(data)[0])
        if batch_size == 0:
            raise RuntimeError("Expected at least one element in extend.")
        device = data.device if hasattr(data, "device") else None
        max_size_along0 = self._storage._max_size_along_dim0(batched_data=data)
        index = (
            torch.arange(
                cur_size, batch_size + cur_size, dtype=torch.long, device=device
            )
            % max_size_along0
        )
        # we need to update the cursor first to avoid race conditions between workers
        self._cursor = (batch_size + cur_size) % max_size_along0
        self._write_count += batch_size
        # Replicate index requires the shape of the storage to be known
        # Other than that, a "flat" (1d) index is ok to write the data
        self._storage.set(index, data)
        index = self._replicate_index(index)
        for ent in self._storage._attached_entities_iter():
            ent.mark_update(index)
        return index

    def state_dict(self) -> dict[str, Any]:
        return {"_cursor": self._cursor}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self._cursor = state_dict["_cursor"]

    def _empty(self):
        self._cursor = 0
        self._write_count = 0

    @property
    def _cursor(self):
        _cursor_value = self.__dict__.get("_cursor_value", None)
        if not self._compilable:
            if _cursor_value is None:
                _cursor_value = self._cursor_value = mp.Value("i", 0)
            return _cursor_value.value
        else:
            if _cursor_value is None:
                _cursor_value = self._cursor_value = 0
            return _cursor_value

    @_cursor.setter
    def _cursor(self, value):
        if not self._compilable:
            _cursor_value = self.__dict__.get("_cursor_value", None)
            if _cursor_value is None:
                _cursor_value = self._cursor_value = mp.Value("i", 0)
            _cursor_value.value = value
        else:
            self._cursor_value = value

    @property
    def _write_count(self):
        _write_count = self.__dict__.get("_write_count_value", None)
        if not self._compilable:
            if _write_count is None:
                _write_count = self._write_count_value = mp.Value("i", 0)
            return _write_count.value
        else:
            if _write_count is None:
                _write_count = self._write_count_value = 0
            return _write_count

    @_write_count.setter
    def _write_count(self, value):
        if not self._compilable:
            _write_count = self.__dict__.get("_write_count_value", None)
            if _write_count is None:
                _write_count = self._write_count_value = mp.Value("i", 0)
            _write_count.value = value
        else:
            self._write_count_value = value

    def __getstate__(self):
        state = super().__getstate__()
        if get_spawning_popen() is None:
            cursor = self._cursor
            del state["_cursor_value"]
            state["cursor__context"] = cursor
        return state

    def __setstate__(self, state):
        cursor = state.pop("cursor__context", None)
        if cursor is not None:
            if not state["_compilable"]:
                _cursor_value = mp.Value("i", cursor)
            else:
                _cursor_value = cursor
            state["_cursor_value"] = _cursor_value
        self.__dict__.update(state)

    def __repr__(self):
        return f"{self.__class__.__name__}(cursor={int(self._cursor)}, full_storage={self._storage._is_full})"


class TensorDictRoundRobinWriter(RoundRobinWriter):
    """A RoundRobin Writer class for composable, tensordict-based replay buffers."""

    def add(self, data: Any) -> int | torch.Tensor:
        index = self._cursor
        # we need to update the cursor first to avoid race conditions between workers
        max_size_along_dim0 = self._storage._max_size_along_dim0(single_data=data)
        self._cursor = (index + 1) % max_size_along_dim0
        self._write_count += 1
        if not is_tensorclass(data):
            data.set(
                "index",
                expand_as_right(
                    torch.as_tensor(index, device=data.device, dtype=torch.long), data
                ),
            )
        self._storage.set(index, data)
        index = self._replicate_index(index)
        for ent in self._storage._attached_entities_iter():
            ent.mark_update(index)
        return index

    def extend(self, data: Sequence) -> torch.Tensor:
        cur_size = self._cursor
        batch_size = len(data)
        device = data.device if hasattr(data, "device") else None
        max_size_along_dim0 = self._storage._max_size_along_dim0(batched_data=data)
        index = (
            torch.arange(
                cur_size, batch_size + cur_size, dtype=torch.long, device=device
            )
            % max_size_along_dim0
        )
        # we need to update the cursor first to avoid race conditions between workers
        self._cursor = (batch_size + cur_size) % max_size_along_dim0
        self._write_count += batch_size
        # storage must convert the data to the appropriate format if needed
        if not is_tensorclass(data):
            data.set(
                "index",
                expand_as_right(
                    torch.as_tensor(index, device=data.device, dtype=torch.long), data
                ),
            )
        # Replicate index requires the shape of the storage to be known
        # Other than that, a "flat" (1d) index is ok to write the data
        self._storage.set(index, data)
        index = self._replicate_index(index)
        for ent in self._storage._attached_entities_iter():
            ent.mark_update(index)
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

    .. note::
        This class isn't compatible with storages with more than one dimension.
        This doesn't mean that storing trajectories is prohibited, but that
        the trajectories stored must be stored on a per-trajectory basis.
        Here are some examples of valid and invalid usages of the class.
        First, a flat buffer where we store individual transitions:

            >>> from torchrl.data import TensorStorage
            >>> # Simplest use case: data comes in 1d and is stored as such
            >>> data = TensorDict({
            ...     "obs": torch.zeros(10, 3),
            ...     "reward": torch.zeros(10, 1),
            ... }, batch_size=[10])
            >>> rb = TensorDictReplayBuffer(
            ...     storage=LazyTensorStorage(max_size=100),
            ...     writer=TensorDictMaxValueWriter(rank_key="reward")
            ... )
            >>> # We initialize the buffer: a total of 100 *transitions* can be stored
            >>> rb.extend(data)
            >>> # Samples 5 *transitions* at random
            >>> sample = rb.sample(5)
            >>> assert sample.shape == (5,)

        Second, a buffer where we store trajectories. The max signal is aggregated
        in each batch (e.g. the reward of each rollout is summed):

            >>> # One can also store batches of data, each batch being a sub-trajectory
            >>> env = ParallelEnv(2, lambda: GymEnv("Pendulum-v1"))
            >>> # Get a batch of [2, 10] -- format is [Batch, Time]
            >>> rollout = env.rollout(max_steps=10)
            >>> rb = TensorDictReplayBuffer(
            ...     storage=LazyTensorStorage(max_size=100),
            ...     writer=TensorDictMaxValueWriter(rank_key="reward")
            ... )
            >>> # We initialize the buffer: a total of 100 *trajectories* (!) can be stored
            >>> rb.extend(rollout)
            >>> # Sample 5 trajectories at random
            >>> sample = rb.sample(5)
            >>> assert sample.shape == (5, 10)

        If data come in batch but a flat buffer is needed, we can simply flatten
        the data before extending the buffer:

            >>> rb = TensorDictReplayBuffer(
            ...     storage=LazyTensorStorage(max_size=100),
            ...     writer=TensorDictMaxValueWriter(rank_key="reward")
            ... )
            >>> # We initialize the buffer: a total of 100 *transitions* can be stored
            >>> rb.extend(rollout.reshape(-1))
            >>> # Sample 5 trajectories at random
            >>> sample = rb.sample(5)
            >>> assert sample.shape == (5,)

        It is not possible to create a buffer that is extended along the time
        dimension, which is usually the recommended way of using buffers with
        batches of trajectories. Since trajectories are overlapping, it's hard
        if not impossible to aggregate the reward values and compare them.
        This constructor isn't valid (notice the ndim argument):

            >>> rb = TensorDictReplayBuffer(
            ...     storage=LazyTensorStorage(max_size=100, ndim=2),  # Breaks!
            ...     writer=TensorDictMaxValueWriter(rank_key="reward")
            ... )

    """

    def __init__(self, rank_key=None, reduction: str = "sum", **kwargs) -> None:
        super().__init__(**kwargs)
        self._cursor = 0
        self._current_top_values = []
        self._rank_key = rank_key
        self._reduction = reduction
        if self._rank_key is None:
            self._rank_key = ("next", "reward")

    def register_storage(self, storage: Storage) -> None:
        if storage.ndim > 1:
            raise ValueError(
                "TensorDictMaxValueWriter is not compatible with storages with more than one dimension. "
                "See the docstring constructor note about storing trajectories with TensorDictMaxValueWriter."
            )
        return super().register_storage(storage)

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
        rank_data = data.get(self._rank_key)

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

    @property
    def _write_count(self):
        _write_count = self.__dict__.get("_write_count_value", None)
        if _write_count is None:
            _write_count = self._write_count_value = mp.Value("i", 0)
        return _write_count.value

    @_write_count.setter
    def _write_count(self, value):
        _write_count = self.__dict__.get("_write_count_value", None)
        if _write_count is None:
            _write_count = self._write_count_value = mp.Value("i", 0)
        _write_count.value = value

    def add(self, data: Any) -> int | torch.Tensor:
        """Inserts a single element of data at an appropriate index, and returns that index.

        The ``rank_key`` in the data passed to this module should be structured as [].
        If it has more dimensions, it will be reduced to a single value using the ``reduction`` method.
        """
        index = self.get_insert_index(data)
        if index is not None:
            data.set("index", index)
            self._write_count += 1
            # Replicate index requires the shape of the storage to be known
            # Other than that, a "flat" (1d) index is ok to write the data
            self._storage.set(index, data)
            index = self._replicate_index(index)
            for ent in self._storage._attached_entities_iter():
                ent.mark_update(index)
        return index

    def extend(self, data: TensorDictBase) -> None:
        """Inserts a series of data points at appropriate indices.

        The ``rank_key`` in the data passed to this module should be structured as [B].
        If it has more dimensions, it will be reduced to a single value using the ``reduction`` method.
        """
        # a map of [idx_in_storage, idx_in_data]
        data_to_replace = {}
        for data_idx, sample in enumerate(data):
            storage_idx = self.get_insert_index(sample)
            if storage_idx is not None:
                self._write_count += 1
                data_to_replace[storage_idx] = data_idx

        # -1 will be interpreted as invalid by prioritized buffers
        # Replace the data in the storage all at once
        if len(data_to_replace) > 0:
            storage_idx, data_idx = zip(*data_to_replace.items())
            index = data.get("index", None)
            dtype = index.dtype if index is not None else torch.long
            device = index.device if index is not None else data.device
            out_index = torch.full(data.shape, -1, dtype=torch.long, device=device)
            data_idx = torch.as_tensor(data_idx, dtype=dtype, device=device)
            storage_idx = torch.as_tensor(storage_idx, dtype=dtype, device=device)
            out_index[data_idx] = storage_idx
            self._storage.set(storage_idx, data[data_idx])
        else:
            device = getattr(self._storage, "device", None)
            out_index = torch.full(data.shape, -1, dtype=torch.long, device=device)
        index = self._replicate_index(out_index)
        for ent in self._storage._attached_entities_iter():
            ent.mark_update(index)
        return index

    def _empty(self) -> None:
        self._cursor = 0
        self._current_top_values = []

    def __getstate__(self):
        if get_spawning_popen() is not None:
            raise RuntimeError(
                f"Writers of type {type(self)} cannot be shared between processes. "
                f"Please submit an issue at https://github.com/pytorch/rl if this feature is needed."
            )
        state = super().__getstate__()
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
        with open(path / "metadata.json") as file:
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

    def state_dict(self) -> dict[str, Any]:
        raise NotImplementedError

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        raise NotImplementedError

    def __repr__(self):
        return f"{self.__class__.__name__}(cursor={int(self._cursor)}, full_storage={self._storage._is_full}, rank_key={self._rank_key}, reduction={self._reduction})"


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
        self._rng_private = None
        self._writers = writers

    @property
    def _rng(self):
        return self._rng_private

    @_rng.setter
    def _rng(self, value):
        self._rng_private = value
        for writer in self._writers:
            writer._rng = value

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
                    "A floating point index was received when an integer dtype was expected."
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

    def state_dict(self) -> dict[str, Any]:
        raise NotImplementedError

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        raise NotImplementedError
