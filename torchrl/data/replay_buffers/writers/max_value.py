# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import heapq
import json
from multiprocessing.context import get_spawning_popen
from pathlib import Path
from typing import Any

import torch
from tensordict import is_tensor_collection, MemoryMappedTensor, TensorDictBase
from torch import multiprocessing as mp
from torchrl._utils import _STRDTYPE2DTYPE

try:
    from torch.compiler import disable as compile_disable
except ImportError:
    from torch._dynamo import disable as compile_disable

try:
    from torch.utils._pytree import tree_leaves
except ImportError:
    from torch.utils._pytree import tree_flatten

    def tree_leaves(data):  # noqa: D103
        tree_flat, _ = tree_flatten(data)
        return tree_flat


from torchrl.data.replay_buffers.storages import Storage
from torchrl.data.replay_buffers.utils import _reduce

# Generation buffers for storages up to this many slots are allocated in one
# shot, so their shape is stable and the ``torch.compile`` extend/sample path
# does not recompile. Larger (or effectively unbounded -- ``ListStorage`` with
# no ``max_size`` reports ``torch.iinfo(torch.int64).max``) capacities grow
# geometrically on demand instead of trying to allocate the whole thing.
_GENERATION_EAGER_ALLOC_LIMIT = 2**20
_GENERATION_MIN_ALLOC = 1024

# Attribute under which the per-slot generation buffer is stored *on the
# storage*. It belongs to the storage, not to the writer: two buffers sharing
# one storage overwrite each other's slots, so a per-writer counter would let
# buffer A's handles look live after buffer B overwrote the slot -- exactly the
# staleness the feature exists to detect.
_SLOT_GENERATIONS_ATTR = "_slot_generations"


from .base import Writer


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
        _write_count = getattr(self, "_write_count_value", None)
        if _write_count is None:
            _write_count = self._write_count_value = mp.Value("i", 0)
        return _write_count.value

    @_write_count.setter
    def _write_count(self, value):
        _write_count = getattr(self, "_write_count_value", None)
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
        self._mark_update_entities(index)
        return index

    # TODO: Workaround for PyTorch nightly regression where compiler can't handle
    # method calls on objects returned from _attached_entities_iter()
    @compile_disable()
    def _mark_update_entities(self, index: torch.Tensor) -> None:
        """Mark entities as updated with the given index."""
        for ent in self._storage._attached_entities_iter():
            ent.mark_update(index)

    def _empty(self, empty_write_count: bool = True) -> None:
        self._cursor = 0
        self._current_top_values = []
        if empty_write_count:
            self._write_count = 0

    def __getstate__(self):
        if get_spawning_popen() is not None:
            raise RuntimeError(
                f"Writers of type {type(self)} cannot be shared between processes. "
                f"Please submit an issue at https://github.com/pytorch/rl if this feature is needed."
            )
        state = super().__getstate__()
        # Handle the mp.Value object for pickling
        if "_write_count_value" in state:
            write_count = self._write_count
            del state["_write_count_value"]
            state["write_count__context"] = write_count
        return state

    def __setstate__(self, state):
        write_count = state.pop("write_count__context", None)
        if write_count is not None:
            state["_write_count_value"] = mp.Value("i", write_count)
        self.__dict__.update(state)

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
                    "write_count": self._write_count,
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
            write_count = metadata.get("write_count")
            if write_count is not None:
                self._write_count = write_count
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
