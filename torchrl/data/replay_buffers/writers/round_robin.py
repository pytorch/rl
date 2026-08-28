# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import json
from collections.abc import Sequence
from multiprocessing.context import get_spawning_popen
from pathlib import Path
from typing import Any

import torch
from tensordict import is_tensor_collection, MemoryMappedTensor
from tensordict.utils import expand_as_right, is_tensorclass
from torch import multiprocessing as mp
from torchrl._utils import _make_ordinal_device, _STRDTYPE2DTYPE

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
from torchrl.data.replay_buffers.utils import _is_int

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


class RoundRobinWriter(Writer):
    """A RoundRobin Writer class for composable replay buffers.

    See also :class:`~torchrl.trainers.algorithms.configs.RoundRobinWriterConfig`.

    Args:
        compilable (bool, optional): whether the writer is compilable.
            If ``True``, the writer cannot be shared between multiple processes.
            Defaults to ``False``.

    Keyword Args:
        track_generations (bool, optional): if ``True``, stamp every storage
            slot with a counter that advances each time the slot is written, so
            a consumer holding an index can tell whether the slot still holds
            the data it sampled. Reads are exposed through
            :meth:`generations_of`, and :meth:`~torchrl.data.ReplayBuffer.sample`
            adds an ``"index_generation"`` entry to its ``info`` (and, for
            tensordict buffers, to the sample). Defaults to ``False``: enabling
            it allocates one ``int64`` slot per storage slot and adds a key to
            the sampler output, so it is opt-in.

    .. note::
        The generation buffer lives on the *storage*, not on the writer, so two
        buffers sharing one storage observe each other's writes. It is
        process-local: a slot overwritten in another process is not reflected
        here. See :ref:`ref_buffers_generations`.

    Examples:
        >>> import torch
        >>> from torchrl.data import LazyTensorStorage, ReplayBuffer, RoundRobinWriter
        >>> rb = ReplayBuffer(
        ...     storage=LazyTensorStorage(4),
        ...     writer=RoundRobinWriter(track_generations=True),
        ... )
        >>> index = rb.extend(torch.arange(4))
        >>> rb.writer.generations_of(index)
        tensor([0, 0, 0, 0])
        >>> _ = rb.extend(torch.arange(4, 6))  # overwrites slots 0 and 1
        >>> rb.writer.generations_of(index)
        tensor([1, 1, 0, 0])
    """

    def __init__(
        self, compilable: bool = False, *, track_generations: bool = False
    ) -> None:
        super().__init__(compilable=compilable)
        self._cursor = 0
        self._write_count  # noqa
        self._track_generations = track_generations
        # Holds the buffer until a storage is registered (dumps/loads and
        # load_state_dict can both run on a storage-less writer).
        self._pending_generation = None

    @property
    def tracks_generations(self) -> bool:
        return self._track_generations

    @property
    def _generation(self) -> torch.Tensor | None:
        if self._storage is None:
            return self._pending_generation
        return getattr(self._storage, _SLOT_GENERATIONS_ATTR, None)

    @_generation.setter
    def _generation(self, value: torch.Tensor | None) -> None:
        if self._storage is None:
            self._pending_generation = value
        else:
            setattr(self._storage, _SLOT_GENERATIONS_ATTR, value)

    def register_storage(self, storage: Storage) -> None:
        super().register_storage(storage)
        pending, self._pending_generation = self._pending_generation, None
        # A buffer restored from a checkpoint carries its stamps in the writer;
        # a storage already shared with another buffer carries the live ones and
        # wins, so the two writers cannot disagree about a slot's generation.
        if pending is not None and self._generation is None:
            self._generation = pending
        self._align_generation_device()

    def _generation_device(self, index: int | torch.Tensor) -> torch.device:
        # The generation buffer follows the storage: sampled indices are built on
        # the storage device, so lookups stay sync-free on the sampling path.
        device = getattr(self._storage, "device", None)
        if device is None or device == "auto":
            if isinstance(index, torch.Tensor):
                return _make_ordinal_device(index.device)
            return torch.device("cpu")
        return _make_ordinal_device(torch.device(device))

    def _align_generation_device(self) -> None:
        generation = self._generation
        if generation is None:
            return
        device = self._generation_device(generation)
        if generation.device != device:
            self._generation = generation.to(device)

    def _ensure_generation(
        self, capacity: int, min_size: int, device: torch.device
    ) -> None:
        generation = self._generation
        if generation is not None and generation.device != device:
            generation = generation.to(device)
            self._generation = generation
        current = 0 if generation is None else generation.numel()
        if current >= min_size:
            return
        if capacity <= _GENERATION_EAGER_ALLOC_LIMIT:
            # One allocation covering every slot: the shape never changes again.
            size = capacity
        else:
            # Too large (or unbounded) to allocate up front -- grow geometrically
            # and stay within the storage's capacity.
            size = min(capacity, max(min_size, 2 * current, _GENERATION_MIN_ALLOC))
        new_generation = torch.full((size,), -1, dtype=torch.int64, device=device)
        if generation is not None:
            new_generation[:current] = generation
        # Deliberately not shared across processes: the buffer is replaced (not
        # mutated) whenever it grows, so a shared mapping would silently stop
        # tracking after the first growth. Cross-process staleness detection
        # needs a storage-owned, fixed-size mapping -- see the docs.
        self._generation = new_generation

    def _bump_generation(self, index: int | torch.Tensor, data: Any) -> None:
        # A writer that did not opt into generation tracking must still update a
        # buffer installed on a shared storage by another writer. It does not
        # allocate the buffer itself or expose generations in its samples.
        if not self._track_generations and self._generation is None:
            return
        device = self._generation_device(index)
        if _is_int(index):
            capacity = self._storage._max_size_along_dim0(single_data=data)
            self._ensure_generation(capacity, int(index) + 1, device)
            self._generation[int(index)] += 1
        else:
            index = torch.as_tensor(index, dtype=torch.long).reshape(-1)
            if index.numel() == 0:
                return
            capacity = self._storage._max_size_along_dim0(batched_data=data)
            if capacity <= _GENERATION_EAGER_ALLOC_LIMIT:
                min_size = capacity
            else:
                # Only reached for capacities we cannot allocate up front, so the
                # device sync from ``.max()`` is not on the common extend path.
                min_size = int(index.max()) + 1
            self._ensure_generation(capacity, min_size, device)
            index = index.to(device)
            self._generation.index_put_(
                (index,), torch.ones_like(index), accumulate=True
            )

    def generations_of(self, index: int | torch.Tensor) -> torch.Tensor:
        if not self._track_generations:
            return super().generations_of(index)
        if isinstance(index, tuple):
            index = index[0]
        elif (
            isinstance(index, torch.Tensor)
            # Only a batch of coordinate vectors, i.e. ndim >= 2, can be
            # unambiguously distinguished from a batch of dim-0 indices: a 1-D
            # tensor of length storage.ndim is far more likely to be several
            # slot indices than one coordinate. Pass a tuple for the latter.
            and index.ndim >= 2
            and self._storage is not None
            and self._storage.ndim > 1
            and index.shape[-1] == self._storage.ndim
        ):
            index = index[..., 0]
        index = torch.as_tensor(index, dtype=torch.long)
        if self._generation is None:
            return torch.full(index.shape, -1, dtype=torch.int64, device=index.device)
        idx = index.to(self._generation.device)
        n = self._generation.numel()
        if not n:
            return torch.full(index.shape, -1, dtype=torch.int64, device=index.device)
        gen = self._generation[idx.clamp(min=0, max=n - 1)]
        valid = (idx >= 0) & (idx < n)
        gen = torch.where(valid, gen, torch.full_like(gen, -1))
        return gen.to(index.device)

    def dumps(self, path):
        path = Path(path).absolute()
        path.mkdir(exist_ok=True)
        metadata = {
            "cursor": self._cursor,
            "write_count": self._write_count,
        }
        generation = self._generation if self._track_generations else None
        if generation is not None:
            generation = generation.cpu()
            try:
                MemoryMappedTensor.from_filename(
                    filename=path / "generation.memmap",
                    shape=generation.shape,
                    dtype=generation.dtype,
                ).copy_(generation)
            except FileNotFoundError:
                MemoryMappedTensor.from_tensor(
                    generation, filename=path / "generation.memmap"
                )
            metadata["generation_shape"] = list(generation.shape)
            metadata["generation_dtype"] = str(generation.dtype)
        with open(path / "metadata.json", "w") as file:
            json.dump(metadata, file)

    def loads(self, path):
        path = Path(path).absolute()
        with open(path / "metadata.json") as file:
            metadata = json.load(file)
            self._cursor = metadata["cursor"]
            write_count = metadata.get("write_count")
            if write_count is not None:
                self._write_count = write_count
            generation_shape = metadata.get("generation_shape")
        if generation_shape is not None:
            generation = MemoryMappedTensor.from_filename(
                filename=path / "generation.memmap",
                dtype=_STRDTYPE2DTYPE[metadata["generation_dtype"]],
                shape=torch.Size(generation_shape),
            ).clone()
            self._generation = generation
            self._align_generation_device()

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
        self._bump_generation(_cursor, data)
        index = self._replicate_index(index)
        self._mark_update_entities(index)
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
            raise RuntimeError(f"Expected at least one element in extend. Got {data=}")
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
        self._bump_generation(index, data)
        index = self._replicate_index(index)
        self._mark_update_entities(index)
        return index

    def write_at(self, index: int | torch.Tensor, data: Any) -> int | torch.Tensor:
        """Writes data at explicit storage indices without moving the cursor.

        The generation of every written slot is bumped, so handles previously
        handed out for those slots are stale once this returns.
        """
        if _is_int(index):
            batch_size = 1
        else:
            index = torch.as_tensor(index, dtype=torch.long)
            if hasattr(data, "device") and data.device is not None:
                index = index.to(data.device)
            batch_size = index.numel()
        self._write_count += batch_size
        self._storage.set(index, data, set_cursor=False)
        self._bump_generation(index, data)
        self._update_storage_len_for_write_at(index)
        index = self._replicate_index(index)
        self._mark_update_entities(index)
        return index

    def _update_storage_len_for_write_at(self, index: int | torch.Tensor) -> None:
        if not hasattr(self._storage, "_len"):
            return
        if _is_int(index):
            max_index = int(index)
        else:
            index = torch.as_tensor(index)
            if index.numel() == 0:
                return
            max_index = int(index.max().item())
        self._storage._len = min(
            max(len(self._storage), max_index + 1), self._storage.max_size
        )

    def state_dict(self) -> dict[str, Any]:
        state_dict = {"_cursor": self._cursor, "_write_count": self._write_count}
        if self._track_generations and self._generation is not None:
            state_dict["_generation"] = self._generation.clone()
        return state_dict

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self._cursor = state_dict["_cursor"]
        write_count = state_dict.get("_write_count")
        if write_count is not None:
            self._write_count = write_count
        generation = state_dict.get("_generation")
        if generation is not None:
            self._generation = generation.clone()
            self._align_generation_device()

    def _empty(self, empty_write_count: bool = True) -> None:
        self._cursor = 0
        # Emptying through any writer invalidates handles held by tracking
        # writers that share this storage.
        generation = self._generation
        if generation is not None:
            # Emptying invalidates every handle, so stamps advance rather than
            # reset -- a reset would make pre-empty handles look live again.
            # Never-written slots keep the -1 sentinel.
            generation[generation >= 0] += 1
        if empty_write_count:
            self._write_count = 0

    # TODO: Workaround for PyTorch nightly regression where compiler can't handle
    # method calls on objects returned from _attached_entities_iter()
    @compile_disable()
    def _mark_update_entities(self, index: torch.Tensor) -> None:
        """Mark entities as updated with the given index."""
        for ent in self._storage._attached_entities_iter():
            ent.mark_update(index)

    @property
    def _cursor(self):
        _cursor_value = getattr(self, "_cursor_value", None)
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
            _cursor_value = getattr(self, "_cursor_value", None)
            if _cursor_value is None:
                _cursor_value = self._cursor_value = mp.Value("i", 0)
            _cursor_value.value = value
        else:
            self._cursor_value = value

    @property
    def _write_count(self):
        _write_count = getattr(self, "_write_count_value", None)
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
            _write_count = getattr(self, "_write_count_value", None)
            if _write_count is None:
                _write_count = self._write_count_value = mp.Value("i", 0)
            _write_count.value = value
        else:
            self._write_count_value = value

    def __getstate__(self):
        state = super().__getstate__()
        if get_spawning_popen() is None:
            cursor = self._cursor
            write_count = self._write_count
            del state["_cursor_value"]
            del state["_write_count_value"]
            state["cursor__context"] = cursor
            state["write_count__context"] = write_count
        return state

    def __setstate__(self, state):
        cursor = state.pop("cursor__context", None)
        write_count = state.pop("write_count__context", None)
        if cursor is not None:
            if not state["_compilable"]:
                _cursor_value = mp.Value("i", cursor)
            else:
                _cursor_value = cursor
            state["_cursor_value"] = _cursor_value
        if write_count is not None:
            if not state["_compilable"]:
                _write_count_value = mp.Value("i", write_count)
            else:
                _write_count_value = write_count
            state["_write_count_value"] = _write_count_value
        self.__dict__.update(state)

    def __repr__(self):
        return f"{self.__class__.__name__}(cursor={int(self._cursor)}, full_storage={self._storage._is_full})"


class TensorDictRoundRobinWriter(RoundRobinWriter):
    """A RoundRobin Writer class for composable, tensordict-based replay buffers.

    See Also:
    :class:`~torchrl.trainers.algorithms.configs.TensorDictRoundRobinWriterConfig`.

    Takes the same arguments as :class:`RoundRobinWriter`, including
    ``track_generations``. When enabled, ``"index_generation"`` is written into
    the sampled tensordict alongside ``"index"``.
    """

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
        self._bump_generation(index, data)
        index = self._replicate_index(index)
        self._mark_update_entities(index)
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
        self._bump_generation(index, data)
        index = self._replicate_index(index)
        self._mark_update_entities(index)
        return index

    def write_at(self, index: int | torch.Tensor, data: Any) -> int | torch.Tensor:
        if _is_int(index):
            batch_size = 1
            index_tensor = torch.as_tensor(index, device=data.device, dtype=torch.long)
        else:
            index_tensor = torch.as_tensor(index, device=data.device, dtype=torch.long)
            batch_size = index_tensor.numel()
        self._write_count += batch_size
        if not is_tensorclass(data):
            data.set("index", expand_as_right(index_tensor, data))
        self._storage.set(index_tensor, data, set_cursor=False)
        self._bump_generation(index_tensor, data)
        self._update_storage_len_for_write_at(index_tensor)
        index = self._replicate_index(index_tensor)
        self._mark_update_entities(index)
        return index
