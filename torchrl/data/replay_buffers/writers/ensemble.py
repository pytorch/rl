# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import textwrap
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tensordict import TensorDictBase

from torchrl.data.replay_buffers.storages import StorageEnsemble


try:
    from torch.utils._pytree import tree_leaves
except ImportError:
    from torch.utils._pytree import tree_flatten

    def tree_leaves(data):  # noqa: D103
        tree_flat, _ = tree_flatten(data)
        return tree_flat


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


class WriterEnsemble(Writer):
    """An ensemble of writers.

    This class is designed to work with :class:`~torchrl.data.replay_buffers.replay_buffers.ReplayBufferEnsemble`.
    It contains the member writers used by a replay-buffer ensemble.

    Args:
        writers (sequence of Writer): the writers to make the composite writer.

    .. warning::
       This class does not write directly. A routed
       :class:`~torchrl.data.ReplayBufferEnsemble` dispatches writes to its
       member writers; otherwise, index the parent ensemble before writing.

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

    @property
    def tracks_generations(self) -> bool:
        return all(writer.tracks_generations for writer in self._writers)

    @property
    def _write_count(self) -> int:
        return sum(getattr(writer, "_write_count", 0) for writer in self._writers)

    def register_storage(self, storage: StorageEnsemble) -> None:
        if not isinstance(storage, StorageEnsemble):
            raise TypeError("WriterEnsemble requires a StorageEnsemble.")
        if len(storage._storages) != len(self._writers):
            raise ValueError(
                "WriterEnsemble and StorageEnsemble must have the same number "
                "of members."
            )
        self._storage = storage
        for writer, member_storage in zip(self._writers, storage._storages):
            writer.register_storage(member_storage)

    def generations_of(self, index: TensorDictBase) -> torch.Tensor:
        if not isinstance(index, TensorDictBase):
            raise TypeError("WriterEnsemble generations require routed index metadata.")
        buffer_ids = index.get("buffer_ids")
        local_indices = index.get("index")
        if buffer_ids.ndim != 1 or local_indices.shape[0] != buffer_ids.shape[0]:
            raise ValueError(
                "WriterEnsemble expects one member id per leading local-index row."
            )
        if buffer_ids.dtype == torch.bool or buffer_ids.is_floating_point():
            raise TypeError("Replay-buffer member ids must be integers.")
        if buffer_ids.numel() and (
            (buffer_ids < 0).any() or (buffer_ids >= len(self._writers)).any()
        ):
            raise ValueError(
                f"Replay-buffer member ids must lie in [0, {len(self._writers) - 1}]."
            )
        generations = [None] * buffer_ids.numel()
        for member_id in buffer_ids.unique(sorted=True).tolist():
            positions = (buffer_ids == member_id).nonzero().flatten()
            member_generations = self._writers[member_id].generations_of(
                local_indices[positions.to(local_indices.device)]
            )
            if member_generations.shape[0] != positions.numel():
                raise RuntimeError(
                    f"Writer {member_id} returned incompatible generation stamps."
                )
            for position, generation in zip(
                positions.tolist(), member_generations.unbind(0)
            ):
                generations[position] = generation
        if not generations:
            return torch.empty(
                buffer_ids.shape, dtype=torch.int64, device=buffer_ids.device
            )
        if all(
            generation.shape == generations[0].shape for generation in generations[1:]
        ):
            return torch.stack(generations)
        return torch.nested.nested_tensor(generations)

    def _empty(self, empty_write_count: bool = True) -> None:
        for writer in self._writers:
            writer._empty(empty_write_count=empty_write_count)

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
        return OrderedDict(
            (str(index), writer.state_dict())
            for index, writer in enumerate(self._writers)
        )

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        for index, writer in enumerate(self._writers):
            writer.load_state_dict(state_dict[str(index)])
