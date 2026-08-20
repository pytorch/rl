# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from copy import copy
from typing import Any

import torch


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


class Writer(ABC):
    """A ReplayBuffer base Writer class."""

    _storage: Storage
    _rng: torch.Generator | None = None

    def __init__(self, compilable: bool = False) -> None:
        self._storage = None
        self._compilable = compilable

    #: Whether this writer stamps storage slots with a reuse generation. Always
    #: ``False`` unless the writer both supports generation tracking and was
    #: constructed with it enabled (see
    #: :class:`~torchrl.data.RoundRobinWriter`).
    tracks_generations: bool = False

    def register_storage(self, storage: Storage) -> None:
        self._storage = storage

    def generations_of(self, index: int | torch.Tensor) -> torch.Tensor:
        """Returns the generation stamp for each physical slot in ``index``.

        A slot's stamp advances once per write it receives, so a single
        ``extend`` that wraps the storage advances a reused slot once per write.
        Comparing a stamp captured at sampling time against the current stamp
        tells you whether the slot still holds the data you sampled.

        Writers that do not track slot reuse -- and writers constructed with
        ``track_generations=False``, which is the default -- report ``-1``
        everywhere. Never-written slots also report ``-1``, so ``-1`` means
        "no usable stamp" rather than "generation zero".

        Args:
            index (int or torch.Tensor): dim-0 slot indices. A 1-D tensor is
                always read as a batch of slot indices; for a storage with
                ``ndim > 1``, pass a ``tuple`` of per-dimension indices (as
                :meth:`~torchrl.data.ReplayBuffer.extend` returns) to identify
                a single cell -- only its dim-0 component is used, since a
                generation stamps a whole dim-0 slot.

        Returns:
            torch.Tensor: ``int64`` stamps shaped like the dim-0 component of
            ``index``, on ``index``'s device.
        """
        index = torch.as_tensor(index)
        return torch.full(index.shape, -1, dtype=torch.int64, device=index.device)

    @abstractmethod
    def add(self, data: Any) -> int:
        """Inserts one piece of data at an appropriate index, and returns that index."""
        ...

    @abstractmethod
    def extend(self, data: Sequence) -> torch.Tensor:
        """Inserts a series of data points at appropriate indices, and returns a tensor containing the indices."""
        ...

    @abstractmethod
    def _empty(self, empty_write_count: bool = True) -> None:
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
                *(torch.arange(dim, device=device) for dim in self._storage.shape[1:]),
                indexing="ij",
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

    def _empty(self, empty_write_count: bool = True) -> None:
        raise RuntimeError(self.WRITING_ERR)

    def dumps(self, path):
        ...

    def loads(self, path):
        ...

    def state_dict(self) -> dict[str, Any]:
        return {}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        return
