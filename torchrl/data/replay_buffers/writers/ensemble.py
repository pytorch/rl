# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import textwrap
from pathlib import Path
from typing import Any

import numpy as np
import torch


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

    def _empty(self, empty_write_count: bool = True) -> None:
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
