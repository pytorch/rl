# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import textwrap
from typing import Any

import numpy as np
import torch

from torchrl.data.replay_buffers.checkpointers import StorageEnsembleCheckpointer

from .base import Storage


class StorageEnsemble(Storage):
    """An ensemble of storages.

    This class is designed to work with :class:`~torchrl.data.replay_buffers.replay_buffers.ReplayBufferEnsemble`.

    Args:
        storages (sequence of Storage): the storages to make the composite storage.

    Keyword Args:
        transforms (list of :class:`~torchrl.envs.Transform`, optional): a list of
            transforms of the same length as storages.

    .. warning::
      This class signatures for :meth:`get` does not match other storages, as
      it will return a tuple ``(buffer_id, samples)`` rather than just the samples.

    .. warning::
       This class does not support writing (similarly to :class:`~torchrl.data.replay_buffers.writers.WriterEnsemble`).
       To extend one of the replay buffers, simply index the parent
       :class:`~torchrl.data.ReplayBufferEnsemble` object.

    """

    _default_checkpointer = StorageEnsembleCheckpointer

    def __init__(
        self,
        *storages: Storage,
        transforms: list[Transform] = None,  # noqa: F821
    ):
        self._rng_private = None
        self._storages = storages
        self._transforms = transforms
        if transforms is not None and len(transforms) != len(storages):
            raise TypeError(
                "transforms must have the same length as the storages provided."
            )

    @property
    def _rng(self):
        return self._rng_private

    @_rng.setter
    def _rng(self, value):
        self._rng_private = value
        for storage in self._storages:
            storage._rng = value

    def extend(self, value):
        raise RuntimeError

    def add(self, value):
        raise RuntimeError

    def get(self, item):
        # we return the buffer id too to be able to track the appropriate collate_fn
        buffer_ids = item.get("buffer_ids")
        index = item.get("index")
        results = []
        for buffer_id, sample in zip(buffer_ids, index):
            buffer_id = self._convert_id(buffer_id)
            results.append((buffer_id, self._get_storage(buffer_id).get(sample)))
        if self._transforms is not None:
            results = [
                (buffer_id, self._transforms[buffer_id](result))
                if self._transforms[buffer_id] is not None
                else (buffer_id, result)
                for buffer_id, result in results
            ]
        return results

    def _convert_id(self, sub):
        if isinstance(sub, torch.Tensor):
            sub = sub.item()
        return sub

    def _get_storage(self, sub):
        return self._storages[sub]

    def state_dict(self) -> dict[str, Any]:
        raise NotImplementedError

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        raise NotImplementedError

    _INDEX_ERROR = "Expected an index of type torch.Tensor, range, np.ndarray, int, slice or ellipsis, got {} instead."

    def __getitem__(self, index):
        if isinstance(index, tuple):
            if index[0] is Ellipsis:
                index = (slice(None), index[1:])
            result = self[index[0]]
            if len(index) > 1:
                if result is self:
                    # then index[0] is an ellipsis/slice(None)
                    sample = [storage[index[1:]] for storage in self._storages]
                    return sample
                if isinstance(result, StorageEnsemble):
                    new_index = (slice(None), *index[1:])
                    return result[new_index]
                return result[index[1:]]
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
                return self._storages[index]
            except IndexError:
                raise IndexError(self._INDEX_ERROR.format(type(index)))
        if isinstance(index, torch.Tensor):
            index = index.tolist()
            storages = [self._storages[i] for i in index]
            transforms = (
                [self._transforms[i] for i in index]
                if self._transforms is not None
                else [None] * len(index)
            )
        else:
            # slice
            storages = self._storages[index]
            transforms = (
                self._transforms[index]
                if self._transforms is not None
                else [None] * len(storages)
            )

        return StorageEnsemble(*storages, transforms=transforms)

    def __len__(self):
        return len(self._storages)

    def __repr__(self):
        storages = textwrap.indent(f"storages={self._storages}", " " * 4)
        transforms = textwrap.indent(f"transforms={self._transforms}", " " * 4)
        return f"StorageEnsemble(\n{storages}, \n{transforms})"
