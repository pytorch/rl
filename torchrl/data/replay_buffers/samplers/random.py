# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tensordict import TensorDict
from torchrl.data.replay_buffers.storages import Storage
from torchrl.data.replay_buffers.utils import _is_int, unravel_index

_EMPTY_STORAGE_ERROR = "Cannot sample from an empty storage."


from .base import Sampler


class RandomSampler(Sampler):
    """A uniformly random sampler for composable replay buffers.

    Keyword Args:
        replacement (bool, optional): if ``False``, the call is dispatched to
            :class:`SamplerWithoutReplacement`, and any additional keyword
            arguments (e.g. ``drop_last``, ``shuffle``) are forwarded to its
            constructor. Defaults to ``True``.

    Examples:
        >>> from torchrl.data import RandomSampler, SamplerWithoutReplacement
        >>> isinstance(RandomSampler(), RandomSampler)
        True
        >>> isinstance(RandomSampler(replacement=False), SamplerWithoutReplacement)
        True
        >>> isinstance(
        ...     RandomSampler(replacement=False, drop_last=True),
        ...     SamplerWithoutReplacement,
        ... )
        True

    """

    def sample(self, storage: Storage, batch_size: int) -> tuple[torch.Tensor, dict]:
        if len(storage) == 0:
            raise RuntimeError(_EMPTY_STORAGE_ERROR)
        index = storage._rand_given_ndim(batch_size)
        return index, {}

    def _empty(self):
        pass

    def dumps(self, path):
        # no op
        ...

    def loads(self, path):
        # no op
        ...

    def state_dict(self) -> dict[str, Any]:
        return {}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        return


class ConsumingSampler(Sampler):
    """A random sampler that consumes entries after they have been sampled.

    ``ConsumingSampler`` tracks how many times each storage index has been
    returned by :meth:`sample`. Once an index has been returned
    ``max_sample_count`` times, it is removed from the set of sampleable
    indices until that slot is overwritten by the replay-buffer writer. When
    used through :class:`~torchrl.data.ReplayBuffer`, consumed indices are kept
    in a free-list so writes can reuse those slots before advancing the normal
    writer cursor.

    Args:
        max_sample_count (int, optional): number of returned samples after which
            an item is consumed. Defaults to ``1``.

    Examples:
        >>> import torch
        >>>
        >>> from torchrl.data import ConsumingSampler, ListStorage, ReplayBuffer
        >>> rb = ReplayBuffer(
        ...     storage=ListStorage(10),
        ...     sampler=ConsumingSampler(),
        ...     batch_size=3,
        ... )
        >>> rb.extend([torch.tensor(i) for i in range(4)])
        tensor([0, 1, 2, 3])
        >>> sample = rb.sample()
        >>> len(sample)
        3
        >>> len(rb)
        1

    .. note::
        ``ConsumingSampler`` only supports 1-dimensional storages and uniform
        random sampling without replacement within each sampled batch.
        Prefetching and prioritized replay are not supported.
    """

    def __init__(self, max_sample_count: int = 1):
        if isinstance(max_sample_count, bool) or not isinstance(
            max_sample_count, (int, np.integer)
        ):
            raise TypeError("max_sample_count must be a positive integer.")
        if max_sample_count < 1:
            raise ValueError("max_sample_count must be a positive integer.")
        self.max_sample_count = int(max_sample_count)
        self._sample_count = None
        self._live_mask = None
        self._known_storage_len = 0
        self._free_indices = None
        self._free_head = 0
        self._ran_out = False
        self._remaining_batches = 0

    @staticmethod
    def _storage_device(storage: Storage | None):
        if storage is None:
            return None
        device = getattr(storage, "device", None)
        if device == "auto":
            return None
        return device

    def _ensure_state(
        self,
        storage: Storage | None = None,
        *,
        min_capacity: int = 0,
    ) -> None:
        if storage is not None:
            if storage.ndim != 1:
                raise ValueError(
                    f"{type(self).__name__} only supports 1-dimensional storages, "
                    f"got storage.ndim={storage.ndim}."
                )
            storage_len = len(storage)
            min_capacity = max(min_capacity, storage_len)
            device = self._storage_device(storage)
        else:
            storage_len = self._known_storage_len
            device = (
                self._sample_count.device if self._sample_count is not None else None
            )

        current_capacity = (
            0 if self._sample_count is None else self._sample_count.numel()
        )
        capacity = max(current_capacity, int(min_capacity))

        needs_init = self._sample_count is None or self._live_mask is None
        needs_resize = current_capacity < capacity
        needs_device = (
            not needs_init
            and device is not None
            and self._sample_count.device != torch.device(device)
        )

        if needs_init or needs_resize or needs_device:
            old_sample_count = self._sample_count
            old_live_mask = self._live_mask
            old_free_indices = self._free_indices
            self._sample_count = torch.zeros(capacity, dtype=torch.long, device=device)
            self._live_mask = torch.zeros(capacity, dtype=torch.bool, device=device)
            if old_sample_count is not None and old_live_mask is not None:
                copy_len = min(old_sample_count.numel(), capacity)
                self._sample_count[:copy_len] = old_sample_count[:copy_len].to(
                    self._sample_count.device
                )
                self._live_mask[:copy_len] = old_live_mask[:copy_len].to(
                    self._live_mask.device
                )
            if old_free_indices is not None:
                self._free_indices = old_free_indices.to(self._sample_count.device)

        if storage is not None:
            if storage_len > self._known_storage_len:
                self._sample_count[self._known_storage_len : storage_len] = 0
                self._live_mask[self._known_storage_len : storage_len] = True
            elif storage_len < self._known_storage_len:
                self._sample_count[storage_len : self._known_storage_len] = 0
                self._live_mask[storage_len : self._known_storage_len] = False
            self._known_storage_len = storage_len

        if self._live_mask is not None and self._sample_count is not None:
            self._live_mask &= self._sample_count < self.max_sample_count

    def _compact_free_indices(self) -> None:
        if self._free_indices is None:
            self._free_head = 0
            return
        if self._free_head:
            self._free_indices = self._free_indices[self._free_head :]
            self._free_head = 0
        if not self._free_indices.numel():
            self._free_indices = None

    def _append_free_indices(self, index: torch.Tensor) -> None:
        if index.numel() == 0:
            return
        index = index.to(self._sample_count.device, dtype=torch.long).reshape(-1)
        self._compact_free_indices()
        if self._free_indices is None:
            self._free_indices = index.clone()
        else:
            self._free_indices = torch.cat([self._free_indices, index])

    def _rebuild_free_indices(self) -> None:
        if self._live_mask is None or self._sample_count is None:
            self._free_indices = None
            self._free_head = 0
            return
        consumed_mask = ~self._live_mask[: self._known_storage_len] & (
            self._sample_count[: self._known_storage_len] >= self.max_sample_count
        )
        self._free_indices = torch.nonzero(consumed_mask, as_tuple=False).flatten()
        self._free_head = 0
        if not self._free_indices.numel():
            self._free_indices = None

    def _index_to_tensor(self, index: int | torch.Tensor | tuple) -> torch.Tensor:
        if isinstance(index, tuple):
            raise ValueError(
                f"{type(self).__name__} only supports flat 1-dimensional indices."
            )
        if _is_int(index):
            min_capacity = int(index) + 1
        else:
            index = torch.as_tensor(index, dtype=torch.long)
            min_capacity = int(index.max().item()) + 1 if index.numel() else 0
        self._ensure_state(min_capacity=min_capacity)
        if _is_int(index):
            return torch.as_tensor(
                [int(index)], dtype=torch.long, device=self._sample_count.device
            )
        return torch.as_tensor(
            index, dtype=torch.long, device=self._sample_count.device
        ).reshape(-1)

    def _mark_indices_live(self, index: int | torch.Tensor | tuple) -> None:
        index = self._index_to_tensor(index)
        if index.numel() == 0:
            return
        self._sample_count[index] = 0
        self._live_mask[index] = True
        self._known_storage_len = max(
            self._known_storage_len, int(index.max().item()) + 1
        )
        self._ran_out = False

    def add(self, index: int) -> None:
        self._mark_indices_live(index)

    def extend(self, index: torch.Tensor) -> None:
        self._mark_indices_live(index)

    def mark_update(
        self, index: int | torch.Tensor | tuple, *, storage: Storage | None = None
    ) -> None:
        if storage is not None:
            min_capacity = len(storage)
            if not _is_int(index) and not isinstance(index, tuple):
                index_tensor = torch.as_tensor(index)
                if index_tensor.numel():
                    min_capacity = max(
                        min_capacity, int(index_tensor.reshape(-1).max().item()) + 1
                    )
            elif _is_int(index):
                min_capacity = max(min_capacity, int(index) + 1)
            self._ensure_state(storage, min_capacity=min_capacity)
        self._mark_indices_live(index)

    def _num_sampleable(self, storage: Storage | None = None) -> int:
        self._ensure_state(storage)
        if self._live_mask is None:
            return 0
        return int(self._live_mask[: self._known_storage_len].sum().item())

    def _pop_consumed_indices(
        self, storage: Storage | None = None, max_count: int | None = None
    ) -> torch.Tensor:
        self._ensure_state(storage)
        if (
            self._live_mask is None
            or self._sample_count is None
            or self._free_indices is None
        ):
            return torch.zeros(0, dtype=torch.long)

        popped = []
        while self._free_head < self._free_indices.numel():
            if max_count is not None and len(popped) >= max_count:
                break
            candidate = self._free_indices[self._free_head]
            self._free_head += 1
            if candidate >= self._known_storage_len:
                continue
            if self._live_mask[candidate]:
                continue
            if self._sample_count[candidate] < self.max_sample_count:
                continue
            popped.append(candidate)

        if self._free_head >= self._free_indices.numel():
            self._free_indices = None
            self._free_head = 0

        if not popped:
            device = self._sample_count.device
            return torch.zeros(0, dtype=torch.long, device=device)
        return torch.stack(popped)

    def _update_remaining_batches(self, batch_size: int, storage: Storage) -> None:
        num_sampleable = self._num_sampleable(storage)
        self._remaining_batches = -(num_sampleable // -batch_size)
        self._ran_out = num_sampleable == 0

    def sample(self, storage: Storage, batch_size: int) -> tuple[torch.Tensor, dict]:
        self._ensure_state(storage)
        if len(storage) == 0 or self._num_sampleable(storage) == 0:
            self._ran_out = True
            self._remaining_batches = 0
            raise RuntimeError(_EMPTY_STORAGE_ERROR)

        live_indices = torch.nonzero(
            self._live_mask[: self._known_storage_len], as_tuple=False
        ).flatten()
        sample_size = min(batch_size, live_indices.numel())
        permutation = torch.randperm(
            live_indices.numel(),
            generator=self._rng,
            device=live_indices.device,
        )[:sample_size]
        index = live_indices[permutation]

        self._sample_count[index] += 1
        self._live_mask[index] = self._sample_count[index] < self.max_sample_count
        newly_consumed = index[~self._live_mask[index]]
        self._append_free_indices(newly_consumed)
        self._update_remaining_batches(batch_size, storage)
        return index, {}

    @property
    def ran_out(self):
        return self._ran_out

    @ran_out.setter
    def ran_out(self, value):
        self._ran_out = value

    def _empty(self):
        self._sample_count = None
        self._live_mask = None
        self._known_storage_len = 0
        self._free_indices = None
        self._free_head = 0
        self._ran_out = False
        self._remaining_batches = 0

    def dumps(self, path):
        path = Path(path)
        path.mkdir(exist_ok=True)
        torch.save(self.state_dict(), path / "consuming_sampler.pt")

    def loads(self, path):
        self.load_state_dict(torch.load(Path(path) / "consuming_sampler.pt"))

    def state_dict(self) -> dict[str, Any]:
        return OrderedDict(
            max_sample_count=self.max_sample_count,
            _sample_count=self._sample_count,
            _live_mask=self._live_mask,
            _known_storage_len=self._known_storage_len,
            _free_indices=(
                None
                if self._free_indices is None
                else self._free_indices[self._free_head :]
            ),
            _free_head=0,
            _ran_out=self._ran_out,
        )

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        # clone incoming tensors to decouple the sampler state from the
        # caller's objects (which may e.g. be mmap-backed by a checkpoint file)
        def _clone(value):
            return value.clone() if isinstance(value, torch.Tensor) else value

        self.max_sample_count = int(state_dict["max_sample_count"])
        self._sample_count = _clone(state_dict["_sample_count"])
        self._live_mask = _clone(state_dict["_live_mask"])
        self._known_storage_len = int(state_dict["_known_storage_len"])
        self._free_indices = _clone(state_dict.get("_free_indices"))
        self._free_head = int(state_dict.get("_free_head", 0))
        self._ran_out = bool(state_dict["_ran_out"])
        if "_free_indices" not in state_dict:
            self._rebuild_free_indices()
        if self._live_mask is not None:
            num_sampleable = int(
                self._live_mask[: self._known_storage_len].sum().item()
            )
            self._remaining_batches = num_sampleable
        else:
            self._remaining_batches = 0

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"max_sample_count={self.max_sample_count}, "
            f"sampleable={self._num_sampleable()})"
        )


class SamplerWithoutReplacement(Sampler):
    """A data-consuming sampler that ensures that the same sample is not present in consecutive batches.

    Args:
        drop_last (bool, optional): if ``True``, the last incomplete sample (if any) will be dropped.
            If ``False``, this last sample will be kept and (unlike with torch dataloaders)
            completed with other samples from a fresh indices permutation.
            Defaults to ``False``.
        shuffle (bool, optional): if ``False``, the items are not randomly
            permuted. This enables to iterate over the replay buffer in the
            order the data was collected. Defaults to ``True``.

    *Caution*: If the size of the storage changes in between two calls, the samples will be re-shuffled
    (as we can't generally keep track of which samples have been sampled before and which haven't).

    Similarly, it is expected that the storage content remains the same in between two calls,
    but this is not enforced.

    When the sampler reaches the end of the list of available indices, a new sample order
    will be generated and the resulting indices will be completed with this new draw, which
    can lead to duplicated indices, unless the :obj:`drop_last` argument is set to ``True``.

    """

    def __init__(self, drop_last: bool = False, shuffle: bool = True):
        self._sample_list = None
        self.len_storage = 0
        self.drop_last = drop_last
        self._ran_out = False
        self.shuffle = shuffle

    def dumps(self, path):
        path = Path(path)
        path.mkdir(exist_ok=True)

        TensorDict(self.state_dict()).memmap(path)

    def loads(self, path):
        sd = TensorDict.load_memmap(path).to_dict()
        self.load_state_dict(sd)

    def _get_sample_list(self, storage: Storage, len_storage: int, batch_size: int):
        if storage is None:
            device = self._sample_list.device
        else:
            device = storage.device if hasattr(storage, "device") else None

        if self.shuffle:
            _sample_list = torch.randperm(
                len_storage, device=device, generator=self._rng
            )
        else:
            _sample_list = torch.arange(len_storage, device=device)
        self._sample_list = _sample_list
        if self.drop_last:
            self._remaining_batches = self._sample_list.numel() // batch_size
        else:
            self._remaining_batches = -(self._sample_list.numel() // -batch_size)

    def _single_sample(self, len_storage, batch_size):
        index = self._sample_list[:batch_size]
        self._sample_list = self._sample_list[batch_size:]
        if self.drop_last:
            self._remaining_batches = self._sample_list.numel() // batch_size
        else:
            self._remaining_batches = -(self._sample_list.numel() // -batch_size)

        # check if we have enough elements for one more batch, assuming same batch size
        # will be used each time sample is called
        if self._sample_list.shape[0] == 0 or (
            self.drop_last and len(self._sample_list) < batch_size
        ):
            self.ran_out = True
            self._get_sample_list(
                storage=None, len_storage=len_storage, batch_size=batch_size
            )
        else:
            self.ran_out = False
        return index

    def _storage_len(self, storage):
        return len(storage)

    def sample(
        self, storage: Storage, batch_size: int
    ) -> tuple[Any, dict]:  # noqa: F811
        len_storage = self._storage_len(storage)
        if len_storage == 0:
            raise RuntimeError(_EMPTY_STORAGE_ERROR)
        if not len_storage:
            raise RuntimeError("An empty storage was passed")
        if self.len_storage != len_storage or self._sample_list is None:
            self._get_sample_list(storage, len_storage, batch_size=batch_size)
        if len_storage < batch_size and self.drop_last:
            raise ValueError(
                f"The batch size ({batch_size}) is greater than the storage capacity ({len_storage}). "
                "This makes it impossible to return a sample without repeating indices. "
                "Consider changing the sampler class or turn the 'drop_last' argument to False."
            )
        self.len_storage = len_storage
        index = self._single_sample(len_storage, batch_size)
        if storage.ndim > 1:
            index = unravel_index(index, storage.shape)
        # we 'always' return the indices. The 'drop_last' just instructs the
        # sampler to turn to `ran_out = True` whenever the next sample
        # will be too short. This will be read by the replay buffer
        # as a signal for an early break of the __iter__().
        return index, {}

    @property
    def ran_out(self):
        return self._ran_out

    @ran_out.setter
    def ran_out(self, value):
        self._ran_out = value

    def _empty(self):
        self._sample_list = None
        self.len_storage = 0
        self._ran_out = False

    def state_dict(self) -> dict[str, Any]:
        return OrderedDict(
            len_storage=self.len_storage,
            _sample_list=self._sample_list,
            drop_last=self.drop_last,
            _ran_out=self._ran_out,
        )

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.len_storage = state_dict["len_storage"]
        # clone to decouple the sampler state from the caller's tensor
        _sample_list = state_dict["_sample_list"]
        self._sample_list = (
            _sample_list.clone()
            if isinstance(_sample_list, torch.Tensor)
            else _sample_list
        )
        self.drop_last = state_dict["drop_last"]
        self._ran_out = state_dict["_ran_out"]

    def __repr__(self):
        if self._sample_list is not None:
            perc = len(self._sample_list) / self.len_storage * 100
        else:
            perc = 0.0
        return f"{self.__class__.__name__}({perc: 4.4f}% sampled)"


def _default_staleness_weight(s: torch.Tensor) -> torch.Tensor:
    """Default freshness weighting: 1 / (staleness + 1)."""
    return 1.0 / (s.float() + 1.0)
