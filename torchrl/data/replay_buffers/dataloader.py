# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

from collections.abc import Mapping
from itertools import islice
from typing import Any, TYPE_CHECKING

import torch
from tensordict import is_tensor_collection, LazyStackedTensorDict
from torch.utils.data import get_worker_info, IterableDataset
from torch.utils.data._utils.collate import default_collate

from torchrl.data.replay_buffers.samplers import Sampler

if TYPE_CHECKING:
    from torch.utils.data._utils.worker import WorkerInfo

    from torchrl.data.replay_buffers.replay_buffers import ReplayBuffer

__all__ = ["ReplayBufferDataset", "tensordict_collate"]


def tensordict_collate(batch: Any) -> Any:
    """Collate function for a :class:`torch.utils.data.DataLoader` reading TorchRL storages or buffers.

    A batch that is already a tensor, a tensor collection or a tuple of them,
    as fetched from a storage or yielded by :class:`ReplayBufferDataset`, is
    returned unchanged. A list of samples, as produced by per-item storages or
    by composing storages with :class:`torch.utils.data.ConcatDataset`, is
    stacked: tensor collections lazily when their shapes differ, tensors
    densely, and mappings or tuples element-wise. The default torch collation
    iterates a tensordict over its batch dimension and cannot be used.

    Args:
        batch (Tensor, TensorDictBase or list): the fetched batch.

    Examples:
        >>> import torch
        >>> from tensordict import TensorDict
        >>> from torch.utils.data import DataLoader
        >>> from torchrl.data import LazyTensorStorage, ReplayBuffer, tensordict_collate
        >>> rb = ReplayBuffer(storage=LazyTensorStorage(100))
        >>> _ = rb.extend(TensorDict({"obs": torch.arange(100)}, [100]))
        >>> loader = DataLoader(
        ...     rb.storage, batch_size=4, shuffle=True, collate_fn=tensordict_collate
        ... )
        >>> next(iter(loader))["obs"].shape
        torch.Size([4])
    """
    if not isinstance(batch, list) or not batch:
        return batch
    first = batch[0]
    if is_tensor_collection(first):
        return LazyStackedTensorDict.maybe_dense_stack(list(batch))
    if isinstance(first, torch.Tensor):
        return torch.stack(list(batch))
    if isinstance(first, Mapping):
        return {key: tensordict_collate([item[key] for item in batch]) for key in first}
    if isinstance(first, tuple):
        return tuple(tensordict_collate(list(items)) for items in zip(*batch))
    return default_collate(list(batch))


class ReplayBufferDataset(IterableDataset):
    """A :class:`torch.utils.data.IterableDataset` streaming batches from a replay buffer.

    Iterating the dataset iterates the buffer: every item is one batch of
    ``replay_buffer.batch_size`` elements with the buffer transforms applied.
    Pass the dataset to a :class:`torch.utils.data.DataLoader` with
    ``batch_size=None`` and :func:`tensordict_collate` to sample in worker
    processes. Each worker holds its own copy of the buffer, so the sampler and
    the transforms run in the worker and ``num_batches`` is split between
    workers. Buffer prefetching is disabled in workers and prefetched batches
    are never serialized to them, the DataLoader prefetches instead. A buffer
    built with a :class:`torch.Generator` is
    reseeded once per worker from the worker seed, so sampling in workers is
    reproducible when the DataLoader is seeded (``torch.manual_seed`` or
    ``DataLoader(generator=...)``). Samplers whose
    :attr:`~torchrl.data.replay_buffers.Sampler.requires_shared_state` is
    ``True`` (without replacement, prioritized, consuming, staleness-aware,
    streaming and prompt-group samplers) are rejected when workers are used.

    Args:
        replay_buffer (ReplayBuffer): the buffer to sample from. Its
            ``batch_size`` must be set.

    Keyword Args:
        num_batches (int or None, optional): number of batches yielded by one
            iterator, shared between DataLoader workers. ``None`` streams
            batches until the sampler runs out, which samplers with replacement
            never do. Defaults to ``None``.

    Examples:
        >>> import torch
        >>> from tensordict import TensorDict
        >>> from torch.utils.data import DataLoader
        >>> from torchrl.data import (
        ...     LazyMemmapStorage,
        ...     SliceSampler,
        ...     TensorDictReplayBuffer,
        ...     tensordict_collate,
        ... )
        >>> rb = TensorDictReplayBuffer(
        ...     storage=LazyMemmapStorage(1000),
        ...     sampler=SliceSampler(num_slices=4, traj_key="episode", cache_values=True),
        ...     batch_size=32,
        ... )
        >>> _ = rb.extend(
        ...     TensorDict(
        ...         {"obs": torch.randn(1000, 3), "episode": torch.arange(1000) // 50},
        ...         [1000],
        ...     )
        ... )
        >>> loader = DataLoader(
        ...     rb.as_dataset(num_batches=8),
        ...     batch_size=None,
        ...     num_workers=4,
        ...     persistent_workers=True,
        ...     collate_fn=tensordict_collate,
        ... )
        >>> batches = list(loader)
        >>> len(batches), batches[0]["obs"].shape
        (8, torch.Size([32, 3]))
    """

    def __init__(
        self, replay_buffer: ReplayBuffer, *, num_batches: int | None = None
    ) -> None:
        self.replay_buffer = replay_buffer
        self.num_batches = num_batches
        self._worker_seed = None

    def __getstate__(self) -> dict[str, Any]:
        replay_buffer = self.replay_buffer
        self._check_sampler(replay_buffer.sampler)
        return {
            **self.__dict__,
            "replay_buffer": (
                type(replay_buffer),
                replay_buffer._state_without_prefetch(),
            ),
        }

    def __setstate__(self, state: dict[str, Any]) -> None:
        buffer_cls, buffer_state = state.pop("replay_buffer")
        replay_buffer = buffer_cls.__new__(buffer_cls)
        replay_buffer.__setstate__(buffer_state)
        self.__dict__.update(state, replay_buffer=replay_buffer)

    def _check_sampler(self, sampler: Sampler) -> None:
        if sampler.requires_shared_state:
            raise RuntimeError(
                f"{type(sampler).__name__} keeps sampling state that DataLoader "
                "workers cannot share. Use num_workers=0 or a sampler without "
                "cross-process state."
            )

    def _setup_worker(self) -> WorkerInfo | None:
        worker = get_worker_info()
        if worker is None:
            return None
        replay_buffer = self.replay_buffer
        self._check_sampler(replay_buffer.sampler)
        replay_buffer._reset_worker_state()
        rng = replay_buffer._rng
        if rng is not None and self._worker_seed != worker.seed:
            replay_buffer.set_rng(
                torch.Generator(device=rng.device).manual_seed(worker.seed)
            )
            self._worker_seed = worker.seed
        return worker

    def __iter__(self):
        worker = self._setup_worker()
        num_batches = self.num_batches
        if worker is not None and num_batches is not None:
            num_batches = len(range(worker.id, num_batches, worker.num_workers))
        yield from islice(self.replay_buffer, num_batches)

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(replay_buffer={self.replay_buffer!r}, "
            f"num_batches={self.num_batches})"
        )
