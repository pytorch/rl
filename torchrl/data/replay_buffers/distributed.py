# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import Any


class _SamplingStrategy(ABC):
    @abstractmethod
    def sample(
        self,
        client: Any,
        global_batch_size: int,
        world_size: int,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        raise NotImplementedError


class _OwnerSerializedSampling(_SamplingStrategy):
    def sample(
        self,
        client: Any,
        global_batch_size: int,
        world_size: int,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        local_batch_size = global_batch_size // world_size
        return client._sample_data_parallel(local_batch_size, *args, **kwargs)


class DataParallelReplayBufferClient:
    """A rank-aware, lifecycle-free view of a replay-buffer service client.

    Sampling batch sizes are interpreted globally. Each call made by one rank
    requests ``batch_size // world_size`` items from the service owner. Because
    the owner serializes these requests, stateful sampler mutations, writes, and
    priority updates remain coordinated in one place. Independent rank calls are
    distribution-equivalent to a global draw, but their order is not guaranteed.

    Args:
        client: a replay-buffer service client. In Phase 1 this is a client
            returned by :meth:`~torchrl.data.RayReplayBuffer.client`.
        rank (int): the rank represented by this view.
        world_size (int): the number of data-parallel ranks.

    Example:
        >>> import torch
        >>> from torchrl.data import DataParallelReplayBufferClient, ReplayBuffer
        >>> replay_buffer = ReplayBuffer(batch_size=4)
        >>> _ = replay_buffer.extend(torch.arange(8))
        >>> client = DataParallelReplayBufferClient(
        ...     replay_buffer, rank=0, world_size=2
        ... )
        >>> client.sample().shape
        torch.Size([2])

    .. note::
        Shared iteration is intentionally unsupported. A finite sampler epoch
        cannot be coordinated safely through independent ``next`` calls.

    .. warning::
        Owner-side prefetching is unsupported. Construct the replay-buffer
        owner with ``prefetch=0``.
    """

    _LIFECYCLE_METHODS = frozenset({"client", "close", "shutdown", "start"})

    def __init__(
        self,
        client: Any,
        *,
        rank: int,
        world_size: int,
    ) -> None:
        if isinstance(rank, bool) or not isinstance(rank, int):
            raise TypeError(f"rank must be an integer, got {type(rank).__name__}.")
        if isinstance(world_size, bool) or not isinstance(world_size, int):
            raise TypeError(
                "world_size must be an integer, " f"got {type(world_size).__name__}."
            )
        if world_size <= 0:
            raise ValueError(f"world_size must be positive, got {world_size}.")
        if rank < 0 or rank >= world_size:
            raise ValueError(
                f"rank must satisfy 0 <= rank < world_size, got rank={rank} "
                f"and world_size={world_size}."
            )
        self._client = client
        self._rank = rank
        self._world_size = world_size
        # Treat the configured batch size as fixed for the lifetime of this view.
        # A local snapshot avoids an extra service round trip on every default
        # sample call.
        self._batch_size = client.batch_size
        self._sampling_strategy: _SamplingStrategy = _OwnerSerializedSampling()

    @property
    def rank(self) -> int:
        """The rank represented by this client view."""
        return self._rank

    @property
    def world_size(self) -> int:
        """The number of data-parallel ranks."""
        return self._world_size

    @property
    def batch_size(self) -> int | None:
        """The configured global batch size of the replay buffer."""
        return self._batch_size

    def sample(
        self,
        batch_size: int | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Sample this rank's share of a global batch.

        Args:
            batch_size (int, optional): global batch size. If omitted, the
                replay buffer's configured batch size is used.
            *args: additional positional arguments forwarded to ``sample``.
            **kwargs: keyword arguments forwarded to ``sample``, including
                ``return_info``.

        Returns:
            This rank's local sample, with leading size
            ``batch_size // world_size``.
        """
        if batch_size is None:
            batch_size = self.batch_size
        if batch_size is None:
            raise RuntimeError(
                "batch_size not specified. Configure a global batch size on the "
                "replay buffer or pass one to sample()."
            )
        if isinstance(batch_size, bool) or not isinstance(batch_size, int):
            raise TypeError(
                "batch_size must be an integer, " f"got {type(batch_size).__name__}."
            )
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}.")
        if batch_size % self.world_size:
            raise ValueError(
                "The global batch size must be divisible by world_size, "
                f"got batch_size={batch_size} and world_size={self.world_size}."
            )
        return self._sampling_strategy.sample(
            self._client,
            batch_size,
            self.world_size,
            *args,
            **kwargs,
        )

    def extend(self, *args: Any, **kwargs: Any) -> Any:
        """Forward an ``extend`` call unchanged to the service owner."""
        return self._client.extend(*args, **kwargs)

    def add(self, *args: Any, **kwargs: Any) -> Any:
        """Forward an ``add`` call unchanged to the service owner."""
        return self._client.add(*args, **kwargs)

    def update_priority(self, *args: Any, **kwargs: Any) -> Any:
        """Forward a priority update unchanged to the service owner."""
        return self._client.update_priority(*args, **kwargs)

    def __len__(self) -> int:
        """Return the global number of sampleable items at the owner."""
        return len(self._client)

    def next(self) -> Any:
        """Reject unsafe shared finite-sampler iteration."""
        raise RuntimeError(
            "DataParallelReplayBufferClient does not support shared next() or "
            "iteration. Call sample() explicitly on every rank."
        )

    def __iter__(self) -> Iterator[Any]:
        """Reject unsafe shared finite-sampler iteration."""
        raise RuntimeError(
            "DataParallelReplayBufferClient does not support shared next() or "
            "iteration. Call sample() explicitly on every rank."
        )

    def __getitem__(self, index: Any) -> Any:
        return self._client[index]

    def __setitem__(self, index: Any, value: Any) -> None:
        self._client[index] = value

    def __getattr__(self, name: str) -> Any:
        if name in self._LIFECYCLE_METHODS:
            raise AttributeError(
                f"{type(self).__name__} has no lifecycle capability {name!r}."
            )
        if name.startswith("_"):
            raise AttributeError(f"{type(self).__name__} has no attribute {name!r}.")
        return getattr(self._client, name)
