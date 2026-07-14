# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import torch
from tensordict import TensorDictBase

from torchrl.data.replay_buffers.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.storages import LazyTensorStorage, Storage


class OfflineToOnlineReplayBuffer:
    """A replay buffer combining an immutable offline dataset with a growing online buffer.

    :meth:`extend` routes new experience to the online buffer only; the offline
    dataset is never modified.  :meth:`sample` draws **exactly**
    ``round(offline_fraction * batch_size)`` transitions from the offline
    dataset and the remainder from the online buffer, concatenated into a flat
    ``[batch_size]`` TensorDict.

    The split is deterministic per batch (not merely correct in expectation),
    so ``offline_fraction`` is honored on every single :meth:`sample` call.

    When the online buffer is empty (i.e. before any :meth:`extend` call), or
    once ``offline_fraction`` has been annealed to 0, :meth:`sample` draws from
    a single buffer only.

    .. note:: Offline and online data must share a compatible key structure so
        the two sampled batches can be concatenated.  This is automatic when
        both come from the same environment (TED format).

    Args:
        offline_dataset (str or ReplayBuffer): an offline dataset object (e.g.
            :class:`~torchrl.data.datasets.MinariExperienceReplay`) or a
            prefixed ID string such as ``"minari:mujoco/hopper/expert-v0"`` or
            ``"d4rl:halfcheetah-medium-v2"`` resolved via
            :func:`~torchrl.data.datasets.load_dataset`.

    Keyword Args:
        online_storage (Storage, optional): storage backend for the online
            buffer.  Mutually exclusive with ``online_capacity``.
        online_capacity (int, optional): shorthand that creates a
            :class:`~torchrl.data.LazyTensorStorage` of this size.
            Mutually exclusive with ``online_storage``.
        offline_fraction (float, optional): fraction of each batch drawn from
            the offline dataset.  Must be in ``(0, 1)``.  Default: ``0.5``.
        batch_size (int, optional): default batch size for :meth:`sample`. Required
            when ``offline_dataset`` is a string, and forwarded to the dataset
            constructor.
        transform (Callable, optional): applied to the concatenated sample
            batch on the read side.
        **dataset_kwargs: forwarded to the dataset constructor when
            ``offline_dataset`` is a string.

    Examples:
        >>> import torch
        >>> from tensordict import TensorDict
        >>> from torchrl.data import (
        ...     OfflineToOnlineReplayBuffer, ReplayBuffer, LazyTensorStorage)
        >>> offline = ReplayBuffer(storage=LazyTensorStorage(1000))
        >>> _ = offline.extend(TensorDict({"observation": torch.randn(1000, 4)}, [1000]))
        >>> rb = OfflineToOnlineReplayBuffer(
        ...     offline_dataset=offline,
        ...     online_capacity=500,
        ...     offline_fraction=0.5,
        ...     batch_size=32,
        ... )
        >>> _ = rb.extend(TensorDict({"observation": torch.randn(10, 4)}, [10]))
        >>> rb.sample(32).batch_size
        torch.Size([32])
    """

    def __init__(
        self,
        offline_dataset,
        *,
        online_storage: Storage | None = None,
        online_capacity: int | None = None,
        offline_fraction: float = 0.5,
        batch_size: int | None = None,
        transform=None,
        **dataset_kwargs,
    ):
        if online_storage is not None and online_capacity is not None:
            raise ValueError("Provide online_storage OR online_capacity, not both.")
        if online_storage is None and online_capacity is None:
            raise ValueError("Provide one of online_storage or online_capacity.")
        if not (0.0 < offline_fraction < 1.0):
            raise ValueError(
                f"offline_fraction must be in (0, 1), got {offline_fraction}."
            )

        # Resolve offline dataset from string if needed
        if isinstance(offline_dataset, str):
            from torchrl.data.datasets.utils import load_dataset

            if "batch_size" not in dataset_kwargs:
                if batch_size is None:
                    raise ValueError(
                        "batch_size must be provided when offline_dataset is a "
                        "string, so the dataset can be constructed."
                    )
                dataset_kwargs["batch_size"] = batch_size
            offline_dataset = load_dataset(offline_dataset, **dataset_kwargs)
        elif dataset_kwargs:
            raise ValueError(
                "dataset_kwargs are only forwarded when offline_dataset is a "
                "string. Pass them directly to your dataset constructor instead."
            )

        # Build online buffer
        if online_capacity is not None:
            online_storage = LazyTensorStorage(online_capacity)
        online_rb = ReplayBuffer(storage=online_storage)

        self._offline_buffer = offline_dataset
        self._online_buffer = online_rb
        # Current fraction may be lowered by anneal(); base fraction is the
        # value we anneal away from.
        self._offline_fraction = offline_fraction
        self._base_offline_fraction = offline_fraction
        self._batch_size = batch_size
        self._transform = transform

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def extend(self, data) -> torch.Tensor:
        """Add new online experience to the online buffer.

        Args:
            data: a TensorDict (or compatible sequence) to add.

        Returns:
            Indices at which the data was stored in the online buffer.
        """
        return self._online_buffer.extend(data)

    def sample(self, batch_size: int | None = None) -> TensorDictBase:
        """Sample a flat ``[batch_size]`` batch split between the two buffers.

        Draws ``round(offline_fraction * batch_size)`` from the offline dataset
        and the rest from the online buffer.  Falls back to a single buffer
        when the online buffer is empty or the offline split rounds to 0.

        Args:
            batch_size (int, optional): number of samples to draw.  Falls back
                to the ``batch_size`` set in ``__init__``.

        Returns:
            TensorDictBase with batch size ``[batch_size]``.
        """
        if batch_size is None:
            batch_size = self._batch_size
        if batch_size is None:
            raise ValueError(
                "batch_size must be provided either in __init__ or sample()."
            )

        n_offline = round(self._offline_fraction * batch_size)
        n_online = batch_size - n_offline

        online_empty = len(self._online_buffer) == 0

        if online_empty or n_offline >= batch_size:
            out = self._offline_buffer.sample(batch_size)
        elif n_offline == 0:
            out = self._online_buffer.sample(batch_size)
        else:
            offline_batch = self._offline_buffer.sample(n_offline)
            online_batch = self._online_buffer.sample(n_online)
            out = torch.cat([offline_batch, online_batch], dim=0)

        if self._transform is not None:
            out = self._transform(out)
        return out

    def anneal(self, step: int, total_steps: int) -> None:
        """Linearly decay ``offline_fraction`` toward 0 over ``total_steps``.

        Call once per training iteration to gradually shift the sampling
        distribution from offline-dominant to purely online.  Clamps at 0 for
        ``step >= total_steps``.

        Args:
            step (int): current training step (0-indexed).
            total_steps (int): step at which ``offline_fraction`` reaches 0.
        """
        self._offline_fraction = self._base_offline_fraction * max(
            0.0, 1.0 - step / total_steps
        )

    @property
    def offline_fraction(self) -> float:
        """The current offline sampling fraction (after any annealing)."""
        return self._offline_fraction

    @property
    def offline_buffer(self):
        """The immutable offline dataset."""
        return self._offline_buffer

    @property
    def online_buffer(self) -> ReplayBuffer:
        """The mutable online replay buffer."""
        return self._online_buffer

    def __len__(self) -> int:
        return len(self._offline_buffer) + len(self._online_buffer)

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}("
            f"offline={type(self._offline_buffer).__name__}, "
            f"offline_fraction={self._offline_fraction:.3f}, "
            f"online_size={len(self._online_buffer)}, "
            f"batch_size={self._batch_size})"
        )


def prefill_replay_buffer(
    rb: ReplayBuffer,
    dataset: str | ReplayBuffer,
    n_samples: int | None = None,
    chunk_size: int = 1000,
) -> ReplayBuffer:
    """Copy samples from an offline dataset into a mutable replay buffer.

    A simpler alternative to :class:`OfflineToOnlineReplayBuffer` for users
    who want a single flat buffer (no per-batch sampling ratio, slightly higher
    memory usage since offline data is copied).

    Args:
        rb (ReplayBuffer): a mutable replay buffer to seed.
        dataset (str or ReplayBuffer): offline dataset or a prefixed ID string
            (``"minari:..."`` / ``"d4rl:..."``).
        n_samples (int, optional): maximum number of samples to copy.
            Defaults to the full dataset.
        chunk_size (int, optional): number of samples copied per iteration.
            When ``dataset`` is a string, this is also used as the dataset
            constructor batch size. Default: ``1000``.

    Returns:
        ReplayBuffer: ``rb`` mutated in-place (also returned for chaining).

    Examples:
        >>> import torch
        >>> from tensordict import TensorDict
        >>> from torchrl.data import ReplayBuffer, LazyTensorStorage
        >>> from torchrl.data.replay_buffers.offline_to_online import (
        ...     prefill_replay_buffer)
        >>> dataset = ReplayBuffer(storage=LazyTensorStorage(500))
        >>> _ = dataset.extend(TensorDict({"obs": torch.randn(500, 4)}, [500]))
        >>> online_rb = ReplayBuffer(storage=LazyTensorStorage(10_000))
        >>> _ = prefill_replay_buffer(online_rb, dataset, n_samples=200)
        >>> len(online_rb)
        200
    """
    if isinstance(dataset, str):
        from torchrl.data.datasets.utils import load_dataset

        dataset = load_dataset(dataset, batch_size=chunk_size)

    total = min(n_samples, len(dataset)) if n_samples is not None else len(dataset)
    copied = 0

    while copied < total:
        this_chunk = min(chunk_size, total - copied)
        data = dataset.sample(this_chunk)
        rb.extend(data)
        copied += this_chunk

    return rb
