# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from numbers import Real
from typing import Any

import torch
from torchrl.data.replay_buffers.utils import INT_CLASSES

from .base import ReplayBuffer


class BlockingReplayBuffer(ReplayBuffer):
    """A bounded consuming replay buffer whose producers wait for free slots.

    ``BlockingReplayBuffer`` is a specialized
    :class:`~torchrl.data.ReplayBuffer`: sampled items are consumed through the
    existing ``consume_after_n_samples`` mechanism, and a producer blocks
    instead of overwriting an item that is still sampleable. An ``extend`` is
    admitted as a whole, so a timeout or cancellation never applies a partial
    batch.

    The buffer uses its storage capacity as the bound. It deliberately does
    not expose drop, raise, or configurable-watermark policies; ordinary
    :class:`~torchrl.data.ReplayBuffer` retains circular overwrite behavior.

    Keyword Args:
        consume_after_n_samples (int, optional): number of returned samples
            after which an item frees its slot. Defaults to ``1``.
        **kwargs: forwarded to :class:`~torchrl.data.ReplayBuffer`. A custom
            sampler, replay transform, compilable writer, prefetching,
            multidimensional storage, and remote service backend are not
            supported.

    Examples:
        >>> import torch
        >>> from torchrl.data import BlockingReplayBuffer, LazyTensorStorage
        >>> replay = BlockingReplayBuffer(
        ...     storage=LazyTensorStorage(2), batch_size=1
        ... )
        >>> _ = replay.extend(torch.arange(2))
        >>> _ = replay.sample()
        >>> replay.add(torch.tensor(2), timeout=1.0) in (0, 1)
        True

    .. note::
        For a shared buffer, pass the collector's shutdown event as
        ``cancel_event`` to :meth:`add` or :meth:`extend` so a blocked worker
        can exit when collection stops.
    """

    _accepts_transport_backend = False

    def __init__(self, *, consume_after_n_samples: int = 1, **kwargs: Any):
        if kwargs.get("sampler") is not None:
            raise ValueError(
                "BlockingReplayBuffer uses ConsumingSampler through "
                "consume_after_n_samples and does not accept a custom sampler."
            )
        if kwargs.get("compilable"):
            raise ValueError("BlockingReplayBuffer does not support compilable=True.")
        if (
            kwargs.get("transform") is not None
            or kwargs.get("transform_factory") is not None
        ):
            raise ValueError("BlockingReplayBuffer does not support replay transforms.")
        super().__init__(
            consume_after_n_samples=consume_after_n_samples,
            **kwargs,
        )

    @classmethod
    def _ServiceClass(cls, service_backend, *args, **kwargs):
        del args, kwargs
        raise ValueError(
            f"{cls.__name__} does not support service_backend={service_backend!r}."
        )

    def _init(self) -> None:
        was_initialized = self.initialized
        super()._init()
        if was_initialized:
            return
        capacity = getattr(self._storage, "max_size", None)
        if not isinstance(capacity, INT_CLASSES) or capacity < 1:
            raise ValueError(
                "BlockingReplayBuffer requires storage with a finite capacity."
            )
        self._blocking_capacity = int(capacity)
        if self.shared:
            self._sampler._share_memory(self._storage)

    def share(self, shared: bool = True) -> BlockingReplayBuffer:
        super().share(shared)
        if shared and getattr(self, "_initialized", False):
            self._sampler._share_memory(self._storage)
        return self

    @staticmethod
    def _deadline(timeout: float | None) -> float | None:
        if timeout is None:
            return None
        if isinstance(timeout, bool) or not isinstance(timeout, Real):
            raise TypeError("timeout must be a non-negative number or None.")
        if timeout < 0:
            raise ValueError("timeout must be non-negative.")
        return time.monotonic() + float(timeout)

    @contextmanager
    def _admit(
        self,
        num_items: int,
        *,
        timeout: float | None,
        cancel_event: Any | None,
    ) -> Iterator[None]:
        if isinstance(num_items, bool) or not isinstance(num_items, INT_CLASSES):
            raise TypeError("num_items must be a positive integer.")
        num_items = int(num_items)
        if num_items < 1:
            raise ValueError("num_items must be a positive integer.")
        if num_items > self._blocking_capacity:
            raise ValueError(
                f"A write of {num_items} items exceeds replay capacity "
                f"{self._blocking_capacity}."
            )

        deadline = self._deadline(timeout)
        condition = self._readiness_condition
        with condition:
            while True:
                if self._service_shutdown:
                    raise RuntimeError("A shut down replay buffer cannot be written.")
                if cancel_event is not None and cancel_event.is_set():
                    raise RuntimeError("Replay-buffer writing was cancelled.")
                with self._replay_lock, self._write_lock:
                    occupancy = self._sampler._num_sampleable(self._storage)
                if occupancy + num_items <= self._blocking_capacity:
                    # Keep the condition locked through the write so another
                    # producer cannot claim the same capacity.
                    yield
                    return
                wait_time = None
                if deadline is not None:
                    wait_time = deadline - time.monotonic()
                    if wait_time <= 0:
                        raise TimeoutError(
                            "Replay buffer did not obtain write capacity within "
                            f"{timeout} seconds."
                        )
                if cancel_event is not None:
                    wait_time = 0.1 if wait_time is None else min(wait_time, 0.1)
                condition.wait(wait_time)

    def add(
        self,
        data: Any,
        *,
        timeout: float | None = None,
        cancel_event: Any | None = None,
    ) -> Any:
        """Add one item once capacity is available.

        Args:
            data (Any): item to add.

        Keyword Args:
            timeout (float, optional): maximum seconds to wait. ``None`` waits
                indefinitely. Defaults to ``None``.
            cancel_event (optional): event-like object exposing ``is_set()``.
                Setting it cancels the blocked write. Defaults to ``None``.

        Returns:
            The storage index of the added item.
        """
        with self._admit(1, timeout=timeout, cancel_event=cancel_event):
            return super().add(data)

    def extend(
        self,
        data: Sequence,
        *,
        update_priority: bool | None = None,
        timeout: float | None = None,
        cancel_event: Any | None = None,
    ) -> torch.Tensor:
        """Add a complete batch once capacity is available.

        Args:
            data (Sequence): batch to add.

        Keyword Args:
            update_priority (bool, optional): unsupported compatibility
                argument inherited from :class:`ReplayBuffer`. Defaults to
                ``None``.
            timeout (float, optional): maximum seconds to wait. ``None`` waits
                indefinitely. Defaults to ``None``.
            cancel_event (optional): event-like object exposing ``is_set()``.
                Setting it cancels the blocked write. Defaults to ``None``.

        Returns:
            The storage indices of the added items.
        """
        batch_size = self._get_batch_size(data)
        with self._admit(
            batch_size,
            timeout=timeout,
            cancel_event=cancel_event,
        ):
            return super().extend(data, update_priority=update_priority)

    def sample(
        self,
        batch_size: int | None = None,
        return_info: bool = False,
        *,
        wait: bool = False,
        timeout: float | None = None,
        cancel_event: Any | None = None,
    ) -> Any:
        """Sample items and release any capacity they consume.

        Arguments match :meth:`~torchrl.data.ReplayBuffer.sample`. Waiting
        producers are notified after a successful sample.
        """
        try:
            return super().sample(
                batch_size,
                return_info=return_info,
                wait=wait,
                timeout=timeout,
                cancel_event=cancel_event,
            )
        finally:
            self._notify_replay_state_change()
