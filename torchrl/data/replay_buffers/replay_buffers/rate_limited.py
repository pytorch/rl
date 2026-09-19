# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import math
import multiprocessing
import time
from numbers import Real
from typing import Any

from torchrl.data.replay_buffers.utils import INT_CLASSES

from .base import ReplayBuffer


class RateLimitedReplayBuffer(ReplayBuffer):
    """A replay buffer with a cumulative sample-to-insert ratio limit.

    ``RateLimitedReplayBuffer`` permits at most ``samples_per_insert`` sampled
    records for every record written through :meth:`add` or :meth:`extend`.
    The limit uses the replay buffer's cumulative counters rather than its
    current occupancy, so it remains meaningful after circular storage wraps.
    Concurrent callers reserve sample budget atomically.

    This class deliberately controls only replay sampling. Policy-version
    freshness is handled by samplers such as
    :class:`~torchrl.data.StalenessAwareSampler`, while weight publication is
    owned by collectors and trainer weight-update hooks.

    Keyword Args:
        samples_per_insert (float): maximum cumulative number of sampled
            records per inserted record. Must be finite and positive.
        **kwargs: forwarded to :class:`~torchrl.data.ReplayBuffer`. Prefetching
            is not supported because prefetched samples would consume ratio
            budget before the corresponding call reserves it.

    Examples:
        >>> import torch
        >>> from torchrl.data import LazyTensorStorage, RateLimitedReplayBuffer
        >>> replay = RateLimitedReplayBuffer(
        ...     storage=LazyTensorStorage(8),
        ...     batch_size=2,
        ...     samples_per_insert=1.0,
        ... )
        >>> _ = replay.extend(torch.arange(2))
        >>> replay.sample(wait=True, timeout=1.0).shape
        torch.Size([2])
        >>> replay.can_sample()
        False

    .. note::
        Checkpointing keeps the replay counters and ratio state together in
        the normal replay-buffer state dict.
    """

    _accepts_transport_backend = False

    def __init__(self, *, samples_per_insert: float, **kwargs: Any):
        self.samples_per_insert = self._validate_samples_per_insert(samples_per_insert)
        if kwargs.get("prefetch"):
            raise ValueError("RateLimitedReplayBuffer does not support prefetching.")
        self._reserved_sample_count_value = 0
        self._sample_wait_count_value = 0
        super().__init__(**kwargs)

    @classmethod
    def _ServiceClass(cls, service_backend, *args, **kwargs):
        del args, kwargs
        raise ValueError(
            f"{cls.__name__} does not support service_backend={service_backend!r}."
        )

    @staticmethod
    def _validate_samples_per_insert(samples_per_insert: float) -> float:
        if isinstance(samples_per_insert, bool) or not isinstance(
            samples_per_insert, Real
        ):
            raise TypeError("samples_per_insert must be a finite positive number.")
        samples_per_insert = float(samples_per_insert)
        if not math.isfinite(samples_per_insert) or samples_per_insert <= 0:
            raise ValueError("samples_per_insert must be a finite positive number.")
        return samples_per_insert

    def share(self, shared: bool = True) -> RateLimitedReplayBuffer:
        reserved_samples = self._counter_value(
            getattr(self, "_reserved_sample_count_value", 0)
        )
        sample_waits = self._counter_value(getattr(self, "_sample_wait_count_value", 0))
        super().share(shared)
        if shared:
            self._reserved_sample_count_value = multiprocessing.Value(
                "q", reserved_samples
            )
            self._sample_wait_count_value = multiprocessing.Value("q", sample_waits)
        else:
            self._reserved_sample_count_value = reserved_samples
            self._sample_wait_count_value = sample_waits
        return self

    def _resolve_ratio_batch_size(self, batch_size: int | None) -> int:
        if batch_size is None:
            batch_size = self.batch_size
        if batch_size is None:
            raise RuntimeError(
                "batch_size not specified. Configure it on the replay buffer or "
                "pass it to sample()."
            )
        if isinstance(batch_size, bool) or not isinstance(batch_size, INT_CLASSES):
            raise TypeError("batch_size must be a positive integer.")
        if batch_size < 1:
            raise ValueError("batch_size must be a positive integer.")
        return int(batch_size)

    @staticmethod
    def _deadline(timeout: float | None) -> float | None:
        if timeout is None:
            return None
        if isinstance(timeout, bool) or not isinstance(timeout, Real):
            raise TypeError("timeout must be a non-negative number or None.")
        if timeout < 0:
            raise ValueError("timeout must be non-negative.")
        return time.monotonic() + float(timeout)

    def _sample_budget(self) -> int:
        stats = super().stats()
        target_samples = math.floor(int(stats["write_count"]) * self.samples_per_insert)
        return (
            target_samples
            - int(stats["samples_returned"])
            - self._counter_value(self._reserved_sample_count_value)
        )

    def _sample_permitted(self, batch_size: int) -> bool:
        return super().can_sample(batch_size) and self._sample_budget() >= batch_size

    def can_sample(self, batch_size: int | None = None) -> bool:
        """Return whether replay readiness and ratio budget permit a sample."""
        batch_size = self._resolve_ratio_batch_size(batch_size)
        with self._readiness_condition:
            return not self._service_shutdown and self._sample_permitted(batch_size)

    def _reserve_sample(
        self,
        batch_size: int,
        *,
        wait: bool,
        deadline: float | None,
        timeout: float | None,
        cancel_event: Any | None,
    ) -> None:
        condition = self._readiness_condition
        waiting = False
        with condition:
            while True:
                if self._service_shutdown:
                    raise RuntimeError("A shut down replay buffer cannot be sampled.")
                if cancel_event is not None and cancel_event.is_set():
                    raise RuntimeError("Replay-buffer sampling was cancelled.")
                if self._sample_permitted(batch_size):
                    self._increment_counter("_reserved_sample_count_value", batch_size)
                    return
                if not wait:
                    raise RuntimeError(
                        "Replay readiness or the sample-to-insert ratio does not "
                        "currently permit this sample. Pass wait=True to wait."
                    )
                if not waiting:
                    self._increment_counter("_sample_wait_count_value", 1)
                    waiting = True
                wait_time = None
                if deadline is not None:
                    wait_time = deadline - time.monotonic()
                    if wait_time <= 0:
                        raise TimeoutError(
                            "Replay buffer did not obtain sample-ratio budget within "
                            f"{timeout} seconds."
                        )
                if cancel_event is not None:
                    wait_time = 0.1 if wait_time is None else min(wait_time, 0.1)
                condition.wait(wait_time)

    def sample(
        self,
        batch_size: int | None = None,
        return_info: bool = False,
        *,
        wait: bool = False,
        timeout: float | None = None,
        cancel_event: Any | None = None,
    ) -> Any:
        """Sample after replay readiness and ratio budget permit the request."""
        batch_size = self._resolve_ratio_batch_size(batch_size)
        deadline = self._deadline(timeout)
        self._reserve_sample(
            batch_size,
            wait=wait,
            deadline=deadline,
            timeout=timeout,
            cancel_event=cancel_event,
        )
        try:
            remaining = None
            if deadline is not None:
                remaining = max(0.0, deadline - time.monotonic())
            return super().sample(
                batch_size,
                return_info=return_info,
                wait=wait,
                timeout=remaining,
                cancel_event=cancel_event,
            )
        finally:
            with self._readiness_condition:
                self._increment_counter("_reserved_sample_count_value", -batch_size)
                self._readiness_condition.notify_all()

    def stats(self) -> dict[str, int | float | bool]:
        """Return replay statistics with cumulative ratio state."""
        with self._readiness_condition:
            stats = super().stats()
            write_count = int(stats["write_count"])
            samples_returned = int(stats["samples_returned"])
            stats.update(
                target_samples_per_insert=self.samples_per_insert,
                samples_per_insert=(
                    samples_returned / write_count if write_count else 0.0
                ),
                sample_budget=self._sample_budget(),
                reserved_samples=self._counter_value(self._reserved_sample_count_value),
                sample_wait_count=self._counter_value(self._sample_wait_count_value),
            )
            return stats

    def state_dict(self) -> dict[str, Any]:
        """Return replay contents, counters, and rate-limit state."""
        state = super().state_dict()
        state["_rate_limit"] = {
            "samples_per_insert": self.samples_per_insert,
            "sample_wait_count": self._counter_value(self._sample_wait_count_value),
        }
        return state

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restore replay contents, counters, and rate-limit state."""
        rate_limit = state_dict["_rate_limit"]
        super().load_state_dict(state_dict)
        with self._readiness_condition:
            self.samples_per_insert = self._validate_samples_per_insert(
                rate_limit["samples_per_insert"]
            )
            self._set_counter("_reserved_sample_count_value", 0)
            self._set_counter(
                "_sample_wait_count_value", rate_limit["sample_wait_count"]
            )
            self._readiness_condition.notify_all()
