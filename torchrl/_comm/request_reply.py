# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Typed request/reply channels and their shared serving loop."""
from __future__ import annotations

import abc
import struct
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import Future
from dataclasses import dataclass, field
from statistics import mean
from typing import Any, Literal, TypeAlias

from tensordict.base import TensorDictBase

MetadataValue: TypeAlias = bool | int | float | str
MessageMetadata: TypeAlias = Mapping[str, MetadataValue]
MetadataDType: TypeAlias = Literal["bool", "int64", "float64"]


@dataclass(frozen=True)
class MetadataField:
    """One scalar field carried in a fixed transport header."""

    name: str
    dtype: MetadataDType


@dataclass(frozen=True)
class MetadataSpec:
    """Fixed schema for request metadata used by tensor transports."""

    fields: tuple[MetadataField, ...] = ()

    def __post_init__(self) -> None:
        names = [metadata_field.name for metadata_field in self.fields]
        if len(names) != len(set(names)):
            raise ValueError("Metadata field names must be unique.")

    def validate(self, metadata: MessageMetadata) -> None:
        expected = {metadata_field.name for metadata_field in self.fields}
        actual = set(metadata)
        if actual != expected:
            raise ValueError(
                "Metadata does not match the declared schema; "
                f"missing={sorted(expected - actual)}, "
                f"extra={sorted(actual - expected)}."
            )
        for metadata_field in self.fields:
            value = metadata[metadata_field.name]
            if metadata_field.dtype == "bool" and not isinstance(value, bool):
                raise TypeError(f"Metadata field {metadata_field.name!r} must be bool.")
            if metadata_field.dtype == "int64" and (
                not isinstance(value, int) or isinstance(value, bool)
            ):
                raise TypeError(f"Metadata field {metadata_field.name!r} must be int.")
            if metadata_field.dtype == "float64" and not isinstance(
                value, (int, float)
            ):
                raise TypeError(
                    f"Metadata field {metadata_field.name!r} must be float."
                )

    def encode(self, metadata: MessageMetadata) -> list[int]:
        self.validate(metadata)
        result = []
        for metadata_field in self.fields:
            value = metadata[metadata_field.name]
            if metadata_field.dtype == "float64":
                result.append(struct.unpack("q", struct.pack("d", float(value)))[0])
            else:
                result.append(int(value))
        return result

    def decode(self, values: Sequence[int]) -> dict[str, MetadataValue]:
        if len(values) != len(self.fields):
            raise ValueError(
                f"Expected {len(self.fields)} metadata values, got {len(values)}."
            )
        result: dict[str, MetadataValue] = {}
        for metadata_field, value in zip(self.fields, values):
            if metadata_field.dtype == "bool":
                result[metadata_field.name] = bool(value)
            elif metadata_field.dtype == "int64":
                result[metadata_field.name] = int(value)
            else:
                result[metadata_field.name] = struct.unpack(
                    "d", struct.pack("q", int(value))
                )[0]
        return result


@dataclass(frozen=True)
class Message:
    """One request/reply-channel message.

    Args:
        payload: TensorDict payload owned by the domain protocol.
        metadata: Small control-plane values interpreted by the domain, not
            by the physical transport.

    """

    payload: TensorDictBase
    metadata: MessageMetadata = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", dict(self.metadata))


def _as_message(
    payload: Message | TensorDictBase,
    metadata: MessageMetadata | None = None,
) -> Message:
    """Normalize a payload and optional metadata into a :class:`Message`."""
    if isinstance(payload, Message):
        if metadata:
            raise ValueError("metadata cannot be added to an existing Message.")
        return payload
    return Message(payload, {} if metadata is None else metadata)


class RequestReplyClient:
    """Lightweight client for a :class:`RequestReplyTransport`.

    Args:
        transport: Bound request/reply channel.

    """

    supports_metadata = False

    def __init__(self, transport: RequestReplyTransport) -> None:
        self._transport = transport
        self.supports_metadata = bool(getattr(transport, "supports_metadata", False))

    def submit(
        self,
        payload: TensorDictBase,
        *,
        metadata: MessageMetadata | None = None,
    ) -> Future[TensorDictBase]:
        """Submit a request and immediately return its future."""
        if metadata is None:
            return self._transport.submit(payload)
        return self._transport.submit(payload, metadata=metadata)

    def __call__(
        self,
        payload: TensorDictBase,
        timeout: float | None = None,
        *,
        metadata: MessageMetadata | None = None,
    ) -> TensorDictBase:
        """Submit a request and block for its reply."""
        return self.submit(payload, metadata=metadata).result(timeout=timeout)


class RequestReplyTransport(abc.ABC):
    """Physical transport bound to one typed request/reply channel.

    Domain services consume this interface and remain independent of whether
    payloads travel by reference, queue, shared memory, object store, or
    ``torch.distributed`` point-to-point operations.

    """

    @abc.abstractmethod
    def submit(
        self,
        payload: TensorDictBase,
        *,
        metadata: MessageMetadata | None = None,
    ) -> Future[TensorDictBase]:
        """Submit one request and return a future for its reply."""

    @abc.abstractmethod
    def drain(self, max_items: int) -> tuple[list[TensorDictBase], list[Any]]:
        """Drain payloads and opaque reply callbacks."""

    @abc.abstractmethod
    def wait_for_work(self, timeout: float) -> None:
        """Wait until work is available or ``timeout`` expires."""

    @abc.abstractmethod
    def resolve(self, callback: Any, result: TensorDictBase) -> None:
        """Resolve one callback with a successful reply."""

    @abc.abstractmethod
    def resolve_exception(self, callback: Any, exc: BaseException) -> None:
        """Resolve one callback with an exception."""

    def drain_messages_with_timing(
        self, max_items: int
    ) -> tuple[list[Message], list[Any], list[float | None]]:
        """Drain messages, callbacks, and optional submission timestamps.

        The default adapter preserves compatibility with transports that only
        implement the historical payload-only ``drain`` contract.
        """
        drain_with_timing = getattr(self, "drain_with_timing", None)
        if drain_with_timing is None:
            payloads, callbacks = self.drain(max_items)
            submitted_at = [None] * len(payloads)
        else:
            payloads, callbacks, submitted_at = drain_with_timing(max_items)
        return [_as_message(payload) for payload in payloads], callbacks, submitted_at

    def _set_peer_alive(self, alive_event) -> None:  # noqa: B027
        """Attach an optional liveness flag used by process-aware clients."""

    def client(self) -> RequestReplyClient:
        """Return a restricted client bound to this channel."""
        return RequestReplyClient(self)


BatchHandler: TypeAlias = Callable[[Sequence[Message]], Sequence[TensorDictBase]]


class ChannelServer:
    """Shared lifecycle, batching, and fan-out loop for request/reply services.

    The server is intentionally domain-neutral. The handler receives a batch
    of :class:`Message` objects and must return one TensorDict reply per
    request. Inference, replay buffers, and future TorchRL services can share
    the same queueing and failure semantics while keeping their domain logic
    in small handlers.

    Args:
        channel: Bound request/reply transport.
        handler: Callable returning one reply per input message.

    Keyword Args:
        max_batch_size: Maximum requests handled together. Defaults to ``64``.
        min_batch_size: Minimum batch size accumulated before dispatch.
            Defaults to ``1``.
        timeout: Maximum accumulation delay in seconds. Defaults to ``0.01``.
        before_wait: Optional non-blocking maintenance hook.
        collect_stats: Whether to collect batching and latency statistics.
            Defaults to ``True``.
        stats_window_size: Number of recent samples retained. Defaults to
            ``1024``.
        shutdown_event: Optional externally owned event.
        name: Worker-thread name prefix. Defaults to ``"ChannelServer"``.

    """

    def __init__(
        self,
        channel: RequestReplyTransport,
        handler: BatchHandler,
        *,
        max_batch_size: int = 64,
        min_batch_size: int = 1,
        timeout: float = 0.01,
        before_wait: Callable[[], None] | None = None,
        collect_stats: bool = True,
        stats_window_size: int = 1024,
        shutdown_event=None,
        name: str = "ChannelServer",
    ) -> None:
        if max_batch_size < 1:
            raise ValueError("max_batch_size must be at least 1.")
        if min_batch_size < 1 or min_batch_size > max_batch_size:
            raise ValueError("min_batch_size must be between 1 and max_batch_size.")
        if timeout < 0:
            raise ValueError("timeout must be non-negative.")
        if stats_window_size < 1:
            raise ValueError("stats_window_size must be at least 1.")
        self.channel = channel
        self.handler = handler
        self.max_batch_size = max_batch_size
        self.min_batch_size = min_batch_size
        self.timeout = timeout
        self.before_wait = before_wait
        self.collect_stats = collect_stats
        self.stats_window_size = stats_window_size
        self._shutdown_event = (
            threading.Event() if shutdown_event is None else shutdown_event
        )
        self._name = name
        self._worker: threading.Thread | None = None
        self._stats_lock = threading.Lock()
        self._reset_stats()

    def _reset_stats(self) -> None:
        self._stats_started_at = time.monotonic()
        self._num_requests = 0
        self._num_batches = 0
        self._batch_sizes: list[int] = []
        self._queue_wait_ms: list[float] = []
        self._handler_ms: list[float] = []

    @staticmethod
    def _percentile(values: list[float], percentile: float) -> float:
        if not values:
            return 0.0
        values = sorted(values)
        index = int(round((len(values) - 1) * percentile))
        return float(values[index])

    def _extend_window(self, target: list, values: list) -> None:
        target.extend(values)
        excess = len(target) - self.stats_window_size
        if excess > 0:
            del target[:excess]

    def _record_stats(
        self,
        batch_size: int,
        queue_wait_ms: Sequence[float],
        handler_ms: float,
    ) -> None:
        if not self.collect_stats:
            return
        with self._stats_lock:
            self._num_requests += batch_size
            self._num_batches += 1
            self._extend_window(self._batch_sizes, [batch_size])
            self._extend_window(self._queue_wait_ms, queue_wait_ms)
            self._extend_window(self._handler_ms, [handler_ms])

    def stats(self, *, reset: bool = False) -> dict[str, float | int]:
        """Return generic request, batching, queue, and handler statistics."""
        with self._stats_lock:
            elapsed = max(time.monotonic() - self._stats_started_at, 1e-12)
            requests = self._num_requests
            batches = self._num_batches
            result = {
                "requests": requests,
                "batches": batches,
                "requests_per_s": requests / elapsed,
                "batches_per_s": batches / elapsed,
                "avg_batch_size": (
                    float(mean(self._batch_sizes)) if self._batch_sizes else 0.0
                ),
                "p50_queue_ms": self._percentile(self._queue_wait_ms, 0.50),
                "p95_queue_ms": self._percentile(self._queue_wait_ms, 0.95),
                "p50_handler_ms": self._percentile(self._handler_ms, 0.50),
                "p95_handler_ms": self._percentile(self._handler_ms, 0.95),
            }
            if reset:
                self._reset_stats()
            return result

    def start(self) -> ChannelServer:
        """Start the background serve loop."""
        if self.is_alive:
            raise RuntimeError("Server is already running.")
        self._shutdown_event.clear()
        self._worker = threading.Thread(
            target=self._run,
            daemon=True,
            name=f"{self._name}-worker",
        )
        self._worker.start()
        return self

    def shutdown(self, timeout: float | None = 5.0) -> None:
        """Stop the serve loop and reject work that has not started."""
        self._shutdown_event.set()
        if self._worker is not None:
            self._worker.join(timeout=timeout)
            self._worker = None

    @property
    def is_alive(self) -> bool:
        """Whether the serve-loop thread is alive."""
        return self._worker is not None and self._worker.is_alive()

    def client(self) -> Any:
        """Return a restricted client for the bound channel."""
        return self.channel.client()

    def _drain(
        self, max_items: int
    ) -> tuple[list[Message], list[Any], list[float | None]]:
        return self.channel.drain_messages_with_timing(max_items)

    def _run(self) -> None:
        try:
            while not self._shutdown_event.is_set():
                if self.before_wait is not None:
                    self.before_wait()
                self.channel.wait_for_work(timeout=self.timeout)
                messages, callbacks, submitted_at = self._drain(self.max_batch_size)
                if not messages:
                    continue
                if len(messages) < self.min_batch_size:
                    deadline = time.monotonic() + self.timeout
                    while len(messages) < self.min_batch_size:
                        remaining = deadline - time.monotonic()
                        if remaining <= 0:
                            break
                        self.channel.wait_for_work(timeout=remaining)
                        more_messages, more_callbacks, more_submitted_at = self._drain(
                            self.max_batch_size - len(messages)
                        )
                        messages.extend(more_messages)
                        callbacks.extend(more_callbacks)
                        submitted_at.extend(more_submitted_at)
                try:
                    now = time.monotonic()
                    queue_wait_ms = [
                        (now - timestamp) * 1000.0
                        for timestamp in submitted_at
                        if timestamp is not None
                    ]
                    started = time.monotonic()
                    results = list(self.handler(messages))
                    handler_ms = (time.monotonic() - started) * 1000.0
                    if len(results) != len(callbacks):
                        raise RuntimeError(
                            f"Channel handler returned {len(results)} replies for "
                            f"{len(callbacks)} requests."
                        )
                    self._record_stats(len(callbacks), queue_wait_ms, handler_ms)
                    for callback, result in zip(callbacks, results):
                        self.channel.resolve(callback, result)
                except Exception as exc:
                    for callback in callbacks:
                        self.channel.resolve_exception(callback, exc)
        finally:
            self._drain_pending_on_shutdown()

    def _drain_pending_on_shutdown(self) -> None:
        error = RuntimeError(f"{self._name} is shutting down.")
        while True:
            messages, callbacks, _ = self._drain(self.max_batch_size)
            if not messages:
                break
            for callback in callbacks:
                self.channel.resolve_exception(callback, error)

    def __enter__(self) -> ChannelServer:
        return self.start()

    def __exit__(self, *exc_info) -> None:
        self.shutdown()

    def __del__(self) -> None:
        worker = getattr(self, "_worker", None)
        if worker is not None and worker.is_alive():
            self.shutdown(timeout=1.0)
