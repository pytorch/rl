# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Shared base classes for queue-based inference transports.

:class:`QueueBasedTransport` factors out the common submit/drain/resolve logic
used by :class:`~torchrl.modules.inference_server.MPTransport`,
:class:`~torchrl.modules.inference_server.RayTransport`, and
:class:`~torchrl.modules.inference_server.MonarchTransport`.  Each concrete
subclass only needs to supply the queue objects (request + per-actor response).
"""
from __future__ import annotations

import queue
import threading
import time
from typing import Any

from tensordict.base import TensorDictBase

from torchrl.modules.inference_server._transport import InferenceTransport

_SENTINEL = object()


class _QueueFuture:
    """Future-like object backed by a per-actor response queue.

    The future retrieves its result by request-id so that out-of-order
    ``result()`` calls work correctly.

    Args:
        client: the :class:`_QueueInferenceClient` that created this future.
        req_id: the unique request identifier within that client.
    """

    def __init__(self, client: _QueueInferenceClient, req_id: int):
        self._client = client
        self._req_id = req_id
        self._result: Any = _SENTINEL

    def done(self) -> bool:
        """Return ``True`` if the result is available without blocking."""
        if self._result is not _SENTINEL:
            return True
        try:
            self._result = self._client._get_result(self._req_id, timeout=0)
        except queue.Empty:
            return False
        return True

    def result(self, timeout: float | None = None) -> TensorDictBase:
        """Block until the result is available.

        Args:
            timeout: seconds to wait.  ``None`` waits indefinitely.

        Raises:
            queue.Empty: if *timeout* expires before a result arrives.
            Exception: if the server set an exception instead of a result.
        """
        if self._result is _SENTINEL:
            self._result = self._client._get_result(self._req_id, timeout=timeout)
        if isinstance(self._result, BaseException):
            raise self._result
        return self._result


class _QueueInferenceClient:
    """Actor-side client for :class:`QueueBasedTransport`.

    Each client owns a dedicated response queue and routes results by
    request-id.

    Args:
        request_queue: the shared request queue (any object with ``.put()``).
        response_queue: this client's dedicated response queue.
        actor_id: the unique identifier assigned by the transport.
    """

    def __init__(self, request_queue, response_queue, actor_id: int):
        self._request_queue = request_queue
        self._response_queue = response_queue
        self._actor_id = actor_id
        self._next_req_id = 0
        self._buffered: dict[int, Any] = {}

    def __call__(self, td: TensorDictBase) -> TensorDictBase:
        """Submit a request and block until the result is ready."""
        return self.submit(td).result()

    def submit(self, td: TensorDictBase) -> _QueueFuture:
        """Submit a request and return a :class:`_QueueFuture`."""
        req_id = self._next_req_id
        self._next_req_id += 1
        self._request_queue.put((self._actor_id, req_id, td))
        return _QueueFuture(self, req_id)

    # -- internal -------------------------------------------------------------

    def _get_result(self, req_id: int, timeout: float | None = None) -> Any:
        """Return the result for *req_id*, buffering any earlier arrivals."""
        if req_id in self._buffered:
            return self._buffered.pop(req_id)
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            remaining = None
            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise queue.Empty(f"Timeout waiting for result of request {req_id}")
            try:
                rid, result = self._response_queue.get(timeout=remaining)
            except Exception as e:
                raise queue.Empty(
                    f"Timeout waiting for result of request {req_id}"
                ) from e
            if rid == req_id:
                return result
            self._buffered[rid] = result


class QueueBasedTransport(InferenceTransport):
    """Base class for transports that use a request queue and per-actor response queues.

    Subclasses must set the following attributes in ``__init__`` (before or
    after calling ``super().__init__()``):

    * ``_request_queue`` -- the shared request queue (any object with
      ``.put()``, ``.get(timeout=...)``, and ``.get(block=False)``).
    * ``_response_queues`` -- a ``dict[int, <queue>]`` mapping actor ids to
      per-actor response queues.

    Subclasses must implement:

    * :meth:`_make_response_queue` -- factory for creating a new response queue.

    .. note::
        ``wait_for_work`` uses a blocking ``get`` to detect new work.  The
        retrieved item is stored in ``_peeked`` and consumed by the next
        ``drain`` call, preserving FIFO ordering.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._next_actor_id = 0
        self._peeked = None

    # -- to be implemented by subclasses --------------------------------------

    def _make_response_queue(self):
        """Create a new response queue for an actor."""
        raise NotImplementedError

    # -- actor API ------------------------------------------------------------

    def client(self) -> _QueueInferenceClient:
        """Create an actor-side client with a dedicated response queue.

        Returns:
            A :class:`_QueueInferenceClient`.
        """
        with self._lock:
            actor_id = self._next_actor_id
            self._next_actor_id += 1
        response_queue = self._make_response_queue()
        self._response_queues[actor_id] = response_queue
        return _QueueInferenceClient(self._request_queue, response_queue, actor_id)

    def submit(self, td: TensorDictBase):
        """Not supported -- use :meth:`client` to obtain an actor handle."""
        raise RuntimeError(
            f"{type(self).__name__}.submit() is not supported. "
            "Call transport.client() to create a client."
        )

    # -- server API -----------------------------------------------------------

    def drain(
        self, max_items: int
    ) -> tuple[list[TensorDictBase], list[tuple[int, int]]]:
        """Dequeue up to *max_items* pending requests (non-blocking)."""
        items: list[TensorDictBase] = []
        callbacks: list[tuple[int, int]] = []
        peeked = self._peeked
        if peeked is not None:
            self._peeked = None
            actor_id, req_id, td = peeked
            items.append(td)
            callbacks.append((actor_id, req_id))
        for _ in range(max_items - len(items)):
            try:
                actor_id, req_id, td = self._request_queue.get(block=False)
            except Exception:
                break
            items.append(td)
            callbacks.append((actor_id, req_id))
        return items, callbacks

    def wait_for_work(self, timeout: float) -> None:
        """Block until at least one request is available or *timeout* elapses."""
        if self._peeked is not None:
            return
        try:
            self._peeked = self._request_queue.get(timeout=timeout)
        except Exception:
            pass

    def resolve(self, callback: tuple[int, int], result: TensorDictBase) -> None:
        """Route the result to the correct actor's response queue."""
        actor_id, req_id = callback
        self._response_queues[actor_id].put((req_id, result))

    def resolve_exception(self, callback: tuple[int, int], exc: BaseException) -> None:
        """Route an exception to the correct actor's response queue."""
        actor_id, req_id = callback
        self._response_queues[actor_id].put((req_id, exc))
