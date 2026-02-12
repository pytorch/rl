# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import multiprocessing as mp
import queue
import threading
import time
from typing import Any

from tensordict.base import TensorDictBase

from torchrl.modules.inference_server._transport import InferenceTransport

_SENTINEL = object()


class _MPFuture:
    """Future-like object backed by a per-actor response queue.

    The future retrieves its result by request-id so that out-of-order
    ``result()`` calls work correctly.

    Args:
        client: the :class:`_MPInferenceClient` that created this future.
        req_id: the unique request identifier within that client.
    """

    def __init__(self, client: _MPInferenceClient, req_id: int):
        self._client = client
        self._req_id = req_id
        self._result: Any = _SENTINEL

    def result(self, timeout: float | None = None) -> TensorDictBase:
        """Block until the result is available.

        Args:
            timeout: seconds to wait.  ``None`` waits indefinitely.

        Raises:
            queue.Empty: if *timeout* expires before a result arrives.
            Exception: if the server set an exception instead of a result.
        """
        if self._result is _SENTINEL:
            item = self._client._get_result(self._req_id, timeout=timeout)
            if isinstance(item, BaseException):
                raise item
            self._result = item
        return self._result


class _MPInferenceClient:
    """Actor-side client for :class:`MPTransport`.

    Each client owns a dedicated response queue and routes results by
    request-id.  Instances are created by :meth:`MPTransport.client` and
    must be created **before** spawning child processes so that the
    underlying queues are inherited.

    Args:
        request_queue: the shared request queue.
        response_queue: this client's dedicated response queue.
        actor_id: the unique identifier assigned by the transport.
    """

    def __init__(
        self,
        request_queue: mp.Queue,
        response_queue: mp.Queue,
        actor_id: int,
    ):
        self._request_queue = request_queue
        self._response_queue = response_queue
        self._actor_id = actor_id
        self._next_req_id = 0
        self._buffered: dict[int, Any] = {}

    def __call__(self, td: TensorDictBase) -> TensorDictBase:
        """Submit a request and block until the result is ready."""
        return self.submit(td).result()

    def submit(self, td: TensorDictBase) -> _MPFuture:
        """Submit a request and return an :class:`_MPFuture`."""
        req_id = self._next_req_id
        self._next_req_id += 1
        self._request_queue.put((self._actor_id, req_id, td))
        return _MPFuture(self, req_id)

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
            rid, result = self._response_queue.get(timeout=remaining)
            if rid == req_id:
                return result
            self._buffered[rid] = result


class MPTransport(InferenceTransport):
    """Cross-process transport using :mod:`multiprocessing` queues.

    Response routing uses per-actor queues (one per :meth:`client` call) so
    that no ``mp.Queue`` object is ever serialised through another queue.
    Clients must be created with :meth:`client` **before** spawning child
    processes.

    Args:
        ctx: a multiprocessing context (e.g. ``mp.get_context("spawn")``).
            Defaults to ``mp.get_context("spawn")``.

    Example:
        >>> import multiprocessing as mp
        >>> transport = MPTransport()
        >>> client = transport.client()           # creates response queue
        >>> p = mp.Process(target=actor_fn, args=(client,))
        >>> p.start()                             # queue inherited
    """

    def __init__(self, ctx: mp.context.BaseContext | None = None):
        self._ctx = ctx if ctx is not None else mp.get_context("spawn")
        self._request_queue: mp.Queue = self._ctx.Queue()
        self._response_queues: dict[int, mp.Queue] = {}
        self._lock = threading.Lock()
        self._next_actor_id = 0

    # -- actor API (called before fork) ---------------------------------------

    def client(self) -> _MPInferenceClient:
        """Create an actor-side client with a dedicated response queue.

        Must be called in the parent process **before** spawning children.

        Returns:
            An :class:`_MPInferenceClient` that can be passed to a child
            process as an argument to :class:`multiprocessing.Process`.
        """
        with self._lock:
            actor_id = self._next_actor_id
            self._next_actor_id += 1
        response_queue: mp.Queue = self._ctx.Queue()
        self._response_queues[actor_id] = response_queue
        return _MPInferenceClient(self._request_queue, response_queue, actor_id)

    def submit(self, td: TensorDictBase):
        """Not supported -- use :meth:`client` to obtain an actor handle."""
        raise RuntimeError(
            "MPTransport.submit() is not supported. "
            "Call transport.client() to create an _MPInferenceClient."
        )

    # -- server API -----------------------------------------------------------

    def drain(
        self, max_items: int
    ) -> tuple[list[TensorDictBase], list[tuple[int, int]]]:
        """Dequeue up to *max_items* pending ``(actor_id, req_id, td)`` tuples."""
        items: list[TensorDictBase] = []
        callbacks: list[tuple[int, int]] = []
        for _ in range(max_items):
            try:
                actor_id, req_id, td = self._request_queue.get_nowait()
                items.append(td)
                callbacks.append((actor_id, req_id))
            except queue.Empty:
                break
        return items, callbacks

    def wait_for_work(self, timeout: float) -> None:
        """Block until at least one request is available or *timeout* elapses."""
        try:
            item = self._request_queue.get(timeout=timeout)
            # Put it back so drain() can consume it.
            self._request_queue.put(item)
        except queue.Empty:
            pass

    def resolve(self, callback: tuple[int, int], result: TensorDictBase) -> None:
        """Route the result to the correct actor's response queue."""
        actor_id, req_id = callback
        self._response_queues[actor_id].put((req_id, result))

    def resolve_exception(self, callback: tuple[int, int], exc: BaseException) -> None:
        """Route an exception to the correct actor's response queue."""
        actor_id, req_id = callback
        self._response_queues[actor_id].put((req_id, exc))
