# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import queue
import threading
import time
from collections.abc import Callable
from typing import Any

_MISSING = object()


class MailboxFuture:
    """Future-like result associated with one mailbox request."""

    def __init__(self, client: MailboxClient, request_id: int) -> None:
        self._client = client
        self._request_id = request_id
        self._result: Any = _MISSING

    def done(self) -> bool:
        """Return ``True`` when the result can be read without blocking."""
        if self._result is not _MISSING:
            return True
        try:
            self._result = self._client._get_result(self._request_id, timeout=0)
        except queue.Empty:
            return False
        return True

    def result(self, timeout: float | None = None) -> Any:
        """Return the request result or raise its remote exception."""
        if self._result is _MISSING:
            self._result = self._client._get_result(self._request_id, timeout=timeout)
        if isinstance(self._result, BaseException):
            raise self._result
        return self._result


class MailboxClient:
    """Picklable client for an N:1 request mailbox."""

    def __init__(self, request_queue, response_queue, client_id: int) -> None:
        self._request_queue = request_queue
        self._response_queue = response_queue
        self._client_id = client_id
        self._next_request_id = 0
        self._buffered: dict[int, Any] = {}
        self._request_lock = threading.Lock()
        self._response_lock = threading.Lock()

    @property
    def client_id(self) -> int:
        """The identifier assigned by the owning mailbox."""
        return self._client_id

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_request_lock"] = None
        state["_response_lock"] = None
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._request_lock = threading.Lock()
        self._response_lock = threading.Lock()

    def submit(self, payload: Any) -> MailboxFuture:
        """Submit a payload and immediately return its future."""
        with self._request_lock:
            request_id = self._next_request_id
            self._next_request_id += 1
            self._request_queue.put(
                (self._client_id, request_id, time.monotonic(), payload)
            )
        return MailboxFuture(self, request_id)

    def __call__(self, payload: Any, timeout: float | None = None) -> Any:
        """Submit a request and block for its result."""
        return self.submit(payload).result(timeout=timeout)

    def _get_result(self, request_id: int, timeout: float | None = None) -> Any:
        with self._response_lock:
            return self._get_result_unlocked(request_id, timeout=timeout)

    def _get_result_unlocked(
        self, request_id: int, timeout: float | None = None
    ) -> Any:
        if request_id in self._buffered:
            return self._buffered.pop(request_id)
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            remaining = None if deadline is None else deadline - time.monotonic()
            if remaining is not None and remaining <= 0:
                raise queue.Empty(
                    f"Timeout waiting for result of request {request_id}."
                )
            try:
                response_id, result = self._response_queue.get(timeout=remaining)
            except queue.Empty:
                raise
            except Exception as err:
                raise queue.Empty(
                    f"Timeout waiting for result of request {request_id}."
                ) from err
            if response_id == request_id:
                return result
            self._buffered[response_id] = result


class Mailbox:
    """N:1 queue with per-client response routing.

    Queue implementations are injected so the same request/reply machinery can
    be used with threads, multiprocessing, Ray, and Monarch.
    """

    def __init__(
        self,
        request_queue,
        response_queue_factory: Callable[[], Any],
        *,
        response_queues=None,
    ) -> None:
        self.request_queue = request_queue
        self.response_queue_factory = response_queue_factory
        self.response_queues = {} if response_queues is None else response_queues
        self._next_client_id = 0
        self._client_lock = threading.Lock()
        self._peeked = None

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_client_lock"] = None
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._client_lock = threading.Lock()

    def client(self) -> MailboxClient:
        """Create a client with a dedicated response queue."""
        with self._client_lock:
            client_id = self._next_client_id
            self._next_client_id += 1
        response_queue = self.response_queue_factory()
        self.response_queues[client_id] = response_queue
        return MailboxClient(self.request_queue, response_queue, client_id)

    def wait_for_work(self, timeout: float) -> None:
        """Block until a request is ready or the timeout expires."""
        if self._peeked is not None:
            return
        try:
            request = self.request_queue.get(timeout=timeout)
        except Exception:
            return
        if request is not None:
            self._peeked = request

    def drain(
        self, max_items: int
    ) -> tuple[list[Any], list[tuple[int, int]], list[float | None]]:
        """Drain requests, callbacks, and submission timestamps."""
        payloads: list[Any] = []
        callbacks: list[tuple[int, int]] = []
        submitted_at: list[float | None] = []

        def append(request) -> None:
            if len(request) == 4:
                client_id, request_id, request_time, payload = request
            else:
                client_id, request_id, payload = request
                request_time = None
            payloads.append(payload)
            callbacks.append((client_id, request_id))
            submitted_at.append(request_time)

        if self._peeked is not None:
            request = self._peeked
            self._peeked = None
            append(request)
        for _ in range(max_items - len(payloads)):
            try:
                request = self.request_queue.get(block=False)
            except Exception:
                break
            if request is not None:
                append(request)
        return payloads, callbacks, submitted_at

    def resolve(self, callback: tuple[int, int], result: Any) -> None:
        """Resolve a request through its client's response queue."""
        client_id, request_id = callback
        self.response_queues[client_id].put((request_id, result))

    def reject(self, callback: tuple[int, int], error: BaseException) -> None:
        """Resolve a request with an exception."""
        self.resolve(callback, error)
