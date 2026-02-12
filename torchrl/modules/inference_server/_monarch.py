# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import queue
import threading
import time
from typing import Any

from tensordict.base import TensorDictBase

from torchrl.modules.inference_server._transport import InferenceTransport

_SENTINEL = object()


class _MonarchFuture:
    """Future-like object for Monarch transport results.

    Args:
        client: the :class:`_MonarchInferenceClient` that created this future.
        req_id: the unique request identifier within that client.
    """

    def __init__(self, client: _MonarchInferenceClient, req_id: int):
        self._client = client
        self._req_id = req_id
        self._result: Any = _SENTINEL

    def result(self, timeout: float | None = None) -> TensorDictBase:
        """Block until the result is available."""
        if self._result is _SENTINEL:
            item = self._client._get_result(self._req_id, timeout=timeout)
            if isinstance(item, BaseException):
                raise item
            self._result = item
        return self._result


class _MonarchInferenceClient:
    """Actor-side client for :class:`MonarchTransport`.

    Each client owns a dedicated response queue and routes results by
    request-id.

    Args:
        request_queue: the shared Monarch queue for requests.
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

    def submit(self, td: TensorDictBase) -> _MonarchFuture:
        """Submit a request and return a :class:`_MonarchFuture`."""
        req_id = self._next_req_id
        self._next_req_id += 1
        self._request_queue.put((self._actor_id, req_id, td))
        return _MonarchFuture(self, req_id)

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
            except Exception:
                raise queue.Empty(f"Timeout waiting for result of request {req_id}")
            if rid == req_id:
                return result
            self._buffered[rid] = result


class MonarchTransport(InferenceTransport):
    """Transport using Monarch for distributed inference on GPU clusters.

    Uses Monarch's actor model and RDMA-capable channels for efficient
    cross-node communication.  Monarch is imported lazily at instantiation
    time; importing the class itself does not require Monarch.

    .. note::
        This transport requires ``monarch`` to be installed.  It is designed
        for large-scale GPU clusters where Monarch is the preferred
        communication layer.

    Keyword Args:
        max_queue_size (int): maximum size of the request queue.
            Default: ``1000``.
    """

    def __init__(self, *, max_queue_size: int = 1000):
        try:
            import monarch  # noqa: F401
            from monarch.tools.queue import MonarchQueue
        except ImportError:
            raise ImportError(
                "Monarch is required for MonarchTransport. "
                "Install it following the Monarch documentation."
            )
        self._request_queue = MonarchQueue(maxsize=max_queue_size)
        self._response_queues: dict[int, Any] = {}
        self._lock = threading.Lock()
        self._next_actor_id = 0
        self._MonarchQueue = MonarchQueue

    # -- actor API ------------------------------------------------------------

    def client(self) -> _MonarchInferenceClient:
        """Create an actor-side client with a dedicated response queue.

        Returns:
            A :class:`_MonarchInferenceClient` that can be passed to a Monarch
            actor.
        """
        with self._lock:
            actor_id = self._next_actor_id
            self._next_actor_id += 1
        response_queue = self._MonarchQueue(maxsize=1000)
        self._response_queues[actor_id] = response_queue
        return _MonarchInferenceClient(self._request_queue, response_queue, actor_id)

    def submit(self, td: TensorDictBase):
        """Not supported -- use :meth:`client` to obtain an actor handle."""
        raise RuntimeError(
            "MonarchTransport.submit() is not supported. "
            "Call transport.client() to create a _MonarchInferenceClient."
        )

    # -- server API -----------------------------------------------------------

    def drain(
        self, max_items: int
    ) -> tuple[list[TensorDictBase], list[tuple[int, int]]]:
        """Dequeue up to *max_items* pending requests (non-blocking)."""
        items: list[TensorDictBase] = []
        callbacks: list[tuple[int, int]] = []
        for _ in range(max_items):
            try:
                actor_id, req_id, td = self._request_queue.get(block=False)
                items.append(td)
                callbacks.append((actor_id, req_id))
            except Exception:
                break
        return items, callbacks

    def wait_for_work(self, timeout: float) -> None:
        """Block until at least one request is available or *timeout* elapses."""
        try:
            item = self._request_queue.get(timeout=timeout)
            self._request_queue.put(item)
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
