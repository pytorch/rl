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


class _RayFuture:
    """Future-like object backed by a per-actor response queue.

    Works identically to :class:`_MPFuture` in the multiprocessing transport
    but uses a :class:`ray.util.queue.Queue` under the hood.

    Args:
        client: the :class:`_RayInferenceClient` that created this future.
        req_id: the unique request identifier within that client.
    """

    def __init__(self, client: _RayInferenceClient, req_id: int):
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
        """Block until the result is available."""
        if self._result is _SENTINEL:
            self._result = self._client._get_result(self._req_id, timeout=timeout)
        if isinstance(self._result, BaseException):
            raise self._result
        return self._result


class _RayInferenceClient:
    """Actor-side client for :class:`RayTransport`.

    Each client owns a dedicated Ray response queue and routes results by
    request-id. Instances are created by :meth:`RayTransport.client`.

    Args:
        request_queue: the shared ``ray.util.queue.Queue`` for requests.
        response_queue: this client's dedicated Ray response queue.
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

    def submit(self, td: TensorDictBase) -> _RayFuture:
        """Submit a request and return a :class:`_RayFuture`."""
        req_id = self._next_req_id
        self._next_req_id += 1
        self._request_queue.put((self._actor_id, req_id, td))
        return _RayFuture(self, req_id)

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


class RayTransport(InferenceTransport):
    """Transport using Ray queues for distributed inference.

    Uses ``ray.util.queue.Queue`` for both request submission and response
    routing.  Per-actor response queues ensure correct result routing without
    serialising Queue objects through other queues.

    Ray is imported lazily at instantiation time; importing the class itself
    does not require Ray.

    Keyword Args:
        max_queue_size (int): maximum size of the request queue.
            Default: ``1000``.

    Example:
        >>> import ray
        >>> ray.init()
        >>> transport = RayTransport()
        >>> client = transport.client()
        >>> # pass *client* to a Ray actor for remote inference requests
    """

    def __init__(self, *, max_queue_size: int = 1000):
        try:
            import ray
            import ray.util.queue
        except ImportError:
            raise ImportError(
                "Ray is required for RayTransport. Install it with: pip install ray"
            )
        self._ray = ray
        self._request_queue = ray.util.queue.Queue(maxsize=max_queue_size)
        self._response_queues: dict[int, ray.util.queue.Queue] = {}
        self._lock = threading.Lock()
        self._next_actor_id = 0

    # -- actor API ------------------------------------------------------------

    def client(self) -> _RayInferenceClient:
        """Create an actor-side client with a dedicated Ray response queue.

        Returns:
            A :class:`_RayInferenceClient` that can be used inside any Ray
            actor or the driver process.
        """
        import ray.util.queue

        with self._lock:
            actor_id = self._next_actor_id
            self._next_actor_id += 1
        response_queue = ray.util.queue.Queue(maxsize=1000)
        self._response_queues[actor_id] = response_queue
        return _RayInferenceClient(self._request_queue, response_queue, actor_id)

    def submit(self, td: TensorDictBase):
        """Not supported -- use :meth:`client` to obtain an actor handle."""
        raise RuntimeError(
            "RayTransport.submit() is not supported. "
            "Call transport.client() to create a _RayInferenceClient."
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
                actor_id, req_id, td = self._request_queue.get(
                    block=False,
                )
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
