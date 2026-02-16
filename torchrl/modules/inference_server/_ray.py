# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import threading

from torchrl.modules.inference_server._queue_transport import (
    _QueueInferenceClient,
    QueueBasedTransport,
)


class _RayRequestQueue:
    """Wrapper around ``ray.util.queue.Queue`` that signals a :class:`threading.Event` on put.

    Also adapts the Ray queue API (``get(block=False)``) to the standard
    ``get_nowait()`` expected by :class:`QueueBasedTransport`.
    """

    def __init__(self, ray_queue, has_work: threading.Event):
        self._queue = ray_queue
        self._has_work = has_work

    def put(self, item):
        self._queue.put(item)
        self._has_work.set()

    def get(self, timeout=None):
        return self._queue.get(timeout=timeout)

    def get_nowait(self):
        return self._queue.get(block=False)


class _RayResponseQueue:
    """Thin wrapper around ``ray.util.queue.Queue`` that adapts the get API."""

    def __init__(self, ray_queue):
        self._queue = ray_queue

    def put(self, item):
        self._queue.put(item)

    def get(self, timeout=None):
        return self._queue.get(timeout=timeout)


class RayTransport(QueueBasedTransport):
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
        super().__init__()
        try:
            import ray.util.queue
        except ImportError:
            raise ImportError(
                "Ray is required for RayTransport. Install it with: pip install ray"
            )
        self._has_work = threading.Event()
        self._request_queue = _RayRequestQueue(
            ray.util.queue.Queue(maxsize=max_queue_size), self._has_work
        )
        self._response_queues: dict[int, _RayResponseQueue] = {}
        self._ray_queue_module = ray.util.queue

    def _make_response_queue(self) -> _RayResponseQueue:
        return _RayResponseQueue(self._ray_queue_module.Queue(maxsize=1000))

    def client(self) -> _QueueInferenceClient:
        """Create an actor-side client with a dedicated Ray response queue.

        Returns:
            A :class:`_QueueInferenceClient` that can be used inside any Ray
            actor or the driver process.
        """
        return super().client()
