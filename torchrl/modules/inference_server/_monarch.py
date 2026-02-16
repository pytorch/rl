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


class _MonarchRequestQueue:
    """Wrapper around ``MonarchQueue`` that signals a :class:`threading.Event` on put.

    Also adapts the Monarch queue API (``get(block=False)``) to the standard
    ``get_nowait()`` expected by :class:`QueueBasedTransport`.
    """

    def __init__(self, monarch_queue, has_work: threading.Event):
        self._queue = monarch_queue
        self._has_work = has_work

    def put(self, item):
        self._queue.put(item)
        self._has_work.set()

    def get(self, timeout=None):
        return self._queue.get(timeout=timeout)

    def get_nowait(self):
        return self._queue.get(block=False)


class _MonarchResponseQueue:
    """Thin wrapper adapting the MonarchQueue get API."""

    def __init__(self, monarch_queue):
        self._queue = monarch_queue

    def put(self, item):
        self._queue.put(item)

    def get(self, timeout=None):
        return self._queue.get(timeout=timeout)


class MonarchTransport(QueueBasedTransport):
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
        super().__init__()
        try:
            import monarch  # noqa: F401
            from monarch.tools.queue import MonarchQueue
        except ImportError:
            raise ImportError(
                "Monarch is required for MonarchTransport. "
                "Install it following the Monarch documentation."
            )
        self._has_work = threading.Event()
        self._request_queue = _MonarchRequestQueue(
            MonarchQueue(maxsize=max_queue_size), self._has_work
        )
        self._response_queues: dict[int, _MonarchResponseQueue] = {}
        self._MonarchQueue = MonarchQueue

    def _make_response_queue(self) -> _MonarchResponseQueue:
        return _MonarchResponseQueue(self._MonarchQueue(maxsize=1000))

    def client(self) -> _QueueInferenceClient:
        """Create an actor-side client with a dedicated response queue.

        Returns:
            A :class:`_QueueInferenceClient` that can be passed to a Monarch
            actor.
        """
        return super().client()
