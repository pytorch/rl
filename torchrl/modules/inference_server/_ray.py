# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

from torchrl.modules.inference_server._queue_transport import (
    _QueueInferenceClient,
    QueueBasedTransport,
)


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
        self._request_queue = ray.util.queue.Queue(maxsize=max_queue_size)
        self._response_queues: dict[int, ray.util.queue.Queue] = {}
        self._ray_queue_module = ray.util.queue

    def _make_response_queue(self):
        return self._ray_queue_module.Queue(maxsize=1000)

    def client(self) -> _QueueInferenceClient:
        """Create an actor-side client with a dedicated Ray response queue.

        Returns:
            A :class:`_QueueInferenceClient` that can be used inside any Ray
            actor or the driver process.
        """
        return super().client()
