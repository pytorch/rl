# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import multiprocessing as mp

from torchrl.modules.inference_server._queue_transport import (
    _QueueInferenceClient,
    QueueBasedTransport,
)


class MPTransport(QueueBasedTransport):
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
        super().__init__()
        self._ctx = ctx if ctx is not None else mp.get_context("spawn")
        self._request_queue: mp.Queue = self._ctx.Queue()
        self._response_queues: dict[int, mp.Queue] = {}

    def _make_response_queue(self) -> mp.Queue:
        return self._ctx.Queue()

    def client(self) -> _QueueInferenceClient:
        """Create an actor-side client with a dedicated response queue.

        Must be called in the parent process **before** spawning children.

        Returns:
            A :class:`_QueueInferenceClient` that can be passed to a child
            process as an argument to :class:`multiprocessing.Process`.
        """
        return super().client()
