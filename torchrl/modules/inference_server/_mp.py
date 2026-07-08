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
        use_manager (bool, optional): if ``True``, back the request and
            response queues with a multiprocessing manager. This is useful
            when clients are forwarded through another spawned process.
            Defaults to ``False``.

    Example:
        >>> import multiprocessing as mp
        >>> transport = MPTransport()
        >>> client = transport.client()           # creates response queue
        >>> p = mp.Process(target=actor_fn, args=(client,))
        >>> p.start()                             # queue inherited
    """

    def __init__(
        self, ctx: mp.context.BaseContext | None = None, *, use_manager: bool = False
    ):
        super().__init__()
        self._ctx = ctx if ctx is not None else mp.get_context("spawn")
        self._use_manager = bool(use_manager)
        self._manager = self._ctx.Manager() if use_manager else None
        if self._manager is None:
            self._request_queue: mp.Queue = self._ctx.Queue()
            self._response_queues: dict[int, mp.Queue] = {}
        else:
            self._request_queue = self._manager.Queue()
            self._response_queues = self._manager.dict()
        # Server-liveness flag baked into every client. It starts set
        # (optimistically alive) and is cleared by a process-backed server
        # owner when the server process exits, so blocked clients raise
        # MailboxPeerClosedError instead of waiting forever on a reply that
        # will never come.
        peer_alive = self._ctx.Event()
        peer_alive.set()
        self._set_peer_alive(peer_alive)

    def _make_response_queue(self) -> mp.Queue:
        if self._manager is not None:
            return self._manager.Queue()
        if self._use_manager:
            # The manager handle is dropped by __getstate__: an unpickled copy
            # cannot create manager-backed response queues, and falling back to
            # a raw mp.Queue would fail later (and confusingly) when registered
            # in the manager-backed response-queue dict.
            raise RuntimeError(
                "Cannot create a client from an unpickled manager-backed "
                "MPTransport: the manager handle is not serialized. Clients of "
                "a manager-backed MPTransport must be created with "
                "transport.client() in the owning process before "
                "pickling/sending the transport."
            )
        return self._ctx.Queue()

    def __getstate__(self) -> dict:
        state = super().__getstate__()
        state["_manager"] = None
        return state

    def close(self) -> None:
        """Release transport resources.

        Shuts down the multiprocessing manager backing the request/response
        queues when the transport was built with ``use_manager=True``
        (a no-op otherwise). The process that owns the transport must call
        this once the server and all clients are done with it:
        :class:`~torchrl.modules.inference_server.ProcessInferenceServer`
        does not close the transport on ``shutdown()``.
        """
        if self._manager is not None:
            self._manager.shutdown()

    def client(self) -> _QueueInferenceClient:
        """Create an actor-side client with a dedicated response queue.

        Must be called in the parent process **before** spawning children.
        In particular, a manager-backed transport (``use_manager=True``)
        loses its manager handle when pickled, so calling this on an
        unpickled copy raises a :class:`RuntimeError`.

        Returns:
            A :class:`_QueueInferenceClient` that can be passed to a child
            process as an argument to :class:`multiprocessing.Process`.
        """
        return super().client()
