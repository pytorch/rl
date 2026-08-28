# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import threading
import time
from concurrent.futures import Future

from tensordict.base import TensorDictBase

from torchrl.modules.inference_server._transport import InferenceTransport


class ThreadingTransport(InferenceTransport):
    """In-process transport for actors that are threads.

    Uses a shared list protected by a :class:`threading.Condition` as the
    request queue and :class:`~concurrent.futures.Future` objects for response
    routing.

    This is the simplest backend and is appropriate when all actors live in the
    same process (e.g. running in a :class:`~concurrent.futures.ThreadPoolExecutor`).
    """

    def __init__(self):
        self._queue: list[TensorDictBase] = []
        self._futures: list[Future] = []
        self._submitted_at: list[float] = []
        self._cond = threading.Condition(threading.Lock())

    def submit(self, td: TensorDictBase) -> Future[TensorDictBase]:
        """Enqueue a request and return a Future for the result."""
        fut: Future[TensorDictBase] = Future()
        with self._cond:
            self._queue.append(td)
            self._futures.append(fut)
            self._submitted_at.append(time.monotonic())
            self._cond.notify()
        return fut

    def drain(self, max_items: int) -> tuple[list[TensorDictBase], list[Future]]:
        """Dequeue up to *max_items* pending requests."""
        items, futs, _submitted_at = self.drain_with_timing(max_items)
        return items, futs

    def drain_with_timing(
        self, max_items: int
    ) -> tuple[list[TensorDictBase], list[Future], list[float]]:
        """Dequeue requests with actor-side submission timestamps."""
        with self._cond:
            n = min(len(self._queue), max_items)
            items = self._queue[:n]
            futs = self._futures[:n]
            submitted_at = self._submitted_at[:n]
            del self._queue[:n]
            del self._futures[:n]
            del self._submitted_at[:n]
        return items, futs, submitted_at

    def wait_for_work(self, timeout: float) -> None:
        """Block until at least one request is enqueued or *timeout* elapses."""
        with self._cond:
            if not self._queue:
                self._cond.wait(timeout=timeout)

    def resolve(self, callback: Future, result: TensorDictBase) -> None:
        """Set the result on the actor's Future."""
        callback.set_result(result)

    def resolve_exception(self, callback: Future, exc: BaseException) -> None:
        """Set an exception on the actor's Future."""
        callback.set_exception(exc)
