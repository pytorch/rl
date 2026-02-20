# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import threading
from collections.abc import Callable
from concurrent.futures import Future

import torch
from tensordict import lazy_stack
from tensordict.base import TensorDictBase
from torch import nn

from torchrl.modules.inference_server._transport import InferenceTransport


class InferenceServer:
    """Auto-batching inference server.

    Actors submit individual TensorDicts via the *transport* and receive
    results asynchronously. A background worker drains the transport queue,
    batches inputs, runs the model, and fans results back to the callers.

    Args:
        model (nn.Module or Callable): a callable that maps a batched
            TensorDictBase to a batched TensorDictBase (e.g. a
            :class:`~tensordict.nn.TensorDictModule`).
        transport (InferenceTransport): the communication backend.

    Keyword Args:
        max_batch_size (int, optional): upper bound on the number of requests
            processed in a single forward pass. Default: ``64``.
        timeout (float, optional): seconds to wait for new work before
            dispatching a partial batch. Default: ``0.01``.
        collate_fn (Callable, optional): function used to stack a list of
            TensorDicts into a batch. Default: :func:`~tensordict.lazy_stack`.
        device (torch.device or str, optional): device to move batches to
            before calling the model. ``None`` means no device transfer.
        weight_sync: an optional
            :class:`~torchrl.weight_update.WeightSyncScheme` used to receive
            updated model weights from a trainer. When set, the server polls
            for new weights between inference batches.

    Example:
        >>> from tensordict.nn import TensorDictModule
        >>> from torchrl.modules.inference_server import (
        ...     InferenceServer,
        ...     ThreadingTransport,
        ... )
        >>> import torch.nn as nn
        >>> policy = TensorDictModule(
        ...     nn.Linear(4, 2), in_keys=["obs"], out_keys=["act"]
        ... )
        >>> transport = ThreadingTransport()
        >>> server = InferenceServer(policy, transport, max_batch_size=8)
        >>> server.start()
        >>> client = transport.client()
        >>> # client(td) can now be called from any thread
        >>> server.shutdown()
    """

    def __init__(
        self,
        model: nn.Module,
        transport: InferenceTransport,
        *,
        max_batch_size: int = 64,
        timeout: float = 0.01,
        collate_fn: Callable | None = None,
        device: torch.device | str | None = None,
        weight_sync=None,
    ):
        self.model = model
        self.transport = transport
        self.max_batch_size = max_batch_size
        self.timeout = timeout
        self.collate_fn = collate_fn if collate_fn is not None else lazy_stack
        self.device = torch.device(device) if device is not None else None
        self.weight_sync = weight_sync

        self._shutdown_event = threading.Event()
        self._worker: threading.Thread | None = None

    # -- lifecycle ------------------------------------------------------------

    def start(self) -> InferenceServer:
        """Start the background inference loop.

        Returns:
            self, for fluent chaining.
        """
        if self._worker is not None and self._worker.is_alive():
            raise RuntimeError("Server is already running.")
        self._shutdown_event.clear()
        self._worker = threading.Thread(
            target=self._run, daemon=True, name="InferenceServer-worker"
        )
        self._worker.start()
        return self

    def shutdown(self, timeout: float | None = 5.0) -> None:
        """Signal the background worker to stop and wait for it to finish.

        Args:
            timeout (float or None): seconds to wait for the worker thread to
                join. ``None`` waits indefinitely.
        """
        self._shutdown_event.set()
        if self._worker is not None:
            self._worker.join(timeout=timeout)
            self._worker = None

    @property
    def is_alive(self) -> bool:
        """Whether the background worker thread is running."""
        return self._worker is not None and self._worker.is_alive()

    # -- background loop ------------------------------------------------------

    @torch.no_grad()
    def _run(self) -> None:
        try:
            while not self._shutdown_event.is_set():
                self.transport.wait_for_work(timeout=self.timeout)

                items, callbacks = self.transport.drain(self.max_batch_size)
                if not items:
                    continue

                batch = self.collate_fn(items)
                if self.device is not None:
                    batch = batch.to(self.device)

                try:
                    results = self.model(batch).unbind(0)
                    if len(results) != len(callbacks):
                        raise RuntimeError(
                            f"Model returned {len(results)} results for a "
                            f"batch of {len(callbacks)} inputs."
                        )
                    for cb, res in zip(callbacks, results):
                        self.transport.resolve(cb, res)
                except Exception as exc:
                    for cb in callbacks:
                        self.transport.resolve_exception(cb, exc)
        finally:
            self._drain_pending_on_shutdown()

    def _drain_pending_on_shutdown(self) -> None:
        """Resolve all pending requests with an error during shutdown."""
        shutdown_exc = RuntimeError("InferenceServer is shutting down.")
        while True:
            items, callbacks = self.transport.drain(self.max_batch_size)
            if not items:
                break
            for cb in callbacks:
                self.transport.resolve_exception(cb, shutdown_exc)

    # -- context manager ------------------------------------------------------

    def __enter__(self) -> InferenceServer:
        return self.start()

    def __exit__(self, *exc_info) -> None:
        self.shutdown()

    def __del__(self) -> None:
        if self._worker is not None and self._worker.is_alive():
            self.shutdown(timeout=1.0)


class InferenceClient:
    """Actor-side handle for an :class:`InferenceServer`.

    Wraps a transport's :meth:`~InferenceTransport.submit` so that calling
    ``client(td)`` looks like a regular synchronous policy call, while the
    actual computation is batched on the server.

    Args:
        transport (InferenceTransport): the transport shared with the server.

    Example:
        >>> client = transport.client()
        >>> td_out = client(td_in)          # blocking
        >>> future = client.submit(td_in)   # non-blocking
        >>> td_out = future.result()
    """

    def __init__(self, transport: InferenceTransport):
        self._transport = transport

    def __call__(self, td: TensorDictBase) -> TensorDictBase:
        """Submit a request and block until the result is ready."""
        return self._transport.submit(td).result()

    def submit(self, td: TensorDictBase) -> Future[TensorDictBase]:
        """Submit a request and return a Future immediately."""
        return self._transport.submit(td)
