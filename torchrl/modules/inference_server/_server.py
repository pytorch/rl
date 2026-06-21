# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import threading
import time
from collections.abc import Callable
from concurrent.futures import Future
import multiprocessing as mp
from multiprocessing.synchronize import Event as MPEvent
from statistics import mean

import torch
from tensordict import lazy_stack
from tensordict.base import TensorDictBase
from torch import nn

from torchrl.modules.inference_server._config import (
    InferenceDeviceConfig,
    InferenceServerConfig,
)
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
        min_batch_size (int, optional): minimum number of requests to
            accumulate before dispatching a batch.  After the first request
            arrives the server keeps draining for up to ``timeout`` seconds
            until at least this many items are collected.  ``1`` (default)
            dispatches immediately.
        timeout (float, optional): seconds to wait for new work before
            dispatching a partial batch. Default: ``0.01``.
        collate_fn (Callable, optional): function used to stack a list of
            TensorDicts into a batch. Default: :func:`~tensordict.lazy_stack`.
        device (torch.device or str, optional): device to move batches to
            before calling the model. This is kept as an alias for
            ``policy_device`` for backward compatibility. ``None`` means no
            device transfer.
        policy_device (torch.device or str, optional): device that owns the
            policy and receives batched requests before model execution.
            If omitted, ``device`` is used.
        output_device (torch.device or str, optional): device where individual
            inference results are moved before being returned to actors. This
            is useful when a CUDA policy serves CPU environment workers.
        collect_stats (bool, optional): if ``True``, collect lightweight
            batching, queue-wait, and forward-latency statistics. Defaults to
            ``True``.
        stats_window_size (int, optional): number of recent timing samples
            kept for percentile statistics. Defaults to ``1024``.
        weight_sync: an optional
            :class:`~torchrl.weight_update.WeightSyncScheme` used to receive
            updated model weights from a trainer. When set, the server polls
            for new weights between inference batches.
        weight_sync_model_id (str, optional): the model identifier used when
            initialising the weight sync scheme on the receiver side.
            Default: ``"policy"``.
        server_config (InferenceServerConfig, optional): structured server
            configuration. Mutually exclusive with non-default batching and
            stats keyword arguments.
        device_config (InferenceDeviceConfig, optional): structured device
            placement configuration. Mutually exclusive with ``device``,
            ``policy_device``, and ``output_device``.

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
        min_batch_size: int = 1,
        timeout: float = 0.01,
        collate_fn: Callable | None = None,
        device: torch.device | str | None = None,
        policy_device: torch.device | str | None = None,
        output_device: torch.device | str | None = None,
        collect_stats: bool = True,
        stats_window_size: int = 1024,
        weight_sync=None,
        weight_sync_model_id: str = "policy",
        server_config: InferenceServerConfig | None = None,
        device_config: InferenceDeviceConfig | None = None,
        shutdown_event: threading.Event | MPEvent | None = None,
    ):
        if server_config is not None:
            if (
                max_batch_size,
                min_batch_size,
                timeout,
                collect_stats,
                stats_window_size,
            ) != (64, 1, 0.01, True, 1024):
                raise ValueError(
                    "server_config is mutually exclusive with non-default "
                    "batching and stats keyword arguments."
                )
            max_batch_size = server_config.max_batch_size
            min_batch_size = server_config.min_batch_size
            timeout = server_config.timeout
            collect_stats = server_config.collect_stats
            stats_window_size = server_config.stats_window_size
        if device_config is not None:
            if (
                device is not None
                or policy_device is not None
                or output_device is not None
            ):
                raise ValueError(
                    "device_config is mutually exclusive with device, "
                    "policy_device, and output_device."
                )
            policy_device = device_config.policy_device
            output_device = device_config.server_output_device()
        self.model = model
        self.transport = transport
        self.max_batch_size = max_batch_size
        self.min_batch_size = min_batch_size
        self.timeout = timeout
        self.collate_fn = collate_fn if collate_fn is not None else lazy_stack
        policy_device = device if policy_device is None else policy_device
        self.policy_device = (
            torch.device(policy_device) if policy_device is not None else None
        )
        self.device = self.policy_device
        self.output_device = (
            torch.device(output_device) if output_device is not None else None
        )
        self.weight_sync = weight_sync
        self._weight_sync_model_id = weight_sync_model_id
        self.collect_stats = collect_stats
        self.stats_window_size = stats_window_size

        self._shutdown_event = (
            threading.Event() if shutdown_event is None else shutdown_event
        )
        self._worker: threading.Thread | None = None
        # Protects model access during weight updates
        self._model_lock = threading.Lock()
        self._stats_lock = threading.Lock()
        self._reset_stats()

        if self.policy_device is not None and hasattr(self.model, "to"):
            self.model.to(self.policy_device)

    # -- stats ---------------------------------------------------------------

    def _reset_stats(self) -> None:
        self._stats_started_at = time.monotonic()
        self._num_requests = 0
        self._num_batches = 0
        self._batch_sizes: list[int] = []
        self._queue_wait_ms: list[float] = []
        self._forward_ms: list[float] = []

    @staticmethod
    def _percentile(values: list[float], percentile: float) -> float:
        if not values:
            return 0.0
        sorted_values = sorted(values)
        index = int(round((len(sorted_values) - 1) * percentile))
        return float(sorted_values[index])

    def _extend_window(self, target: list, values: list) -> None:
        target.extend(values)
        excess = len(target) - self.stats_window_size
        if excess > 0:
            del target[:excess]

    def _record_batch_stats(
        self,
        *,
        batch_size: int,
        queue_wait_ms: list[float],
        forward_ms: float,
    ) -> None:
        if not self.collect_stats:
            return
        with self._stats_lock:
            self._num_requests += batch_size
            self._num_batches += 1
            self._extend_window(self._batch_sizes, [batch_size])
            self._extend_window(self._queue_wait_ms, queue_wait_ms)
            self._extend_window(self._forward_ms, [forward_ms])

    def stats(self, *, reset: bool = False) -> dict[str, float | int]:
        """Return lightweight inference-server throughput statistics.

        Args:
            reset (bool, optional): if ``True``, clear counters after taking
                the snapshot. Defaults to ``False``.

        Returns:
            A dictionary with request/batch counts, rates, average batch size,
            and p50/p95 queue and forward latencies in milliseconds.
        """
        with self._stats_lock:
            elapsed = max(time.monotonic() - self._stats_started_at, 1e-12)
            num_requests = self._num_requests
            num_batches = self._num_batches
            batch_sizes = list(self._batch_sizes)
            queue_wait_ms = list(self._queue_wait_ms)
            forward_ms = list(self._forward_ms)
            result = {
                "requests": num_requests,
                "batches": num_batches,
                "requests_per_s": num_requests / elapsed,
                "batches_per_s": num_batches / elapsed,
                "avg_batch_size": float(mean(batch_sizes)) if batch_sizes else 0.0,
                "p50_queue_ms": self._percentile(queue_wait_ms, 0.50),
                "p95_queue_ms": self._percentile(queue_wait_ms, 0.95),
                "p50_forward_ms": self._percentile(forward_ms, 0.50),
                "p95_forward_ms": self._percentile(forward_ms, 0.95),
            }
            if reset:
                self._reset_stats()
            return result

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

    def _init_weight_sync(self) -> None:
        """Initialise the weight sync scheme on the receiver (server) side."""
        ws = self.weight_sync
        if ws is None:
            return
        if not ws.initialized_on_receiver:
            ws.init_on_receiver(
                model_id=self._weight_sync_model_id,
                model=self.model,
                worker_idx=0,
            )
        if not ws.synchronized_on_receiver:
            ws.connect(worker_idx=0)

    def _poll_weight_update(self) -> None:
        """Non-blocking check for fresh weights from the trainer."""
        ws = self.weight_sync
        if ws is None:
            return
        with self._model_lock:
            ws.receive(timeout=0.0)

    @torch.no_grad()
    def _run(self) -> None:
        self._init_weight_sync()

        try:
            while not self._shutdown_event.is_set():
                self._poll_weight_update()

                self.transport.wait_for_work(timeout=self.timeout)

                drain_with_timing = getattr(self.transport, "drain_with_timing", None)
                if drain_with_timing is None:
                    items, callbacks = self.transport.drain(self.max_batch_size)
                    submitted_at = [None] * len(items)
                else:
                    items, callbacks, submitted_at = drain_with_timing(
                        self.max_batch_size
                    )
                if not items:
                    continue

                # Accumulate up to min_batch_size (or until timeout expires)
                if len(items) < self.min_batch_size:
                    deadline = time.monotonic() + self.timeout
                    while len(items) < self.min_batch_size:
                        remaining = deadline - time.monotonic()
                        if remaining <= 0:
                            break
                        self.transport.wait_for_work(timeout=remaining)
                        if drain_with_timing is None:
                            more_items, more_cbs = self.transport.drain(
                                self.max_batch_size - len(items)
                            )
                            more_submitted_at = [None] * len(more_items)
                        else:
                            more_items, more_cbs, more_submitted_at = (
                                drain_with_timing(self.max_batch_size - len(items))
                            )
                        items.extend(more_items)
                        callbacks.extend(more_cbs)
                        submitted_at.extend(more_submitted_at)

                batch = self.collate_fn(items)
                if self.policy_device is not None:
                    batch = batch.to(self.policy_device)

                try:
                    now = time.monotonic()
                    queue_wait_ms = [
                        (now - item_submitted_at) * 1000.0
                        for item_submitted_at in submitted_at
                        if item_submitted_at is not None
                    ]
                    forward_start = time.monotonic()
                    with self._model_lock:
                        result_batch = self.model(batch)
                    if self.output_device is not None:
                        result_batch = result_batch.to(self.output_device)
                    forward_ms = (time.monotonic() - forward_start) * 1000.0
                    self._record_batch_stats(
                        batch_size=len(callbacks),
                        queue_wait_ms=queue_wait_ms,
                        forward_ms=forward_ms,
                    )
                    results = result_batch.unbind(0)
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


def _process_server_entry(
    policy_factory: Callable[[], nn.Module],
    transport: InferenceTransport,
    server_kwargs: dict,
    shutdown_event: MPEvent,
    ready_queue,
) -> None:
    """Run an :class:`InferenceServer` loop inside a child process."""
    try:
        model = policy_factory()
        server = InferenceServer(
            model=model,
            transport=transport,
            shutdown_event=shutdown_event,
            **server_kwargs,
        )
        ready_queue.put((True, None))
        server._run()
    except BaseException as exc:
        ready_queue.put((False, repr(exc)))
        raise


class ProcessInferenceServer:
    """Dedicated-process wrapper around :class:`InferenceServer`.

    This server is intended for actor/env workers that communicate through a
    queue-based transport such as
    :class:`~torchrl.modules.inference_server.MPTransport`. Clients must be
    created from the transport before :meth:`start` so that the child process
    inherits their response queues.

    Args:
        policy_factory (Callable[[], nn.Module]): picklable factory that creates
            the policy inside the server process.
        transport (InferenceTransport): transport shared with actor clients.

    Keyword Args:
        max_batch_size (int, optional): maximum requests per forward pass.
        min_batch_size (int, optional): minimum requests to accumulate before
            dispatching a partial batch.
        timeout (float, optional): wait timeout in seconds.
        collate_fn (Callable, optional): collate function for requests.
        device (torch.device or str, optional): alias for ``policy_device``.
        policy_device (torch.device or str, optional): policy execution device.
        output_device (torch.device or str, optional): actor response device.
        collect_stats (bool, optional): forwarded to :class:`InferenceServer`.
        stats_window_size (int, optional): forwarded to :class:`InferenceServer`.
        weight_sync: optional weight synchronization scheme.
        weight_sync_model_id (str, optional): model id for weight sync.
        server_config (InferenceServerConfig, optional): structured server
            configuration. Mutually exclusive with non-default batching and
            stats keyword arguments.
        device_config (InferenceDeviceConfig, optional): structured device
            placement configuration. Mutually exclusive with ``device``,
            ``policy_device``, and ``output_device``.
        mp_context: multiprocessing context or start-method name. Defaults to
            ``"spawn"``.

    Examples:
        >>> import multiprocessing as mp
        >>> import torch.nn as nn
        >>> from tensordict.nn import TensorDictModule
        >>> from torchrl.modules.inference_server import MPTransport
        >>> def make_policy():
        ...     return TensorDictModule(
        ...         nn.Linear(4, 2), in_keys=["observation"], out_keys=["action"]
        ...     )
        >>> ctx = mp.get_context("spawn")
        >>> transport = MPTransport(ctx=ctx)
        >>> client = transport.client()
        >>> server = ProcessInferenceServer(
        ...     policy_factory=make_policy,
        ...     transport=transport,
        ...     mp_context=ctx,
        ... )
        >>> server.start()
        >>> server.shutdown()
    """

    def __init__(
        self,
        *,
        policy_factory: Callable[[], nn.Module],
        transport: InferenceTransport,
        max_batch_size: int = 64,
        min_batch_size: int = 1,
        timeout: float = 0.01,
        collate_fn: Callable | None = None,
        device: torch.device | str | None = None,
        policy_device: torch.device | str | None = None,
        output_device: torch.device | str | None = None,
        collect_stats: bool = True,
        stats_window_size: int = 1024,
        weight_sync=None,
        weight_sync_model_id: str = "policy",
        server_config: InferenceServerConfig | None = None,
        device_config: InferenceDeviceConfig | None = None,
        mp_context: str | mp.context.BaseContext | None = None,
    ) -> None:
        if server_config is not None:
            if (
                max_batch_size,
                min_batch_size,
                timeout,
                collect_stats,
                stats_window_size,
            ) != (64, 1, 0.01, True, 1024):
                raise ValueError(
                    "server_config is mutually exclusive with non-default "
                    "batching and stats keyword arguments."
                )
            max_batch_size = server_config.max_batch_size
            min_batch_size = server_config.min_batch_size
            timeout = server_config.timeout
            collect_stats = server_config.collect_stats
            stats_window_size = server_config.stats_window_size
        if device_config is not None:
            if (
                device is not None
                or policy_device is not None
                or output_device is not None
            ):
                raise ValueError(
                    "device_config is mutually exclusive with device, "
                    "policy_device, and output_device."
                )
            policy_device = device_config.policy_device
            output_device = device_config.server_output_device()
        self.policy_factory = policy_factory
        self.transport = transport
        if isinstance(mp_context, str):
            self._ctx = mp.get_context(mp_context)
        elif mp_context is None:
            self._ctx = mp.get_context("spawn")
        else:
            self._ctx = mp_context
        self._shutdown_event = self._ctx.Event()
        self._ready_queue = self._ctx.Queue()
        self._process: mp.Process | None = None
        self._server_kwargs = {
            "max_batch_size": max_batch_size,
            "min_batch_size": min_batch_size,
            "timeout": timeout,
            "collate_fn": collate_fn,
            "device": device,
            "policy_device": policy_device,
            "output_device": output_device,
            "collect_stats": collect_stats,
            "stats_window_size": stats_window_size,
            "weight_sync": weight_sync,
            "weight_sync_model_id": weight_sync_model_id,
        }

    def start(self) -> ProcessInferenceServer:
        """Start the child process and wait until the policy is initialized."""
        if self.is_alive:
            raise RuntimeError("Server is already running.")
        self._shutdown_event.clear()
        self._process = self._ctx.Process(
            target=_process_server_entry,
            kwargs={
                "policy_factory": self.policy_factory,
                "transport": self.transport,
                "server_kwargs": self._server_kwargs,
                "shutdown_event": self._shutdown_event,
                "ready_queue": self._ready_queue,
            },
            daemon=True,
            name="ProcessInferenceServer",
        )
        self._process.start()
        ok, payload = self._ready_queue.get(timeout=30.0)
        if not ok:
            self.shutdown(timeout=1.0)
            raise RuntimeError(f"ProcessInferenceServer failed to start: {payload}")
        return self

    def shutdown(self, timeout: float | None = 5.0) -> None:
        """Signal the child process to stop and wait for it to exit."""
        self._shutdown_event.set()
        process = self._process
        if process is None:
            return
        process.join(timeout=timeout)
        if process.is_alive():
            process.terminate()
            process.join(timeout=timeout)
        self._process = None

    @property
    def is_alive(self) -> bool:
        """Whether the child process is alive."""
        return self._process is not None and self._process.is_alive()

    def stats(self, *, reset: bool = False) -> dict[str, float | int]:
        """Return process-server stats.

        Live stats are not shared across processes yet, so this currently
        returns an empty dictionary.
        """
        return {}

    def __enter__(self) -> ProcessInferenceServer:
        return self.start()

    def __exit__(self, *exc_info) -> None:
        self.shutdown()

    def __del__(self) -> None:
        if self.is_alive:
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
