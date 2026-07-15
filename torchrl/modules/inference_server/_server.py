# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import contextlib
import importlib.util
import inspect
import multiprocessing as mp

import queue
import threading
import time
from collections.abc import Callable
from concurrent.futures import Future
from dataclasses import replace
from multiprocessing.synchronize import Event as MPEvent
from statistics import mean
from typing import Any, Literal

import torch
from tensordict import lazy_stack, TensorDict
from tensordict.base import TensorDictBase
from tensordict.nn.probabilistic import InteractionType, set_interaction_type
from tensordict.utils import NestedKey
from torch import nn

from torchrl._comm import CommandChannel, Mailbox, watch_process_liveness
from torchrl._comm.ray_runtime import _RayRuntimeLease, _set_ray_client_liveness
from torchrl.modules.inference_server._client import (
    _NO_INTERACTION_TYPE_CODE,
    _REMOTE_INTERACTION_TYPE_KEY,
)
from torchrl.modules.inference_server._config import (
    _resolve_device_config,
    InferenceDeviceConfig,
    InferenceServerConfig,
)
from torchrl.modules.inference_server._factory import (
    _inference_transport_kind,
    _make_inference_transport,
    _validate_inference_transport_selection,
)
from torchrl.modules.inference_server._threading import ThreadingTransport
from torchrl.modules.inference_server._transport import InferenceTransport
from torchrl.weight_update import (
    SharedMemWeightSyncScheme,
    WeightStrategy,
    WeightSyncScheme,
)
from torchrl.weight_update.utils import _weight_tensor_signature

_CODE_TO_INTERACTION_TYPE = {
    0: InteractionType.MODE,
    1: InteractionType.MEDIAN,
    2: InteractionType.MEAN,
    3: InteractionType.RANDOM,
    4: InteractionType.DETERMINISTIC,
}
_has_ray = importlib.util.find_spec("ray") is not None


class _InferenceServerMeta(type):
    """Select a private owner while keeping InferenceServer as the API."""

    def __call__(cls, *args, **kwargs):
        if cls is not InferenceServer:
            return super().__call__(*args, **kwargs)

        if len(args) > 2:
            raise TypeError(
                "InferenceServer accepts at most model and transport positionally."
            )
        model = args[0] if args else kwargs.pop("model", None)
        transport = args[1] if len(args) > 1 else kwargs.pop("transport", None)
        policy_factory = kwargs.pop("policy_factory", None)
        server_config = kwargs.get("server_config")
        configured_backend = (
            server_config.service_backend if server_config is not None else "thread"
        )
        service_backend = kwargs.pop("service_backend", None)
        if service_backend is None:
            service_backend = configured_backend
        elif server_config is not None and configured_backend not in (
            "thread",
            service_backend,
        ):
            raise ValueError(
                "service_backend conflicts with server_config.service_backend."
            )
        if service_backend not in ("thread", "process", "ray"):
            raise ValueError(
                "InferenceServer service_backend must be 'thread', 'process', or 'ray'."
            )

        service_options = dict(kwargs.pop("service_backend_options", None) or {})
        transport_options = kwargs.pop("transport_options", None)
        request_spec = kwargs.pop("request_spec", None)
        response_spec = kwargs.pop("response_spec", None)
        num_clients = kwargs.pop("num_clients", None)
        _validate_inference_transport_selection(
            transport,
            service_backend=service_backend,
            transport_options=transport_options,
            request_spec=request_spec,
            response_spec=response_spec,
        )

        if service_backend == "ray":
            if model is not None:
                raise ValueError(
                    "service_backend='ray' requires policy_factory so the policy "
                    "is constructed on the Ray actor."
                )
            if policy_factory is None:
                raise ValueError(
                    "policy_factory is required for service_backend='ray'."
                )
            return _RayInferenceServer(
                policy_factory=policy_factory,
                transport=transport,
                transport_options=transport_options,
                service_backend_options=service_options,
                request_spec=request_spec,
                response_spec=response_spec,
                num_clients=num_clients,
                **kwargs,
            )

        resolved_transport = _make_inference_transport(
            transport,
            service_backend=service_backend,
            transport_options=transport_options,
            request_spec=request_spec,
            response_spec=response_spec,
            num_clients=num_clients,
        )
        if service_backend == "process":
            if model is not None:
                raise ValueError(
                    "service_backend='process' requires policy_factory so the policy "
                    "is constructed in the child process."
                )
            if policy_factory is None:
                raise ValueError(
                    "policy_factory is required for service_backend='process'."
                )
            allowed_options = {"mp_context", "startup_timeout"}
            extra_options = set(service_options) - allowed_options
            if extra_options:
                raise ValueError(
                    "Unsupported process service_backend_options: "
                    f"{sorted(extra_options)}."
                )
            return ProcessInferenceServer(
                policy_factory=policy_factory,
                transport=resolved_transport,
                **service_options,
                **kwargs,
            )

        if service_options:
            raise ValueError(
                "service_backend_options are only valid for process or Ray services."
            )
        if model is None:
            if policy_factory is None:
                raise ValueError("Either model or policy_factory must be provided.")
            model = policy_factory()
        elif policy_factory is not None:
            raise ValueError("model and policy_factory are mutually exclusive.")
        return super().__call__(model, resolved_transport, **kwargs)


def _normalize_tensordict_device_metadata(data: TensorDictBase) -> TensorDictBase:
    target_device = data.device
    if target_device is not None:
        target_device = torch.device(target_device)
    for value in data.values(include_nested=True, leaves_only=True):
        value_device = getattr(value, "device", None)
        if value_device is None:
            continue
        value_device = torch.device(value_device)
        if target_device is None:
            target_device = value_device
        elif value_device != target_device:
            return data

    if target_device is None:
        return data

    needs_normalization = data.device != target_device
    if not needs_normalization:
        for value in data.values(include_nested=True, leaves_only=False):
            if isinstance(value, TensorDictBase) and value.device != target_device:
                needs_normalization = True
                break
    if not needs_normalization:
        return data

    data = data.copy()
    data.clear_device_()
    return data.to(target_device)


def _default_collate(items: list[TensorDictBase]) -> TensorDictBase:
    return lazy_stack(
        [
            _normalize_tensordict_device_metadata(item)
            if isinstance(item, TensorDictBase)
            else item
            for item in items
        ]
    )


class InferenceServer(metaclass=_InferenceServerMeta):
    """Auto-batching inference server.

    Actors submit individual TensorDicts via the *transport* and receive
    results asynchronously. A background worker drains the transport queue,
    batches inputs, runs the model, and fans results back to the callers.

    Args:
        model (nn.Module or Callable, optional): callable that maps a batched
            TensorDictBase to a batched TensorDictBase (e.g. a
            :class:`~tensordict.nn.TensorDictModule`). Pass ``policy_factory``
            instead when a process or Ray actor owns the policy.
        transport (InferenceTransport or str, optional): payload transport.
            ``"auto"`` selects a backend-appropriate transport and is the
            recommended default.

    Keyword Args:
        policy_factory (Callable, optional): zero-argument policy constructor.
            Required for ``service_backend="process"`` and
            ``service_backend="ray"`` so policy parameters are created by the
            process that owns them.
        service_backend (str, optional): where inference runs: ``"thread"``,
            ``"process"``, or ``"ray"``. Defaults to ``"thread"``.
        service_backend_options (dict, optional): owner configuration. The Ray
            backend accepts ``ray_init_config`` and ``remote_config``; the
            process backend accepts ``mp_context`` and ``startup_timeout``.
        transport_options (dict, optional): options forwarded to the selected
            transport. For ``"distributed"``, ``backend`` selects ``"gloo"``
            or ``"nccl"``. Explicit selectors never fall back to another
            transport.
        request_spec (TensorDictBase, optional): static request layout for
            shared-memory or process-owned distributed transports. Ray-owned
            distributed transports infer and bind this layout on first use.
        response_spec (TensorDictBase, optional): static response layout paired
            with ``request_spec``.
        num_clients (int, optional): expected concurrent client count for
            transports that allocate a fixed number of slots.
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
            configuration. Mutually exclusive with the ``max_batch_size``,
            ``min_batch_size``, ``timeout``, ``collect_stats``, and
            ``stats_window_size`` keyword arguments (passing any of them
            alongside a config raises, even when the value equals the
            default).
        device_config (InferenceDeviceConfig, optional): structured device
            placement configuration. Mutually exclusive with ``device``,
            ``policy_device``, and ``output_device``. The server consumes
            ``policy_device`` and ``output_device`` only; ``env_device`` is
            used as a fallback for ``output_device`` and ``storing_device``
            is rejected (it is a collector-level setting).
        policy_version (int, optional): initial behavior-policy version
            attached to inference outputs. Defaults to ``0``.
        policy_version_key (NestedKey or None, optional): TensorDict key used
            for behavior-policy version annotations. ``None`` disables
            annotations. Defaults to ``"policy_version"``.

    Example:
        >>> import torch
        >>> from tensordict import TensorDict
        >>> from tensordict.nn import TensorDictModule
        >>> from torchrl.modules.inference_server import InferenceServer
        >>> import torch.nn as nn
        >>> policy = TensorDictModule(
        ...     nn.Linear(4, 2), in_keys=["obs"], out_keys=["act"]
        ... )
        >>> with InferenceServer(policy, transport="auto", max_batch_size=8) as server:
        ...     result = server.client()(TensorDict({"obs": torch.randn(4)}))
        >>> result["act"].shape
        torch.Size([2])
    """

    def __init__(
        self,
        model: nn.Module | Callable[[TensorDictBase], TensorDictBase] | None = None,
        transport: InferenceTransport
        | Literal[
            "auto",
            "thread",
            "process",
            "ray",
            "shared_memory",
            "direct",
            "distributed",
        ] = "auto",
        *,
        policy_factory: Callable[
            [], nn.Module | Callable[[TensorDictBase], TensorDictBase]
        ]
        | None = None,
        service_backend: Literal["thread", "process", "ray"] = "thread",
        service_backend_options: dict[str, Any] | None = None,
        transport_options: dict[str, Any] | None = None,
        request_spec: TensorDictBase | None = None,
        response_spec: TensorDictBase | None = None,
        num_clients: int | None = None,
        max_batch_size: int | None = None,
        min_batch_size: int | None = None,
        timeout: float | None = None,
        collate_fn: Callable | None = None,
        device: torch.device | str | None = None,
        policy_device: torch.device | str | None = None,
        output_device: torch.device | str | None = None,
        collect_stats: bool | None = None,
        stats_window_size: int | None = None,
        weight_sync=None,
        weight_sync_model_id: str = "policy",
        server_config: InferenceServerConfig | None = None,
        device_config: InferenceDeviceConfig | None = None,
        shutdown_event: threading.Event | MPEvent | None = None,
        policy_version: int = 0,
        policy_version_key: NestedKey | None = "policy_version",
    ):
        # Deployment keywords are consumed by the metaclass before this local
        # implementation is constructed. Keeping them in the signature makes
        # the canonical API discoverable to help() and static tooling.
        del (
            policy_factory,
            service_backend,
            service_backend_options,
            transport_options,
            request_spec,
            response_spec,
            num_clients,
        )
        if model is None or not isinstance(transport, InferenceTransport):
            raise RuntimeError(
                "InferenceServer deployment arguments were not resolved before "
                "constructing the local server."
            )
        if server_config is not None and any(
            kwarg is not None
            for kwarg in (
                max_batch_size,
                min_batch_size,
                timeout,
                collect_stats,
                stats_window_size,
            )
        ):
            raise ValueError(
                "server_config is mutually exclusive with the max_batch_size, "
                "min_batch_size, timeout, collect_stats, and stats_window_size "
                "keyword arguments."
            )
        # Unset kwargs fall back to the (given or default) config values, so
        # the signature carries no duplicated default literals.
        _server_defaults = (
            server_config if server_config is not None else InferenceServerConfig()
        )
        if max_batch_size is None:
            max_batch_size = _server_defaults.max_batch_size
        if min_batch_size is None:
            min_batch_size = _server_defaults.min_batch_size
        if timeout is None:
            timeout = _server_defaults.timeout
        if collect_stats is None:
            collect_stats = _server_defaults.collect_stats
        if stats_window_size is None:
            stats_window_size = _server_defaults.stats_window_size
        _devices = _resolve_device_config(
            device_config,
            device=device,
            policy_device=policy_device,
            output_device=output_device,
            allow_storing_device=False,
        )
        self.model = model
        self.transport = transport
        self.max_batch_size = max_batch_size
        self.min_batch_size = min_batch_size
        self.timeout = timeout
        self.collate_fn = collate_fn if collate_fn is not None else _default_collate
        self.policy_device = _devices.policy_device
        self.device = self.policy_device
        self.output_device = _devices.output_device
        self.weight_sync = weight_sync
        self._weight_sync_model_id = weight_sync_model_id
        self._policy_version = int(policy_version)
        # Optional multiprocessing.Value mirror so a parent process can read
        # the live version of a server running in a child process.
        self._policy_version_shared = None
        self.policy_version_key = policy_version_key
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
        self._num_weight_updates = 0
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
                "policy_version": self._policy_version,
                "weight_updates": self._num_weight_updates,
            }
            if reset:
                self._reset_stats()
            return result

    @property
    def policy_version(self) -> int:
        """The current behavior-policy version served with inference outputs."""
        return self._policy_version

    def _mark_weight_update(self, model_version: int | None = None) -> None:
        with self._stats_lock:
            if model_version is None:
                self._policy_version += 1
            else:
                self._policy_version = int(model_version)
            self._num_weight_updates += 1
            if self._policy_version_shared is not None:
                self._policy_version_shared.value = self._policy_version

    def update_policy_weights_(self, model_id=None, policy_or_weights=None, **kwargs):
        """Weight-sync cascade hook: record an applied weight update.

        Weight-sync schemes cascade to their ``context`` after applying
        weights to the registered model. The server installs itself as the
        scheme context (when none is set) so that the policy version is
        bumped exactly when weights are actually applied -- including
        shared-memory schemes whose background receiver thread applies
        weights outside the server's polling loop.
        """
        self._mark_weight_update()

    def update_model(
        self,
        update_fn: Callable[[nn.Module], Any],
        *,
        mark_weight_update: bool = True,
    ) -> Any:
        """Apply an in-place update to the served model under the model lock.

        Args:
            update_fn (Callable): function called with ``self.model`` while
                inference is blocked by the server's model lock.
            mark_weight_update (bool, optional): if ``True``, increment the
                behavior-policy version and weight-update counter after
                ``update_fn`` succeeds. Defaults to ``True``.

        Returns:
            The value returned by ``update_fn``.
        """
        with self._model_lock:
            result = update_fn(self.model)
            if mark_weight_update:
                self._mark_weight_update()
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

    @property
    def service_backend(self) -> str:
        """Execution backend that owns the policy."""
        return "thread"

    @property
    def transport_kind(self) -> str:
        """Physical transport used for inference payloads."""
        return _inference_transport_kind(self.transport)

    def client(self) -> Any:
        """Return a restricted inference client from the owned transport."""
        return self.transport.client()

    def clients(self, num_clients: int) -> list[Any]:
        """Return one independently routed client per concurrent consumer."""
        if isinstance(num_clients, bool) or not isinstance(num_clients, int):
            raise TypeError("num_clients must be an integer.")
        if num_clients < 1:
            raise ValueError("num_clients must be at least 1.")
        return [self.client() for _ in range(num_clients)]

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
        # Ride the scheme's post-application cascade so the version is bumped
        # when weights are actually applied (see update_policy_weights_).
        if isinstance(ws, WeightSyncScheme) and ws.context is None:
            ws.context = self

    def _poll_weight_update(self) -> None:
        """Non-blocking check for fresh weights from the trainer."""
        ws = self.weight_sync
        if ws is None:
            return
        if isinstance(ws, SharedMemWeightSyncScheme):
            # Shared-memory schemes apply weights in place through a
            # background receiver thread started at connect() time; polling
            # receive() here would re-apply and re-count the same shared
            # buffer on every server iteration. Version bumps arrive through
            # the update_policy_weights_ cascade instead.
            return
        with self._model_lock:
            weights = ws.receive(timeout=0.0)
            if weights is not None and getattr(ws, "context", None) is not self:
                # When the server is the scheme context, receive() already
                # cascaded into update_policy_weights_; do not count twice.
                self._mark_weight_update()

    def _set_policy_version(self, result_batch: TensorDictBase) -> TensorDictBase:
        """Annotate inference outputs with the behavior policy version."""
        if self.policy_version_key is None:
            return result_batch
        device = result_batch.device
        if device is None:
            device = self.output_device or self.policy_device or torch.device("cpu")
        version = torch.full(
            result_batch.batch_size,
            self.policy_version,
            dtype=torch.long,
            device=device,
        )
        return result_batch.set(self.policy_version_key, version)

    def _interaction_type_context(self, batch: TensorDictBase):
        code = batch.get(_REMOTE_INTERACTION_TYPE_KEY, default=None)
        if code is None:
            return contextlib.nullcontext(), batch
        if not isinstance(code, torch.Tensor):
            interaction_code = int(code)
        else:
            flat_code = code.reshape(-1)
            if flat_code.numel() == 0:
                return contextlib.nullcontext(), batch.exclude(
                    _REMOTE_INTERACTION_TYPE_KEY, inplace=False
                )
            interaction_code = int(flat_code[0].item())
            if not flat_code.eq(interaction_code).all():
                raise RuntimeError(
                    "InferenceServer received a mixed interaction-type batch. "
                    "Use homogeneous server requests or a smaller max_batch_size."
                )
        batch = batch.exclude(_REMOTE_INTERACTION_TYPE_KEY, inplace=False)
        if interaction_code == _NO_INTERACTION_TYPE_CODE:
            # Sentinel: the caller had no active interaction context.
            return contextlib.nullcontext(), batch
        interaction_type_value = _CODE_TO_INTERACTION_TYPE[interaction_code]
        return set_interaction_type(interaction_type_value), batch

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
                            more_items, more_cbs, more_submitted_at = drain_with_timing(
                                self.max_batch_size - len(items)
                            )
                        items.extend(more_items)
                        callbacks.extend(more_cbs)
                        submitted_at.extend(more_submitted_at)

                try:
                    now = time.monotonic()
                    queue_wait_ms = [
                        (now - item_submitted_at) * 1000.0
                        for item_submitted_at in submitted_at
                        if item_submitted_at is not None
                    ]
                    batch = self.collate_fn(items)
                    if self.policy_device is not None:
                        batch = batch.to(self.policy_device)
                    forward_start = time.monotonic()
                    with self._model_lock:
                        interaction_context, batch = self._interaction_type_context(
                            batch
                        )
                        with interaction_context:
                            result_batch = self.model(batch)
                        if self.output_device is not None:
                            result_batch = result_batch.to(self.output_device)
                        result_batch = self._set_policy_version(result_batch)
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
        # getattr: __del__ also runs on instances whose __init__ raised
        # before attribute assignment (e.g. config-validation errors).
        worker = getattr(self, "_worker", None)
        if worker is not None and worker.is_alive():
            self.shutdown(timeout=1.0)


_inference_server_signature = inspect.signature(InferenceServer.__init__)
InferenceServer.__signature__ = _inference_server_signature.replace(
    parameters=tuple(_inference_server_signature.parameters.values())[1:]
)


def _process_server_entry(
    policy_factory: Callable[[], nn.Module],
    transport: InferenceTransport,
    server_kwargs: dict,
    shutdown_event: MPEvent,
    ready_queue,
    control_channel: CommandChannel,
    policy_version_value=None,
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
        # Mirror the live policy version into shared memory so the parent's
        # ProcessInferenceServer.policy_version stays accurate.
        server._policy_version_shared = policy_version_value
        server.start()
    except BaseException as exc:
        # The ready queue is only used for the startup handshake. Failures
        # after a successful handshake propagate through the child's exit
        # code instead, so that a restart cannot pick up a stale sentinel.
        ready_queue.put((False, repr(exc)))
        raise
    ready_queue.put((True, None))
    try:
        while not shutdown_event.is_set():
            if not server.is_alive:
                # The serve loop died (its own finally already drained and
                # rejected pending requests). Exit with an error so the
                # parent sees a dead process instead of a healthy control
                # plane fronting a dead server.
                raise RuntimeError(
                    "InferenceServer serve loop died inside the server process."
                )
            request = control_channel.receive(timeout=0.05)
            if request is None:
                continue
            verb = request.verb
            payload_in = request.payload
            try:
                if verb == "stats":
                    payload = server.stats(**payload_in)
                elif verb == "health":
                    payload = {
                        "alive": server.is_alive,
                        "policy_version": server.policy_version,
                    }
                elif verb == "update_model_weights":
                    weights = payload_in["weights"]
                    mark_weight_update = payload_in.get("mark_weight_update", True)
                    with server._model_lock:
                        if hasattr(server.model, "load_policy_weights"):
                            server.model.load_policy_weights(weights)
                        else:
                            WeightStrategy(extract_as="tensordict").apply_weights(
                                server.model,
                                weights.to(server.policy_device)
                                if server.policy_device is not None
                                else weights,
                            )
                        if mark_weight_update:
                            server._mark_weight_update()
                    payload = {"accepted": True}
                elif verb == "shutdown":
                    shutdown_event.set()
                    payload = {"accepted": True}
                else:
                    raise RuntimeError(f"Unknown process-server verb: {verb}")
            except Exception as exc:
                control_channel.reject(
                    request,
                    RuntimeError(
                        f"ProcessInferenceServer command {verb!r} failed: {exc!r}"
                    ),
                )
            else:
                control_channel.resolve(request, payload)
    finally:
        # Join the serve loop without a deadline: its shutdown path drains
        # and rejects pending requests, which must not be skipped even when
        # a slow forward pass is in flight (clients would hang forever).
        # The parent enforces the hard deadline via process.join/terminate.
        try:
            server.shutdown(timeout=None)
        finally:
            control_channel.close(
                RuntimeError("ProcessInferenceServer control channel closed.")
            )


class ProcessInferenceServer:
    """Dedicated-process wrapper around :class:`InferenceServer`.

    This server is intended for actor/env workers that communicate through a
    queue-based transport such as
    :class:`~torchrl.modules.inference_server.MPTransport`. The restricted
    client returned by :meth:`client` is created before the server process is
    spawned so its response queue is inherited safely.

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
            configuration. Mutually exclusive with the ``max_batch_size``,
            ``min_batch_size``, ``timeout``, ``collect_stats``, and
            ``stats_window_size`` keyword arguments.
        device_config (InferenceDeviceConfig, optional): structured device
            placement configuration. Mutually exclusive with ``device``,
            ``policy_device``, and ``output_device``. Same field subset as
            :class:`InferenceServer`: ``storing_device`` is rejected.
        policy_version (int, optional): initial behavior-policy version
            attached to inference outputs. Defaults to ``0``.
        policy_version_key (NestedKey or None, optional): TensorDict key used
            for behavior-policy version annotations. ``None`` disables
            annotations. Defaults to ``"policy_version"``.
        mp_context: multiprocessing context or start-method name. Defaults to
            ``"spawn"``.
        startup_timeout (float, optional): seconds :meth:`start` waits for the
            child process to build the policy and report readiness. Increase
            this when the policy factory loads a large checkpoint. Defaults to
            ``300.0``.

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
        >>> server = ProcessInferenceServer(
        ...     policy_factory=make_policy,
        ...     transport=transport,
        ...     mp_context=ctx,
        ... )
        >>> server.start()
        >>> client = server.client()
        >>> server.shutdown()
    """

    def __init__(
        self,
        *,
        policy_factory: Callable[[], nn.Module],
        transport: InferenceTransport,
        max_batch_size: int | None = None,
        min_batch_size: int | None = None,
        timeout: float | None = None,
        collate_fn: Callable | None = None,
        device: torch.device | str | None = None,
        policy_device: torch.device | str | None = None,
        output_device: torch.device | str | None = None,
        collect_stats: bool | None = None,
        stats_window_size: int | None = None,
        weight_sync=None,
        weight_sync_model_id: str = "policy",
        server_config: InferenceServerConfig | None = None,
        device_config: InferenceDeviceConfig | None = None,
        policy_version: int = 0,
        policy_version_key: NestedKey | None = "policy_version",
        mp_context: str | mp.context.BaseContext | None = None,
        startup_timeout: float = 300.0,
    ) -> None:
        if server_config is not None and any(
            kwarg is not None
            for kwarg in (
                max_batch_size,
                min_batch_size,
                timeout,
                collect_stats,
                stats_window_size,
            )
        ):
            raise ValueError(
                "server_config is mutually exclusive with the max_batch_size, "
                "min_batch_size, timeout, collect_stats, and stats_window_size "
                "keyword arguments."
            )
        _server_defaults = (
            server_config if server_config is not None else InferenceServerConfig()
        )
        if max_batch_size is None:
            max_batch_size = _server_defaults.max_batch_size
        if min_batch_size is None:
            min_batch_size = _server_defaults.min_batch_size
        if timeout is None:
            timeout = _server_defaults.timeout
        if collect_stats is None:
            collect_stats = _server_defaults.collect_stats
        if stats_window_size is None:
            stats_window_size = _server_defaults.stats_window_size
        _devices = _resolve_device_config(
            device_config,
            device=device,
            policy_device=policy_device,
            output_device=output_device,
            allow_storing_device=False,
        )
        self.policy_factory = policy_factory
        self.transport = transport
        self.startup_timeout = startup_timeout
        if isinstance(mp_context, str):
            self._ctx = mp.get_context(mp_context)
        elif mp_context is None:
            self._ctx = mp.get_context("spawn")
        else:
            self._ctx = mp_context
        # Server-liveness flag consulted by blocking client waits. Reuse the
        # transport's flag when it has one (MPTransport creates it eagerly so
        # clients created before this server also see it); otherwise create
        # one and attach it. A monitor thread clears it when the server
        # process exits so blocked clients raise MailboxPeerClosedError
        # instead of hanging forever.
        peer_alive = getattr(transport, "_peer_alive", None)
        if peer_alive is None:
            peer_alive = self._ctx.Event()
            peer_alive.set()
            transport._set_peer_alive(peer_alive)
        self._peer_alive = peer_alive
        self._process_monitor: threading.Thread | None = None
        self._service_client = transport.client()
        self._shutdown_event = self._ctx.Event()
        self._ready_queue = self._ctx.Queue()
        control_request_queue = self._ctx.Queue()
        control_mailbox = Mailbox(
            control_request_queue,
            self._ctx.Queue,
        )
        self._control_channel = CommandChannel(control_mailbox)
        self._control_client = self._control_channel.client()
        self._process: mp.Process | None = None
        self._server_kwargs = {
            "max_batch_size": max_batch_size,
            "min_batch_size": min_batch_size,
            "timeout": timeout,
            "collate_fn": collate_fn,
            # Devices are pre-resolved here; the child server's own resolution
            # is a no-op on these values.
            "policy_device": _devices.policy_device,
            "output_device": _devices.output_device,
            "collect_stats": collect_stats,
            "stats_window_size": stats_window_size,
            "weight_sync": weight_sync,
            "weight_sync_model_id": weight_sync_model_id,
            "policy_version": policy_version,
            "policy_version_key": policy_version_key,
        }
        # Live mirror of the child's policy version ("q" = signed 64-bit).
        self._policy_version_value = self._ctx.Value("q", int(policy_version))

    @property
    def policy_version(self) -> int:
        """The live behavior-policy version of the child server."""
        return int(self._policy_version_value.value)

    def start(self) -> ProcessInferenceServer:
        """Start the child process and wait until the policy is initialized."""
        if self.is_alive:
            raise RuntimeError("Server is already running.")
        previous_monitor = self._process_monitor
        if previous_monitor is not None:
            previous_monitor.join(timeout=self.startup_timeout)
            if previous_monitor.is_alive():
                raise RuntimeError(
                    "The previous ProcessInferenceServer monitor did not stop."
                )
            self._process_monitor = None
        self._shutdown_event.clear()
        self._peer_alive.set()
        self._process = self._ctx.Process(
            target=_process_server_entry,
            kwargs={
                "policy_factory": self.policy_factory,
                "transport": self.transport,
                "server_kwargs": self._server_kwargs,
                "shutdown_event": self._shutdown_event,
                "ready_queue": self._ready_queue,
                "control_channel": self._control_channel,
                "policy_version_value": self._policy_version_value,
            },
            daemon=True,
            name="ProcessInferenceServer",
        )
        self._process.start()
        self._process_monitor = threading.Thread(
            target=watch_process_liveness,
            args=(self._process.sentinel, self._peer_alive),
            daemon=True,
            name="ProcessInferenceServerMonitor",
        )
        self._process_monitor.start()
        try:
            ok, payload = self._ready_queue.get(timeout=self.startup_timeout)
        except queue.Empty:
            self.shutdown(timeout=1.0)
            raise TimeoutError(
                f"ProcessInferenceServer did not report readiness within "
                f"{self.startup_timeout} seconds. If the policy factory loads a "
                f"large checkpoint, increase startup_timeout."
            ) from None
        if not ok:
            self.shutdown(timeout=1.0)
            raise RuntimeError(f"ProcessInferenceServer failed to start: {payload}")
        return self

    def _request_control(
        self,
        verb: Literal["stats", "health", "update_model_weights", "shutdown"],
        payload: dict | None = None,
        timeout: float = 5.0,
    ):
        """One control-plane round trip: verb + payload out, reply back.

        Messages use the generic command-channel shape (request ``{"id",
        "verb", "payload"}``, reply ``{"id", "ok", "payload"}``) so this can
        later ride a shared CommandChannel abstraction unchanged.
        """
        if self._process is None:
            raise RuntimeError("ProcessInferenceServer is not running.")
        if not self._process.is_alive():
            raise RuntimeError(
                "ProcessInferenceServer process is not alive "
                f"(exitcode={self._process.exitcode})."
            )
        try:
            return self._control_client.call(verb, payload or {}, timeout=timeout)
        except queue.Empty:
            raise TimeoutError(
                f"Timed out waiting for ProcessInferenceServer {verb!r}."
            ) from None

    def shutdown(self, timeout: float | None = 5.0) -> None:
        """Signal the child process to stop and wait for it to exit."""
        if self.is_alive:
            try:
                self._request_control("shutdown", timeout=timeout or 5.0)
            except Exception:
                pass
        self._shutdown_event.set()
        process = self._process
        if process is None:
            return
        process.join(timeout=timeout)
        if process.is_alive():
            process.terminate()
            process.join(timeout=timeout)
        monitor = self._process_monitor
        if monitor is not None:
            monitor.join(timeout=timeout)
            if not monitor.is_alive():
                self._process_monitor = None
        self._process = None

    @property
    def is_alive(self) -> bool:
        """Whether the child process is alive."""
        return self._process is not None and self._process.is_alive()

    @property
    def service_backend(self) -> str:
        """Execution backend that owns the policy."""
        return "process"

    @property
    def transport_kind(self) -> str:
        """Physical transport used for inference payloads."""
        return _inference_transport_kind(self.transport)

    def client(self) -> Any:
        """Return a restricted inference client from the owned transport."""
        return self._service_client

    def clients(self, num_clients: int) -> list[Any]:
        """Return one independently routed client per concurrent consumer."""
        if isinstance(num_clients, bool) or not isinstance(num_clients, int):
            raise TypeError("num_clients must be an integer.")
        if num_clients < 1:
            raise ValueError("num_clients must be at least 1.")
        # MPTransport routes replies per client, so reserve a fresh endpoint.
        return [self.transport.client() for _ in range(num_clients)]

    def stats(
        self, *, reset: bool = False, timeout: float = 5.0
    ) -> dict[str, float | int]:
        """Return process-server stats from the child process.

        This is a blocking control-plane round trip: it can take up to
        ``timeout`` seconds and raises :class:`TimeoutError` when the child
        does not answer in time, or :class:`RuntimeError` when the child is
        not running.

        Args:
            reset (bool, optional): if ``True``, reset counters in the child
                process after taking the snapshot.
            timeout (float, optional): seconds to wait for the child's
                answer. Defaults to ``5.0``.
        """
        return self._request_control("stats", {"reset": reset}, timeout=timeout)

    def update_model_weights(
        self,
        weights: TensorDictBase,
        *,
        mark_weight_update: bool = True,
        timeout: float = 300.0,
    ) -> dict[str, bool]:
        """Apply TensorDict weights to the model hosted by the child process.

        This is a blocking control-plane round trip; large models can take a
        while to transfer and apply, hence the generous default timeout.

        Args:
            weights (TensorDictBase): weights to apply to the child's model.
            mark_weight_update (bool, optional): whether to bump the child's
                behavior-policy version. Defaults to ``True``.
            timeout (float, optional): seconds to wait for the child to apply
                the weights. Defaults to ``300.0``.
        """
        return self._request_control(
            "update_model_weights",
            {"weights": weights, "mark_weight_update": mark_weight_update},
            timeout=timeout,
        )

    def health(self, *, timeout: float = 5.0) -> dict[str, int | bool | None]:
        """Return a lightweight child-process health snapshot.

        Never raises on a dead or unresponsive child; degraded fields are
        reported in the returned dictionary instead (``process_alive`` /
        ``control_error``), so this is safe to call from monitoring loops.

        Args:
            timeout (float, optional): seconds to wait for the child's
                answer. Defaults to ``5.0``.
        """
        process = self._process
        result = {
            "process_alive": process.is_alive() if process is not None else False,
            "pid": process.pid if process is not None else None,
            "exitcode": process.exitcode if process is not None else None,
        }
        if process is not None and process.is_alive():
            try:
                result.update(self._request_control("health", timeout=timeout))
            except (RuntimeError, TimeoutError) as exc:
                # The child may have died between the liveness check and the
                # control round trip; a health probe reports that instead of
                # raising.
                result["process_alive"] = (
                    process.is_alive() if process is not None else False
                )
                result["exitcode"] = process.exitcode if process is not None else None
                result["control_error"] = repr(exc)
        return result

    def __enter__(self) -> ProcessInferenceServer:
        return self.start()

    def __exit__(self, *exc_info) -> None:
        self.shutdown()

    def __del__(self) -> None:
        # getattr: __del__ also runs on instances whose __init__ raised
        # before attribute assignment (e.g. config-validation errors).
        process = getattr(self, "_process", None)
        if process is not None and process.is_alive():
            self.shutdown(timeout=1.0)


class _RayInferenceServerActor:
    """Ray actor that owns a policy, transport, and local inference loop."""

    def __init__(
        self,
        policy_factory: Callable[[], nn.Module],
        transport: InferenceTransport | str | None,
        transport_options: dict[str, Any] | None,
        request_spec: TensorDictBase | None,
        response_spec: TensorDictBase | None,
        num_clients: int | None,
        server_kwargs: dict[str, Any],
    ) -> None:
        self.model = policy_factory()
        self._transport = transport
        self._transport_options = transport_options
        self._num_clients = num_clients
        self._bootstrap_lock = threading.Lock()
        config = server_kwargs.get("server_config")
        if config is not None and config.service_backend != "thread":
            server_kwargs["server_config"] = replace(config, service_backend="thread")
        self._server_kwargs = server_kwargs
        self._receiver_schemes: dict[str, WeightSyncScheme] = {}
        self.server = None
        if request_spec is not None or response_spec is not None:
            if request_spec is None or response_spec is None:
                raise ValueError(
                    "request_spec and response_spec must be provided together."
                )
            self._start_server(request_spec, response_spec)
        elif transport != "distributed":
            resolved_transport = _make_inference_transport(
                transport,
                service_backend="ray",
                transport_options=transport_options,
                request_spec=None,
                response_spec=None,
                num_clients=num_clients,
            )
            self.server = InferenceServer(
                self.model,
                resolved_transport,
                service_backend="thread",
                **server_kwargs,
            ).start()

    def _start_server(
        self, request_spec: TensorDictBase, response_spec: TensorDictBase
    ) -> None:
        resolved_transport = _make_inference_transport(
            self._transport,
            service_backend="ray",
            transport_options=self._transport_options,
            request_spec=request_spec,
            response_spec=response_spec,
            num_clients=self._num_clients,
        )
        self.server = InferenceServer(
            self.model,
            resolved_transport,
            service_backend="thread",
            **self._server_kwargs,
        ).start()

    def bootstrap_distributed(self, request: TensorDictBase):
        """Bind the distributed layout on the first endpoint generation."""
        with self._bootstrap_lock:
            if self.server is None:
                # Reuse InferenceServer's collation, placement, and versioning
                # rules to derive the reply layout without publishing a second
                # public configuration surface.
                probe = InferenceServer(
                    self.model,
                    ThreadingTransport(),
                    service_backend="thread",
                    **self._server_kwargs,
                )
                # Collation may return a lazy stack backed by the request. A
                # TensorDictModule then writes its output keys into that stack,
                # which must not widen the request schema used by the transport.
                batch = probe.collate_fn([request.clone()])
                if probe.policy_device is not None:
                    batch = batch.to(probe.policy_device)
                interaction_context, batch = probe._interaction_type_context(batch)
                with interaction_context:
                    response = self.model(batch)
                if probe.output_device is not None:
                    response = response.to(probe.output_device)
                response = probe._set_policy_version(response).unbind(0)[0]
                self._start_server(request, response)
            return self.server.client()

    def client(self):
        if self.server is None:
            raise RuntimeError(
                "The distributed transport has not established its schema yet."
            )
        return self.server.client()

    def stats(self, reset: bool = False) -> dict[str, float | int]:
        if self.server is None:
            return {"num_requests": 0, "num_batches": 0}
        return self.server.stats(reset=reset)

    def health(self) -> dict[str, Any]:
        return {
            "is_alive": self.server is None or self.server.is_alive,
            "policy_version": (
                int(self._server_kwargs.get("policy_version", 0))
                if self.server is None
                else self.server.policy_version
            ),
            "transport": (
                "distributed"
                if self.server is None
                else _inference_transport_kind(self.server.transport)
            ),
        }

    def update_model_weights(
        self, weights: TensorDictBase, mark_weight_update: bool = True
    ) -> None:
        if self.server is None:
            weights.to_module(self.model)
            if mark_weight_update:
                self._server_kwargs["policy_version"] = (
                    int(self._server_kwargs.get("policy_version", 0)) + 1
                )
        else:
            self.server.update_model(
                lambda model: weights.to_module(model),
                mark_weight_update=mark_weight_update,
            )

    def register_scheme_receiver(
        self,
        weight_recv_schemes: dict[str, WeightSyncScheme],
        *,
        synchronize_weights: bool = True,
    ) -> None:
        """Install restricted weight receivers for learner-rank publication."""
        for model_id, scheme in weight_recv_schemes.items():
            previous_scheme = self._receiver_schemes.get(model_id)
            if previous_scheme is not None and previous_scheme is not scheme:
                previous_scheme.shutdown()
            scheme.init_on_receiver(
                model_id=model_id,
                worker_idx=0,
                model=self.model,
            )
            self._receiver_schemes[model_id] = scheme
        if synchronize_weights:
            for scheme in self._receiver_schemes.values():
                scheme.connect(worker_idx=0)

    def _weight_sync_signature(
        self, model_id: str
    ) -> tuple[tuple[tuple[str, ...], tuple[int, ...], str], ...]:
        """Return the ordered tensor schema expected by this service."""
        if model_id != "policy":
            raise KeyError(f"Unknown inference model_id {model_id!r}.")
        weights = TensorDict.from_module(self.model)
        return _weight_tensor_signature(weights)

    def _receive_weights_scheme(self, model_version: int | None = None) -> None:
        if not self._receiver_schemes:
            raise RuntimeError("No inference weight receiver is configured.")
        if self.server is None:
            for scheme in self._receiver_schemes.values():
                scheme.receive()
            current_version = int(self._server_kwargs.get("policy_version", 0))
            self._server_kwargs["policy_version"] = (
                current_version + 1 if model_version is None else model_version
            )
        else:
            with self.server._model_lock:
                for scheme in self._receiver_schemes.values():
                    scheme.receive()
                self.server._mark_weight_update(model_version)

    def _connect_weights_scheme(self, model_version: int | None = None) -> None:
        if not self._receiver_schemes:
            raise RuntimeError("No inference weight receiver is configured.")
        if self.server is None:
            for scheme in self._receiver_schemes.values():
                if not scheme.synchronized_on_receiver:
                    scheme.connect(worker_idx=0)
            current_version = int(self._server_kwargs.get("policy_version", 0))
            self._server_kwargs["policy_version"] = (
                current_version + 1 if model_version is None else model_version
            )
        else:
            with self.server._model_lock:
                for scheme in self._receiver_schemes.values():
                    if not scheme.synchronized_on_receiver:
                        scheme.connect(worker_idx=0)
                self.server._mark_weight_update(model_version)

    def shutdown(self) -> None:
        for scheme in self._receiver_schemes.values():
            scheme.shutdown()
        self._receiver_schemes.clear()
        if self.server is not None:
            self.server.shutdown(timeout=None)
            close = getattr(self.server.transport, "close", None)
            if close is not None:
                close()


class _RayDistributedInferenceClient:
    """Restricted client that lazily binds a distributed TensorDict schema."""

    def __init__(self, actor) -> None:
        self._actor = actor
        self._client = None

    def __call__(
        self, payload: TensorDictBase, timeout: float | None = None
    ) -> TensorDictBase:
        if self._client is None:
            import ray

            self._client = ray.get(self._actor.bootstrap_distributed.remote(payload))
            _set_ray_client_liveness(self._client, self._actor)
        return self._client(payload, timeout=timeout)


class _RayInferenceServer(InferenceServer):
    """Private Ray owner returned by ``InferenceServer`` dispatch."""

    def __init__(
        self,
        *,
        policy_factory: Callable[[], nn.Module],
        transport: InferenceTransport | str | None = "auto",
        transport_options: dict[str, Any] | None = None,
        service_backend_options: dict[str, Any] | None = None,
        request_spec: TensorDictBase | None = None,
        response_spec: TensorDictBase | None = None,
        num_clients: int | None = None,
        **server_kwargs,
    ) -> None:
        if not _has_ray:
            raise ImportError("Ray is required for service_backend='ray'.")
        import ray

        options = dict(service_backend_options or {})
        ray_init_config = options.pop("ray_init_config", None)
        remote_config = dict(options.pop("remote_config", None) or {})
        if options:
            raise ValueError(
                f"Unsupported Ray service_backend_options: {sorted(options)}."
            )
        if remote_config.get("num_gpus", 0):
            if (
                server_kwargs.get("device") is None
                and server_kwargs.get("policy_device") is None
                and server_kwargs.get("device_config") is None
            ):
                server_kwargs["policy_device"] = "cuda:0"
            if (
                server_kwargs.get("output_device") is None
                and server_kwargs.get("device_config") is None
            ):
                server_kwargs["output_device"] = "cpu"

        self._runtime_lease = _RayRuntimeLease.acquire(ray_init_config)
        self._actor = None
        self._lazy_distributed = (
            transport == "distributed"
            and request_spec is None
            and response_spec is None
        )
        self._distributed = transport == "distributed"
        try:
            actor_cls = ray.remote(**remote_config)(_RayInferenceServerActor)
            self._actor = actor_cls.remote(
                policy_factory,
                transport,
                transport_options,
                request_spec,
                response_spec,
                num_clients,
                server_kwargs,
            )
            ray.get(self._actor.health.remote())
        except BaseException:
            if self._actor is not None:
                with contextlib.suppress(Exception):
                    ray.kill(self._actor, no_restart=True)
                self._actor = None
            self._runtime_lease.release()
            raise

    @property
    def service_backend(self) -> str:
        return "ray"

    @property
    def is_alive(self) -> bool:
        actor = getattr(self, "_actor", None)
        if actor is None:
            return False
        import ray

        try:
            return bool(ray.get(actor.health.remote())["is_alive"])
        except Exception:
            return False

    @property
    def policy_version(self) -> int:
        return int(self.health()["policy_version"])

    @property
    def transport_kind(self) -> str:
        return str(self.health()["transport"])

    def start(self) -> _RayInferenceServer:
        if not self.is_alive:
            raise RuntimeError("The Ray inference server is not alive.")
        return self

    def client(self):
        if self._actor is None:
            raise RuntimeError("The Ray inference server is closed.")
        if self._lazy_distributed:
            return _RayDistributedInferenceClient(self._actor)
        import ray

        client = ray.get(self._actor.client.remote())
        if self._distributed:
            _set_ray_client_liveness(client, self._actor)
        return client

    def clients(self, num_clients: int) -> list[Any]:
        if isinstance(num_clients, bool) or not isinstance(num_clients, int):
            raise TypeError("num_clients must be an integer.")
        if num_clients < 1:
            raise ValueError("num_clients must be at least 1.")
        return [self.client() for _ in range(num_clients)]

    def stats(self, *, reset: bool = False) -> dict[str, float | int]:
        if self._actor is None:
            raise RuntimeError("The Ray inference server is closed.")
        import ray

        return ray.get(self._actor.stats.remote(reset))

    def health(self) -> dict[str, Any]:
        if self._actor is None:
            return {"is_alive": False}
        import ray

        return ray.get(self._actor.health.remote())

    def update_model_weights(
        self,
        weights: TensorDictBase,
        *,
        mark_weight_update: bool = True,
    ) -> None:
        if self._actor is None:
            raise RuntimeError("The Ray inference server is closed.")
        import ray

        ray.get(self._actor.update_model_weights.remote(weights, mark_weight_update))

    def update_policy_weights_(
        self, model_id=None, policy_or_weights=None, **kwargs
    ) -> None:
        del kwargs
        if (
            policy_or_weights is None
            and model_id is not None
            and not isinstance(model_id, str)
        ):
            policy_or_weights = model_id
        if isinstance(policy_or_weights, nn.Module):
            policy_or_weights = TensorDict.from_module(policy_or_weights).data
        if policy_or_weights is None:
            raise ValueError("Policy weights must be provided.")
        self.update_model_weights(policy_or_weights)

    def shutdown(self, timeout: float | None = 5.0) -> None:
        actor = getattr(self, "_actor", None)
        if actor is None:
            return
        import ray

        try:
            ray.get(actor.shutdown.remote(), timeout=timeout)
        except Exception:
            pass
        try:
            ray.kill(actor, no_restart=True)
        except Exception:
            pass
        self._actor = None
        self._runtime_lease.release()

    close = shutdown

    def __enter__(self) -> _RayInferenceServer:
        return self.start()

    def __exit__(self, *exc_info) -> None:
        self.shutdown()

    def __del__(self) -> None:
        if getattr(self, "_actor", None) is not None:
            with contextlib.suppress(BaseException):
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
