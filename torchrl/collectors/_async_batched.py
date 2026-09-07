# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import contextlib
import os
import queue
import threading
import time
from collections import deque, OrderedDict
from collections.abc import Callable, Iterator, Sequence
from typing import Literal

import torch
from tensordict import (
    lazy_stack,
    LazyStackedTensorDict,
    maybe_dense_stack,
    NestedKey,
    TensorDictBase,
)

from torchrl._comm import MailboxTransportError
from torchrl._utils import _maybe_record_function_decorator, logger as torchrl_logger
from torchrl.collectors._base import BaseCollector
from torchrl.envs import AsyncEnvPool, EnvBase
from torchrl.envs.async_envs import _validate_cpu_affinity
from torchrl.modules.inference_server import (
    InferenceDeviceConfig,
    InferenceServer,
    InferenceServerConfig,
    PolicyClientModule,
    ProcessInferenceServer,
    ThreadingTransport,
)
from torchrl.modules.inference_server._config import _resolve_device_config
from torchrl.modules.inference_server._transport import InferenceTransport

_ENV_IDX_KEY = "env_index"

_POLICY_BACKENDS = ("threading", "multiprocessing", "ray", "monarch")
_ENV_BACKENDS = ("threading", "multiprocessing")
_PauseRequest = tuple[threading.Barrier, threading.Event]


def _make_transport(
    policy_backend: str, num_slots: int | None = None
) -> InferenceTransport:
    """Create an :class:`InferenceTransport` from a backend name.

    Args:
        policy_backend: one of ``"threading"``, ``"multiprocessing"``,
            ``"ray"``, or ``"monarch"``.
        num_slots: when set and ``policy_backend="threading"``, a
            :class:`~torchrl.modules.SlotTransport` is created instead of
            the generic :class:`~torchrl.modules.ThreadingTransport`.
    """
    if policy_backend == "threading":
        if num_slots is not None:
            from torchrl.modules.inference_server._slot import SlotTransport

            return SlotTransport(num_slots)
        return ThreadingTransport()
    if policy_backend == "multiprocessing":
        from torchrl.modules.inference_server._mp import MPTransport

        return MPTransport()
    if policy_backend == "ray":
        from torchrl.modules.inference_server._ray import RayTransport

        return RayTransport()
    if policy_backend == "monarch":
        from torchrl.modules.inference_server._monarch import MonarchTransport

        return MonarchTransport()
    raise ValueError(
        f"Unknown policy_backend {policy_backend!r}. "
        f"Expected one of {_POLICY_BACKENDS}."
    )


def _wait_while_paused(pause_request: list[_PauseRequest | None]) -> None:
    """Park one coordinator and report when it reaches the pause boundary."""
    request = pause_request[0]
    if request is None:
        return
    barrier, resume_event = request
    try:
        barrier.wait()
    except threading.BrokenBarrierError:
        return
    resume_event.wait()


def _env_loop(
    pool: AsyncEnvPool,
    env_id: int,
    transport: InferenceTransport | None,
    client: Callable | None,
    result_queue: queue.Queue,
    shutdown_event: threading.Event,
    pause_request: list[_PauseRequest | None],
    env_device: torch.device | None,
    storing_device: torch.device | None,
):
    """Per-env worker thread using a pool slot and inference-server policy.

    Each thread owns one slot in the :class:`~torchrl.envs.AsyncEnvPool` and
    one inference client.  The pool handles the actual environment execution in
    whatever backend it was configured with (threading, multiprocessing, etc.),
    while this thread coordinates the send/recv cycle and inference submission.

        reset -> infer -> step_send -> step_recv -> put transition -> infer -> ...
    """
    if client is None:
        client = transport.client()

    try:
        pool.async_reset_send(env_index=env_id)
        obs = pool.async_reset_recv(env_index=env_id)

        while not shutdown_event.is_set():
            _wait_while_paused(pause_request)
            if shutdown_event.is_set():
                break

            action_td = client(obs)
            if env_device is not None:
                action_td = action_td.to(env_device)
            pool.async_step_and_maybe_reset_send(action_td, env_index=env_id)
            cur_td, obs = pool.async_step_and_maybe_reset_recv(env_index=env_id)
            cur_td.set(_ENV_IDX_KEY, env_id)
            if storing_device is not None:
                cur_td = cur_td.to(storing_device)
            result_queue.put(cur_td)
    except Exception as exc:
        if not shutdown_event.is_set():
            result_queue.put(exc)


def _env_ids(tensordict: TensorDictBase) -> list[int]:
    """Read one environment index for each item in the leading batch dimension."""
    values = tensordict.get(_ENV_IDX_KEY)
    if hasattr(values, "data") and not isinstance(values, torch.Tensor):
        values = values.data
    if hasattr(values, "tolist"):
        values = values.tolist()
    if not isinstance(values, list):
        values = [values]
    result = []
    for value in values:
        while isinstance(value, list) and len(value) == 1:
            value = value[0]
        result.append(int(value))
    return result


def _env_batch_loop(
    pool: AsyncEnvPool,
    clients: list[PolicyClientModule],
    result_queue: queue.Queue,
    shutdown_event: threading.Event,
    pause_request: list[_PauseRequest | None],
    env_device: torch.device | None,
    storing_device: torch.device | None,
):
    """Coordinate a shared-memory env pool from one thread in ready batches."""
    exchange_keys = pool.exchange_keys
    ready_observations = {}
    pending_actions = {}
    policy_outputs = {}
    stepping = 0
    poll_interval = 0.001

    try:
        env_ids = list(range(pool.num_envs))
        pool.async_reset_send(env_index=env_ids)
        resetting = pool.num_envs

        while not shutdown_event.is_set():
            if resetting:
                try:
                    observations = pool.async_reset_recv(
                        min_get=1, max_get=resetting, timeout=poll_interval
                    )
                except TimeoutError:
                    pass
                else:
                    reset_ids = _env_ids(observations)
                    resetting -= len(reset_ids)
                    ready_observations.update(zip(reset_ids, observations.unbind(0)))
            if pause_request[0] is None and ready_observations:
                for env_id, observation in ready_observations.items():
                    pending_actions[env_id] = clients[env_id].submit(observation)
                ready_observations.clear()

            if (
                pause_request[0] is not None
                and not resetting
                and not pending_actions
                and not stepping
            ):
                _wait_while_paused(pause_request)
                continue

            ready_ids = [
                env_id for env_id, future in pending_actions.items() if future.done()
            ]
            if ready_ids:
                env_inputs = []
                for env_id in ready_ids:
                    action = pending_actions.pop(env_id).result()
                    if env_device is not None:
                        action = action.to(env_device)
                    policy_outputs[env_id] = action
                    env_inputs.append(
                        action.select(*exchange_keys, strict=False)
                        if exchange_keys
                        else action
                    )
                pool.async_step_and_maybe_reset_send(
                    lazy_stack(env_inputs), env_index=ready_ids
                )
                stepping += len(ready_ids)

            if shutdown_event.is_set():
                break
            if not stepping:
                shutdown_event.wait(poll_interval)
                continue

            try:
                transitions, next_observations = pool.async_step_and_maybe_reset_recv(
                    min_get=1,
                    max_get=pool.num_envs,
                    timeout=poll_interval,
                )
            except TimeoutError:
                continue

            completed_ids = _env_ids(next_observations)
            stepping -= len(completed_ids)

            # Submit before copying transitions so policy inference overlaps
            # the shared-memory copy. During a pause, retain observations until
            # all in-flight inference and environment work has drained.
            for env_id, observation in zip(completed_ids, next_observations.unbind(0)):
                if pause_request[0] is None:
                    pending_actions[env_id] = clients[env_id].submit(observation)
                else:
                    ready_observations[env_id] = observation

            outputs = lazy_stack(
                [policy_outputs.pop(env_id) for env_id in completed_ids]
            )
            # Dense pool stacking already copied the transition tensors out of
            # shared memory before the slots can be reused. A lazy stack still
            # aliases the slots and must be cloned before actions are sent back.
            if isinstance(transitions, LazyStackedTensorDict):
                transitions = transitions.clone()
            transitions.update(outputs.exclude(*transitions.keys(True, True)))
            if storing_device is not None:
                transitions = transitions.to(storing_device)
            result_queue.put(transitions)
    except Exception as exc:
        if not shutdown_event.is_set():
            result_queue.put(exc)


class AsyncBatchedCollector(BaseCollector):
    """Asynchronous collector with env slots and a policy server.

    The collector pairs environment coordinators with an
    :class:`~torchrl.envs.AsyncEnvPool` and an
    :class:`~torchrl.modules.InferenceServer`.

    Unlike :class:`~torchrl.collectors.Collector`, this collector fully
    decouples environment stepping from policy inference:

    * An :class:`~torchrl.envs.AsyncEnvPool` runs *N* environments using
      whatever backend the user chooses (``"threading"``,
      ``"multiprocessing"``).
    * With a shared-memory exchange or grouped workers, one coordinator drains
      whichever environments are ready and submits their observations without
      blocking. Other exchanges use one lightweight coordinator thread per
      environment.
    * The :class:`~torchrl.modules.InferenceServer` running in a background
      thread continuously drains observation submissions, batches them, runs
      a single forward pass, and fans actions back out.

    There is **no global synchronisation barrier**: fast environments keep
    stepping while slow ones wait for inference, and the server always
    processes whatever observations have accumulated.

    The user simply provides env factories and a policy; the collector
    handles all wiring internally.

    Args:
        create_env_fn (list[Callable[[], EnvBase]]): a list of callables, each
            returning an :class:`~torchrl.envs.EnvBase` instance.  The list
            length determines the number of parallel environments.

    Keyword Args:
        policy (nn.Module or Callable, optional): the policy module.
            Mutually exclusive with ``policy_factory``.
        policy_factory (Callable[[], Callable], optional): a zero-argument
            callable that returns the policy.  Useful when the policy cannot
            be pickled.  Mutually exclusive with ``policy``.
        frames_per_batch (int): number of environment frames to collect per
            batch.  Required.
        total_frames (int, optional): total number of frames the collector
            should return during its lifespan.  ``-1`` means endless.
            Defaults to ``-1``.
        max_batch_size (int, optional): upper bound on the number of
            requests the inference server processes in a single forward pass.
            Defaults to ``64``.
        min_batch_size (int, optional): minimum number of requests the
            inference server accumulates before dispatching a batch.  After
            the first request arrives the server keeps draining for up to
            ``server_timeout`` seconds until this many items are collected.
            ``1`` (default) dispatches immediately.
        server_timeout (float, optional): seconds the server waits for work
            before dispatching a partial batch.  Defaults to ``0.01``.
        transport (InferenceTransport, optional): a pre-built transport
            object.  When provided, it takes precedence over
            ``policy_backend``.  When ``None`` (default) a transport is
            created automatically from the resolved ``policy_backend``.
        device (torch.device or str, optional): device for policy inference
            (shorthand for ``InferenceDeviceConfig(policy_device=...)``).
            Defaults to ``None``.
        server_config (InferenceServerConfig, optional): structured server
            configuration: execution ``backend`` (``"thread"`` runs the serve
            loop in this process, ``"process"`` a dedicated server process
            requiring ``policy_factory``), batching, optional static
            CUDA-graph execution, and stats settings.
            Mutually exclusive with the ``max_batch_size``,
            ``min_batch_size``, and ``server_timeout`` keyword arguments.
        device_config (InferenceDeviceConfig, optional): structured device
            placement (``policy_device``, ``output_device``, ``env_device``,
            ``storing_device``) for the whole collection pipeline. Mutually
            exclusive with ``device``.
        policy_version (int, optional): initial behavior-policy version
            attached to server outputs. Defaults to ``0``.
        policy_version_key (NestedKey or None, optional): TensorDict key used
            for behavior-policy version annotations. ``None`` disables
            annotations. Defaults to ``"policy_version"``.
        backend (str, optional): global default backend for both
            environments and policy inference.  Specific overrides
            ``env_backend`` and ``policy_backend`` take precedence when set.
            One of ``"threading"``, ``"multiprocessing"``, ``"ray"``, or
            ``"monarch"``.  Defaults to ``"threading"``.
        env_backend (str, optional): backend for the
            :class:`~torchrl.envs.AsyncEnvPool` that runs environments.  One
            of ``"threading"`` or ``"multiprocessing"``.  Falls back to
            ``backend`` when ``None``.  The coordinator threads are always
            Python threads regardless of this setting.  Defaults to ``None``.
        env_exchange (str, optional): data exchange of a multiprocessing
            :class:`~torchrl.envs.AsyncEnvPool`, one of ``"queue"``, ``"shm"``
            or ``"auto"``. The shared-memory exchange also enables batched
            coordination from one thread. Defaults to ``"queue"``.
        envs_per_worker (int, optional): Number of environments hosted by each
            multiprocessing worker. Grouped workers share one coordinator that
            drains ready environments without waiting for a complete group.
            Defaults to ``1``.
        policy_backend (str, optional): backend for the inference transport
            used to communicate with the
            :class:`~torchrl.modules.InferenceServer`.  One of
            ``"threading"``, ``"multiprocessing"``, ``"ray"``, or
            ``"monarch"``.  Falls back to ``backend`` when ``None``.
            Defaults to ``None``.
        reset_at_each_iter (bool, optional): whether to reset all envs at the
            start of every collection batch.  Defaults to ``False``.
        postproc (Callable, optional): post-processing transform applied to
            each collected batch before yielding.  Defaults to ``None``.
        yield_completed_trajectories (bool, optional): if ``True``, the
            collector yields individual completed trajectories as they finish
            rather than fixed-size batches.  ``frames_per_batch`` acts as the
            *minimum* number of frames to accumulate before yielding.
            The synchronous and multi-process collectors expose the same
            capability through the ``trajs_per_batch`` keyword argument (a
            trajectory count rather than a flag).
            Defaults to ``False``.
        weight_sync: an optional
            :class:`~torchrl.weight_update.WeightSyncScheme` forwarded to the
            inference server for receiving weight updates.
        weight_sync_model_id (str, optional): model id for weight sync.
            Defaults to ``"policy"``.
        verbose (bool, optional): if ``True``, log progress messages.
            Defaults to ``False``.
        create_env_kwargs (dict or list[dict], optional): keyword arguments
            forwarded to each environment factory.  A single dict is broadcast
            to all factories.
        worker_affinity (Sequence[Sequence[int]] or Callable[[int], Sequence[int]], optional):
            Optional Linux CPU placement forwarded to the multiprocessing
            :class:`~torchrl.envs.AsyncEnvPool`. Use it when worker scheduling
            on CPUs reserved for simulators or driver work causes contention or
            step-time jitter; most users should leave it unset. TorchRL knows
            which CPUs are available, but not the application's intended CPU
            partition or each environment's thread requirements, so it cannot
            choose these masks automatically. Provide one mask per worker process,
            or a callable mapping each worker index to its mask. Defaults
            to ``None``. See :ref:`async_batched_collector_cpu_affinity` for a
            complete collector example.
        driver_affinity (Sequence[int], optional): Linux CPU affinity mask for
            the inference-server and coordinator threads, plus
            parent-side multiprocessing queue feeder threads. Dedicated
            process-backed inference servers are not covered. The thread
            constructing the collector is restored to its original affinity
            after startup. Defaults to ``None``. See
            :ref:`async_batched_collector_cpu_affinity` for an example.

    Examples:
        >>> from torchrl.collectors import AsyncBatchedCollector
        >>> from torchrl.envs import GymEnv
        >>> from tensordict.nn import TensorDictModule
        >>> import torch.nn as nn
        >>> policy = TensorDictModule(
        ...     nn.Linear(4, 2), in_keys=["observation"], out_keys=["action"]
        ... )
        >>> collector = AsyncBatchedCollector(
        ...     create_env_fn=[lambda: GymEnv("CartPole-v1")] * 4,
        ...     policy=policy,
        ...     frames_per_batch=200,
        ...     total_frames=1000,
        ... )
        >>> for batch in collector:
        ...     print(batch.shape)
        ...     break
        >>> collector.shutdown()
    """

    def __init__(
        self,
        create_env_fn: list[Callable[[], EnvBase]],
        *,
        policy: Callable | None = None,
        policy_factory: Callable[[], Callable] | None = None,
        frames_per_batch: int,
        total_frames: int = -1,
        max_batch_size: int | None = None,
        min_batch_size: int | None = None,
        server_timeout: float | None = None,
        transport: InferenceTransport | None = None,
        device: torch.device | str | None = None,
        backend: Literal[
            "threading", "multiprocessing", "ray", "monarch"
        ] = "threading",
        env_backend: Literal["threading", "multiprocessing"] | None = None,
        env_exchange: Literal["queue", "shm", "auto"] = "queue",
        envs_per_worker: int = 1,
        policy_backend: (
            Literal["threading", "multiprocessing", "ray", "monarch"] | None
        ) = None,
        reset_at_each_iter: bool = False,
        postproc: Callable[[TensorDictBase], TensorDictBase] | None = None,
        yield_completed_trajectories: bool = False,
        weight_sync=None,
        weight_sync_model_id: str = "policy",
        verbose: bool = False,
        create_env_kwargs: dict | list[dict] | None = None,
        worker_affinity: Sequence[Sequence[int]]
        | Callable[[int], Sequence[int]]
        | None = None,
        driver_affinity: Sequence[int] | None = None,
        server_config: InferenceServerConfig | None = None,
        device_config: InferenceDeviceConfig | None = None,
        policy_version: int = 0,
        policy_version_key: NestedKey | None = "policy_version",
    ):
        if policy is not None and policy_factory is not None:
            raise TypeError("policy and policy_factory are mutually exclusive.")
        if policy is None and policy_factory is None:
            raise TypeError("One of policy or policy_factory must be provided.")
        if server_config is not None and any(
            kwarg is not None
            for kwarg in (max_batch_size, min_batch_size, server_timeout)
        ):
            raise ValueError(
                "server_config is mutually exclusive with the max_batch_size, "
                "min_batch_size, and server_timeout keyword arguments."
            )
        _server_defaults = (
            server_config if server_config is not None else InferenceServerConfig()
        )
        server_backend = _server_defaults.service_backend
        if server_backend == "process" and policy_factory is None:
            raise TypeError(
                "InferenceServerConfig(service_backend='process') requires "
                "policy_factory so the policy can be constructed inside the "
                "server process."
            )
        if max_batch_size is None:
            max_batch_size = _server_defaults.max_batch_size
        if min_batch_size is None:
            min_batch_size = _server_defaults.min_batch_size
        if server_timeout is None:
            server_timeout = _server_defaults.timeout
        _devices = _resolve_device_config(device_config, device=device)
        policy_device = _devices.policy_device
        output_device = _devices.output_device
        self._env_device = _devices.env_device
        self._storing_device = _devices.storing_device

        # ---- resolve policy ---------------------------------------------------
        self._policy_factory = policy_factory
        if policy_factory is not None and server_backend != "process":
            policy = policy_factory()
        self._policy = policy

        # ---- env config -------------------------------------------------------
        if not isinstance(create_env_fn, Sequence):
            raise TypeError("create_env_fn must be a list of env factories.")
        self._create_env_fn = list(create_env_fn)
        self._num_envs = len(create_env_fn)
        self._create_env_kwargs = create_env_kwargs

        # ---- resolve backends -------------------------------------------------
        effective_env_backend = env_backend if env_backend is not None else backend
        effective_policy_backend = (
            policy_backend if policy_backend is not None else backend
        )
        if effective_env_backend not in _ENV_BACKENDS:
            raise ValueError(
                f"env_backend={effective_env_backend!r} is not supported. "
                f"Expected one of {_ENV_BACKENDS}."
            )
        if worker_affinity is not None and effective_env_backend != "multiprocessing":
            raise ValueError(
                "worker_affinity is only supported with "
                "env_backend='multiprocessing'."
            )
        self._env_backend = effective_env_backend
        if env_exchange not in ("queue", "shm", "auto"):
            raise ValueError(
                f"env_exchange={env_exchange!r} is not supported. Expected one of "
                "('queue', 'shm', 'auto')."
            )
        if env_exchange == "shm" and effective_env_backend != "multiprocessing":
            raise ValueError(
                "env_exchange='shm' requires env_backend='multiprocessing'."
            )
        self._env_exchange = env_exchange
        if (
            isinstance(envs_per_worker, bool)
            or not isinstance(envs_per_worker, int)
            or envs_per_worker < 1
        ):
            raise ValueError(
                "envs_per_worker must be a positive integer, got "
                f"{envs_per_worker!r}."
            )
        if envs_per_worker != 1 and effective_env_backend != "multiprocessing":
            raise ValueError(
                "envs_per_worker is only supported with "
                "env_backend='multiprocessing'."
            )
        self._envs_per_worker = envs_per_worker
        if worker_affinity is None:
            self._worker_affinity = None
        else:
            num_workers = (self._num_envs + envs_per_worker - 1) // envs_per_worker
            if callable(worker_affinity):
                affinity_masks = [
                    worker_affinity(worker_index) for worker_index in range(num_workers)
                ]
            else:
                affinity_masks = list(worker_affinity)
                if len(affinity_masks) != num_workers:
                    raise ValueError(
                        "worker_affinity must provide one CPU mask per "
                        f"worker process, got {len(affinity_masks)} masks for "
                        f"{num_workers} worker processes."
                    )
            self._worker_affinity = [
                _validate_cpu_affinity(
                    affinity,
                    option_name=f"worker_affinity[{worker_index}]",
                )
                for worker_index, affinity in enumerate(affinity_masks)
            ]
        self._driver_affinity = (
            _validate_cpu_affinity(driver_affinity, option_name="driver_affinity")
            if driver_affinity is not None
            else None
        )
        self._server_backend = server_backend
        if server_backend == "process":
            if policy_backend not in (None, "multiprocessing"):
                raise ValueError(
                    "InferenceServerConfig(service_backend='process') requires "
                    "policy_backend=None or 'multiprocessing'."
                )
            effective_policy_backend = "multiprocessing"
        self._policy_backend = effective_policy_backend

        # ---- build transport --------------------------------------------------
        if transport is None:
            transport = _make_transport(
                effective_policy_backend, num_slots=self._num_envs
            )
        self._transport = transport

        # ---- build inference server -------------------------------------------
        if server_backend == "process":
            self._server = ProcessInferenceServer(
                policy_factory=policy_factory,
                transport=transport,
                max_batch_size=max_batch_size,
                static_batch_size=_server_defaults.static_batch_size,
                min_batch_size=min_batch_size,
                timeout=server_timeout,
                collate_fn=maybe_dense_stack,
                policy_device=policy_device,
                output_device=output_device,
                weight_sync=weight_sync,
                weight_sync_model_id=weight_sync_model_id,
                collect_stats=_server_defaults.collect_stats,
                stats_window_size=_server_defaults.stats_window_size,
                policy_version=policy_version,
                policy_version_key=policy_version_key,
            )
        else:
            self._server = InferenceServer(
                model=policy,
                transport=transport,
                max_batch_size=max_batch_size,
                static_batch_size=_server_defaults.static_batch_size,
                min_batch_size=min_batch_size,
                timeout=server_timeout,
                collate_fn=maybe_dense_stack,
                policy_device=policy_device,
                output_device=output_device,
                weight_sync=weight_sync,
                weight_sync_model_id=weight_sync_model_id,
                collect_stats=_server_defaults.collect_stats,
                stats_window_size=_server_defaults.stats_window_size,
                policy_version=policy_version,
                policy_version_key=policy_version_key,
            )
        self._policy_version_key = policy_version_key
        self._max_inflight_per_env = _server_defaults.max_inflight_per_env

        # ---- collector settings -----------------------------------------------
        self.requested_frames_per_batch = frames_per_batch
        self.frames_per_batch = frames_per_batch
        self.total_frames = total_frames
        self.reset_at_each_iter = reset_at_each_iter
        self.yield_completed_trajectories = yield_completed_trajectories
        self._postproc = postproc
        self.verbose = verbose

        self._frames = 0
        self._iter = -1

        # ---- runtime state (created lazily) -----------------------------------
        self._shutdown_event: threading.Event | None = None
        self._result_queue: queue.Queue | None = None
        self._env_pool: AsyncEnvPool | None = None
        self._workers: list[threading.Thread] = []
        self._clients: list[Callable] | None = None
        self._uses_batched_coordinator = False
        self._pause_request: list[_PauseRequest | None] = [None]
        self._pause_lock = threading.Lock()
        self._transition_carry: deque[TensorDictBase] = deque()

        # Per-env trajectory accumulators (for yield_completed_trajectories)
        self._yield_queues: list[deque] = [deque() for _ in range(self._num_envs)]
        self._trajectory_queue: deque = deque()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _ensure_started(self) -> None:
        """Create the env pool, start the server and coordinator threads."""
        if self._workers and all(w.is_alive() for w in self._workers):
            return

        original_affinity = None
        try:
            if self._driver_affinity is not None:
                original_affinity = os.sched_getaffinity(0)
                os.sched_setaffinity(0, self._driver_affinity)

            # Build the pool under the driver mask so feeder threads inherit it.
            kwargs = {}
            if self._create_env_kwargs is not None:
                kwargs["create_env_kwargs"] = self._create_env_kwargs
            self._env_pool = AsyncEnvPool(
                self._create_env_fn,
                backend=self._env_backend,
                envs_per_worker=self._envs_per_worker,
                # Explicit so the pool's default-change FutureWarning is not
                # emitted from library code.
                exchange=self._env_exchange,
                worker_affinity=self._worker_affinity,
                **kwargs,
            )

            # Create clients before a process server starts so response queues are
            # inherited by the child process.
            if self._clients is None:
                self._clients = [
                    PolicyClientModule(
                        self._transport.client(),
                        max_inflight=self._max_inflight_per_env,
                    )
                    for _ in range(self._num_envs)
                ]

            if self._server.static_batch_size is not None:
                request_spec = self._env_pool.fake_tensordict()[0]
                self._server.prepare_cudagraph(request_spec)

            # Start inference server
            if not self._server.is_alive:
                self._server.start()

            # Start coordinator threads. Shared slots can be drained safely in
            # ready batches, avoiding one Python thread and one clone per env.
            self._result_queue = queue.Queue()
            self._shutdown_event = threading.Event()

            self._workers = []
            if self._env_pool.resolved_exchange == "shm" or self._envs_per_worker > 1:
                self._uses_batched_coordinator = True
                thread = threading.Thread(
                    target=_env_batch_loop,
                    kwargs={
                        "pool": self._env_pool,
                        "clients": self._clients,
                        "result_queue": self._result_queue,
                        "shutdown_event": self._shutdown_event,
                        "pause_request": self._pause_request,
                        "env_device": self._env_device,
                        "storing_device": self._storing_device,
                    },
                    daemon=True,
                    name="AsyncBatchedCollector-env-batch",
                )
                self._workers.append(thread)
                thread.start()
                return

            self._uses_batched_coordinator = False
            for i in range(self._num_envs):
                t = threading.Thread(
                    target=_env_loop,
                    kwargs={
                        "pool": self._env_pool,
                        "env_id": i,
                        "transport": self._transport,
                        "client": self._clients[i],
                        "result_queue": self._result_queue,
                        "shutdown_event": self._shutdown_event,
                        "pause_request": self._pause_request,
                        "env_device": self._env_device,
                        "storing_device": self._storing_device,
                    },
                    daemon=True,
                    name=f"AsyncBatchedCollector-env-{i}",
                )
                self._workers.append(t)
                t.start()
        except Exception:
            self.shutdown(raise_on_error=False)
            raise
        finally:
            if original_affinity is not None:
                os.sched_setaffinity(0, original_affinity)

    @contextlib.contextmanager
    def pause(self, timeout: float | None = 30.0) -> Iterator[None]:
        """Pause environment coordination and policy inference.

        In-flight policy and environment requests finish before the context is
        entered. The coordinator threads then remain parked until the context
        exits, leaving the inference server idle. This provides a quiescent
        boundary for operations such as a lazy :func:`torch.compile` call.

        Compile and warm up modules before starting collection whenever
        possible. Use this context when compilation after collection has
        started is unavoidable.

        Args:
            timeout (float or None): maximum seconds to wait for the
                coordinator threads to pause. ``None`` waits indefinitely.
                Defaults to ``30.0``.

        Raises:
            RuntimeError: if another pause is active or a coordinator exits.
            TimeoutError: if the coordinator threads do not park within
                ``timeout`` seconds.
        """
        if not self._pause_lock.acquire(blocking=False):
            raise RuntimeError("AsyncBatchedCollector is already paused.")

        workers = tuple(self._workers)
        request = None
        try:
            if not workers:
                yield None
                return
            if not all(worker.is_alive() for worker in workers):
                raise RuntimeError(
                    "AsyncBatchedCollector cannot pause because a coordinator "
                    "thread has exited."
                )

            request = (threading.Barrier(len(workers) + 1), threading.Event())
            self._pause_request[0] = request
            try:
                request[0].wait(timeout=timeout)
            except threading.BrokenBarrierError:
                if not all(worker.is_alive() for worker in workers):
                    raise RuntimeError(
                        "AsyncBatchedCollector cannot pause because a "
                        "coordinator thread exited while pausing."
                    ) from None
                raise TimeoutError(
                    "Timed out while waiting for AsyncBatchedCollector "
                    "coordinator threads to pause."
                ) from None

            yield None
        finally:
            try:
                if request is not None:
                    if self._pause_request[0] is request:
                        self._pause_request[0] = None
                    request[0].abort()
                    request[1].set()
            finally:
                self._pause_lock.release()

    @property
    def env(self) -> AsyncEnvPool:
        """The underlying :class:`AsyncEnvPool`."""
        self._ensure_started()
        return self._env_pool

    @property
    def policy(self) -> Callable:
        """The policy passed to the inference server.

        With ``InferenceServerConfig(service_backend="process")`` the policy only exists inside the
        server process, so this returns the ``policy_factory`` instead.
        """
        if self._policy is not None:
            return self._policy
        return self._policy_factory

    def server_stats(self, *, reset: bool = False) -> dict[str, float | int]:
        """Return inference-server statistics when available."""
        stats = getattr(self._server, "stats", None)
        if stats is None:
            return {}
        return stats(reset=reset)

    @property
    def policy_version(self) -> int:
        """The live behavior-policy version of the inference server."""
        return self._server.policy_version

    # ------------------------------------------------------------------
    # Rollout: drain the result queue
    # ------------------------------------------------------------------

    _SERVER_DEATH_GRACE_S = 2.0

    def _check_worker_result(self, item):
        """Re-raise exceptions propagated from coordinator threads.

        Worker threads may observe a dying server before the liveness
        watchdog does: their transport read errors out first, and process
        teardown is asynchronous, so a killed server can still report alive
        for a moment. Mailbox transport failures identify a dead peer by
        construction and are attributed to the server immediately; other
        worker errors give liveness a short grace window so the failure is
        attributed to the dead server whenever that is the actual cause.
        """
        if isinstance(item, BaseException):
            if isinstance(item, MailboxTransportError):
                raise RuntimeError(
                    "The inference server died while the collector was "
                    "waiting for transitions. Check the server process "
                    "logs (e.g. OOM kills or exceptions in the policy)."
                ) from item
            deadline = time.monotonic() + self._SERVER_DEATH_GRACE_S
            while True:
                if not self._server.is_alive:
                    raise RuntimeError(
                        "The inference server died while the collector was "
                        "waiting for transitions. Check the server process "
                        "logs (e.g. OOM kills or exceptions in the policy)."
                    ) from item
                if time.monotonic() >= deadline:
                    break
                time.sleep(0.1)
            raise RuntimeError(
                "A collector worker thread raised an exception."
            ) from item

    _LIVENESS_POLL_S = 1.0

    def _next_result(self) -> TensorDictBase:
        """Block for the next transition, watching server and worker liveness.

        A dead inference server (e.g. an OOM-killed server process) would
        otherwise leave every coordinator thread blocked on a response and
        this method blocked on the queue, hanging the iterator forever with
        no error.
        """
        rq = self._result_queue
        while True:
            try:
                td = rq.get(timeout=self._LIVENESS_POLL_S)
            except queue.Empty:
                if not self._server.is_alive:
                    raise RuntimeError(
                        "The inference server died while the collector was "
                        "waiting for transitions. Check the server process "
                        "logs (e.g. OOM kills or exceptions in the policy)."
                    ) from None
                if self._workers and not any(w.is_alive() for w in self._workers):
                    raise RuntimeError(
                        "All collector worker threads exited while the "
                        "collector was waiting for transitions."
                    ) from None
                continue
            self._check_worker_result(td)
            return td

    @_maybe_record_function_decorator("AsyncBatchedCollector._rollout_frames")
    def _rollout_frames(self) -> TensorDictBase:
        """Drain ``frames_per_batch`` transitions from the workers."""
        rq = self._result_queue
        frames_to_collect = self.frames_per_batch
        if self.total_frames >= 0:
            frames_to_collect = min(frames_to_collect, self.total_frames - self._frames)
        collected = 0
        transitions: list[TensorDictBase] = []

        while self._transition_carry and collected < frames_to_collect:
            transition = self._transition_carry.popleft()
            transitions.append(transition)
            collected += transition.numel()

        while collected < frames_to_collect:
            # Block for at least one transition
            td = self._next_result()
            if self._uses_batched_coordinator:
                for transition in td.unbind(0):
                    if collected < frames_to_collect:
                        transitions.append(transition)
                        collected += transition.numel()
                    else:
                        self._transition_carry.append(transition)
            else:
                transitions.append(td)
                collected += td.numel()
            # Batch-drain any additional items already in the queue
            while collected < frames_to_collect:
                try:
                    td = rq.get_nowait()
                except queue.Empty:
                    break
                self._check_worker_result(td)
                if self._uses_batched_coordinator:
                    for transition in td.unbind(0):
                        if collected < frames_to_collect:
                            transitions.append(transition)
                            collected += transition.numel()
                        else:
                            self._transition_carry.append(transition)
                else:
                    transitions.append(td)
                    collected += td.numel()
            if self.verbose:
                torchrl_logger.debug(
                    f"AsyncBatchedCollector: {collected}/{self.frames_per_batch} frames"
                )

        return lazy_stack(transitions)

    @_maybe_record_function_decorator("AsyncBatchedCollector._rollout_yield_trajs")
    def _rollout_yield_trajs(self) -> TensorDictBase:
        """Drain transitions until a complete trajectory is available."""
        while not self._trajectory_queue:
            td = self._next_result()
            if self._uses_batched_coordinator:
                for transition in td.unbind(0):
                    self._record_trajectory_transition(transition)
            else:
                self._record_trajectory_transition(td)

        result = self._trajectory_queue.popleft()
        return result.reshape(-1)

    def _record_trajectory_transition(self, td: TensorDictBase) -> None:
        """Record one environment transition for trajectory yielding."""
        env_id = 0
        if td.get(_ENV_IDX_KEY, default=None) is not None:
            env_id = _env_ids(td)[0]

        self._yield_queues[env_id].append(td)
        if td["next", "done"].any():
            self._trajectory_queue.append(
                lazy_stack(list(self._yield_queues[env_id]), -1)
            )
            self._yield_queues[env_id].clear()

    @property
    def rollout(self) -> Callable[[], TensorDictBase]:
        if self.yield_completed_trajectories:
            return self._rollout_yield_trajs
        return self._rollout_frames

    # ------------------------------------------------------------------
    # BaseCollector interface
    # ------------------------------------------------------------------

    def iterator(self) -> Iterator[TensorDictBase]:
        """Iterate over collected batches."""
        self._ensure_started()

        total = self.total_frames
        while total < 0 or self._frames < total:
            self._iter += 1
            td = self.rollout()
            if not self.yield_completed_trajectories and total >= 0:
                remaining = total - self._frames
                if td.numel() > remaining:
                    td = td.reshape(-1)[:remaining]
            self._frames += td.numel()
            if self._postproc is not None:
                td = self._postproc(td)
            yield td

    def shutdown(
        self,
        timeout: float | None = None,
        close_env: bool = True,
        raise_on_error: bool = True,
    ) -> None:
        """Shut down the collector, inference server, threads and env pool."""
        if self._shutdown_event is not None:
            self._shutdown_event.set()
        request = self._pause_request[0]
        self._pause_request[0] = None
        if request is not None:
            request[0].abort()
            request[1].set()
        self._transition_carry.clear()
        _timeout = timeout or 5.0
        for w in self._workers:
            w.join(timeout=_timeout)
        self._workers = []
        self._server.shutdown(timeout=_timeout)
        if close_env and self._env_pool is not None:
            self._env_pool.close(raise_if_closed=raise_on_error)
            self._env_pool = None

    def set_seed(self, seed: int, static_seed: bool = False) -> int:
        """Set the seed (no-op; envs are created inside the pool)."""
        return seed

    def state_dict(self) -> OrderedDict:
        return OrderedDict()

    def load_state_dict(self, state_dict: OrderedDict) -> None:
        pass

    def __del__(self) -> None:
        if getattr(self, "_workers", None):
            try:
                self.shutdown(timeout=2.0, raise_on_error=False)
            except Exception:
                pass
