# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import queue
import threading
from collections import deque, OrderedDict
from collections.abc import Callable, Iterator, Sequence
from typing import Literal

import torch
from tensordict import lazy_stack, TensorDictBase

from torchrl._utils import logger as torchrl_logger
from torchrl.collectors._base import BaseCollector
from torchrl.envs import AsyncEnvPool, EnvBase
from torchrl.modules.inference_server import InferenceServer, ThreadingTransport
from torchrl.modules.inference_server._transport import InferenceTransport

_ENV_IDX_KEY = "env_index"

_POLICY_BACKENDS = ("threading", "multiprocessing", "ray", "monarch")
_ENV_BACKENDS = ("threading", "multiprocessing")


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


def _env_loop(
    pool: AsyncEnvPool,
    env_id: int,
    transport: InferenceTransport,
    result_queue: queue.Queue,
    shutdown_event: threading.Event,
):
    """Per-env worker thread using pool slot for env execution and InferenceServer for policy.

    Each thread owns one slot in the :class:`~torchrl.envs.AsyncEnvPool` and
    one inference client.  The pool handles the actual environment execution in
    whatever backend it was configured with (threading, multiprocessing, etc.),
    while this thread coordinates the send/recv cycle and inference submission.

        reset -> infer (blocking) -> step_send -> step_recv -> put transition -> infer -> ...
    """
    client = transport.client()

    try:
        pool.async_reset_send(env_index=env_id)
        obs = pool.async_reset_recv(env_index=env_id)
        action_td = client(obs)

        while not shutdown_event.is_set():
            pool.async_step_and_maybe_reset_send(action_td, env_index=env_id)
            cur_td, next_obs = pool.async_step_and_maybe_reset_recv(env_index=env_id)
            cur_td.set(_ENV_IDX_KEY, env_id)
            result_queue.put(cur_td)
            if shutdown_event.is_set():
                break
            action_td = client(next_obs)
    except Exception as exc:
        if not shutdown_event.is_set():
            result_queue.put(exc)


class AsyncBatchedCollector(BaseCollector):
    """Asynchronous collector that pairs per-env threads with an :class:`~torchrl.envs.AsyncEnvPool` and an :class:`~torchrl.modules.InferenceServer`.

    Unlike :class:`~torchrl.collectors.Collector`, this collector fully
    decouples environment stepping from policy inference:

    * An :class:`~torchrl.envs.AsyncEnvPool` runs *N* environments using
      whatever backend the user chooses (``"threading"``,
      ``"multiprocessing"``).
    * *N* lightweight coordinator threads -- one per environment -- each own
      a slot in the pool and an inference client.  A thread sends its env's
      observation to the :class:`~torchrl.modules.InferenceServer`, blocks
      until the batched action is returned, then sends the action back to
      the pool for stepping.
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
        device (torch.device or str, optional): device for policy inference.
            Passed to the inference server.  Defaults to ``None``.
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
        max_batch_size: int = 64,
        min_batch_size: int = 1,
        server_timeout: float = 0.01,
        transport: InferenceTransport | None = None,
        device: torch.device | str | None = None,
        backend: Literal[
            "threading", "multiprocessing", "ray", "monarch"
        ] = "threading",
        env_backend: Literal["threading", "multiprocessing"] | None = None,
        policy_backend: Literal["threading", "multiprocessing", "ray", "monarch"]
        | None = None,
        reset_at_each_iter: bool = False,
        postproc: Callable[[TensorDictBase], TensorDictBase] | None = None,
        yield_completed_trajectories: bool = False,
        weight_sync=None,
        weight_sync_model_id: str = "policy",
        verbose: bool = False,
        create_env_kwargs: dict | list[dict] | None = None,
    ):
        if policy is not None and policy_factory is not None:
            raise TypeError("policy and policy_factory are mutually exclusive.")
        if policy is None and policy_factory is None:
            raise TypeError("One of policy or policy_factory must be provided.")

        # ---- resolve policy ---------------------------------------------------
        if policy_factory is not None:
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
        self._env_backend = effective_env_backend
        self._policy_backend = effective_policy_backend

        # ---- build transport --------------------------------------------------
        if transport is None:
            transport = _make_transport(
                effective_policy_backend, num_slots=self._num_envs
            )
        self._transport = transport

        # ---- build inference server -------------------------------------------
        self._server = InferenceServer(
            model=policy,
            transport=transport,
            max_batch_size=max_batch_size,
            min_batch_size=min_batch_size,
            timeout=server_timeout,
            device=device,
            weight_sync=weight_sync,
            weight_sync_model_id=weight_sync_model_id,
        )

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

        # Per-env trajectory accumulators (for yield_completed_trajectories)
        self._yield_queues: list[deque] = [deque() for _ in range(self._num_envs)]
        self._trajectory_queue: deque = deque()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _ensure_started(self) -> None:
        """Create the env pool, start the server and per-env threads."""
        if self._workers and all(w.is_alive() for w in self._workers):
            return

        # Build env pool
        kwargs = {}
        if self._create_env_kwargs is not None:
            kwargs["create_env_kwargs"] = self._create_env_kwargs
        self._env_pool = AsyncEnvPool(
            self._create_env_fn,
            backend=self._env_backend,
            **kwargs,
        )

        # Start inference server
        if not self._server.is_alive:
            self._server.start()

        # Start per-env coordinator threads
        self._result_queue = queue.Queue()
        self._shutdown_event = threading.Event()

        self._workers = []
        for i in range(self._num_envs):
            t = threading.Thread(
                target=_env_loop,
                kwargs={
                    "pool": self._env_pool,
                    "env_id": i,
                    "transport": self._transport,
                    "result_queue": self._result_queue,
                    "shutdown_event": self._shutdown_event,
                },
                daemon=True,
                name=f"AsyncBatchedCollector-env-{i}",
            )
            self._workers.append(t)
            t.start()

    @property
    def env(self) -> AsyncEnvPool:
        """The underlying :class:`AsyncEnvPool`."""
        self._ensure_started()
        return self._env_pool

    @property
    def policy(self) -> Callable:
        """The policy passed to the inference server."""
        return self._policy

    # ------------------------------------------------------------------
    # Rollout: drain the result queue
    # ------------------------------------------------------------------

    @staticmethod
    def _check_worker_result(item):
        """Re-raise exceptions propagated from coordinator threads."""
        if isinstance(item, BaseException):
            raise RuntimeError(
                "A collector worker thread raised an exception."
            ) from item

    def _rollout_frames(self) -> TensorDictBase:
        """Drain ``frames_per_batch`` transitions from the workers."""
        rq = self._result_queue
        collected = 0
        transitions: list[TensorDictBase] = []

        while collected < self.frames_per_batch:
            # Block for at least one transition
            td = rq.get()
            self._check_worker_result(td)
            transitions.append(td)
            collected += td.numel()
            # Batch-drain any additional items already in the queue
            while collected < self.frames_per_batch:
                try:
                    td = rq.get_nowait()
                except queue.Empty:
                    break
                transitions.append(td)
                collected += td.numel()
            if self.verbose:
                torchrl_logger.debug(
                    f"AsyncBatchedCollector: {collected}/{self.frames_per_batch} frames"
                )

        return lazy_stack(transitions)

    def _rollout_yield_trajs(self) -> TensorDictBase:
        """Drain transitions until a complete trajectory is available."""
        rq = self._result_queue

        while not self._trajectory_queue:
            td = rq.get()
            self._check_worker_result(td)
            env_id = 0
            eid = td.get(_ENV_IDX_KEY, default=None)
            if eid is not None:
                # Unwrap NonTensorData / NonTensorStack / list wrappers
                if hasattr(eid, "data"):
                    eid = eid.data
                while isinstance(eid, (list,)) and len(eid) == 1:
                    eid = eid[0]
                env_id = int(eid)

            self._yield_queues[env_id].append(td)
            if td["next", "done"].any():
                self._trajectory_queue.append(
                    lazy_stack(list(self._yield_queues[env_id]), -1)
                )
                self._yield_queues[env_id].clear()

        result = self._trajectory_queue.popleft()
        return result.reshape(-1)

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
