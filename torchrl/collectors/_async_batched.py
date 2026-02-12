# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import queue
import threading
from collections import deque, OrderedDict
from collections.abc import Callable, Iterator, Sequence

import torch
from tensordict import lazy_stack, TensorDictBase

from torchrl._utils import logger as torchrl_logger
from torchrl.collectors._base import BaseCollector
from torchrl.envs import AsyncEnvPool, EnvBase
from torchrl.modules.inference_server import InferenceServer, ThreadingTransport
from torchrl.modules.inference_server._transport import InferenceTransport


def _extract_env_ids(td: TensorDictBase, env_idx_key: str) -> list[int]:
    """Extract scalar env indices from a batched TensorDict returned by AsyncEnvPool."""
    raw = td.get(env_idx_key).tolist()
    ids = []
    for eid in raw:
        while isinstance(eid, list) and len(eid) == 1:
            eid = eid[0]
        ids.append(int(eid))
    return ids


def _direct_env_loop(
    env_factory: Callable,
    create_env_kwargs: dict,
    transport: InferenceTransport,
    result_queue: queue.Queue,
    shutdown_event: threading.Event,
):
    """Per-env worker that submits directly to the InferenceServer.

    Each worker owns one environment and one inference client.  The
    client blocks until the server has batched and processed the
    observation, so the worker loop is simply:

        reset -> infer (blocking) -> step -> put transition -> infer -> ...

    This eliminates the coordinator thread and its serialization overhead.
    """
    env = env_factory(**create_env_kwargs)
    client = transport.client()

    try:
        obs = env.reset()
        action_td = client(obs)

        while not shutdown_event.is_set():
            cur_td, next_obs = env.step_and_maybe_reset(action_td)
            result_queue.put(cur_td)
            if shutdown_event.is_set():
                break
            action_td = client(next_obs)
    except Exception:
        if not shutdown_event.is_set():
            raise
    finally:
        env.close()


def _coordinator_loop(
    pool: AsyncEnvPool,
    transport: InferenceTransport,
    result_queue: queue.Queue,
    shutdown_event: threading.Event,
    num_envs: int,
):
    """Single-threaded coordinator that pipelines env stepping and batched inference.

    Architecture
    ------------
    ``AsyncEnvPool`` manages N envs in whatever backend the user chose
    (multiprocessing, threading, asyncio).  This coordinator thread bridges
    the pool with the ``InferenceServer``:

    * When an env finishes stepping (``recv``), its observation is submitted
      to the server via ``client.submit()`` (non-blocking ``Future``).
    * When a ``Future`` resolves (action ready), the action is sent back to
      the pool so the env can step again.
    * The server thread batches all pending observations into efficient
      forward passes -- this is where the throughput comes from.
    * Transitions are pushed to ``result_queue`` for the main thread to drain.

    The coordinator never lets *all* envs wait on inference with none stepping,
    which would deadlock the ``recv`` call.  When that edge case arises it
    explicitly waits for an inference ``Future`` to complete first.
    """
    # One client per env so that submit() calls are independent
    clients = {i: transport.client() for i in range(num_envs)}
    # pending[env_id] = Future  -- envs waiting for inference results
    pending: dict[int, object] = {}
    num_stepping = 0

    def _send_action(pool, action_td, env_id):
        """Send an action back to the pool, ensuring correct batch shape."""
        pool.async_step_and_maybe_reset_send(action_td.unsqueeze(0), env_index=env_id)

    # ---- Prime: reset all envs, run initial inference, send for stepping -----
    initial_obs = pool.reset()
    for i, obs in enumerate(initial_obs.unbind(0)):
        pending[i] = clients[i].submit(obs)

    # Wait for all initial actions and kick off stepping
    while pending:
        done_ids = [eid for eid, f in pending.items() if f.done()]
        for eid in done_ids:
            action_td = pending.pop(eid).result()
            _send_action(pool, action_td, eid)
            num_stepping += 1
        if not done_ids:
            # Busy-wait briefly (all futures are still being processed)
            threading.Event().wait(0.0001)

    # ---- Main loop -----------------------------------------------------------
    while not shutdown_event.is_set():
        # Safety: if every env is waiting on inference and none is stepping,
        # recv() would block forever.  Drain at least one future first.
        while num_stepping == 0 and pending:
            done_ids = [eid for eid, f in pending.items() if f.done()]
            for eid in done_ids:
                action_td = pending.pop(eid).result()
                _send_action(pool, action_td, eid)
                num_stepping += 1
            if not done_ids:
                threading.Event().wait(0.0001)

        if num_stepping == 0:
            break  # nothing in-flight -- shutdown or error

        # 1. Recv completed env steps (blocks until >= 1 is ready)
        cur_output, next_output = pool.async_step_and_maybe_reset_recv(
            min_get=1,
        )
        env_ids = _extract_env_ids(cur_output, pool._env_idx_key)
        num_stepping -= len(env_ids)

        # 2. Record transitions & submit observations to the server
        for eid, cur_td, next_td in zip(
            env_ids, cur_output.unbind(0), next_output.unbind(0)
        ):
            result_queue.put(cur_td)
            pending[eid] = clients[eid].submit(next_td)

        # 3. Send back every env whose inference already completed
        done_ids = [eid for eid, f in pending.items() if f.done()]
        for eid in done_ids:
            action_td = pending.pop(eid).result()
            _send_action(pool, action_td, eid)
            num_stepping += 1


class AsyncBatchedCollector(BaseCollector):
    """Asynchronous collector that pairs :class:`~torchrl.envs.AsyncEnvPool` with an :class:`~torchrl.modules.InferenceServer`.

    Unlike :class:`~torchrl.collectors.Collector`, this collector fully
    decouples environment stepping from policy inference:

    * An :class:`~torchrl.envs.AsyncEnvPool` runs *N* environments in
      parallel using whatever backend the user chooses (``"multiprocessing"``,
      ``"threading"``, ``"asyncio"``).
    * An :class:`~torchrl.modules.InferenceServer` running in a background
      thread continuously drains observation submissions, batches them, runs
      a single forward pass, and fans actions back out.
    * A lightweight coordinator thread bridges the two: whenever an env
      finishes stepping its observation is submitted to the server
      (non-blocking), and whenever an action is ready the env is sent back
      for stepping -- all without a global synchronisation barrier.

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
        server_timeout (float, optional): seconds the server waits for work
            before dispatching a partial batch.  Defaults to ``0.01``.
        transport (InferenceTransport, optional): a pre-built transport
            backend.  When ``None`` (default) a
            :class:`~torchrl.modules.ThreadingTransport` is created
            automatically.
        device (torch.device or str, optional): device for policy inference.
            Passed to the inference server.  Defaults to ``None``.
        env_backend (str, optional): backend for :class:`AsyncEnvPool`.
            One of ``"threading"``, ``"multiprocessing"`` or ``"asyncio"``.
            Defaults to ``"multiprocessing"``.
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
        ...     env_backend="threading",
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
        server_timeout: float = 0.01,
        transport: InferenceTransport | None = None,
        device: torch.device | str | None = None,
        env_backend: str = "multiprocessing",
        reset_at_each_iter: bool = False,
        postproc: Callable[[TensorDictBase], TensorDictBase] | None = None,
        yield_completed_trajectories: bool = False,
        weight_sync=None,
        weight_sync_model_id: str = "policy",
        verbose: bool = False,
        create_env_kwargs: dict | list[dict] | None = None,
        direct: bool = False,
    ):
        if policy is not None and policy_factory is not None:
            raise TypeError("policy and policy_factory are mutually exclusive.")
        if policy is None and policy_factory is None:
            raise TypeError("One of policy or policy_factory must be provided.")

        # ---- resolve policy ---------------------------------------------------
        if policy_factory is not None:
            policy = policy_factory()
        self._policy = policy

        # ---- build transport --------------------------------------------------
        if transport is None:
            transport = ThreadingTransport()
        self._transport = transport

        # ---- build inference server -------------------------------------------
        self._server = InferenceServer(
            model=policy,
            transport=transport,
            max_batch_size=max_batch_size,
            timeout=server_timeout,
            device=device,
            weight_sync=weight_sync,
            weight_sync_model_id=weight_sync_model_id,
        )

        # ---- env config -------------------------------------------------------
        if not isinstance(create_env_fn, Sequence):
            raise TypeError("create_env_fn must be a list of env factories.")
        self._create_env_fn = list(create_env_fn)
        self._num_envs = len(create_env_fn)
        self._env_backend = env_backend
        self._create_env_kwargs = create_env_kwargs

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
        self._shutdown_event = threading.Event()
        self._result_queue: queue.Queue | None = None
        self._coordinator: threading.Thread | None = None
        self._env_pool: AsyncEnvPool | None = None
        self._workers: list[threading.Thread] = []
        self._direct = direct

        # Per-env trajectory accumulators (for yield_completed_trajectories)
        self._yield_queues: list[deque] = [deque() for _ in range(self._num_envs)]
        self._trajectory_queue: deque = deque()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _ensure_started(self) -> None:
        """Create the env pool, start the server and coordinator/worker threads."""
        if self._direct:
            if self._workers and all(w.is_alive() for w in self._workers):
                return

            if not self._server.is_alive:
                self._server.start()

            self._result_queue = queue.Queue()
            self._shutdown_event.clear()

            env_kwargs = self._create_env_kwargs
            if env_kwargs is None:
                env_kwargs = [{}] * self._num_envs
            elif isinstance(env_kwargs, dict):
                env_kwargs = [env_kwargs] * self._num_envs

            self._workers = []
            for i in range(self._num_envs):
                t = threading.Thread(
                    target=_direct_env_loop,
                    kwargs={
                        "env_factory": self._create_env_fn[i],
                        "create_env_kwargs": env_kwargs[i],
                        "transport": self._transport,
                        "result_queue": self._result_queue,
                        "shutdown_event": self._shutdown_event,
                    },
                    daemon=True,
                    name=f"AsyncBatchedCollector-env-{i}",
                )
                self._workers.append(t)
                t.start()
            return

        if self._coordinator is not None and self._coordinator.is_alive():
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

        # Start coordinator thread
        self._result_queue = queue.Queue()
        self._shutdown_event.clear()
        self._coordinator = threading.Thread(
            target=_coordinator_loop,
            kwargs={
                "pool": self._env_pool,
                "transport": self._transport,
                "result_queue": self._result_queue,
                "shutdown_event": self._shutdown_event,
                "num_envs": self._num_envs,
            },
            daemon=True,
            name="AsyncBatchedCollector-coordinator",
        )
        self._coordinator.start()

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

    def _rollout_frames(self) -> TensorDictBase:
        """Drain ``frames_per_batch`` transitions from the coordinator."""
        rq = self._result_queue
        collected = 0
        transitions: list[TensorDictBase] = []

        while collected < self.frames_per_batch:
            td = rq.get()
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
            # Infer worker id from the env_index key if present, else round-robin
            env_id = 0
            if self._env_pool is not None:
                eid = td.get(self._env_pool._env_idx_key, default=None)
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
        """Shut down the collector, inference server, coordinator and env pool."""
        self._shutdown_event.set()
        if self._direct:
            for w in self._workers:
                w.join(timeout=timeout or 5.0)
            self._workers = []
        else:
            if self._coordinator is not None:
                self._coordinator.join(timeout=timeout or 5.0)
                self._coordinator = None
            if close_env and self._env_pool is not None:
                self._env_pool.close(raise_if_closed=raise_on_error)
                self._env_pool = None
        self._server.shutdown(timeout=timeout or 5.0)

    def set_seed(self, seed: int, static_seed: bool = False) -> int:
        """Set the seed (no-op; envs are created inside the pool)."""
        return seed

    def state_dict(self) -> OrderedDict:
        return OrderedDict()

    def load_state_dict(self, state_dict: OrderedDict) -> None:
        pass

    def __del__(self) -> None:
        if getattr(self, "_coordinator", None) is not None:
            try:
                self.shutdown(timeout=2.0, raise_on_error=False)
            except Exception:
                pass
