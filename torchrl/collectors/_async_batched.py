# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import multiprocessing as mp
import queue
import threading
from collections import deque, OrderedDict
from collections.abc import Callable, Iterator, Sequence
from typing import Literal

import torch
from tensordict import lazy_stack, TensorDictBase

from torchrl._utils import logger as torchrl_logger
from torchrl.collectors._base import BaseCollector
from torchrl.data.utils import CloudpickleWrapper
from torchrl.envs import EnvBase
from torchrl.modules.inference_server import InferenceServer, ThreadingTransport
from torchrl.modules.inference_server._mp import MPTransport
from torchrl.modules.inference_server._transport import InferenceTransport

_ENV_IDX_KEY = "env_index"


def _threading_env_loop(
    env_factory: Callable,
    create_env_kwargs: dict,
    transport: InferenceTransport,
    result_queue: queue.Queue,
    shutdown_event: threading.Event,
    env_id: int,
):
    """Per-env worker thread that submits directly to the InferenceServer.

    Each worker owns one environment and one inference client.  The
    client blocks until the server has batched and processed the
    observation, so the worker loop is simply:

        reset -> infer (blocking) -> step -> put transition -> infer -> ...
    """
    env = env_factory(**create_env_kwargs)
    client = transport.client()

    try:
        obs = env.reset()
        action_td = client(obs)

        while not shutdown_event.is_set():
            cur_td, next_obs = env.step_and_maybe_reset(action_td)
            cur_td.set(_ENV_IDX_KEY, env_id)
            result_queue.put(cur_td)
            if shutdown_event.is_set():
                break
            action_td = client(next_obs)
    except Exception:
        if not shutdown_event.is_set():
            raise
    finally:
        env.close()


def _mp_env_loop(
    env_factory: Callable,
    create_env_kwargs: dict,
    client,
    result_queue,
    shutdown_event,
    env_id: int,
):
    """Per-env worker process that submits directly to the InferenceServer.

    Identical to :func:`_threading_env_loop` but designed for
    :class:`multiprocessing.Process` workers.  The ``client`` is a
    pre-created :class:`_MPInferenceClient` whose underlying
    ``mp.Queue`` handles are inherited by the child process.
    """
    if isinstance(env_factory, CloudpickleWrapper):
        env_factory = env_factory.fn
    env = env_factory(**create_env_kwargs)

    try:
        obs = env.reset()
        action_td = client(obs)

        while not shutdown_event.is_set():
            cur_td, next_obs = env.step_and_maybe_reset(action_td)
            cur_td.set(_ENV_IDX_KEY, env_id)
            result_queue.put(cur_td)
            if shutdown_event.is_set():
                break
            action_td = client(next_obs)
    except Exception:
        if not shutdown_event.is_set():
            raise
    finally:
        env.close()


class AsyncBatchedCollector(BaseCollector):
    """Asynchronous collector that pairs per-env workers with an :class:`~torchrl.modules.InferenceServer`.

    Unlike :class:`~torchrl.collectors.Collector`, this collector fully
    decouples environment stepping from policy inference:

    * Each environment runs in its own worker (thread or process) and
      submits observations directly to the inference server.
    * An :class:`~torchrl.modules.InferenceServer` running in a background
      thread continuously drains observation submissions, batches them, runs
      a single forward pass, and fans actions back out.
    * Workers block on a ``Future`` while waiting for inference, releasing
      the GIL so other workers and the server can proceed.

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
        server_timeout (float, optional): seconds the server waits for work
            before dispatching a partial batch.  Defaults to ``0.01``.
        transport (InferenceTransport, optional): a pre-built transport
            backend.  When ``None`` (default) one is created automatically
            to match the ``backend`` (``ThreadingTransport`` for
            ``"threading"``, ``MPTransport`` for ``"multiprocessing"``).
            Pass a :class:`~torchrl.modules.RayTransport` or
            :class:`~torchrl.modules.MonarchTransport` for distributed
            setups (workers will be spawned as threads that hold
            Ray/Monarch clients).
        device (torch.device or str, optional): device for policy inference.
            Passed to the inference server.  Defaults to ``None``.
        backend (str, optional): how to run per-env workers.  One of
            ``"threading"`` or ``"multiprocessing"``.  Defaults to
            ``"threading"``.
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
        server_timeout: float = 0.01,
        transport: InferenceTransport | None = None,
        device: torch.device | str | None = None,
        backend: Literal["threading", "multiprocessing"] = "threading",
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
        self._backend = backend
        self._create_env_kwargs = create_env_kwargs

        # ---- build transport --------------------------------------------------
        if transport is None:
            transport = (
                MPTransport() if backend == "multiprocessing" else ThreadingTransport()
            )
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
        self._shutdown_event: threading.Event | mp.Event = None
        self._result_queue: queue.Queue | mp.Queue = None
        self._workers: list = []

        # Per-env trajectory accumulators (for yield_completed_trajectories)
        self._yield_queues: list[deque] = [deque() for _ in range(self._num_envs)]
        self._trajectory_queue: deque = deque()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _normalise_env_kwargs(self) -> list[dict]:
        env_kwargs = self._create_env_kwargs
        if env_kwargs is None:
            return [{}] * self._num_envs
        if isinstance(env_kwargs, dict):
            return [env_kwargs] * self._num_envs
        return list(env_kwargs)

    def _ensure_started(self) -> None:
        """Start the inference server and spawn per-env workers."""
        if self._workers and all(
            (w.is_alive() if hasattr(w, "is_alive") else True) for w in self._workers
        ):
            return

        if not self._server.is_alive:
            self._server.start()

        env_kwargs = self._normalise_env_kwargs()

        if self._backend == "multiprocessing":
            self._start_mp_workers(env_kwargs)
        else:
            self._start_threading_workers(env_kwargs)

    def _start_threading_workers(self, env_kwargs: list[dict]) -> None:
        self._result_queue = queue.Queue()
        self._shutdown_event = threading.Event()

        self._workers = []
        for i in range(self._num_envs):
            t = threading.Thread(
                target=_threading_env_loop,
                kwargs={
                    "env_factory": self._create_env_fn[i],
                    "create_env_kwargs": env_kwargs[i],
                    "transport": self._transport,
                    "result_queue": self._result_queue,
                    "shutdown_event": self._shutdown_event,
                    "env_id": i,
                },
                daemon=True,
                name=f"AsyncBatchedCollector-env-{i}",
            )
            self._workers.append(t)
            t.start()

    def _start_mp_workers(self, env_kwargs: list[dict]) -> None:
        ctx = mp.get_context("spawn")
        self._result_queue = ctx.Queue()
        self._shutdown_event = ctx.Event()

        # Pre-create one client per env before spawning (queues are inherited)
        clients = [self._transport.client() for _ in range(self._num_envs)]

        self._workers = []
        for i in range(self._num_envs):
            env_fn = self._create_env_fn[i]
            if not isinstance(env_fn, EnvBase) and env_fn.__class__.__name__ != "EnvCreator":
                env_fn = CloudpickleWrapper(env_fn)

            p = ctx.Process(
                target=_mp_env_loop,
                kwargs={
                    "env_factory": env_fn,
                    "create_env_kwargs": env_kwargs[i],
                    "client": clients[i],
                    "result_queue": self._result_queue,
                    "shutdown_event": self._shutdown_event,
                    "env_id": i,
                },
                daemon=True,
                name=f"AsyncBatchedCollector-env-{i}",
            )
            self._workers.append(p)
            p.start()

    @property
    def policy(self) -> Callable:
        """The policy passed to the inference server."""
        return self._policy

    # ------------------------------------------------------------------
    # Rollout: drain the result queue
    # ------------------------------------------------------------------

    def _rollout_frames(self) -> TensorDictBase:
        """Drain ``frames_per_batch`` transitions from the workers."""
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
        """Shut down the collector, inference server and workers."""
        if self._shutdown_event is not None:
            self._shutdown_event.set()
        _timeout = timeout or 5.0
        for w in self._workers:
            w.join(timeout=_timeout)
        # Terminate any stragglers (multiprocessing only)
        for w in self._workers:
            if hasattr(w, "terminate") and w.is_alive():
                w.terminate()
        self._workers = []
        self._server.shutdown(timeout=_timeout)

    def set_seed(self, seed: int, static_seed: bool = False) -> int:
        """Set the seed (no-op; envs are created inside workers)."""
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
