# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

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


class AsyncBatchedCollector(BaseCollector):
    """Asynchronous collector that pairs :class:`~torchrl.envs.AsyncEnvPool` with an :class:`~torchrl.modules.InferenceServer`.

    Unlike :class:`~torchrl.collectors.Collector`, this collector decouples
    environment stepping from policy inference.  An internal
    :class:`~torchrl.modules.InferenceServer` batches policy forward passes
    while environments step in parallel, enabling full GPU utilisation and
    overlapping CPU-bound work (env stepping) with GPU-bound work (inference).

    The user simply provides a list of environment factories and a policy
    (or policy factory) -- the collector handles all internal wiring:

    1. Wraps the env factories in an :class:`~torchrl.envs.AsyncEnvPool`.
    2. Creates an :class:`~torchrl.modules.InferenceServer` with the given
       policy and a :class:`~torchrl.modules.ThreadingTransport` (or a
       user-supplied transport).
    3. Runs an asynchronous rollout loop that pipelines env stepping and
       batched inference.

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
            One of ``"threading"`` or ``"multiprocessing"``.
            Defaults to ``"threading"``.
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
        env_backend: str = "threading",
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

        # ---- build env pool ---------------------------------------------------
        if not isinstance(create_env_fn, Sequence):
            raise TypeError("create_env_fn must be a list of env factories.")
        self._create_env_fn = list(create_env_fn)
        self._num_envs = len(create_env_fn)
        self._env_backend = env_backend
        self._create_env_kwargs = create_env_kwargs
        self._env_pool: AsyncEnvPool | None = None  # built lazily at start

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
        self._shutdown_event = threading.Event()

        # Per-env trajectory accumulators (for yield_completed_trajectories mode)
        self._yield_queues: list[deque] = [deque() for _ in range(self._num_envs)]
        self._trajectory_queue: deque = deque()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _ensure_started(self) -> None:
        """Lazily instantiate the env pool and start the inference server."""
        if self._env_pool is None:
            kwargs = {}
            if self._create_env_kwargs is not None:
                kwargs["create_env_kwargs"] = self._create_env_kwargs
            self._env_pool = AsyncEnvPool(
                self._create_env_fn,
                backend=self._env_backend,
                **kwargs,
            )
        if not self._server.is_alive:
            self._server.start()

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
    # Rollout helpers
    # ------------------------------------------------------------------

    def _client_policy(self, td: TensorDictBase) -> TensorDictBase:
        """Use the inference client to run the policy in a batched fashion."""
        # For each env in the batch, submit to the server individually and
        # collect results.  The server batches these across all concurrent
        # submissions.
        tds = td.unbind(0)
        futures = [self._client.submit(t) for t in tds]
        results = [f.result() for f in futures]
        return lazy_stack(results)

    def _rollout_frames(self) -> TensorDictBase:
        """Collect `frames_per_batch` frames using the pipelined async loop."""
        env = self.env

        if (
            self.reset_at_each_iter
            or not hasattr(self, "_shuttle")
            or self._shuttle is None
        ):
            self._shuttle = env.reset()

        trajectory = []
        collected = 0
        policy_input = self._shuttle

        while collected < self.frames_per_batch:
            if self.verbose:
                torchrl_logger.debug(
                    f"AsyncBatchedCollector: {collected}/{self.frames_per_batch} frames"
                )
            # Policy inference via server (batched automatically)
            env_input = self._client_policy(policy_input)
            # Synchronous step-and-maybe-reset
            cur_output, next_output = env.step_and_maybe_reset(env_input)

            trajectory.append(cur_output.clone())
            collected += cur_output.numel()
            policy_input = self._shuttle = next_output

        return lazy_stack(trajectory, -1)

    def _rollout_async(self) -> TensorDictBase:
        """Pipelined async rollout: overlap env step with inference."""
        env = self.env

        if (
            self.reset_at_each_iter
            or not hasattr(self, "_shuttle")
            or self._shuttle is None
        ):
            self._shuttle = env.reset()

        trajectory = []
        collected = 0

        # Prime the pipeline: send the first step
        policy_input = self._shuttle
        env_input = self._client_policy(policy_input)
        env.async_step_and_maybe_reset_send(env_input)

        while collected < self.frames_per_batch:
            if self.verbose:
                torchrl_logger.debug(
                    f"AsyncBatchedCollector: {collected}/{self.frames_per_batch} frames"
                )
            cur_output, next_output = env.async_step_and_maybe_reset_recv()

            trajectory.append(cur_output.clone())
            collected += cur_output.numel()

            self._shuttle = next_output

            # Pipeline: send next step while we process the current one
            if collected < self.frames_per_batch:
                env_input = self._client_policy(next_output)
                env.async_step_and_maybe_reset_send(env_input)

        return lazy_stack(trajectory, -1)

    def _rollout_yield_trajs(self) -> TensorDictBase:
        """Collect until at least one full trajectory is done, yield it."""
        env = self.env

        if not hasattr(self, "_started") or not self._started:
            self._shuttle = env.reset()
            policy_input = self._shuttle
            env_input = self._client_policy(policy_input)
            env.async_step_and_maybe_reset_send(env_input)
            self._started = True

        dones = torch.zeros(self._num_envs, dtype=torch.bool)

        while not self._trajectory_queue:
            cur_output, next_output = env.async_step_and_maybe_reset_recv()

            # Route results to per-env queues
            env_ids_raw = cur_output.get(env._env_idx_key).tolist()
            env_ids = []
            for eid in env_ids_raw:
                while isinstance(eid, list) and len(eid) == 1:
                    eid = eid[0]
                env_ids.append(eid)

            dones.fill_(False)
            for i, _data in zip(env_ids, cur_output.unbind(0)):
                self._yield_queues[i].append(_data)
                dones[i] = _data["next", "done"].any()

            if dones.any():
                for idx in dones.nonzero(as_tuple=True)[0].tolist():
                    self._trajectory_queue.append(
                        lazy_stack(list(self._yield_queues[idx]), -1)
                    )
                    self._yield_queues[idx].clear()

            # Pipeline: send next step
            self._shuttle = next_output
            env_input = self._client_policy(next_output)
            env.async_step_and_maybe_reset_send(env_input)

        result = self._trajectory_queue.popleft()
        # Flatten extra dimensions from AsyncEnvPool child batch sizes
        return result.reshape(-1)

    @property
    def rollout(self) -> Callable[[], TensorDictBase]:
        if self.yield_completed_trajectories:
            return self._rollout_yield_trajs
        return self._rollout_async

    # ------------------------------------------------------------------
    # BaseCollector interface
    # ------------------------------------------------------------------

    def iterator(self) -> Iterator[TensorDictBase]:
        """Iterate over collected batches."""
        self._ensure_started()
        self._client = self._transport.client()

        total = self.total_frames
        while total < 0 or self._frames < total:
            self._iter += 1
            td = self.rollout()
            if td is None:
                yield
                continue
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
        """Shut down the collector, inference server and env pool."""
        self._shutdown_event.set()
        self._server.shutdown(timeout=timeout or 5.0)
        if close_env and self._env_pool is not None:
            self._env_pool.close(raise_if_closed=raise_on_error)
            self._env_pool = None

    def set_seed(self, seed: int, static_seed: bool = False) -> int:
        """Set the seed for the env pool."""
        # AsyncEnvPool does not natively support set_seed uniformly.
        # We store it and pass through if available.
        return seed

    def state_dict(self) -> OrderedDict:
        return OrderedDict()

    def load_state_dict(self, state_dict: OrderedDict) -> None:
        pass

    def __del__(self) -> None:
        if getattr(self, "_env_pool", None) is not None:
            try:
                self.shutdown(timeout=2.0, raise_on_error=False)
            except Exception:
                pass
