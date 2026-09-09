# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import contextlib
import functools as ft
import itertools
import multiprocessing as mp
import os
import queue
import threading
import time
import warnings
from collections import deque, OrderedDict
from collections.abc import Callable, Iterator, Mapping, Sequence
from typing import Literal

import torch
from tensordict import (
    lazy_stack,
    LazyStackedTensorDict,
    maybe_dense_stack,
    NestedKey,
    TensorDict,
    TensorDictBase,
)
from torchrl._comm import MailboxTransportError
from torchrl._comm.mailbox import _exit_on_parent_exit
from torchrl._utils import (
    _maybe_record_function_decorator,
    logger as torchrl_logger,
    timeit,
)
from torchrl.collectors._base import BaseCollector
from torchrl.collectors._constants import DEFAULT_EXPLORATION_TYPE
from torchrl.collectors.utils import _maybe_normalize_replay_buffer_tensordict_device
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.utils import CloudpickleWrapper
from torchrl.envs import AsyncEnvPool, EnvBase, EnvCreator
from torchrl.envs.async_envs import _validate_cpu_affinity
from torchrl.envs.utils import ExplorationType
from torchrl.modules.inference_server import (
    InferenceDeviceConfig,
    InferenceServer,
    InferenceServerConfig,
    PolicyClientModule,
    ProcessInferenceServer,
    ProcessSlotTransport,
    ThreadingTransport,
)
from torchrl.modules.inference_server._config import _resolve_device_config
from torchrl.modules.inference_server._transport import InferenceTransport
from torchrl.weight_update.weight_sync_schemes import WeightStrategy

_ENV_IDX_KEY = "env_index"

_POLICY_BACKENDS = ("threading", "multiprocessing", "ray", "monarch")
_ENV_BACKENDS = ("threading", "multiprocessing")
_PauseRequest = tuple[threading.Barrier, threading.Event]


class _CollectorStopped(RuntimeError):
    """Internal signal interrupting a blocked rollout during shutdown."""


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


def _put_result(result_queue, item, shutdown_event, *, pause=None) -> bool:
    """Retain a completed result through backpressure and pause."""
    while not shutdown_event.is_set():
        if pause is not None:
            pause()
        if shutdown_event.is_set():
            break
        try:
            result_queue.put(item, timeout=0.1)
            return True
        except queue.Full:
            continue
    return False


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

            # Initial resets may have no device metadata while later policy
            # results do. Keep requests collatable when streams start or reset
            # at different times; tensor placement is unchanged.
            action_td = client(obs.clone(recurse=False).clear_device_())
            if env_device is not None:
                action_td = action_td.to(env_device)
            pool.async_step_and_maybe_reset_send(action_td, env_index=env_id)
            cur_td, obs = pool.async_step_and_maybe_reset_recv(env_index=env_id)
            cur_td.set(_ENV_IDX_KEY, env_id)
            if storing_device is not None:
                cur_td = cur_td.to(storing_device)
            if not _put_result(
                result_queue,
                cur_td,
                shutdown_event,
                pause=ft.partial(_wait_while_paused, pause_request),
            ):
                break
    except Exception as exc:
        if not shutdown_event.is_set():
            _put_result(result_queue, exc, shutdown_event)


def _process_env_loop(
    env_factory: Callable[[], EnvBase],
    create_env_kwargs: dict,
    env_id: int,
    client: PolicyClientModule,
    result_queue,
    result_capacity,
    shutdown_event,
    pause_event,
    paused_event,
    error_queue,
    worker_affinity: Sequence[int] | None,
    env_device: torch.device | None,
    storing_device: torch.device | None,
    chunk_size: int = 1,
):
    """Run environment stepping and remote inference in one worker process.

    With ``chunk_size > 1`` the worker accumulates that many consecutive
    transitions and sends them as one dense result, so the driver handles one
    message per chunk instead of one per transition. One capacity permit then
    covers the whole chunk under construction.
    """
    env = None
    chunk: list[TensorDictBase] = []
    try:
        if worker_affinity is not None:
            os.sched_setaffinity(0, worker_affinity)
        threading.Thread(target=_exit_on_parent_exit, daemon=True).start()
        torch.set_num_threads(1)
        env = env_factory(**create_env_kwargs)
        observation = env.reset()
        while not shutdown_event.is_set():
            if pause_event.is_set():
                paused_event.set()
                while pause_event.is_set() and not shutdown_event.is_set():
                    shutdown_event.wait(0.01)
                paused_event.clear()
                continue
            # Reserve capacity before inference/stepping. A full result queue
            # must still let the worker observe pause and shutdown requests.
            if not chunk:
                if not result_capacity.acquire(timeout=0.01):
                    continue
                if pause_event.is_set() or shutdown_event.is_set():
                    result_capacity.release()
                    continue
            policy_output = client(observation)
            action_td = observation.update(policy_output)
            if env_device is not None:
                action_td = action_td.to(env_device)
            transition, observation = env.step_and_maybe_reset(action_td)
            if chunk_size == 1:
                transition.set(_ENV_IDX_KEY, env_id)
            else:
                # Chunks are concatenated in the driver; keep the index a tensor
                # like the batched coordinator does.
                transition.set(
                    _ENV_IDX_KEY,
                    torch.full(transition.batch_size, env_id, dtype=torch.long),
                )
            # multiprocessing.Queue serializes on a feeder thread after put()
            # returns. Own the transition storage before the environment or
            # inference response slot can be reused.
            transition = transition.clone()
            if storing_device is not None:
                transition = transition.to(storing_device)
            if chunk_size > 1:
                chunk.append(transition)
                if len(chunk) < chunk_size:
                    continue
                transition = maybe_dense_stack(chunk)
                chunk = []
            if all(
                value.device.type == "cpu"
                for value in transition.values(True, True)
                if isinstance(value, torch.Tensor)
            ):
                # Transfer one shared storage instead of negotiating a file
                # descriptor for every tensor leaf of every transition.
                transition = transition.consolidate()
            result_queue.put(transition, block=False)
    except Exception as exc:
        if not shutdown_event.is_set():
            error_queue.put(RuntimeError(f"Environment {env_id} failed: {exc!r}"))
            shutdown_event.wait()
    finally:
        if env is not None:
            env.close()


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
    pending_results = deque()
    policy_outputs = {}
    stepping = 0
    poll_interval = 0.001

    try:
        env_ids = list(range(pool.num_envs))
        pool.async_reset_send(env_index=env_ids)
        resetting = pool.num_envs

        while not shutdown_event.is_set():
            while pending_results:
                try:
                    result_queue.put_nowait(pending_results[0])
                except queue.Full:
                    break
                pending_results.popleft()
            if pending_results and pause_request[0] is None:
                shutdown_event.wait(poll_interval)
                continue
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
            transitions.set(
                _ENV_IDX_KEY,
                torch.tensor(completed_ids, device=transitions.device, dtype=torch.long)
                .reshape(-1, *([1] * (transitions.ndim - 1)))
                .expand(transitions.batch_size),
            )
            if storing_device is not None:
                transitions = transitions.to(storing_device)
            pending_results.append(transitions)
    except Exception as exc:
        if not shutdown_event.is_set():
            _put_result(result_queue, exc, shutdown_event)


def _env_kwargs(create_env_kwargs, index: int) -> dict:
    if create_env_kwargs is None:
        return {}
    if isinstance(create_env_kwargs, Mapping):
        return dict(create_env_kwargs)
    return dict(create_env_kwargs[index])


def _process_slot_blocker(
    *,
    env_backend: str,
    envs_per_worker: int,
    policy_factory: Callable | None,
    policy_backend: str | None,
    server_backend: str,
    policy_device: torch.device | None,
    env_device: torch.device | None,
    storing_device: torch.device | None,
) -> str | None:
    """Why environment workers cannot reach a dedicated inference process directly.

    Returns ``None`` when a :class:`ProcessSlotTransport` can serve the
    collector, otherwise a short reason for the log.
    """
    if env_backend != "multiprocessing":
        return "environment workers are threads (env_backend='multiprocessing' is required)"
    if envs_per_worker != 1:
        return "the transport hosts one environment per worker process"
    if policy_factory is None:
        return (
            "policy_factory is required to rebuild the policy in the inference process"
        )
    if policy_backend not in (None, "multiprocessing"):
        return f"policy_backend={policy_backend!r} selects another inference transport"
    if server_backend not in ("thread", "process"):
        return f"service_backend={server_backend!r} selects another inference server"
    if policy_device is not None and policy_device.type not in ("cpu", "cuda"):
        # Weight updates cross process boundaries as shared CPU storage or CUDA IPC handles.
        return (
            f"policy_device={policy_device} cannot share weights with another process"
        )
    for name, target_device in (
        ("env_device", env_device),
        ("storing_device", storing_device),
    ):
        if target_device is not None and target_device.type != "cpu":
            return f"{name}={target_device} is not a CPU device"
    return None


def _auto_process_slot_transport(
    env_factory: Callable[..., EnvBase],
    env_kwargs: dict,
    policy: Callable,
    *,
    num_slots: int,
    policy_version_key: NestedKey | None,
) -> tuple[ProcessSlotTransport | None, str | None]:
    """Derive fixed request and response layouts from one environment and one policy pass.

    Returns the transport, or ``None`` with the reason the layouts could not be
    derived.
    """
    in_keys = getattr(policy, "in_keys", None)
    out_keys = getattr(policy, "out_keys", None)
    if not in_keys or not out_keys:
        return None, "the policy does not declare in_keys and out_keys"
    env = env_factory(**env_kwargs)
    try:
        fake = env.fake_tensordict()
    finally:
        env.close()
    missing = [key for key in in_keys if key not in fake.keys(True, True)]
    if missing:
        return None, f"the environment does not produce the policy inputs {missing}"
    request = fake.select(*in_keys, strict=True).cpu()
    reference = None
    if isinstance(policy, torch.nn.Module):
        reference = next(itertools.chain(policy.parameters(), policy.buffers()), None)
    probe_device = reference.device if reference is not None else torch.device("cpu")
    try:
        with torch.no_grad():
            output = policy(request.clone().unsqueeze(0).to(probe_device))
    except Exception as err:  # noqa: BLE001
        return None, f"the policy could not run on a sample request ({err!r})"
    output = output.squeeze(0) if output.batch_dims else output
    missing = [key for key in out_keys if key not in output.keys(True, True)]
    if missing:
        return None, f"the policy did not return its outputs {missing}"
    response = output.select(*out_keys, strict=True).cpu()
    if policy_version_key is not None:
        response.set(policy_version_key, torch.zeros((), dtype=torch.long))
    return ProcessSlotTransport(request, response, num_slots=num_slots), None


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
    * With a :class:`~torchrl.modules.inference_server.ProcessSlotTransport`,
      each multiprocessing environment worker talks directly to the dedicated
      inference process; the driver receives completed transitions only.
      Completed and in-flight results are bounded to twice the environment
      count; ``transition_chunk_size`` sets how many consecutive transitions
      one result holds. Workers and the inference server exit when their
      owner dies.
    * The :class:`~torchrl.modules.InferenceServer` running in a background
      thread continuously drains observation submissions, batches them, runs
      a single forward pass, and fans actions back out.

    There is **no global synchronisation barrier**: fast environments keep
    stepping while slow ones wait for inference, and the server always
    processes whatever observations have accumulated.

    The user simply provides env factories and a policy; the collector
    handles all wiring internally. With ``transport="auto"``, multiprocessing
    environment workers and a ``policy_factory``, it also derives the fixed
    request and response layouts and serves the policy from a dedicated
    process that the workers reach directly.

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
        transport (InferenceTransport, "auto" or "thread", optional): the
            inference transport. A pre-built transport object takes precedence
            over ``policy_backend``; a
            :class:`~torchrl.modules.inference_server.ProcessSlotTransport`
            runs the complete acting loop in environment worker processes and
            implies a process inference server and multiprocessing environment
            workers. ``"auto"`` builds that transport when environment workers
            are processes with one environment each and a ``policy_factory``
            is given: the request and response layouts come from one
            environment's ``fake_tensordict()`` and one policy pass. When
            those conditions do not hold, or the layouts cannot be derived,
            the policy is served from a thread of this process and the reason
            is logged. ``"thread"`` always serves from a thread of this
            process. ``None`` (default) behaves like ``"thread"`` and emits a
            :class:`FutureWarning` when ``"auto"`` would pick process slots:
            in v0.15 the default becomes ``"auto"``.
        device (torch.device or str, optional): device for policy inference
            (shorthand for ``InferenceDeviceConfig(policy_device=...)``).
            Defaults to ``None``.
        server_config (InferenceServerConfig, optional): structured server
            configuration: execution ``backend`` (``"thread"`` runs the serve
            loop in this process, ``"process"`` a dedicated server process
            requiring ``policy_factory``; a
            :class:`~torchrl.modules.inference_server.ProcessSlotTransport`
            turns ``"thread"`` into ``"process"``), batching, optional static
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
            ``"monarch"``.  Defaults to ``None``: environment workers run in
            threads and inference uses the threading transport. In v0.15 the
            environment-worker default changes to ``"multiprocessing"``; a
            :class:`FutureWarning` is emitted until then when neither
            ``backend`` nor ``env_backend`` is given.
        env_backend (str, optional): backend for the
            :class:`~torchrl.envs.AsyncEnvPool` that runs environments.  One
            of ``"threading"`` or ``"multiprocessing"``.  Falls back to
            ``backend`` when ``None``.  The coordinator threads are always
            Python threads regardless of this setting.  Defaults to ``None``.
        env_exchange (str, optional): data exchange of a multiprocessing
            :class:`~torchrl.envs.AsyncEnvPool`, one of ``"queue"``, ``"shm"``
            or ``"auto"``. The shared-memory exchange also enables batched
            coordination from one thread; ``"auto"`` selects it whenever the
            environment schema allows. It does not apply when a
            :class:`~torchrl.modules.inference_server.ProcessSlotTransport`
            owns the worker exchange. Defaults to ``"auto"``.
        envs_per_worker (int, optional): Number of environments hosted by each
            multiprocessing worker. Grouped workers share one coordinator that
            drains ready environments without waiting for a complete group.
            Defaults to ``1``.
        transition_chunk_size (int or "auto", optional): number of consecutive
            transitions each environment worker process accumulates before
            sending them to the driver as one dense message. Requires a
            :class:`~torchrl.modules.inference_server.ProcessSlotTransport`.
            ``"auto"`` (default) uses ``frames_per_batch // len(create_env_fn)``
            with process workers, so each batch holds one contiguous run per
            environment like the synchronous collectors and a transition waits
            at most one batch; it resolves to ``1`` otherwise.
            ``1`` sends every transition as soon as it completes.
            Larger values take the driver off the per-transition path: it
            receives one message per chunk, concatenates whole chunks into
            each batch and writes each batch to ``replay_buffer`` with a
            single routed ``extend``, so its per-transition Python work is
            amortized over the chunk. The cost is latency: a transition
            reaches the driver only once its chunk is complete, and up to
            ``transition_chunk_size - 1`` transitions per environment stay in
            the worker while collection is paused or stopped. Batches are then
            dense :class:`~tensordict.TensorDict` instances and ``env_index``
            is a tensor. Defaults to ``"auto"``.
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
        exploration_type (ExplorationType, optional): interaction mode used
            when collecting data, one of
            ``torchrl.envs.utils.ExplorationType.RANDOM``, ``MODE``, ``MEAN``
            or ``DETERMINISTIC``. Every inference request is stamped with it
            and, when ``static_batch_size`` is set, the CUDA graph is captured
            under it, independently of the process-wide
            :func:`~torchrl.envs.utils.set_exploration_type` context, which a
            learner thread of the same process may change at any time.
            Defaults to ``ExplorationType.RANDOM``.
        replay_buffer (ReplayBuffer, optional): replay buffer to extend in the
            collector's parent thread after post-processing. When provided,
            iteration yields ``None`` instead of full rollout batches.
            Defaults to ``None``.
        post_collect_hook (Callable, optional): callback invoked with each
            post-processed batch before it is normalized and written to replay
            (or yielded when no replay buffer is configured). Defaults to
            ``None``.
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
        transport: InferenceTransport | Literal["auto", "thread"] | None = None,
        device: torch.device | str | None = None,
        backend: Literal["threading", "multiprocessing", "ray", "monarch"]
        | None = None,
        env_backend: Literal["threading", "multiprocessing"] | None = None,
        env_exchange: Literal["queue", "shm", "auto"] = "auto",
        envs_per_worker: int = 1,
        transition_chunk_size: int | Literal["auto"] = "auto",
        policy_backend: (
            Literal["threading", "multiprocessing", "ray", "monarch"] | None
        ) = None,
        reset_at_each_iter: bool = False,
        postproc: Callable[[TensorDictBase], TensorDictBase] | None = None,
        exploration_type: ExplorationType = DEFAULT_EXPLORATION_TYPE,
        replay_buffer: ReplayBuffer | None = None,
        post_collect_hook: Callable[[TensorDictBase], None] | None = None,
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
        super().__init__(post_collect_hook=post_collect_hook)
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

        self._policy_factory = policy_factory

        # ---- env config -------------------------------------------------------
        if not isinstance(create_env_fn, Sequence):
            raise TypeError("create_env_fn must be a list of env factories.")
        self._create_env_fn = list(create_env_fn)
        self._num_envs = len(create_env_fn)
        if create_env_kwargs is not None and not isinstance(create_env_kwargs, Mapping):
            if (
                not isinstance(create_env_kwargs, Sequence)
                or len(create_env_kwargs) != self._num_envs
                or not all(isinstance(kwargs, Mapping) for kwargs in create_env_kwargs)
            ):
                raise ValueError(
                    "create_env_kwargs must be a dict or a list of dicts with "
                    f"length {self._num_envs}."
                )
        self._create_env_kwargs = create_env_kwargs

        # ---- resolve backends -------------------------------------------------
        explicit_transport = isinstance(transport, InferenceTransport)
        if transport is not None and not explicit_transport:
            if transport not in ("auto", "thread"):
                raise ValueError(
                    "transport must be an InferenceTransport, 'auto', 'thread' or "
                    f"None, got {transport!r}."
                )
        if env_backend is not None:
            effective_env_backend = env_backend
        elif backend is not None:
            effective_env_backend = backend
        elif explicit_transport and isinstance(transport, ProcessSlotTransport):
            # The transport only works with environment worker processes.
            effective_env_backend = "multiprocessing"
        else:
            warnings.warn(
                "AsyncBatchedCollector runs environment workers in threads when "
                "neither backend nor env_backend is given. In v0.15 this default "
                "will change to env_backend='multiprocessing'. Pass "
                "env_backend='threading' to keep the current behavior, or "
                "env_backend='multiprocessing' to adopt the future default now.",
                FutureWarning,
                stacklevel=2,
            )
            effective_env_backend = "threading"
        if policy_backend is not None:
            effective_policy_backend = policy_backend
        else:
            effective_policy_backend = backend if backend is not None else "threading"
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
        if transition_chunk_size != "auto" and (
            isinstance(transition_chunk_size, bool)
            or not isinstance(transition_chunk_size, int)
            or transition_chunk_size < 1
        ):
            raise ValueError(
                "transition_chunk_size must be a positive integer or 'auto', got "
                f"{transition_chunk_size!r}."
            )
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
        # ---- resolve the transport --------------------------------------------
        # Environment worker processes can talk to a dedicated inference
        # process directly when the policy can be rebuilt there and no tensor
        # has to cross the driver on the action path.
        blocker = _process_slot_blocker(
            env_backend=effective_env_backend,
            envs_per_worker=envs_per_worker,
            policy_factory=policy_factory,
            policy_backend=policy_backend,
            server_backend=server_backend,
            policy_device=policy_device,
            env_device=self._env_device,
            storing_device=self._storing_device,
        )
        probe_policy = None
        if explicit_transport:
            pass
        elif transport == "auto":
            if blocker is None:
                probe_policy = policy_factory()
                transport, blocker = _auto_process_slot_transport(
                    self._create_env_fn[0],
                    _env_kwargs(create_env_kwargs, 0),
                    probe_policy,
                    num_slots=self._num_envs,
                    policy_version_key=policy_version_key,
                )
            if blocker is not None:
                transport = None
                torchrl_logger.info(
                    "AsyncBatchedCollector(transport='auto') serves the policy "
                    "from a thread of this process: %s.",
                    blocker,
                )
            else:
                torchrl_logger.info(
                    "AsyncBatchedCollector(transport='auto') serves the policy "
                    "from a dedicated process that the %d environment worker "
                    "processes reach directly.",
                    self._num_envs,
                )
        elif transport is None:
            if blocker is None:
                warnings.warn(
                    "AsyncBatchedCollector serves the policy from a thread of this "
                    "process by default. In v0.15, when environment workers are "
                    "processes and a policy_factory is given, the default will "
                    "become transport='auto', which serves the policy from a "
                    "dedicated process that the environment workers reach "
                    "directly. Pass transport='thread' to keep the current "
                    "behavior, or transport='auto' to adopt the future default now.",
                    FutureWarning,
                    stacklevel=2,
                )
        else:  # transport == "thread"
            transport = None

        uses_process_env_workers = isinstance(transport, ProcessSlotTransport)
        if uses_process_env_workers:
            if server_backend == "thread":
                # The transport reaches a dedicated inference process by design.
                server_backend = "process"
            if policy_factory is None:
                raise TypeError(
                    "ProcessSlotTransport requires policy_factory so the policy "
                    "can be constructed inside the inference server process."
                )
            if envs_per_worker != 1:
                raise ValueError("ProcessSlotTransport requires envs_per_worker=1.")
            if (
                server_backend != "process"
                or effective_env_backend != "multiprocessing"
            ):
                raise ValueError(
                    "ProcessSlotTransport requires both a process inference server "
                    "and env_backend='multiprocessing'."
                )
            if env_exchange == "shm":
                raise ValueError(
                    "env_exchange does not apply when ProcessSlotTransport owns "
                    "the environment-worker exchange; leave it as 'auto'."
                )
            if transport._num_slots < self._num_envs:
                raise ValueError(
                    f"ProcessSlotTransport needs at least one slot per environment "
                    f"({self._num_envs}), but has {transport._num_slots}."
                )
            if policy_device is not None and policy_device.type not in ("cpu", "cuda"):
                raise ValueError(
                    "ProcessSlotTransport requires a CPU or CUDA policy_device so "
                    f"weight updates can be shared with the inference process; got "
                    f"{policy_device}."
                )
            for name, target_device in (
                ("env_device", self._env_device),
                ("storing_device", self._storing_device),
            ):
                if target_device is not None and target_device.type != "cpu":
                    raise ValueError(
                        f"ProcessSlotTransport requires a CPU {name}; got "
                        f"{target_device}. Keep CUDA policy execution in the "
                        "inference server process."
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
        if transport is None:
            transport = _make_transport(
                effective_policy_backend, num_slots=self._num_envs
            )
        self._transport = transport
        self._uses_process_env_workers = uses_process_env_workers

        # ---- resolve the chunk size -------------------------------------------
        if transition_chunk_size == "auto":
            # One contiguous run per environment and batch, the layout of the
            # synchronous collectors, so a transition waits at most one batch.
            transition_chunk_size = (
                max(1, frames_per_batch // self._num_envs)
                if uses_process_env_workers
                else 1
            )
        self._transition_chunk_size = transition_chunk_size
        self._uses_chunked_results = transition_chunk_size > 1
        if self._uses_chunked_results and not uses_process_env_workers:
            raise ValueError(
                "transition_chunk_size > 1 requires a ProcessSlotTransport so "
                "that environment worker processes assemble the chunks."
            )

        # ---- resolve policy ---------------------------------------------------
        if policy_factory is not None and server_backend != "process":
            policy = probe_policy if probe_policy is not None else policy_factory()
        self._policy = policy

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
                mp_context=(transport._ctx if self._uses_process_env_workers else None),
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
        self.exploration_type = ExplorationType(
            exploration_type
            if exploration_type is not None
            else DEFAULT_EXPLORATION_TYPE
        )
        self.replay_buffer = replay_buffer
        self.verbose = verbose

        self._frames = 0
        self._iter = -1

        # ---- runtime state (created lazily) -----------------------------------
        self._shutdown_event: object | None = None
        self._result_queue: object | None = None
        self._worker_error_queue = None
        self._env_pool: AsyncEnvPool | None = None
        self._workers: list[threading.Thread | mp.Process] = []
        self._clients: list[Callable] | None = None
        self._uses_batched_coordinator = False
        self._pause_request: list[_PauseRequest | None] = [None]
        self._pause_lock = threading.Lock()
        self._transition_carry: deque[TensorDictBase] = deque()
        self._iteration_lock = threading.Lock()
        self._replay_lock = threading.Lock()
        self._replay_thread: threading.Thread | None = None
        self._replay_error: Exception | None = None
        # Weights received before a process server started; applied at start.
        self._pending_weights: TensorDictBase | None = None

        # Per-env trajectory accumulators (for yield_completed_trajectories)
        self._yield_queues: list[deque] = [deque() for _ in range(self._num_envs)]
        self._trajectory_queue: deque = deque()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _ensure_started(self) -> None:
        """Create the env pool, start the server and coordinator threads."""
        if self._workers:
            if self._uses_process_env_workers:
                self._check_process_workers()
                if self._shutdown_event.is_set():
                    raise RuntimeError(
                        "Environment workers have stopped; create a new collector."
                    )
            if all(w.is_alive() for w in self._workers):
                return

        original_affinity = None
        try:
            if self._driver_affinity is not None:
                original_affinity = os.sched_getaffinity(0)
                os.sched_setaffinity(0, self._driver_affinity)

            if self._uses_process_env_workers:
                if self._clients is None:
                    self._clients = [
                        PolicyClientModule(
                            self._transport.client(),
                            max_inflight=self._max_inflight_per_env,
                            interaction_type=self.exploration_type,
                        )
                        for _ in range(self._num_envs)
                    ]
                if self._server.static_batch_size is not None:
                    self._server.prepare_cudagraph(
                        self._transport._request_slots[0],
                        interaction_type=self.exploration_type,
                    )
                if not self._server.is_alive:
                    self._server.start()
                self._apply_pending_weights()

                create_env_kwargs = self._create_env_kwargs
                if create_env_kwargs is None:
                    create_env_kwargs = [{} for _ in range(self._num_envs)]
                elif isinstance(create_env_kwargs, Mapping):
                    create_env_kwargs = [
                        dict(create_env_kwargs) for _ in range(self._num_envs)
                    ]

                ctx = self._transport._ctx
                # Count completed and in-flight transitions together. Queue
                # capacity alone cannot prevent workers from blocking in put().
                capacity = 2 * self._num_envs
                self._result_queue = ctx.Queue(maxsize=capacity)
                self._result_capacity = ctx.BoundedSemaphore(capacity)
                self._worker_error_queue = ctx.Queue()
                self._worker_check_timer = timeit(
                    "AsyncBatchedCollector.worker_liveness", sync=False
                )
                self._worker_check_timer.start()
                self._shutdown_event = ctx.Event()
                self._process_pause_event = ctx.Event()
                self._process_paused_events = [
                    ctx.Event() for _ in range(self._num_envs)
                ]
                self._workers = []
                self._uses_batched_coordinator = False
                try:
                    for env_id, (env_factory, env_kwargs, client) in enumerate(
                        zip(self._create_env_fn, create_env_kwargs, self._clients)
                    ):
                        if not isinstance(
                            env_factory, (EnvCreator, CloudpickleWrapper)
                        ):
                            env_factory = CloudpickleWrapper(env_factory)
                        process = ctx.Process(
                            target=_process_env_loop,
                            kwargs={
                                "env_factory": env_factory,
                                "create_env_kwargs": env_kwargs,
                                "env_id": env_id,
                                "client": client,
                                "result_queue": self._result_queue,
                                "result_capacity": self._result_capacity,
                                "shutdown_event": self._shutdown_event,
                                "pause_event": self._process_pause_event,
                                "paused_event": self._process_paused_events[env_id],
                                "error_queue": self._worker_error_queue,
                                "worker_affinity": (
                                    self._worker_affinity[env_id]
                                    if self._worker_affinity is not None
                                    else None
                                ),
                                "env_device": self._env_device,
                                "storing_device": self._storing_device,
                                "chunk_size": self._transition_chunk_size,
                            },
                            name=f"AsyncBatchedCollector-env-{env_id}",
                            daemon=True,
                        )
                        process.start()
                        self._workers.append(process)
                except BaseException:
                    self._shutdown_event.set()
                    for process in self._workers:
                        if process.is_alive():
                            process.terminate()
                        process.join(timeout=1.0)
                    self._workers = []
                    self._server.shutdown(timeout=1.0)
                    raise
                return

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
                        interaction_type=self.exploration_type,
                    )
                    for _ in range(self._num_envs)
                ]

            if self._server.static_batch_size is not None:
                request_spec = self._env_pool.fake_tensordict()[0]
                self._server.prepare_cudagraph(
                    request_spec, interaction_type=self.exploration_type
                )

            # Start inference server
            if not self._server.is_alive:
                self._server.start()
            self._apply_pending_weights()

            # Start coordinator threads. Shared slots can be drained safely in
            # ready batches, avoiding one Python thread and one clone per env.
            self._uses_batched_coordinator = (
                self._env_pool.resolved_exchange == "shm" or self._envs_per_worker > 1
            )
            # A ready packet contains at most one transition per environment.
            # Bound the queue by roughly one rollout, plus in-flight completions.
            queue_size = self.frames_per_batch
            if self._uses_batched_coordinator:
                queue_size = max(
                    1, queue_size // self._env_pool.fake_tensordict().numel()
                )
            self._result_queue = queue.Queue(maxsize=queue_size)
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
        entered. The coordinator threads or environment processes then remain
        parked until the context exits, leaving the inference server idle.
        Completed transitions can remain buffered for the next iteration.
        This provides a quiescent boundary for operations such as a lazy
        :func:`torch.compile` call.

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
        replay_locked = False
        try:
            if not workers:
                yield None
                return
            if not all(worker.is_alive() for worker in workers):
                raise RuntimeError(
                    "AsyncBatchedCollector cannot pause because a coordinator "
                    "thread has exited."
                )

            if self._uses_process_env_workers:
                self._process_pause_event.set()
                pause_timer = timeit("AsyncBatchedCollector.process_pause", sync=False)
                pause_timer.start()
                while not all(event.is_set() for event in self._process_paused_events):
                    self._check_process_workers()
                    if not self._server.is_alive:
                        raise RuntimeError("The inference server exited while pausing.")
                    if timeout is not None and pause_timer.elapsed() >= timeout:
                        raise TimeoutError(
                            "Timed out while waiting for environment workers to pause."
                        )
                    self._shutdown_event.wait(0.01)
                replay_locked = self._replay_lock.acquire(
                    timeout=-1 if timeout is None else timeout
                )
                if not replay_locked:
                    raise TimeoutError(
                        "Timed out waiting for the replay writer to pause."
                    )
                yield None
                return

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

            replay_locked = self._replay_lock.acquire(
                timeout=-1 if timeout is None else timeout
            )
            if not replay_locked:
                raise TimeoutError("Timed out waiting for the replay writer to pause.")
            yield None
        finally:
            try:
                if replay_locked:
                    self._replay_lock.release()
                if self._uses_process_env_workers and workers:
                    self._process_pause_event.clear()
                    # Wait for acknowledgement reset before a subsequent pause.
                    while any(event.is_set() for event in self._process_paused_events):
                        if self._shutdown_event.wait(0.01) or not all(
                            worker.is_alive() for worker in workers
                        ):
                            break
                if request is not None:
                    if self._pause_request[0] is request:
                        self._pause_request[0] = None
                    request[0].abort()
                    request[1].set()
            finally:
                self._pause_lock.release()

    @property
    def server_backend(self) -> str:
        """The resolved inference server backend: ``"thread"``, ``"process"`` or ``"ray"``."""
        return self._server_backend

    @property
    def env(self) -> AsyncEnvPool:
        """The underlying :class:`AsyncEnvPool`."""
        self._ensure_started()
        if self._env_pool is None:
            raise RuntimeError(
                "ProcessSlotTransport runs environments directly in worker "
                "processes and does not create an AsyncEnvPool."
            )
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

    def _maybe_fallback_update(
        self,
        policy_or_weights=None,
        *,
        model_id: str | None = None,
    ) -> None:
        if model_id not in (None, "policy"):
            raise KeyError(f"Unknown AsyncBatchedCollector model_id {model_id!r}.")
        if isinstance(policy_or_weights, torch.nn.Module):
            weights = TensorDict.from_module(policy_or_weights)
        elif isinstance(policy_or_weights, TensorDictBase):
            weights = policy_or_weights
        elif isinstance(policy_or_weights, dict):
            weights = TensorDict.from_dict(policy_or_weights, batch_size=[])
        else:
            raise TypeError(
                "AsyncBatchedCollector policy updates require an nn.Module, "
                "TensorDict, or state dictionary."
            )
        weights = weights.detach().clone()
        update_model_weights = getattr(self._server, "update_model_weights", None)
        if update_model_weights is not None:
            if not self._server.is_alive:
                # The server process starts with the first iteration. Keep the
                # latest weights and apply them right after it starts, before
                # any request is served, so a policy rebuilt from a factory acts
                # with the trainer's weights from the first step.
                self._pending_weights = weights
                return
            update_model_weights(weights)
        else:
            self._server.update_model(
                ft.partial(WeightStrategy().apply_weights, weights=weights)
            )

    def _apply_pending_weights(self) -> None:
        if self._pending_weights is None:
            return
        weights, self._pending_weights = self._pending_weights, None
        self._server.update_model_weights(weights)

    # ------------------------------------------------------------------
    # Rollout: drain the result queue
    # ------------------------------------------------------------------

    _SERVER_DEATH_GRACE_S = 2.0

    def _check_worker_result(self, item):
        """Re-raise exceptions propagated from collector workers.

        Workers may observe a dying server before the liveness
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
            worker_kind = "process" if self._uses_process_env_workers else "thread"
            raise RuntimeError(
                f"A collector worker {worker_kind} raised an exception."
            ) from item

    _LIVENESS_POLL_S = 1.0

    def _check_process_workers(self) -> None:
        try:
            error = self._worker_error_queue.get_nowait()
        except queue.Empty:
            pass
        else:
            raise RuntimeError("An environment worker failed.") from error
        if any(not worker.is_alive() for worker in self._workers):
            raise RuntimeError("An environment worker process exited while collecting.")

    def _next_result(self) -> TensorDictBase:
        """Block for the next transition, watching server and worker liveness.

        A dead inference server (e.g. an OOM-killed server process) would
        otherwise leave every coordinator thread blocked on a response and
        this method blocked on the queue, hanging the iterator forever with
        no error.
        """
        rq = self._result_queue
        while True:
            if self._shutdown_event.is_set():
                raise _CollectorStopped(
                    "The collector has stopped collecting transitions."
                )
            if (
                self._uses_process_env_workers
                and self._worker_check_timer.elapsed() >= self._LIVENESS_POLL_S
            ):
                self._worker_check_timer.start()
                self._check_process_workers()
            try:
                td = rq.get(timeout=self._LIVENESS_POLL_S)
            except queue.Empty:
                if self._shutdown_event.is_set():
                    continue
                if not self._server.is_alive:
                    raise RuntimeError(
                        "The inference server died while the collector was "
                        "waiting for transitions. Check the server process "
                        "logs (e.g. OOM kills or exceptions in the policy)."
                    ) from None
                if self._workers and not any(w.is_alive() for w in self._workers):
                    raise RuntimeError(
                        "All collector workers exited while the "
                        "collector was waiting for transitions."
                    ) from None
                if self._uses_process_env_workers:
                    exited = [
                        worker for worker in self._workers if not worker.is_alive()
                    ]
                    if exited:
                        details = ", ".join(
                            f"{worker.name} (exitcode={worker.exitcode})"
                            for worker in exited
                        )
                        raise RuntimeError(
                            "Environment worker process exited while the "
                            f"collector was running: {details}."
                        ) from None
                continue
            except (EOFError, OSError) as exc:
                if not self._uses_process_env_workers:
                    raise
                self._check_process_workers()
                raise RuntimeError(
                    "An environment worker result could not be received."
                ) from exc
            self._check_worker_result(td)
            if self._uses_process_env_workers:
                self._result_capacity.release()
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
            if self._uses_chunked_results:
                collected = self._append_chunk(
                    transition, transitions, collected, frames_to_collect
                )
            else:
                transitions.append(transition)
                collected += transition.numel()

        while collected < frames_to_collect:
            # Block for at least one transition
            td = self._next_result()
            collected = self._append_result(
                td, transitions, collected, frames_to_collect
            )
            # Batch-drain any additional items already in the queue
            while collected < frames_to_collect:
                try:
                    td = rq.get_nowait()
                except queue.Empty:
                    break
                except (EOFError, OSError) as exc:
                    if not self._uses_process_env_workers:
                        raise
                    self._check_process_workers()
                    raise RuntimeError(
                        "An environment worker result could not be received."
                    ) from exc
                self._check_worker_result(td)
                if self._uses_process_env_workers:
                    self._result_capacity.release()
                collected = self._append_result(
                    td, transitions, collected, frames_to_collect
                )
            if self.verbose:
                torchrl_logger.debug(
                    f"AsyncBatchedCollector: {collected}/{self.frames_per_batch} frames"
                )

        if self._uses_chunked_results:
            return self._cat_chunks(transitions)
        return lazy_stack(transitions)

    def _append_result(
        self,
        td: TensorDictBase,
        transitions: list[TensorDictBase],
        collected: int,
        budget: int,
    ) -> int:
        """Append one worker result, carrying frames beyond ``budget``."""
        if self._uses_batched_coordinator:
            for transition in td.unbind(0):
                if collected < budget:
                    transitions.append(transition)
                    collected += transition.numel()
                else:
                    self._transition_carry.append(transition)
            return collected
        if self._uses_chunked_results:
            return self._append_chunk(td, transitions, collected, budget)
        transitions.append(td)
        return collected + td.numel()

    def _append_chunk(
        self,
        chunk: TensorDictBase,
        transitions: list[TensorDictBase],
        collected: int,
        budget: int,
    ) -> int:
        """Append a chunk of consecutive transitions from one environment.

        The chunk stays a dense block; it is only split along its leading
        dimension where the frame budget ends, and the remainder is carried
        over to the next batch in order. Like every other result, a single
        transition is never split, so a budget that is not a multiple of one
        transition's size is overshot by less than one transition.
        """
        steps = chunk.shape[0]
        step_numel = chunk.numel() // steps
        room = max(1, (budget - collected) // step_numel)
        if steps > room:
            self._transition_carry.append(chunk[room:])
            chunk = chunk[:room]
        transitions.append(chunk)
        return collected + chunk.numel()

    @staticmethod
    def _cat_chunks(chunks: list[TensorDictBase]) -> TensorDictBase:
        """Concatenate worker chunks into one dense batch."""
        try:
            return torch.cat(chunks, 0)
        except (KeyError, RuntimeError, TypeError):
            # Different schemas or non-tensor metadata representations may
            # prevent concatenating the chunks into a dense batch.
            return lazy_stack([row for chunk in chunks for row in chunk.unbind(0)])

    @_maybe_record_function_decorator("AsyncBatchedCollector._rollout_yield_trajs")
    def _rollout_yield_trajs(self) -> TensorDictBase:
        """Drain transitions until a complete trajectory is available."""
        while not self._trajectory_queue:
            td = self._next_result()
            if self._uses_batched_coordinator or self._uses_chunked_results:
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

    def start(self) -> None:
        """Collect into replay in a background thread until ``total_frames``.

        Requires ``replay_buffer``. Post-processing and the post-collect hook
        run on the writer thread. Iteration and background collection are
        mutually exclusive. Use :meth:`pause` to quiesce collection and writes,
        and :meth:`async_shutdown` to join the writer and surface its errors.
        """
        if self.replay_buffer is None:
            raise RuntimeError("Background collection requires a replay_buffer.")
        if self._replay_thread is not None or self._iteration_lock.locked():
            raise RuntimeError("The collector is already collecting.")
        self._ensure_started()
        self._replay_error = None
        self._iteration_started = True
        self._replay_thread = threading.Thread(
            target=self._run_replay,
            name="AsyncBatchedCollector-replay",
            daemon=True,
        )
        self._replay_thread.start()

    def _run_replay(self) -> None:
        try:
            for _ in self.iterator():
                pass
        except Exception as exc:
            self._replay_error = exc
            self._shutdown_event.set()

    def __iter__(self) -> Iterator[TensorDictBase | None]:
        if self._replay_thread is not None:
            # Reject concurrent consumption before BaseCollector's iterator
            # error handler shuts down the active background collector.
            raise RuntimeError("Background collection owns the collector iterator.")
        return super().__iter__()

    def iterator(self) -> Iterator[TensorDictBase | None]:
        """Iterate over collected batches."""
        if self._replay_thread is not None and (
            threading.current_thread() is not self._replay_thread
        ):
            raise RuntimeError("Background collection owns the collector iterator.")
        self._ensure_started()
        if not self._iteration_lock.acquire(blocking=False):
            raise RuntimeError("The collector is already collecting.")
        try:
            total = self.total_frames
            while (
                total < 0 or self._frames < total
            ) and not self._shutdown_event.is_set():
                self._iter += 1
                td = self.rollout()
                # The pause boundary includes post-processing and replay writes.
                # Never hold this lock while waiting for a paused environment.
                while not self._replay_lock.acquire(timeout=0.1):
                    if self._shutdown_event.is_set():
                        return
                try:
                    if self._shutdown_event.is_set():
                        return
                    if not self.yield_completed_trajectories and total >= 0:
                        remaining = total - self._frames
                        if td.numel() > remaining:
                            td = td.reshape(-1)[:remaining]
                    self._frames += td.numel()
                    if self._postproc is not None:
                        td = self._postproc(td)
                    if self.post_collect_hook is not None:
                        self.post_collect_hook(td)
                    if self.replay_buffer is not None:
                        td = _maybe_normalize_replay_buffer_tensordict_device(
                            td, self.replay_buffer
                        )
                        self.replay_buffer.extend(td)
                        td = None
                finally:
                    self._replay_lock.release()
                yield td
        except _CollectorStopped:
            return
        finally:
            self._iteration_lock.release()

    def shutdown(
        self,
        timeout: float | None = None,
        close_env: bool = True,
        raise_on_error: bool = True,
    ) -> None:
        """Shut down the collector, inference server, threads and env pool."""
        if self._shutdown_event is not None and self._workers:
            self._shutdown_event.set()
        request = self._pause_request[0]
        self._pause_request[0] = None
        if request is not None:
            request[0].abort()
            request[1].set()
        self._transition_carry.clear()
        _timeout = 5.0 if timeout is None else timeout
        shutdown_timer = timeit("AsyncBatchedCollector.shutdown").start()
        replay_thread = self._replay_thread
        if replay_thread is not None:
            replay_thread.join(timeout=_timeout)
            if not replay_thread.is_alive():
                self._replay_thread = None
        if self._uses_process_env_workers:
            while any(worker.is_alive() for worker in self._workers):
                if shutdown_timer.elapsed() >= _timeout:
                    break
                if self._result_queue is not None:
                    while True:
                        try:
                            self._result_queue.get_nowait()
                        except queue.Empty:
                            break
                        except (EOFError, OSError):
                            # The item is already removed from the queue, but
                            # rebuilding a discarded tensor can fail once its
                            # worker's resource sharer has exited.
                            continue
                for worker in self._workers:
                    worker.join(timeout=0.05)
            for worker in self._workers:
                if worker.is_alive():
                    worker.terminate()
                worker.join(timeout=1.0)
        else:
            for worker in self._workers:
                worker.join(timeout=_timeout)
        self._workers = []
        self._server.shutdown(timeout=max(0.0, _timeout - shutdown_timer.elapsed()))
        if close_env and self._env_pool is not None:
            self._env_pool.close(raise_if_closed=raise_on_error)
            self._env_pool = None
        if self._uses_process_env_workers and self._result_queue is not None:
            self._result_queue.close()
            self._result_queue.join_thread()
            self._result_queue = None
        if self._worker_error_queue is not None:
            self._worker_error_queue.close()
            self._worker_error_queue.join_thread()
            self._worker_error_queue = None

        if raise_on_error:
            if replay_thread is not None and replay_thread.is_alive():
                raise TimeoutError(
                    "The replay writer did not stop before shutdown timed out."
                )
            if self._replay_error is not None:
                error = self._replay_error
                self._replay_error = None
                raise RuntimeError("Background replay collection failed.") from error

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
