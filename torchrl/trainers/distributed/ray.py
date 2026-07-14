# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import importlib.util
import random
import socket
import threading
import time
from collections.abc import Callable, Mapping
from datetime import timedelta
from functools import lru_cache
from typing import Any, Literal, TYPE_CHECKING

import numpy as np
import torch
from tensordict import TensorDict, TensorDictBase

from torchrl.distributed import DataParallelContext
from torchrl.trainers.learners import (
    Learner,
    LearnerContext,
    LearnerGroup,
    LearnerStepRequest,
    LearnerStepResult,
    LearnerWeights,
)

if TYPE_CHECKING:
    from typing import Self

_has_ray = importlib.util.find_spec("ray") is not None


@lru_cache(None)
def _ray():
    if not _has_ray:
        raise ImportError("RayLearnerGroup requires the ray package.")
    import ray

    return ray


class _RayLearnerWorker:
    def __init__(
        self,
        learner_factory: Callable[[LearnerContext], Learner],
        replay_buffer: Any,
        generation: int,
        seed: int | None,
    ) -> None:
        self.learner_factory = learner_factory
        self.replay_buffer = replay_buffer
        self.generation = generation
        self.seed = seed
        self.learner: Learner | None = None

    def probe(self) -> dict[str, Any]:
        ray = _ray()
        return {
            "node_id": ray.get_runtime_context().get_node_id(),
            "node_ip": ray.util.get_node_ip_address(),
            "gpu_ids": list(ray.get_gpu_ids()),
        }

    def reserve_rendezvous(self) -> int:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("", 0))
            return int(sock.getsockname()[1])

    def setup(
        self,
        *,
        rank: int,
        local_rank: int,
        world_size: int,
        device: str,
        backend: str,
        init_method: str,
        timeout: float,
    ) -> dict[str, Any]:
        context = DataParallelContext.from_rendezvous(
            rank=rank,
            world_size=world_size,
            local_rank=local_rank,
            device=device,
            backend=backend,
            init_method=init_method,
            timeout=timedelta(seconds=timeout),
        )
        rank_seed = None if self.seed is None else self.seed + rank
        if rank_seed is not None:
            random.seed(rank_seed)
            np.random.seed(rank_seed)
            torch.manual_seed(rank_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(rank_seed)
        replay_buffer = self.replay_buffer.data_parallel(
            rank=rank, world_size=world_size
        )
        factory_context = LearnerContext(
            rank=rank,
            local_rank=local_rank,
            world_size=world_size,
            device=context.device,
            generation=self.generation,
            replay_buffer=replay_buffer,
            data_parallel_context=context,
            seed=rank_seed,
        )
        try:
            learner = self.learner_factory(factory_context)
            if not isinstance(learner, Learner):
                raise TypeError(
                    "learner_factory must return Learner, got "
                    f"{type(learner).__name__}."
                )
            self.learner = learner.initialize()
        except BaseException:
            context.close()
            raise
        return {
            "rank": rank,
            "local_rank": local_rank,
            "device": str(context.device),
            "model_version": self.learner.model_version,
            "last_round": self.learner.last_round,
        }

    def begin_round(self, request: LearnerStepRequest) -> None:
        self._require_learner()._begin_round(request)

    def step_round_once(self) -> LearnerStepResult:
        return self._require_learner()._step_round_once()

    def finish_round(self) -> LearnerStepResult:
        return self._require_learner()._finish_round()

    def get_weights(self, model_id: str) -> LearnerWeights:
        return self._require_learner().get_weights(model_id)

    def state_dict(self) -> dict[str, Any]:
        return self._require_learner().state_dict()

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> dict[str, int]:
        learner = self._require_learner()
        learner.load_state_dict(state_dict)
        return {
            "last_round": learner.last_round,
            "model_version": learner.model_version,
        }

    def close(self) -> None:
        if self.learner is not None:
            self.learner.close()
            self.learner = None

    def _require_learner(self) -> Learner:
        if self.learner is None:
            raise RuntimeError("Ray learner worker is not initialized.")
        return self.learner


class RayLearnerGroup(LearnerGroup):
    """Own a fixed, synchronously commanded gang of Ray learner actors.

    The group is the collective-safety boundary: callers never receive actor
    handles, all optimizer substeps are dispatched to every rank concurrently,
    and any rank failure invalidates the full generation.

    Args:
        learner_factory (callable): Picklable actor-local learner factory.
        replay_buffer: Lifecycle-free Ray replay-buffer client.
        world_size (int): Number of learner actors.
        global_batch_size (int): Batch size across all actors.
        resources_per_rank (mapping, optional): Ray actor resources using
            ``num_cpus``, ``num_gpus``, and optional custom resource names.
        placement_strategy (str): Ray placement strategy. Defaults to ``"PACK"``.
        backend (str, optional): Process-group backend. Defaults to Gloo for CPU
            actors and NCCL for GPU actors.
        setup_timeout (float): Gang allocation and setup timeout in seconds.
        command_timeout (float): Per-command timeout in seconds.
        seed (int, optional): Base learner seed; each rank receives ``seed + rank``.
        learner_id (str, optional): Stable checkpoint identity. Defaults to the
            factory's qualified name.
        ray_init_config (mapping, optional): Configuration used only when Ray is
            not already initialized. Group teardown never calls ``ray.shutdown``.

    Example:
        A controller starts the gang, issues one global-batch command to every
        rank, and always tears the generation down as one unit:

        >>> from torchrl.trainers import LearnerStepRequest
        >>> from torchrl.trainers.distributed import RayLearnerGroup
        >>> def run_one_round(learner_factory, replay_client):
        ...     group = RayLearnerGroup(
        ...         learner_factory,
        ...         replay_client,
        ...         world_size=2,
        ...         global_batch_size=256,
        ...     ).start()
        ...     try:
        ...         return group.step(LearnerStepRequest(1, 1, 256))
        ...     finally:
        ...         group.shutdown()
    """

    def __init__(
        self,
        learner_factory: Callable[[LearnerContext], Learner],
        replay_buffer: Any,
        *,
        world_size: int,
        global_batch_size: int,
        resources_per_rank: Mapping[str, float] | None = None,
        placement_strategy: Literal[
            "PACK", "SPREAD", "STRICT_PACK", "STRICT_SPREAD"
        ] = "PACK",
        backend: str | None = None,
        setup_timeout: float = 300.0,
        command_timeout: float = 300.0,
        seed: int | None = None,
        learner_id: str | None = None,
        ray_init_config: Mapping[str, Any] | None = None,
    ) -> None:
        if isinstance(world_size, bool) or not isinstance(world_size, int):
            raise TypeError("world_size must be an integer.")
        if world_size <= 0:
            raise ValueError("world_size must be positive.")
        if global_batch_size <= 0 or global_batch_size % world_size:
            raise ValueError(
                "global_batch_size must be positive and divisible by world_size."
            )
        if setup_timeout <= 0 or command_timeout <= 0:
            raise ValueError("setup_timeout and command_timeout must be positive.")
        self.learner_factory = learner_factory
        self.replay_buffer = replay_buffer
        self._world_size = world_size
        self._global_batch_size = global_batch_size
        self.resources_per_rank = dict(
            resources_per_rank or {"num_cpus": 1.0, "num_gpus": 0.0}
        )
        self.placement_strategy = placement_strategy
        self.backend = backend
        self.setup_timeout = float(setup_timeout)
        self.command_timeout = float(command_timeout)
        self.seed = seed
        self.learner_id = learner_id or self._factory_id(learner_factory)
        self.ray_init_config = dict(ray_init_config or {})
        self._lock = threading.RLock()
        self._actors: list[Any] = []
        self._placement_group = None
        self._state = "created"
        self._generation = 0
        self._last_round = 0
        self._model_version = 0

    @property
    def world_size(self) -> int:
        return self._world_size

    @property
    def global_batch_size(self) -> int:
        return self._global_batch_size

    @property
    def is_alive(self) -> bool:
        return self._state == "running" and len(self._actors) == self.world_size

    @property
    def generation(self) -> int:
        """Current learner-group generation."""
        return self._generation

    @property
    def last_round(self) -> int:
        """Last fully completed controller round."""
        return self._last_round

    @property
    def model_version(self) -> int:
        """Version after the last completed optimizer step."""
        return self._model_version

    def start(self) -> Self:
        with self._lock:
            if self.is_alive:
                return self
            if self._state == "failed":
                raise RuntimeError(
                    "A failed RayLearnerGroup cannot be restarted; construct a new "
                    "group generation."
                )
            ray = _ray()
            if not ray.is_initialized():
                ray.init(**self.ray_init_config)
            self._generation += 1
            self._state = "starting"
            try:
                self._start_actors()
            except BaseException as err:
                self._mark_failed()
                raise RuntimeError(
                    f"Ray learner group generation {self.generation} failed during "
                    "startup."
                ) from err
            self._state = "running"
            self._last_round = 0
            self._model_version = 0
            return self

    def step(self, request: LearnerStepRequest) -> LearnerStepResult:
        with self._lock:
            self._ensure_running()
            if request.global_batch_size != self.global_batch_size:
                raise ValueError(
                    "LearnerStepRequest.global_batch_size must match the group."
                )
            expected_round = self.last_round + 1
            if request.round_id != expected_round:
                raise RuntimeError(
                    f"Expected round_id={expected_round}, got {request.round_id}."
                )
            ray = _ray()
            try:
                self._get_ranked(
                    [actor.begin_round.remote(request) for actor in self._actors],
                    timeout=self.command_timeout,
                )
                for _ in range(request.num_steps):
                    self._wait_for_replay(request.global_batch_size)
                    substep_results = self._get_ranked(
                        [actor.step_round_once.remote() for actor in self._actors],
                        timeout=self.command_timeout,
                    )
                    self._validate_results(
                        substep_results, request.round_id, expected_steps=1
                    )
                results = self._get_ranked(
                    [actor.finish_round.remote() for actor in self._actors],
                    timeout=self.command_timeout,
                )
                self._validate_results(
                    results, request.round_id, expected_steps=request.num_steps
                )
            except BaseException as err:
                self._mark_failed()
                raise RuntimeError(
                    "Ray learner group failed: "
                    f"generation={self.generation}, round={request.round_id}, "
                    f"error={err}."
                ) from err
            del ray
            self._last_round = request.round_id
            self._model_version = results[0].model_version
            return LearnerStepResult(
                round_id=request.round_id,
                optim_steps=request.num_steps,
                model_version=self.model_version,
                metrics=self._aggregate_metrics(results),
            )

    def get_weights(self, model_id: str = "policy") -> LearnerWeights:
        with self._lock:
            self._ensure_running()
            try:
                snapshot = self._get(
                    self._actors[0].get_weights.remote(model_id),
                    timeout=self.command_timeout,
                )
            except BaseException as err:
                self._mark_failed()
                raise RuntimeError(
                    "Rank-zero weight extraction failed for learner group "
                    f"generation {self.generation}."
                ) from err
            if snapshot.model_version != self.model_version:
                raise RuntimeError(
                    "Rank-zero snapshot version does not match the learner group: "
                    f"{snapshot.model_version} != {self.model_version}."
                )
            return snapshot

    def state_dict(self) -> dict[str, Any]:
        with self._lock:
            self._ensure_running()
            states = self._get(
                [actor.state_dict.remote() for actor in self._actors],
                timeout=self.command_timeout,
            )
            replicated = dict(states[0])
            replicated.pop("rng", None)
            return {
                "format_version": 1,
                "generation": self.generation,
                "world_size": self.world_size,
                "global_batch_size": self.global_batch_size,
                "last_round": self.last_round,
                "model_version": self.model_version,
                "learner_id": self.learner_id,
                "replicated": replicated,
                "rng_by_rank": [state["rng"] for state in states],
            }

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        with self._lock:
            self._ensure_running()
            if int(state_dict["world_size"]) != self.world_size:
                raise ValueError(
                    "Cannot restore a learner group with a new world size."
                )
            if int(state_dict["global_batch_size"]) != self.global_batch_size:
                raise ValueError("Checkpoint global_batch_size does not match.")
            if state_dict["learner_id"] != self.learner_id:
                raise ValueError(
                    f"Checkpoint learner_id={state_dict['learner_id']!r} does not "
                    f"match {self.learner_id!r}."
                )
            rank_states = []
            for rng in state_dict["rng_by_rank"]:
                rank_state = dict(state_dict["replicated"])
                rank_state["rng"] = rng
                rank_states.append(rank_state)
            results = self._get(
                [
                    actor.load_state_dict.remote(rank_state)
                    for actor, rank_state in zip(self._actors, rank_states)
                ],
                timeout=self.command_timeout,
            )
            if any(result != results[0] for result in results[1:]):
                self._mark_failed()
                raise RuntimeError("Learner ranks disagreed after state restoration.")
            self._last_round = int(state_dict["last_round"])
            self._model_version = int(state_dict["model_version"])

    def shutdown(self, timeout: float | None = None) -> None:
        with self._lock:
            if not self._actors and self._placement_group is None:
                self._state = "stopped"
                return
            ray = _ray()
            if self._state == "running":
                try:
                    self._get(
                        [actor.close.remote() for actor in self._actors],
                        timeout=timeout or min(self.command_timeout, 10.0),
                    )
                except BaseException:
                    pass
            for actor in self._actors:
                try:
                    ray.kill(actor, no_restart=True)
                except BaseException:
                    pass
            self._actors = []
            self._remove_placement_group()
            self._state = "stopped"

    def _start_actors(self) -> None:
        ray = _ray()
        from ray.util.placement_group import placement_group
        from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

        num_cpus = float(self.resources_per_rank.get("num_cpus", 1.0))
        num_gpus = float(self.resources_per_rank.get("num_gpus", 0.0))
        custom_resources = {
            key: float(value)
            for key, value in self.resources_per_rank.items()
            if key not in {"num_cpus", "num_gpus"}
        }
        bundle = {"CPU": num_cpus, **custom_resources}
        if num_gpus:
            bundle["GPU"] = num_gpus
        self._placement_group = placement_group(
            [bundle.copy() for _ in range(self.world_size)],
            strategy=self.placement_strategy,
        )
        self._get(self._placement_group.ready(), timeout=self.setup_timeout)

        remote_worker = ray.remote(_RayLearnerWorker)
        for rank in range(self.world_size):
            strategy = PlacementGroupSchedulingStrategy(
                placement_group=self._placement_group,
                placement_group_bundle_index=rank,
                placement_group_capture_child_tasks=True,
            )
            actor = remote_worker.options(
                num_cpus=num_cpus,
                num_gpus=num_gpus,
                resources=custom_resources,
                max_restarts=0,
                max_task_retries=0,
                scheduling_strategy=strategy,
            ).remote(
                self.learner_factory,
                self.replay_buffer,
                self.generation,
                self.seed,
            )
            self._actors.append(actor)

        probes = self._get(
            [actor.probe.remote() for actor in self._actors],
            timeout=self.setup_timeout,
        )
        local_ranks: dict[str, int] = {}
        assigned_local_ranks = []
        for probe in probes:
            node_id = probe["node_id"]
            local_rank = local_ranks.get(node_id, 0)
            local_ranks[node_id] = local_rank + 1
            assigned_local_ranks.append(local_rank)

        port = self._get(
            self._actors[0].reserve_rendezvous.remote(), timeout=self.setup_timeout
        )
        init_method = f"tcp://{probes[0]['node_ip']}:{port}"
        device = "cuda:0" if num_gpus else "cpu"
        backend = self.backend or ("nccl" if num_gpus else "gloo")
        setup_refs = [
            actor.setup.remote(
                rank=rank,
                local_rank=assigned_local_ranks[rank],
                world_size=self.world_size,
                device=device,
                backend=backend,
                init_method=init_method,
                timeout=self.setup_timeout,
            )
            for rank, actor in enumerate(self._actors)
        ]
        setup_results = self._get_ranked(setup_refs, timeout=self.setup_timeout)
        if [result["rank"] for result in setup_results] != list(range(self.world_size)):
            raise RuntimeError("Ray learners returned inconsistent rank metadata.")

    def _wait_for_replay(self, global_batch_size: int) -> None:
        deadline = time.monotonic() + self.command_timeout
        while len(self.replay_buffer) < global_batch_size:
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    "Timed out waiting for the replay buffer to contain a global "
                    f"batch of {global_batch_size} items."
                )
            time.sleep(0.01)

    @staticmethod
    def _validate_results(
        results: list[LearnerStepResult], round_id: int, *, expected_steps: int
    ) -> None:
        if not results:
            raise RuntimeError("No learner results were returned.")
        first = results[0]
        first_keys = set(first.metrics.keys(True, True))
        first_shapes = {key: tuple(first.metrics.get(key).shape) for key in first_keys}
        for rank, result in enumerate(results):
            if result.round_id != round_id:
                raise RuntimeError(
                    f"Rank {rank} returned round {result.round_id}, expected {round_id}."
                )
            if result.optim_steps != expected_steps:
                raise RuntimeError(
                    f"Rank {rank} returned {result.optim_steps} optimizer steps, "
                    f"expected {expected_steps}."
                )
            if result.model_version != first.model_version:
                raise RuntimeError("Learner model versions diverged.")
            keys = set(result.metrics.keys(True, True))
            shapes = {key: tuple(result.metrics.get(key).shape) for key in keys}
            if keys != first_keys or shapes != first_shapes:
                raise RuntimeError("Learner metric structures diverged.")
            for key in keys:
                value = result.metrics.get(key)
                if value.numel() != 1 or not torch.isfinite(value).all():
                    raise RuntimeError(
                        f"Rank {rank} returned invalid scalar metric {key!r}."
                    )

    @staticmethod
    def _aggregate_metrics(results: list[LearnerStepResult]) -> TensorDictBase:
        if not results:
            return TensorDict(device="cpu")
        output = results[0].metrics.clone()
        for key in output.keys(True, True):
            output.set(
                key,
                torch.stack([result.metrics.get(key) for result in results]).mean(),
            )
        return output

    def _get(self, refs: Any, *, timeout: float) -> Any:
        return _ray().get(refs, timeout=timeout)

    def _get_ranked(self, refs: list[Any], *, timeout: float) -> list[Any]:
        ray = _ray()
        pending = {ref: rank for rank, ref in enumerate(refs)}
        results: list[Any] = [None] * len(refs)
        deadline = time.monotonic() + timeout
        while pending:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                ranks = sorted(pending.values())
                raise TimeoutError(f"Timed out waiting for learner ranks {ranks}.")
            ready, _ = ray.wait(
                list(pending), num_returns=1, timeout=remaining, fetch_local=True
            )
            if not ready:
                ranks = sorted(pending.values())
                raise TimeoutError(f"Timed out waiting for learner ranks {ranks}.")
            ref = ready[0]
            rank = pending.pop(ref)
            try:
                results[rank] = ray.get(ref)
            except BaseException as err:
                raise RuntimeError(f"Learner rank={rank} failed.") from err
        return results

    def _mark_failed(self) -> None:
        self._state = "failed"
        ray = _ray()
        for actor in self._actors:
            try:
                ray.kill(actor, no_restart=True)
            except BaseException:
                pass
        self._actors = []
        self._remove_placement_group()

    def _remove_placement_group(self) -> None:
        if self._placement_group is None:
            return
        try:
            from ray.util.placement_group import remove_placement_group

            remove_placement_group(self._placement_group)
        except BaseException:
            pass
        self._placement_group = None

    def _ensure_running(self) -> None:
        if not self.is_alive:
            raise RuntimeError(
                f"RayLearnerGroup is not running (state={self._state!r})."
            )

    @staticmethod
    def _factory_id(factory: Callable[[LearnerContext], Learner]) -> str:
        module = getattr(factory, "__module__", type(factory).__module__)
        qualname = getattr(factory, "__qualname__", type(factory).__qualname__)
        return f"{module}.{qualname}"


__all__ = ["RayLearnerGroup"]
