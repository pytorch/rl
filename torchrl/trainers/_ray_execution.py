# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Private Ray execution backend for Trainer-owned optimization."""

from __future__ import annotations

import contextlib
import importlib.util
import random
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import torch
from tensordict import TensorDict, TensorDictBase
from tensordict.utils import NestedKey
from torch import optim

from torchrl._comm.ray_runtime import _RayRuntimeLease
from torchrl.objectives.common import LossModule
from torchrl.objectives.utils import TargetNetUpdater
from torchrl.trainers._distributed import (
    _connect_tcp_store,
    _create_tcp_store,
    _DDPProcessGroup,
)
from torchrl.trainers._execution import _ExecutionStep, _Learner
from torchrl.trainers.trainers import OptimizationStepper

_has_ray = importlib.util.find_spec("ray") is not None
if _has_ray:
    import ray
    from ray.util.placement_group import placement_group, remove_placement_group


@dataclass
class _LearnerObjects:
    """One connected object graph serialized to each learner actor."""

    loss_module: LossModule
    optimizer: optim.Optimizer | None
    optimization_stepper: OptimizationStepper | None
    target_net_updater: TargetNetUpdater | None


class _RayLearnerWorker:
    def __init__(self, objects: _LearnerObjects, replay_client: Any) -> None:
        self.objects = objects
        self.replay_client = replay_client
        self.master_store = None
        self.learner = None
        self.weight_sync_scheme = None
        self.generation = 0

    def create_store(self, host: str | None, timeout: float) -> tuple[str, int]:
        self.master_store, coordinates = _create_tcp_store(host=host, timeout=timeout)
        return coordinates

    def setup(
        self,
        *,
        rank: int,
        world_size: int,
        local_batch_size: int,
        coordinates: tuple[str, int],
        backend: Literal["gloo", "nccl"],
        generation: int,
        timeout: float,
        seed: int | None,
        clip_grad_norm: bool,
        clip_norm: float | None,
        update_replay_priority: bool,
    ) -> dict[str, int]:
        if seed is not None:
            rank_seed = seed + rank
            random.seed(rank_seed)
            np.random.seed(rank_seed)
            torch.manual_seed(rank_seed)
        store = (
            self.master_store
            if rank == 0
            else _connect_tcp_store(coordinates, timeout=timeout)
        )
        process_group = _DDPProcessGroup.create(
            rank=rank,
            local_rank=0,
            world_size=world_size,
            store=store,
            backend=backend,
            generation=generation,
            timeout=timeout,
        )
        self.learner = _Learner(
            loss_module=self.objects.loss_module,
            replay_buffer=self.replay_client,
            local_batch_size=local_batch_size,
            optimizer=self.objects.optimizer,
            optimization_stepper=self.objects.optimization_stepper,
            target_net_updater=self.objects.target_net_updater,
            process_group=process_group,
            clip_grad_norm=clip_grad_norm,
            clip_norm=clip_norm,
            update_replay_priority=update_replay_priority,
        ).initialize()
        self.generation = generation
        return {
            "rank": rank,
            "model_version": self.learner.model_version,
            "last_round": self.learner.last_round,
        }

    def step(self, num_steps: int, round_id: int) -> _ExecutionStep:
        return self._require_learner().step(num_steps, round_id)

    def get_weights(
        self, model_id: str, expected_version: int | None
    ) -> TensorDictBase:
        return self._require_learner().get_weights(
            model_id, expected_version=expected_version
        )

    def configure_weight_sync(self, scheme) -> None:
        remote_collectors = scheme._remote_collectors
        scheme.init_on_sender(
            model_id="policy",
            remote_collectors=remote_collectors,
            num_workers=len(remote_collectors),
        )
        self.weight_sync_scheme = scheme

    def publish_weights(
        self,
        *,
        expected_generation: int,
        expected_version: int,
        model_weights_key: NestedKey | None,
        auxiliary_weights: TensorDictBase | None,
    ) -> int:
        if expected_generation != self.generation:
            raise RuntimeError(
                f"Stale learner generation {expected_generation}; active generation "
                f"is {self.generation}."
            )
        if self.weight_sync_scheme is None:
            raise RuntimeError("Learner weight synchronization is not configured.")
        weights = self._require_learner().get_weights(
            "policy", expected_version=expected_version
        )
        if model_weights_key is not None:
            payload = (
                TensorDict()
                if auxiliary_weights is None
                else auxiliary_weights.detach().clone()
            )
            payload.set(model_weights_key, weights)
            weights = payload
        self.weight_sync_scheme._set_model_version(expected_version)
        if not self.weight_sync_scheme.synchronized_on_sender:
            self.weight_sync_scheme.connect(weights=weights)
        else:
            self.weight_sync_scheme.send(weights=weights)
        return expected_version

    def state_dict(self) -> dict[str, Any]:
        return self._require_learner().state_dict()

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> dict[str, int]:
        learner = self._require_learner()
        learner.load_state_dict(state_dict)
        learner.synchronize_after_restore()
        return {
            "model_version": learner.model_version,
            "last_round": learner.last_round,
        }

    def probe(self) -> bool:
        return self.learner is not None

    def close(self) -> None:
        if self.learner is not None:
            self.learner.close()
            self.learner = None
        if self.weight_sync_scheme is not None:
            self.weight_sync_scheme.shutdown()
            self.weight_sync_scheme = None
        self.master_store = None

    def _require_learner(self) -> _Learner:
        if self.learner is None:
            raise RuntimeError("Ray learner worker is not initialized.")
        return self.learner


class _RayTrainerExecution:
    """Gang-scheduled Ray actors implementing the private Trainer boundary."""

    def __init__(
        self,
        *,
        loss_module: LossModule,
        optimizer: optim.Optimizer | None,
        optimization_stepper: OptimizationStepper | None,
        target_net_updater: TargetNetUpdater | None,
        replay_buffer: Any,
        global_batch_size: int,
        options: Mapping[str, Any] | None = None,
        seed: int | None = None,
        clip_grad_norm: bool = True,
        clip_norm: float | None = None,
        update_replay_priority: bool = True,
        weight_sync_scheme: Any | None = None,
        weight_sync_factory: Any | None = None,
    ) -> None:
        if not _has_ray:
            raise ImportError("learner_backend='ray' requires the ray package.")
        options = dict(options or {})
        self.world_size = self._positive_int("world_size", options.pop("world_size", 1))
        self.global_batch_size = self._positive_int(
            "global_batch_size", global_batch_size
        )
        if self.global_batch_size % self.world_size:
            raise ValueError(
                f"Global batch_size ({self.global_batch_size}) must be divisible "
                f"by learner world_size ({self.world_size})."
            )
        self.local_batch_size = self.global_batch_size // self.world_size
        self.resources_per_rank = dict(
            options.pop("resources_per_rank", {"num_cpus": 1, "num_gpus": 0})
        )
        self.placement_strategy = options.pop("placement_strategy", "PACK")
        requested_backend = options.pop("backend", None)
        if requested_backend is None:
            requested_backend = (
                "nccl" if self.resources_per_rank.get("num_gpus", 0) else "gloo"
            )
        if requested_backend not in ("gloo", "nccl"):
            raise ValueError("learner backend must be 'gloo' or 'nccl'.")
        self.backend = requested_backend
        self.setup_timeout = float(options.pop("setup_timeout", 120.0))
        self.command_timeout = float(options.pop("command_timeout", 120.0))
        self.store_host = options.pop("store_host", None)
        self.ray_init_config = dict(options.pop("ray_init_config", {}))
        if options:
            raise TypeError(
                "Unexpected learner_backend_options: " f"{', '.join(sorted(options))}."
            )
        if not hasattr(replay_buffer, "clients"):
            raise TypeError(
                "Ray learner execution requires a replay owner exposing clients()."
            )
        self.replay_buffer = replay_buffer
        self.objects = _LearnerObjects(
            loss_module=loss_module,
            optimizer=optimizer,
            optimization_stepper=optimization_stepper,
            target_net_updater=target_net_updater,
        )
        self.seed = seed
        self.clip_grad_norm = clip_grad_norm
        self.clip_norm = clip_norm
        self.update_replay_priority = update_replay_priority
        self.weight_sync_scheme = weight_sync_scheme
        self.weight_sync_factory = weight_sync_factory
        self.generation = 0
        self.last_round = 0
        self.model_version = 0
        self._actors = []
        self._placement_group = None
        self._runtime_lease = None
        self._failed = False

    def start(self) -> None:
        if self.is_alive():
            return
        self._runtime_lease = _RayRuntimeLease.acquire(self.ray_init_config)
        self.generation += 1
        self.last_round = 0
        self.model_version = 0
        self._failed = False
        try:
            if self.weight_sync_factory is not None:
                self.weight_sync_scheme = self.weight_sync_factory(
                    new_generation=self.generation > 1
                )
            bundle = self._placement_bundle(self.resources_per_rank)
            bundles = [dict(bundle) for _ in range(self.world_size)]
            self._placement_group = placement_group(
                bundles, strategy=self.placement_strategy
            )
            ray.get(self._placement_group.ready(), timeout=self.setup_timeout)
            clients = self.replay_buffer.clients(self.world_size)
            worker_cls = ray.remote(_RayLearnerWorker)
            for rank, client in enumerate(clients):
                actor = worker_cls.options(
                    **self.resources_per_rank,
                    placement_group=self._placement_group,
                    placement_group_bundle_index=rank,
                ).remote(self.objects, client)
                self._actors.append(actor)
            coordinates = ray.get(
                self._actors[0].create_store.remote(
                    self.store_host, self.setup_timeout
                ),
                timeout=self.setup_timeout,
            )
            setup = [
                actor.setup.remote(
                    rank=rank,
                    world_size=self.world_size,
                    local_batch_size=self.local_batch_size,
                    coordinates=coordinates,
                    backend=self.backend,
                    generation=self.generation,
                    timeout=self.setup_timeout,
                    seed=self.seed,
                    clip_grad_norm=self.clip_grad_norm,
                    clip_norm=self.clip_norm,
                    update_replay_priority=self.update_replay_priority,
                )
                for rank, actor in enumerate(self._actors)
            ]
            ray.get(setup, timeout=self.setup_timeout)
            if self.weight_sync_scheme is not None:
                ray.get(
                    self._actors[0].configure_weight_sync.remote(
                        self.weight_sync_scheme
                    ),
                    timeout=self.setup_timeout,
                )
        except BaseException:
            self._failed = True
            self.shutdown()
            raise

    def step(self, num_steps: int) -> _ExecutionStep:
        self._require_alive()
        round_id = self.last_round + 1
        try:
            receipts = ray.get(
                [actor.step.remote(num_steps, round_id) for actor in self._actors],
                timeout=self.command_timeout,
            )
        except BaseException:
            self._failed = True
            self.shutdown()
            raise
        versions = {receipt.model_version for receipt in receipts}
        rounds = {receipt.round_id for receipt in receipts}
        if versions != {self.model_version + num_steps} or rounds != {round_id}:
            self._failed = True
            self.shutdown()
            raise RuntimeError("Learner ranks returned inconsistent step receipts.")
        self.last_round = round_id
        self.model_version = versions.pop()
        metrics = self._average_rank_metrics([receipt.metrics for receipt in receipts])
        return _ExecutionStep(round_id, num_steps, self.model_version, metrics)

    def get_weights(
        self, model_id: str = "policy", *, expected_version: int | None = None
    ) -> TensorDictBase:
        self._require_alive()
        if expected_version is None:
            expected_version = self.model_version
        return ray.get(
            self._actors[0].get_weights.remote(model_id, expected_version),
            timeout=self.command_timeout,
        )

    def publish_weights(
        self,
        *,
        expected_version: int,
        model_weights_key: NestedKey | None = None,
        auxiliary_weights: TensorDictBase | None = None,
    ) -> int:
        self._require_alive()
        try:
            return int(
                ray.get(
                    self._actors[0].publish_weights.remote(
                        expected_generation=self.generation,
                        expected_version=expected_version,
                        model_weights_key=model_weights_key,
                        auxiliary_weights=auxiliary_weights,
                    ),
                    timeout=self.command_timeout,
                )
            )
        except BaseException:
            self._failed = True
            self.shutdown()
            raise

    def state_dict(self) -> dict[str, Any]:
        self._require_alive()
        return {
            "format_version": 1,
            "world_size": self.world_size,
            "global_batch_size": self.global_batch_size,
            "generation": self.generation,
            "model_version": self.model_version,
            "last_round": self.last_round,
            "ranks": ray.get(
                [actor.state_dict.remote() for actor in self._actors],
                timeout=self.command_timeout,
            ),
        }

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        self._require_alive()
        if int(state_dict["world_size"]) != self.world_size:
            raise ValueError(
                "Checkpoint world_size does not match the learner backend."
            )
        if int(state_dict["global_batch_size"]) != self.global_batch_size:
            raise ValueError(
                "Checkpoint batch_size does not match the learner backend."
            )
        rank_states = state_dict["ranks"]
        if len(rank_states) != self.world_size:
            raise ValueError("Checkpoint rank state count does not match world_size.")
        restored = ray.get(
            [
                actor.load_state_dict.remote(rank_state)
                for actor, rank_state in zip(self._actors, rank_states)
            ],
            timeout=self.command_timeout,
        )
        versions = {item["model_version"] for item in restored}
        rounds = {item["last_round"] for item in restored}
        if len(versions) != 1 or len(rounds) != 1:
            raise RuntimeError("Restored learner ranks disagree on semantic state.")
        self.model_version = versions.pop()
        self.last_round = rounds.pop()

    def is_alive(self) -> bool:
        if not self._actors:
            return False
        try:
            return all(
                ray.get(
                    [actor.probe.remote() for actor in self._actors],
                    timeout=min(self.command_timeout, 5.0),
                )
            )
        except Exception:
            return False

    def shutdown(self, timeout: float | None = None) -> None:
        timeout = self.command_timeout if timeout is None else timeout
        actors, self._actors = self._actors, []
        for actor in reversed(actors):
            with contextlib.suppress(Exception):
                ray.get(actor.close.remote(), timeout=timeout)
            with contextlib.suppress(Exception):
                ray.kill(actor, no_restart=True)
        if self._placement_group is not None:
            with contextlib.suppress(Exception):
                remove_placement_group(self._placement_group)
            self._placement_group = None
        if self._runtime_lease is not None:
            self._runtime_lease.release()
            self._runtime_lease = None

    def _require_alive(self) -> None:
        if not self.is_alive():
            state = "failed" if self._failed else "stopped"
            raise RuntimeError(f"Ray learner generation is {state}.")

    @staticmethod
    def _positive_int(name: str, value: Any) -> int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an integer.")
        if value <= 0:
            raise ValueError(f"{name} must be positive.")
        return value

    @staticmethod
    def _placement_bundle(resources_per_rank: Mapping[str, Any]) -> dict[str, float]:
        """Translate Ray actor options into placement-group resource names."""
        bundle = {
            "CPU": float(resources_per_rank.get("num_cpus", 1)),
            "GPU": float(resources_per_rank.get("num_gpus", 0)),
        }
        custom_resources = resources_per_rank.get("resources", {})
        bundle.update(
            {name: float(quantity) for name, quantity in custom_resources.items()}
        )
        return {name: quantity for name, quantity in bundle.items() if quantity}

    @staticmethod
    def _average_rank_metrics(metrics: list[TensorDictBase]) -> TensorDictBase:
        if not metrics:
            return TensorDict(device="cpu")
        keys = set(metrics[0].keys(True, True))
        for item in metrics[1:]:
            keys.intersection_update(item.keys(True, True))
        result = TensorDict(device="cpu")
        for key in keys:
            result.set(key, torch.stack([item.get(key) for item in metrics]).mean())
        return result
