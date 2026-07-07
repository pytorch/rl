# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Shared dummy training loop for the service deployment examples."""

from __future__ import annotations

import importlib.util
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from functools import partial
from pathlib import Path
from typing import Any, Literal

import torch
from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from torch import nn

from torchrl.data import ListStorage, ReplayBuffer, TensorDictReplayBuffer
from torchrl.modules.inference_server import (
    InferenceServer,
    MPTransport,
    PolicyClientModule,
    ProcessInferenceServer,
    RayTransport,
    ThreadingTransport,
)
from torchrl.record import CSVLogger

_has_ray = importlib.util.find_spec("ray") is not None
if _has_ray:
    import ray

ExampleServiceBackend = Literal["direct", "process", "ray"]


def make_policy() -> TensorDictModule:
    """Create the deterministic policy owned by the inference service."""
    torch.manual_seed(0)
    return TensorDictModule(
        nn.Linear(4, 2),
        in_keys=["observation"],
        out_keys=["action"],
    )


def _make_logger(service_backend: ExampleServiceBackend, log_dir: str | Path):
    options = None
    if service_backend == "process":
        options = {"context": "spawn", "max_queue_size": 256}
    elif service_backend == "ray":
        options = {"actor_options": {"num_cpus": 1}}
    return CSVLogger(
        exp_name=f"services-{service_backend}",
        log_dir=log_dir,
        service_backend=service_backend,
        service_backend_options=options,
    )


def _make_replay_buffer(
    service_backend: ExampleServiceBackend,
    *,
    capacity: int,
    batch_size: int,
) -> ReplayBuffer:
    # Replay buffers currently support direct and Ray owners. The process
    # profile keeps its replay buffer in the driver, where actor threads share
    # its identity client.
    replay_backend = "ray" if service_backend == "ray" else "direct"
    options = {"remote_config": {"num_cpus": 1}} if replay_backend == "ray" else None
    return TensorDictReplayBuffer(
        storage=partial(ListStorage, max_size=capacity),
        batch_size=batch_size,
        service_backend=replay_backend,
        service_backend_options=options,
    )


def _make_inference_server(
    service_backend: ExampleServiceBackend,
    *,
    num_actors: int,
):
    if service_backend == "process":
        context = mp.get_context("spawn")
        transport = MPTransport(ctx=context)
        server = ProcessInferenceServer(
            policy_factory=make_policy,
            transport=transport,
            max_batch_size=num_actors,
            mp_context=context,
        )
    else:
        transport = RayTransport() if service_backend == "ray" else ThreadingTransport()
        server = InferenceServer(
            make_policy(),
            transport,
            max_batch_size=num_actors,
        )
    return server, transport


def run_actor(
    actor_id: int,
    steps: int,
    policy: PolicyClientModule,
    replay_buffer: Any,
    logger: Any,
) -> float:
    """Collect synthetic transitions using service clients only."""
    generator = torch.Generator().manual_seed(actor_id)
    total_reward = 0.0
    for step in range(steps):
        observation = torch.randn(4, generator=generator)
        result = policy(TensorDict({"observation": observation}, batch_size=[]))
        action = result["action"]
        reward = -action.square().mean()
        transition = TensorDict(
            {
                "observation": observation,
                "action": action,
                "reward": reward,
                "next": TensorDict(
                    {"observation": observation + action.mean()}, batch_size=[]
                ),
            },
            batch_size=[],
        )
        replay_buffer.add(transition)
        reward_value = float(reward)
        logger.log_scalar(f"actor/{actor_id}/reward", reward_value, step=step)
        total_reward += reward_value
    return total_reward


def _make_actor_inputs(
    *,
    num_actors: int,
    steps_per_actor: int,
    inference_server,
    replay_buffer,
    logger,
):
    return [
        (
            actor_id,
            steps_per_actor,
            PolicyClientModule(
                inference_server,
                in_keys=["observation"],
                out_keys=["action", "policy_version"],
            ),
            replay_buffer.client(),
            logger.client(),
        )
        for actor_id in range(num_actors)
    ]


def _run_actors(service_backend: ExampleServiceBackend, actor_inputs) -> list[float]:
    if service_backend == "ray":
        remote_actor = ray.remote(num_cpus=1)(run_actor)
        return ray.get([remote_actor.remote(*items) for items in actor_inputs])
    with ThreadPoolExecutor(max_workers=len(actor_inputs)) as executor:
        futures = [executor.submit(run_actor, *items) for items in actor_inputs]
        return [future.result() for future in futures]


def run_training(
    *,
    service_backend: ExampleServiceBackend,
    num_actors: int = 4,
    steps_per_actor: int = 8,
    batch_size: int = 8,
    log_dir: str | Path = "/tmp/torchrl-service-example",
) -> dict[str, float | int]:
    """Run the same training loop with a direct, process, or Ray profile."""
    if service_backend == "ray" and not _has_ray:
        raise ImportError("The Ray example requires `pip install ray`.")
    if num_actors < 1 or steps_per_actor < 1 or batch_size < 1:
        raise ValueError("Actor, step, and batch counts must all be positive.")
    capacity = num_actors * steps_per_actor
    if batch_size > capacity:
        raise ValueError("batch_size cannot exceed the number of collected samples.")

    with ExitStack() as owners:
        if service_backend == "ray" and not ray.is_initialized():
            ray.init(
                num_cpus=num_actors + 3,
                ignore_reinit_error=True,
                log_to_driver=False,
            )
            owners.callback(ray.shutdown)

        logger = _make_logger(service_backend, log_dir)
        owners.callback(logger.shutdown)
        replay_buffer = _make_replay_buffer(
            service_backend,
            capacity=capacity,
            batch_size=batch_size,
        )
        owners.callback(replay_buffer.shutdown)
        inference_server, transport = _make_inference_server(
            service_backend,
            num_actors=num_actors,
        )
        close_transport = getattr(transport, "close", None)
        if callable(close_transport):
            owners.callback(close_transport)
        owners.callback(inference_server.shutdown)
        inference_server.start()

        actor_inputs = _make_actor_inputs(
            num_actors=num_actors,
            steps_per_actor=steps_per_actor,
            inference_server=inference_server,
            replay_buffer=replay_buffer,
            logger=logger,
        )
        actor_returns = _run_actors(service_backend, actor_inputs)

        sample = replay_buffer.sample()
        server_stats = inference_server.stats()
        metrics = {
            "replay_size": len(replay_buffer),
            "sample_reward": float(sample["reward"].mean()),
            "mean_actor_return": sum(actor_returns) / len(actor_returns),
            "mean_batch_size": float(server_stats["avg_batch_size"]),
        }
        logger.log_metrics(
            {f"train/{key}": value for key, value in metrics.items()},
            step=capacity,
        )
        logger.flush()
        print(
            f"{service_backend}: sampled {sample.batch_size} from "
            f"{metrics['replay_size']} transitions; inference mean batch "
            f"size={metrics['mean_batch_size']:.2f}"
        )
        return metrics
