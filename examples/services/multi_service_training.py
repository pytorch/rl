# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Run a small actor/replay/trainer loop through TorchRL service clients.

The driver owns an inference server, a process logger, and a replay buffer.
Actor threads receive only restricted clients. The replay buffer is direct by
default so the example runs with core dependencies; pass ``--replay-backend
ray`` to place it in a Ray actor without changing the actor loop.

Run from the repository root:

    python examples/services/multi_service_training.py
    python examples/services/multi_service_training.py --replay-backend ray
"""

from __future__ import annotations

import argparse
import importlib.util
from concurrent.futures import ThreadPoolExecutor
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
    PolicyClientModule,
    ThreadingTransport,
)
from torchrl.record import CSVLogger

_has_ray = importlib.util.find_spec("ray") is not None
if _has_ray:
    import ray

ReplayBackend = Literal["direct", "ray"]


def make_policy() -> TensorDictModule:
    """Create the policy owned by the inference service."""
    torch.manual_seed(0)
    return TensorDictModule(
        nn.Linear(4, 2),
        in_keys=["observation"],
        out_keys=["action"],
    )


def make_replay_buffer(
    backend: ReplayBackend, capacity: int, batch_size: int
) -> ReplayBuffer:
    """Create a replay-buffer owner with a backend-independent call site."""
    options = {"remote_config": {"num_cpus": 1}} if backend == "ray" else None
    return TensorDictReplayBuffer(
        storage=partial(ListStorage, max_size=capacity),
        batch_size=batch_size,
        service_backend=backend,
        service_backend_options=options,
    )


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
        output = policy(TensorDict({"observation": observation}, batch_size=[]))
        action = output["action"]
        reward = -action.square().mean()
        next_observation = observation + action.mean()
        transition = TensorDict(
            {
                "observation": observation,
                "action": action,
                "reward": reward,
                "next": TensorDict({"observation": next_observation}, batch_size=[]),
            },
            batch_size=[],
        )
        replay_buffer.add(transition)
        reward_value = float(reward)
        logger.log_scalar(f"actor/{actor_id}/reward", reward_value, step=step)
        total_reward += reward_value
    return total_reward


def main(
    *,
    replay_backend: ReplayBackend = "direct",
    num_actors: int = 4,
    steps_per_actor: int = 8,
    batch_size: int = 8,
    log_dir: str | Path = "/tmp/torchrl-service-example",
) -> None:
    """Run the example and explicitly shut down every service owner."""
    if replay_backend == "ray" and not _has_ray:
        raise ImportError(
            "The Ray replay-buffer backend requires Ray. Install it with "
            "`pip install ray`."
        )

    started_ray = False
    if replay_backend == "ray" and not ray.is_initialized():
        ray.init(ignore_reinit_error=True)
        started_ray = True

    logger = CSVLogger(
        exp_name="multi-service",
        log_dir=log_dir,
        service_backend="process",
    )
    replay_buffer = make_replay_buffer(
        replay_backend,
        capacity=num_actors * steps_per_actor,
        batch_size=batch_size,
    )
    inference_server = InferenceServer(
        make_policy(),
        ThreadingTransport(),
        max_batch_size=num_actors,
    )
    inference_server.start()

    try:
        # Each actor gets independent transport clients. Remote clients cannot
        # shut down their owners; a direct replay buffer uses identity clients.
        actor_inputs = [
            (
                actor_id,
                steps_per_actor,
                PolicyClientModule(
                    inference_server.client(),
                    in_keys=["observation"],
                    out_keys=["action", "policy_version"],
                ),
                replay_buffer.client(),
                logger.client(),
            )
            for actor_id in range(num_actors)
        ]
        with ThreadPoolExecutor(max_workers=num_actors) as executor:
            futures = [executor.submit(run_actor, *items) for items in actor_inputs]
            actor_returns = [future.result() for future in futures]

        sample = replay_buffer.sample()
        stats = inference_server.stats()
        logger.log_metrics(
            {
                "train/replay_size": len(replay_buffer),
                "train/sample_reward": sample["reward"].mean(),
                "train/mean_actor_return": sum(actor_returns) / len(actor_returns),
                "inference/mean_batch_size": stats["avg_batch_size"],
            },
            step=num_actors * steps_per_actor,
        )
        logger.flush()
        print(
            f"sampled {sample.batch_size} from {len(replay_buffer)} transitions; "
            f"inference mean batch size={stats['avg_batch_size']:.2f}"
        )
    finally:
        # The driver tears owners down in dependency order and closes the
        # logger last.
        inference_server.shutdown()
        replay_buffer.shutdown()
        logger.shutdown()
        if started_ray:
            ray.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-backend", choices=("direct", "ray"), default="direct")
    parser.add_argument("--num-actors", type=int, default=4)
    parser.add_argument("--steps-per-actor", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--log-dir", default="/tmp/torchrl-service-example")
    args = parser.parse_args()
    main(
        replay_backend=args.replay_backend,
        num_actors=args.num_actors,
        steps_per_actor=args.steps_per_actor,
        batch_size=args.batch_size,
        log_dir=args.log_dir,
    )
