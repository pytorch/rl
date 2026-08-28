# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Backend-neutral training loop shared by the service examples."""

from __future__ import annotations

import importlib.util
from contextlib import ExitStack
from functools import partial
from pathlib import Path
from typing import Literal

import torch
from tensordict import TensorDict, TensorDictBase
from tensordict.nn import TensorDictModule
from torch import nn

from torchrl.data import ListStorage, ReplayBuffer, TensorDictReplayBuffer
from torchrl.envs import GymEnv
from torchrl.modules.inference_server import InferenceServer, PolicyClientModule
from torchrl.record import CSVLogger

_has_ray = importlib.util.find_spec("ray") is not None

ExampleServiceBackend = Literal["direct", "process", "ray"]


def make_policy() -> TensorDictModule:
    """Create the policy owned by the inference service."""
    torch.manual_seed(0)
    return TensorDictModule(
        nn.Sequential(nn.Linear(3, 1), nn.Tanh()),
        in_keys=["observation"],
        out_keys=["action"],
    )


class _RewardPredictionLoss(nn.Module):
    """Small trainable loss used to demonstrate TensorDict backpropagation."""

    def __init__(self) -> None:
        super().__init__()
        self.reward_predictor = nn.Linear(3, 1)

    def forward(self, sample: TensorDictBase) -> TensorDict:
        prediction = self.reward_predictor(sample["observation"])
        error = prediction - sample["next", "reward"]
        return TensorDict(
            {"loss": error.square().mean()},
            batch_size=[],
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
    # Replay buffers currently support direct and Ray service backends.
    replay_backend = "ray" if service_backend == "ray" else "direct"
    options = {"remote_config": {"num_cpus": 1}} if replay_backend == "ray" else None
    return TensorDictReplayBuffer(
        storage=partial(ListStorage, max_size=capacity),
        batch_size=batch_size,
        service_backend=replay_backend,
        service_backend_options=options,
    )


def _make_inference_server(service_backend: ExampleServiceBackend):
    if service_backend == "process":
        server = InferenceServer(
            policy_factory=make_policy,
            service_backend="process",
            service_backend_options={"mp_context": "spawn"},
            transport="auto",
        )
    elif service_backend == "ray":
        server = InferenceServer(
            policy_factory=make_policy,
            service_backend="ray",
            service_backend_options={"remote_config": {"num_cpus": 1}},
            transport="auto",
        )
    else:
        server = InferenceServer(make_policy(), transport="auto")
    return server


def run_training(
    *,
    service_backend: ExampleServiceBackend,
    steps: int = 32,
    batch_size: int = 8,
    log_dir: str | Path = "/tmp/torchrl-service-example",
) -> dict[str, float | int]:
    """Run one ordinary TensorDict loop against the selected services."""
    if service_backend == "ray" and not _has_ray:
        raise ImportError("The Ray example requires `pip install ray`.")
    if steps < 1 or batch_size < 1:
        raise ValueError("steps and batch_size must both be positive.")

    with ExitStack() as owners:
        logger_owner = _make_logger(service_backend, log_dir)
        owners.callback(logger_owner.shutdown)
        replay_owner = _make_replay_buffer(
            service_backend,
            capacity=max(steps, batch_size),
            batch_size=batch_size,
        )
        owners.callback(replay_owner.shutdown)
        inference_owner = _make_inference_server(service_backend)
        owners.callback(inference_owner.shutdown)
        inference_owner.start()

        env = GymEnv("Pendulum-v1")
        env.set_seed(0)
        owners.callback(env.close)

        policy = PolicyClientModule(
            inference_owner,
            in_keys=["observation"],
            out_keys=["action", "policy_version"],
        )
        replay_buffer = replay_owner.client()
        logger = logger_owner.client()
        loss_fn = _RewardPredictionLoss()
        optimizer = torch.optim.Adam(loss_fn.parameters(), lr=3e-3)

        td = env.reset()
        # The selected service backend does not appear in the training loop.
        for step in range(steps):
            td = policy(td)
            step_td = env.step(td)
            replay_buffer.add(step_td)
            td = env.step_mdp(step_td)

            sample = replay_buffer.sample()
            optimizer.zero_grad()
            loss = loss_fn(sample)
            loss.sum(reduce=True).backward()
            optimizer.step()

            logger.log_scalar("train/loss", float(loss["loss"].detach()), step=step)
            logger.log_scalar(
                "train/reward", float(step_td["next", "reward"]), step=step
            )
            if bool(step_td["next", "done"].any()):
                td = env.reset()

        logger_owner.flush()
        metrics = {
            "replay_size": len(replay_buffer),
            "loss": float(loss["loss"].detach()),
            "reward": float(step_td["next", "reward"]),
        }
        print(
            f"{service_backend}: trained for {steps} steps with "
            f"{metrics['replay_size']} replay entries"
        )
        return metrics
