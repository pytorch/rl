# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Train DQN with Ray-owned replay, inference, collection, and learners.

The driver constructs ordinary TorchRL objects. TorchRL owns actor placement,
restricted service clients, replay sampling, and DDP rendezvous internally.

Run from the repository root with
``python examples/distributed/ray_dqn_trainer.py``.
"""

from __future__ import annotations

from functools import partial

import torch
from tensordict.nn import TensorDictModule
from torch import nn

from torchrl.collectors.distributed import RayCollector
from torchrl.data import LazyTensorStorage, TensorDictReplayBuffer
from torchrl.envs import GymEnv
from torchrl.modules import QValueActor
from torchrl.modules.inference_server import InferenceServer
from torchrl.objectives import DQNLoss, HardUpdate
from torchrl.trainers.algorithms import DQNTrainer


def make_env() -> GymEnv:
    """Create CartPole inside a collector actor."""
    return GymEnv("CartPole-v1", device="cpu")


def make_policy() -> QValueActor:
    """Create the same policy graph for inference and learner ownership."""
    env = make_env()
    value = TensorDictModule(
        nn.Sequential(nn.Linear(4, 64), nn.ReLU(), nn.Linear(64, 2)),
        in_keys=["observation"],
        out_keys=["action_value"],
    )
    policy = QValueActor(value, in_keys=["observation"], spec=env.action_spec)
    env.close()
    return policy


def main() -> None:
    ray_init_config = {
        "num_cpus": 8,
        "include_dashboard": False,
        "log_to_driver": False,
    }
    replay = TensorDictReplayBuffer(
        storage=partial(LazyTensorStorage, 10_000),
        batch_size=16,
        service_backend="ray",
        service_backend_options={
            "ray_init_config": ray_init_config,
            "remote_config": {"num_cpus": 1},
        },
        transport="auto",
    )
    inference = InferenceServer(
        policy_factory=make_policy,
        service_backend="ray",
        service_backend_options={"remote_config": {"num_cpus": 1}},
        transport="auto",
    )
    collector = RayCollector(
        create_env_fn=[make_env],
        policy=inference,
        replay_buffer=replay,
        collector_class="single",
        frames_per_batch=32,
        total_frames=128,
        remote_configs={"num_cpus": 1, "num_gpus": 0},
        sync=True,
    )

    policy = make_policy()
    proof_env = make_env()
    loss_module = DQNLoss(
        policy,
        action_space=proof_env.action_spec,
        delay_value=True,
    )
    proof_env.close()
    optimizer = torch.optim.Adam(loss_module.parameters(), lr=3e-4)
    target_updater = HardUpdate(loss_module, value_network_update_interval=20)
    trainer = DQNTrainer(
        collector=collector,
        total_frames=128,
        frame_skip=1,
        optim_steps_per_batch=1,
        loss_module=loss_module,
        optimizer=optimizer,
        replay_buffer=replay,
        target_net_updater=target_updater,
        batch_size=16,
        learner_backend="ray",
        learner_backend_options={
            "world_size": 2,
            "resources_per_rank": {"num_cpus": 1, "num_gpus": 0},
        },
        progress_bar=False,
        enable_logging=False,
    )
    try:
        trainer.train()
    finally:
        collector.shutdown()
        inference.shutdown()
        replay.shutdown()


if __name__ == "__main__":
    main()
