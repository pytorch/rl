# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Run DQN with Ray collectors, replay, and a data-parallel learner gang.

The driver is only the controller: it owns scheduling, exploration, logging,
and service lifecycles. Each learner actor constructs its own loss, optimizer,
and target updater. Run locally with two CPU learner ranks using::

    python examples/distributed/replay_buffers/ray_learner_dqn.py

The same recipe can connect to an existing Ray cluster with ``--ray-address``.
"""

from __future__ import annotations

import argparse
from functools import partial

import ray
import torch
from tensordict.nn import TensorDictSequential

from torchrl._utils import logger as torchrl_logger
from torchrl.collectors.distributed.ray import RayCollector
from torchrl.data import (
    Composite,
    LazyTensorStorage,
    OneHot,
    RayReplayBuffer,
    TensorDictReplayBuffer,
)
from torchrl.envs import GymEnv
from torchrl.modules import EGreedyModule, MLP, QValueActor
from torchrl.objectives import DQNLoss
from torchrl.objectives.utils import HardUpdate
from torchrl.trainers import Learner
from torchrl.trainers.algorithms import DQNTrainer
from torchrl.trainers.distributed import RayLearnerGroup


def make_env() -> GymEnv:
    """Create one CPU CartPole environment."""
    return GymEnv("CartPole-v1", device="cpu")


def make_value_network(device: torch.device | str = "cpu") -> QValueActor:
    """Create the policy architecture shared by collectors and learners."""
    return QValueActor(
        module=MLP(
            in_features=4,
            out_features=2,
            num_cells=[64, 64],
            device=device,
        ),
        spec=Composite(action=OneHot(2)).to(device),
        in_keys=["observation"],
    )


def make_learner(
    replay_buffer,
    data_parallel_context,
    *,
    learning_rate: float,
    target_update_interval: int,
) -> Learner:
    """Construct all trainable DQN state inside one Ray learner actor."""
    value_network = make_value_network(data_parallel_context.device)
    loss_module = DQNLoss(
        value_network=value_network,
        action_space="one-hot",
        delay_value=True,
    ).to(data_parallel_context.device)
    optimizer = torch.optim.Adam(loss_module.parameters(), lr=learning_rate)
    target_updater = HardUpdate(
        loss_module, value_network_update_interval=target_update_interval
    )
    return Learner(
        loss_module,
        replay_buffer,
        optimizer=optimizer,
        target_net_updater=target_updater,
        data_parallel_context=data_parallel_context,
        models={"policy": loss_module.value_network},
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ray-address", default=None)
    parser.add_argument("--total-frames", type=int, default=20_000)
    parser.add_argument("--frames-per-batch", type=int, default=256)
    parser.add_argument("--buffer-size", type=int, default=50_000)
    parser.add_argument("--global-batch-size", type=int, default=128)
    parser.add_argument("--learner-world-size", type=int, default=2)
    parser.add_argument("--num-collectors", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--target-update-interval", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ray.init(address=args.ray_address, ignore_reinit_error=True)
    try:
        replay_buffer = RayReplayBuffer(
            replay_buffer_cls=TensorDictReplayBuffer,
            storage=partial(LazyTensorStorage, args.buffer_size),
            batch_size=args.global_batch_size,
            remote_config={"num_cpus": 0},
        )
        learner_group = RayLearnerGroup(
            partial(
                make_learner,
                learning_rate=args.learning_rate,
                target_update_interval=args.target_update_interval,
            ),
            replay_buffer.client(),
            world_size=args.learner_world_size,
            global_batch_size=args.global_batch_size,
            resources_per_rank={"num_cpus": 1, "num_gpus": 0},
            seed=args.seed,
            learner_id="cartpole-dqn-v1",
        )

        value_network = make_value_network()
        greedy_module = EGreedyModule(
            annealing_num_steps=args.total_frames,
            eps_init=1.0,
            eps_end=0.05,
            spec=value_network.spec,
        )
        collection_policy = TensorDictSequential(value_network, greedy_module)
        collector = RayCollector(
            [make_env] * args.num_collectors,
            collection_policy,
            frames_per_batch=args.frames_per_batch,
            total_frames=args.total_frames,
            replay_buffer=replay_buffer,
            sync=False,
            remote_configs={"num_cpus": 1, "num_gpus": 0},
        )
        trainer = DQNTrainer(
            collector=collector,
            total_frames=args.total_frames,
            frame_skip=1,
            optim_steps_per_batch=1,
            loss_module=None,
            optimizer=None,
            learner_group=learner_group,
            replay_buffer=replay_buffer,
            greedy_module=greedy_module,
            async_collection=True,
            progress_bar=True,
            enable_logging=False,
        )
        trainer.train()
        torchrl_logger.info(
            "completed frames=%s optimizer_steps=%s rounds=%s model_version=%s",
            trainer.collected_frames,
            trainer._optim_count,
            trainer._learner_round,
            trainer._published_model_version,
        )
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()
