# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""A minimal torchrun DQN recipe with one Ray-owned replay buffer.

Start a Ray head node, then launch two replicated learners::

    ray start --head
    torchrun --standalone --nproc-per-node=2 \
        examples/distributed/replay_buffers/native_data_parallel_dqn.py

Every process connects to the same Ray cluster. Rank zero owns and populates
the replay service, while all ranks sample a local share of the configured
global batch and explicitly average gradients before the optimizer step.
"""

from __future__ import annotations

import argparse
from functools import partial

import ray
import torch
import torch.distributed as dist
from tensordict import TensorDict

from torchrl._utils import logger as torchrl_logger
from torchrl.data import LazyTensorStorage, RayReplayBuffer, TensorDictReplayBuffer
from torchrl.distributed import DataParallelContext
from torchrl.objectives import DQNLoss


def make_transitions(num_transitions: int, *, seed: int) -> TensorDict:
    """Create a small synthetic transition dataset for the recipe."""
    generator = torch.Generator().manual_seed(seed)
    observation = torch.randn(num_transitions, 4, generator=generator)
    next_observation = observation + 0.05 * torch.randn(
        num_transitions, 4, generator=generator
    )
    action_index = torch.randint(2, (num_transitions,), generator=generator)
    action = torch.nn.functional.one_hot(action_index, 2).to(torch.float32)
    reward = observation[:, :1] - 0.25 * action_index.unsqueeze(-1).to(torch.float32)
    done = torch.zeros(num_transitions, 1, dtype=torch.bool)
    return TensorDict(
        {
            "observation": observation,
            "action": action,
            ("next", "observation"): next_observation,
            ("next", "reward"): reward,
            ("next", "done"): done,
            ("next", "terminated"): done.clone(),
        },
        batch_size=[num_transitions],
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ray-address", default="auto")
    parser.add_argument("--global-batch-size", type=int, default=128)
    parser.add_argument("--buffer-size", type=int, default=4096)
    parser.add_argument("--updates", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    context = DataParallelContext.from_torchrun(device=args.device)
    ray.init(address=args.ray_address, ignore_reinit_error=True)
    owner = None
    try:
        if context.is_rank_zero:
            owner = RayReplayBuffer(
                replay_buffer_cls=TensorDictReplayBuffer,
                storage=partial(LazyTensorStorage, args.buffer_size),
                batch_size=args.global_batch_size,
                remote_config={"num_cpus": 0},
            )
            owner.extend(make_transitions(args.buffer_size, seed=args.seed))
            client = owner.client()
        else:
            client = None

        payload = [client]
        if context.world_size > 1:
            dist.broadcast_object_list(
                payload,
                src=0,
                group=context.process_group,
            )
        replay_buffer = payload[0].data_parallel(
            rank=context.rank, world_size=context.world_size
        )
        if len(replay_buffer) < args.global_batch_size:
            raise RuntimeError(
                "The replay service must contain at least one global batch before "
                "all ranks sample."
            )

        torch.manual_seed(args.seed + context.rank)
        value_network = torch.nn.Linear(4, 2, device=context.device)
        loss_module = DQNLoss(
            value_network,
            action_space="one-hot",
            delay_value=False,
        ).to(context.device)
        context.broadcast_module(loss_module)
        optimizer = torch.optim.Adam(loss_module.parameters(), lr=args.learning_rate)

        for update in range(args.updates):
            batch = replay_buffer.sample().to(context.device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_module(batch)["loss"]
            loss.backward()
            context.sync_gradients(optimizer)
            optimizer.step()
            if context.is_rank_zero:
                torchrl_logger.info(
                    "update=%s loss=%.6f local_batch_size=%s",
                    update,
                    loss.detach().item(),
                    batch.numel(),
                )
    finally:
        # This simple teardown assumes every rank reaches the collectives. If a
        # rank exits early, peers may block until the process-group timeout; a
        # failure-aware controller should coordinate teardown in production.
        context.barrier()
        if owner is not None:
            owner.shutdown()
        context.barrier()
        ray.shutdown()
        context.close()


if __name__ == "__main__":
    main()
