# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""RPC data collection on a single node.

The default configuration works fine on machines equipped with 4 GPUs, but can
be scaled up or down depending on the available configuration.

The number of nodes should not be greater than the number of GPUs minus 1, as
each node will be assigned one GPU to work with, while the main worker will
keep its own GPU (presumably for model training).

"""
from argparse import ArgumentParser

import torch.cuda
import tqdm

from torchrl.collectors.collectors import (
    MultiSyncDataCollector,
    RandomPolicy,
    SyncDataCollector,
)
from torchrl.collectors.distributed import RPCDataCollector
from torchrl.envs import EnvCreator, ParallelEnv
from torchrl.envs.libs.gym import GymEnv

parser = ArgumentParser()
parser.add_argument(
    "--num_workers", default=1, type=int, help="Number of workers in each node."
)
parser.add_argument(
    "--num_nodes", default=3, type=int, help="Number of nodes for the collector (the main "
                                             "worker being excluded from this count)."
)
parser.add_argument(
    "--frames_per_batch",
    default=300,
    type=int,
    help="Number of frames in each batch of data. Must be "
    "divisible by the product of nodes and workers.",
)
parser.add_argument(
    "--total_frames",
    default=1_200_000,
    type=int,
    help="Total number of frames collected by the collector. Must be "
    "divisible by the product of nodes and workers.",
)
parser.add_argument(
    "--sync",
    action="store_true",
    help="whether collection should be synchronous or not.",
)
if __name__ == "__main__":
    args = parser.parse_args()
    num_workers = args.num_workers
    num_nodes = args.num_nodes
    frames_per_batch = args.frames_per_batch
    launcher = "mp"

    device_count = torch.cuda.device_count()

    device_str = "device"
    if device_count:
        if num_nodes > device_count-1:
            raise RuntimeError(
                "Expected at most as many workers as GPU devices (excluded cuda:0 which "
                f"will be used by the main worker). Got {num_workers} workers for {device_count} GPUs."
            )
        collector_kwargs = [
            {device_str: f"cuda:{i}", f"storing_{device_str}": f"cuda:{i}"}
            for i in range(1, num_nodes + 2)
        ]
    else:
        collector_kwargs = {device_str: "cpu", f"storing_{device_str}": "cpu"}

    make_env = EnvCreator(lambda: GymEnv("ALE/Pong-v5"))
    if num_workers == 1:
        action_spec = make_env().action_spec
    else:
        make_env = ParallelEnv(num_workers, make_env)
        action_spec = make_env.action_spec

    collector = RPCDataCollector(
        [make_env] * num_nodes,
        RandomPolicy(action_spec),
        num_workers_per_collector=1,
        frames_per_batch=frames_per_batch,
        total_frames=args.total_frames,
        collector_class=SyncDataCollector,
        collector_kwargs=collector_kwargs,
        sync=args.sync,
        storing_device="cuda:0" if device_count else "cpu",
        launcher=launcher,
        visible_devices=list(range(device_count)) if device_count else None,
    )

    pbar = tqdm.tqdm(total=collector.total_frames)
    for data in collector:
        pbar.update(data.numel())
    collector.shutdown()
