# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from argparse import ArgumentParser

import torch.cuda
import tqdm

from torchrl.collectors.collectors import (
    MultiSyncDataCollector,
    RandomPolicy,
    SyncDataCollector,
)
from torchrl.collectors.distributed import RPCDataCollector
from torchrl.envs import EnvCreator
from torchrl.envs.libs.gym import GymEnv

parser = ArgumentParser()
parser.add_argument(
    "--num_workers", default=1, type=int, help="Number of workers in each node."
)
parser.add_argument(
    "--num_nodes", default=4, type=int, help="Number of nodes for the collector."
)
parser.add_argument(
    "--frames_per_batch",
    default=800,
    type=int,
    help="Number of frames in each batch of data. Must be "
    "divisible by the product of nodes and workers.",
)
parser.add_argument(
    "--total_frames",
    default=2_000_000,
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

    device_str = "device" if num_workers <= 1 else "devices"
    if torch.cuda.device_count():
        collector_kwargs = [
            {device_str: f"cuda:{i}", f"storing_{device_str}": f"cuda:{i}"}
            for i in range(1, num_nodes + 2)
        ]
    else:
        collector_kwargs = {device_str: f"cpu", f"storing_{device_str}": f"cpu"}


    make_env = EnvCreator(lambda: GymEnv("ALE/Pong-v5"))
    action_spec = make_env().action_spec

    collector = RPCDataCollector(
        [make_env] * num_nodes,
        RandomPolicy(action_spec),
        num_workers_per_collector=num_workers,
        frames_per_batch=frames_per_batch,
        total_frames=args.total_frames,
        collector_class=SyncDataCollector
        if num_workers == 1
        else MultiSyncDataCollector,
        collector_kwargs=collector_kwargs,
        sync=args.sync,
        storing_device="cuda:0" if torch.cuda.device_count() else "cpu",
        launcher=launcher,
    )

    pbar = tqdm.tqdm(total=collector.total_frames)
    for data in collector:
        pbar.update(data.numel())
    collector.shutdown()
