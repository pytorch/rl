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

Each node can support multiple workers through the usage of `ParallelEnv`.

The default task is `Pong-v5` but a different one can be picked through the
`--env` flag. Any available gym env will work.

"""
import time
from argparse import ArgumentParser

import gym

import torch.cuda
import tqdm
from torchrl._utils import logger as torchrl_logger

from torchrl.collectors.collectors import SyncDataCollector
from torchrl.collectors.distributed import RPCDataCollector
from torchrl.envs import EnvCreator, ParallelEnv
from torchrl.envs.libs.gym import GymEnv, set_gym_backend
from torchrl.envs.utils import RandomPolicy

parser = ArgumentParser()
parser.add_argument(
    "--num_workers", default=8, type=int, help="Number of workers in each node."
)
parser.add_argument(
    "--num_nodes",
    default=3,
    type=int,
    help="Number of nodes for the collector (the main "
    "worker being excluded from this count).",
)
parser.add_argument(
    "--frames_per_batch",
    default=3000,
    type=int,
    help="Number of frames in each batch of data. Must be "
    "divisible by the product of nodes and workers if sync, by the number of "
    "workers otherwise.",
)
parser.add_argument(
    "--total_frames",
    default=3_000_000,
    type=int,
    help="Total number of frames collected by the collector.",
)
parser.add_argument(
    "--sync",
    action="store_true",
    help="whether collection should be synchronous or not.",
)
parser.add_argument(
    "--env",
    default="ALE/Pong-v5",
    help="Gym environment to be run.",
)
if __name__ == "__main__":
    args = parser.parse_args()
    num_workers = args.num_workers
    num_nodes = args.num_nodes
    frames_per_batch = args.frames_per_batch
    launcher = "mp"

    device_count = torch.cuda.device_count()

    if device_count:
        if num_nodes > device_count - 1:
            raise RuntimeError(
                "Expected at most as many workers as GPU devices (excluded cuda:0 which "
                f"will be used by the main worker). Got {num_workers} workers for {device_count} GPUs."
            )
        collector_kwargs = [
            {"device": f"cuda:{i}", "storing_device": f"cuda:{i}"}
            for i in range(1, num_nodes + 2)
        ]
    else:
        collector_kwargs = {"device": "cpu", "storing_device": "cpu"}

    def gym_make():
        with set_gym_backend(gym):
            return GymEnv(args.env)

    make_env = EnvCreator(gym_make)
    if num_workers == 1:
        action_spec = make_env().action_spec
    else:
        make_env = ParallelEnv(
            num_workers,
            make_env,
            serial_for_single=True,
        )
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
        visible_devices=list(range(1, device_count)) if device_count else None,
    )

    counter = 0
    pbar = tqdm.tqdm(total=collector.total_frames)
    for i, data in enumerate(collector):
        pbar.update(data.numel())
        pbar.set_description(f"data shape: {data.shape}, data device: {data.device}")
        if i >= 10:
            counter += data.numel()
        if i == 10:
            t0 = time.time()
    collector.shutdown()
    t1 = time.time()
    torchrl_logger.info(f"time elapsed: {t1-t0}s, rate: {counter/(t1-t0)} fps")
    exit()
