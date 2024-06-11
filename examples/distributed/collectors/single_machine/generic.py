# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Distributed data collection on a single node with sync and async support.

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

import torch
import tqdm
from torchrl._utils import logger as torchrl_logger

from torchrl.collectors.collectors import (
    MultiaSyncDataCollector,
    MultiSyncDataCollector,
    SyncDataCollector,
)
from torchrl.collectors.distributed import DistributedDataCollector
from torchrl.envs import EnvCreator, ParallelEnv
from torchrl.envs.libs.gym import GymEnv, set_gym_backend
from torchrl.envs.utils import RandomPolicy

parser = ArgumentParser()
parser.add_argument(
    "--num_workers", default=8, type=int, help="Number of workers in each node."
)
parser.add_argument(
    "--num_nodes", default=3, type=int, help="Number of nodes for the collector."
)
parser.add_argument(
    "--frames_per_batch",
    default=3000,
    type=int,
    help="Number of frames in each batch of data. Must be "
    "divisible by the product of nodes and workers.",
)
parser.add_argument(
    "--total_frames",
    default=3_000_000,
    type=int,
    help="Total number of frames collected by the collector. Must be "
    "divisible by the product of nodes and workers.",
)
parser.add_argument(
    "--backend",
    default="nccl",
    help="backend for torch.distributed. Must be one of "
    "'gloo', 'nccl' or 'mpi'. Use 'nccl' for cuda to cuda "
    "data passing.",
)
parser.add_argument(
    "--sync",
    action="store_true",
    help="whether collection should be synchronous or not.",
)
parser.add_argument(
    "--worker_parallelism",
    choices=["env", "collector"],
    default="collector",
    help="Source of parallelism for the nodes' multiprocessing. Can be 'env'"
    " (ie. envs are executed in parallel) or 'collector' (ie each node handles"
    " a multiprocessed data collector). In this example, the former is slower"
    " due to a higher IO throughput.",
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

    def gym_make():
        with set_gym_backend(gym):
            return GymEnv(args.env)

    make_env = EnvCreator(gym_make)
    if args.worker_parallelism == "collector" or num_workers == 1:
        action_spec = make_env().action_spec
    else:
        make_env = ParallelEnv(
            num_workers,
            make_env,
            serial_for_single=True,
        )
        action_spec = make_env.action_spec

    if args.worker_parallelism == "collector" and num_workers > 1:
        sub_collector_class = (
            MultiSyncDataCollector if args.sync else MultiaSyncDataCollector
        )
        num_workers_per_collector = num_workers
        device_str = "devices"  # MultiSyncDataCollector expects a devices kwarg
    else:
        sub_collector_class = SyncDataCollector
        num_workers_per_collector = 1
        device_str = "device"  # SyncDataCollector expects a device kwarg

    if args.backend == "nccl":
        if num_nodes > device_count - 1:
            raise RuntimeError(
                "Expected at most as many workers as GPU devices (excluded cuda:0 which "
                f"will be used by the main worker). Got {num_workers} workers for {device_count} GPUs."
            )
        collector_kwargs = [
            {device_str: f"cuda:{i}", f"storing_{device_str}": f"cuda:{i}"}
            for i in range(1, num_nodes + 2)
        ]
    elif args.backend == "gloo":
        collector_kwargs = {device_str: "cpu", f"storing_{device_str}": "cpu"}
    else:
        raise NotImplementedError(
            f"device assignment not implemented for backend {args.backend}"
        )

    collector = DistributedDataCollector(
        [make_env] * num_nodes,
        RandomPolicy(action_spec),
        num_workers_per_collector=num_workers_per_collector,
        frames_per_batch=frames_per_batch,
        total_frames=args.total_frames,
        collector_class=sub_collector_class,
        collector_kwargs=collector_kwargs,
        sync=args.sync,
        storing_device="cuda:0" if args.backend == "nccl" else "cpu",
        launcher=launcher,
        backend=args.backend,
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
