# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import time
from argparse import ArgumentParser

import torch
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
    "--slurm_partition", default="train", help="Slurm partition to be used."
)
parser.add_argument(
    "--slurm_cpus_per_gpu",
    default=8,
    type=int,
    help="Number of CPUs per GPU on each node.",
)
parser.add_argument(
    "--slurm_gpus_per_task", default=1, type=int, help="Number of GPUs per node."
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
    launcher = "submitit"

    slurm_conf = {
        "timeout_min": 100,
        "slurm_partition": args.slurm_partition,
        "slurm_cpus_per_gpu": args.slurm_cpus_per_gpu,
        "slurm_gpus_per_task": args.slurm_gpus_per_task,
    }
    device_str = "device" if num_workers <= 1 else "devices"
    if torch.cuda.device_count():
        collector_kwargs = {device_str: "cuda:0", f"storing_{device_str}": "cuda:0"}
    else:
        collector_kwargs = {device_str: "cpu", "storing_{device_str}": "cpu"}

    make_env = EnvCreator(lambda: GymEnv(args.env))
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
        slurm_kwargs=slurm_conf,
        sync=args.sync,
        storing_device="cuda:0" if torch.cuda.device_count() else "cpu",
        launcher=launcher,
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
    print(f"time elapsed: {t1-t0}s, rate: {counter/(t1-t0)} fps")
    exit()
