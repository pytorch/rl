# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Multi-node RPC-based distributed data collection with submitit in contexts where jobs can't launch other jobs.

The default configuration will ask for 8 nodes with 1 GPU each and 32 procs / node.

It should reach a collection speed of roughly 15-25K fps, or better depending
on the cluster specs.

The logic of the script is the following: we create a `main()` function that
executes or code (in this case just a data collection but in practice a training
loop should be present).

Since this `main()` function cannot launch sub-jobs by design, we launch the script
from the jump host and pass the slurm specs to submitit.

*Note*:

  Although we don't go in much details into this in this script, the specs of the training
  node and the specs of the inference nodes can differ (look at the DEFAULT_SLURM_CONF
  and DEFAULT_SLURM_CONF_MAIN dictionaries below).

"""
import time
from argparse import ArgumentParser

import tqdm
from torchrl._utils import logger as torchrl_logger

from torchrl.collectors.distributed import RPCDataCollector, submitit_delayed_launcher
from torchrl.collectors.distributed.default_configs import (
    DEFAULT_SLURM_CONF,
    DEFAULT_SLURM_CONF_MAIN,
)
from torchrl.envs import EnvCreator

parser = ArgumentParser()
parser.add_argument("--partition", "-p", help="slurm partition to use")
parser.add_argument("--num_jobs", type=int, default=32, help="Number of jobs")
parser.add_argument("--tcp_port", type=int, default=1234, help="TCP port")
parser.add_argument(
    "--num_workers", type=int, default=1, help="Number of workers per node"
)
parser.add_argument(
    "--gpus_per_node",
    "--gpus-per-node",
    "-G",
    type=int,
    default=0,
    help="Number of GPUs per node. ",
)
parser.add_argument(
    "--cpus_per_task",
    "--cpus-per-task",
    "-c",
    type=int,
    default=8,
    help="Number of CPUs per node.",
)
parser.add_argument(
    "--sync", action="store_true", help="Use --sync to collect data synchronously."
)
parser.add_argument(
    "--frames_per_batch",
    "--frames-per-batch",
    default=4000,
    type=int,
    help="Number of frames in each batch of data. Must be "
    "divisible by the product of nodes and workers if sync, by the number of "
    "workers otherwise.",
)
parser.add_argument(
    "--total_frames",
    "--total-frames",
    default=10_000_000,
    type=int,
    help="Total number of frames collected by the collector.",
)
parser.add_argument(
    "--time",
    "-t",
    default="1:00:00",
    help="Timeout for the nodes",
)

args = parser.parse_args()

slurm_gpus_per_node = args.gpus_per_node
slurm_time = args.time

DEFAULT_SLURM_CONF["slurm_gpus_per_node"] = slurm_gpus_per_node
DEFAULT_SLURM_CONF["slurm_time"] = slurm_time
DEFAULT_SLURM_CONF["slurm_cpus_per_task"] = args.cpus_per_task
DEFAULT_SLURM_CONF["slurm_partition"] = args.partition
DEFAULT_SLURM_CONF_MAIN["slurm_partition"] = args.partition
DEFAULT_SLURM_CONF_MAIN["slurm_time"] = slurm_time

num_jobs = args.num_jobs
tcp_port = args.tcp_port
num_workers = args.num_workers
sync = args.sync
total_frames = args.total_frames
frames_per_batch = args.frames_per_batch


@submitit_delayed_launcher(
    num_jobs=num_jobs,
    tcpport=tcp_port,
    framework="rpc",
)
def main():
    import gym
    from torchrl.collectors import MultiSyncDataCollector, SyncDataCollector
    from torchrl.data import BoundedTensorSpec
    from torchrl.envs.libs.gym import GymEnv, set_gym_backend
    from torchrl.envs.utils import RandomPolicy

    collector_class = SyncDataCollector if num_workers == 1 else MultiSyncDataCollector
    device_str = "device" if num_workers == 1 else "devices"

    def make_env():
        # gymnasium breaks when using multiproc
        with set_gym_backend(gym):
            return GymEnv("ALE/Pong-v5")

    collector = RPCDataCollector(
        [EnvCreator(make_env)] * num_jobs,
        policy=RandomPolicy(BoundedTensorSpec(-1, 1, shape=(1,))),
        launcher="submitit_delayed",
        frames_per_batch=frames_per_batch,
        total_frames=total_frames,
        tcp_port=tcp_port,
        collector_class=collector_class,
        num_workers_per_collector=args.num_workers,
        collector_kwargs={device_str: "cuda:0" if slurm_gpus_per_node else "cpu"},
        storing_device="cuda:0" if slurm_gpus_per_node else "cpu",
        sync=sync,
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
    t1 = time.time()
    torchrl_logger.info(f"time elapsed: {t1-t0}s, rate: {counter/(t1-t0)} fps")
    collector.shutdown()
    exit()


if __name__ == "__main__":
    main()
