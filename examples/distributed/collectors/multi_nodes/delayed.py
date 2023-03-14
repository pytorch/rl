# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from argparse import ArgumentParser

import tqdm

from torchrl.collectors.distributed.generic import (
    DEFAULT_SLURM_CONF,
    DEFAULT_SLURM_CONF_MAIN,
    DistributedDataCollector,
    submitit_delayed_launcher,
)
from torchrl.envs import EnvCreator

parser = ArgumentParser()
parser.add_argument("--partition", help="slurm partition to use")
parser.add_argument("--num_jobs", type=int, default=8, help="Number of jobs")
parser.add_argument("--tcp_port", type=int, default=1234, help="TCP port")
parser.add_argument("--num_workers", type=int, default=1, help="Number of workers per node")

args = parser.parse_args()

DEFAULT_SLURM_CONF["slurm_partition"] = args.partition
DEFAULT_SLURM_CONF_MAIN["slurm_partition"] = args.partition

num_jobs = args.num_jobs
tcp_port = args.tcp_port
num_workers = args.num_workers

@submitit_delayed_launcher(num_jobs=num_jobs, tcpport=tcp_port)
def main():
    from torchrl.collectors.collectors import RandomPolicy
    from torchrl.data import BoundedTensorSpec
    from torchrl.envs.libs.gym import GymEnv
    from torchrl.collectors import MultiSyncDataCollector, SyncDataCollector

    collector_class = SyncDataCollector if num_workers == 1 else MultiSyncDataCollector
    collector = DistributedDataCollector(
        [EnvCreator(lambda: GymEnv("ALE/Pong-v5"))] * num_jobs,
        policy=RandomPolicy(BoundedTensorSpec(-1, 1, shape=(1,))),
        launcher="submitit_delayed",
        frames_per_batch=800,
        total_frames=1_000_000,
        tcp_port=tcp_port,
        collector_class=collector_class,
        num_workers_per_collector=args.num_workers,
    )
    pbar = tqdm.tqdm(total=1_000_000)
    for data in collector:
        pbar.update(data.numel())


if __name__ == "__main__":
    main()
