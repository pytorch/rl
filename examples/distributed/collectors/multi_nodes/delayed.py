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
args = parser.parse_args()

DEFAULT_SLURM_CONF["slurm_partition"] = args.partition
DEFAULT_SLURM_CONF_MAIN["slurm_partition"] = args.partition

num_jobs = 8
tcp_port = 4321


@submitit_delayed_launcher(num_jobs=num_jobs, tcpport=tcp_port)
def main():
    from torchrl.collectors.collectors import RandomPolicy
    from torchrl.data import BoundedTensorSpec
    from torchrl.envs.libs.gym import GymEnv

    collector = DistributedDataCollector(
        [EnvCreator(lambda: GymEnv("ALE/Pong-v5"))] * num_jobs,
        policy=RandomPolicy(BoundedTensorSpec(-1, 1, shape=(1,))),
        launcher="submitit_delayed",
        frames_per_batch=800,
        total_frames=1_000_000,
        tcp_port=tcp_port,
    )
    pbar = tqdm.tqdm(total=1_000_000)
    for data in collector:
        pbar.update(data.numel())


if __name__ == "__main__":
    main()
