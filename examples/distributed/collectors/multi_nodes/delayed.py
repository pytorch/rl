# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from argparse import ArgumentParser

from torchrl.collectors.distributed.generic import submitit_delayed_launcher, \
    DistributedDataCollector, DEFAULT_SLURM_CONF, DEFAULT_SLURM_CONF_MAIN
from torchrl.envs import EnvCreator

parser = ArgumentParser()
parser.add_argument("--partition", help="slurm partition to use")
args = parser.parse_args()

DEFAULT_SLURM_CONF["slurm_partition"] = args.partition
DEFAULT_SLURM_CONF_MAIN["slurm_partition"] = args.partition

num_jobs=2

@submitit_delayed_launcher(num_jobs=num_jobs)
def main():
    from torchrl.envs.libs.gym import GymEnv
    from torchrl.collectors.collectors import RandomPolicy
    from torchrl.data import BoundedTensorSpec
    collector = DistributedDataCollector(
        [EnvCreator(lambda: GymEnv("Pendulum-v1"))] * num_jobs,
        policy=RandomPolicy(BoundedTensorSpec(-1, 1, shape=(1,))),
        launcher="submitit_delayed",
        frames_per_batch=200,
        total_frames=10_000,
    )
    for data in collector:
        print(data)

if __name__ == "__main__":
    main()
