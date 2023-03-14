# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from torchrl.collectors.distributed.generic import submitit_delayed_launcher, \
    DistributedDataCollector
from torchrl.envs import EnvCreator

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
    )
    for data in collector:
        print(data)

if __name__ == "__main__":
    main()
