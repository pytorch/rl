# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Visual representation with DNN data collection benchmark
========================================================

"""

import warnings

warnings.filterwarnings("ignore")

import time

import tqdm

from torchrl.collectors.collectors import MultiaSyncDataCollector, RandomPolicy
from torchrl.envs import EnvCreator, ParallelEnv, R3MTransform, TransformedEnv
from torchrl.envs.libs.dm_control import DMControlEnv

total_frames = 10000
if __name__ == "__main__":

    def make_env():
        return DMControlEnv("cheetah", "run", from_pixels=True)

    def make_transformed_env(env):
        return TransformedEnv(
            env,
            R3MTransform("resnet50", in_keys=["pixels"]),
        )

    for method in range(2):
        if method == 0:

            def parallel_env():
                return make_transformed_env(ParallelEnv(16, EnvCreator(make_env)))

            policy = RandomPolicy(make_env().action_spec)
        else:
            parallel_env = ParallelEnv(
                16, EnvCreator(lambda: make_transformed_env(make_env()))
            )
            policy = RandomPolicy(parallel_env.action_spec)
        collector = MultiaSyncDataCollector(
            [
                parallel_env,
                parallel_env,
            ],
            policy=policy,
            total_frames=total_frames,
            frames_per_batch=64,
            devices=[
                "cuda:0",
                "cuda:1",
            ],
            passing_devices=[
                "cuda:0",
                "cuda:1",
            ],
            split_trajs=False,
        )
        frames = 0
        pbar = tqdm.tqdm(total=total_frames)
        for i, data in enumerate(collector):
            pbar.update(data.numel())
            if i == 10:
                t = time.time()
            if i >= 10:
                frames += data.numel()
        t = time.time() - t
        print(
            f"batched={method==0}, frames per sec: {frames/t: 4.4f} (frames={frames}, t={t})"
        )
        del collector
    exit()
