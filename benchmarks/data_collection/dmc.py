# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
DeepMind control suite data collection benchmark
================================================

"""

import time

import tqdm

from torchrl.collectors.collectors import MultiaSyncDataCollector, RandomPolicy
from torchrl.envs import (
    Compose,
    EnvCreator,
    GrayScale,
    ParallelEnv,
    Resize,
    ToTensorImage,
    TransformedEnv,
)
from torchrl.envs.libs.dm_control import DMControlEnv

total_frames = 100000
if __name__ == "__main__":

    def make_env():
        return DMControlEnv("cheetah", "run", from_pixels=True)

    def make_transformed_env(env):
        return TransformedEnv(
            env,
            Compose(
                ToTensorImage(),
                GrayScale(),
                Resize(84, 84),
            ),
        )

    parallel_env = make_transformed_env(ParallelEnv(8, EnvCreator(make_env)))
    # parallel_env = ParallelEnv(8, EnvCreator(lambda: make_transformed_env(make_env())))
    collector = MultiaSyncDataCollector(
        [
            parallel_env,
            parallel_env,
            parallel_env,
            parallel_env,
        ],
        RandomPolicy(parallel_env.action_spec),
        total_frames=total_frames,
        frames_per_batch=64,
        devices=["cuda:0", "cuda:1", "cuda:2", "cuda:3"],
        passing_devices=["cuda:0", "cuda:1", "cuda:2", "cuda:3"],
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
    print(f"frames per sec: {frames/t: 4.4f} (frames={frames}, t={t})")
    del collector
    exit()
