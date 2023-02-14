# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Replay Buffer usage benchmark
=============================

"""

import warnings

warnings.filterwarnings("ignore")

import time

import tqdm
from torchrl.collectors.collectors import MultiaSyncDataCollector, RandomPolicy

from torchrl.data import LazyMemmapStorage, ListStorage, ReplayBuffer
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

total_frames = 100_000
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

    for method in range(2):

        def parallel_env():
            return make_transformed_env(ParallelEnv(16, EnvCreator(make_env)))

        policy = RandomPolicy(make_env().action_spec)
        if method == 0:
            replay_buffer = ReplayBuffer(
                1_000_000, storage=ListStorage(1_000_000), prefetch=10
            )
        else:
            replay_buffer = ReplayBuffer(
                1_000_000, storage=LazyMemmapStorage(1_000_000), prefetch=10
            )
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
        for _data in collector:
            break
        data = _data
        for i in range(total_frames // data.numel()):
            pbar.update(data.numel())
            if i == 10:
                t = time.time()
            if i >= 10:
                frames += data.numel()
            replay_buffer.extend(data.view(-1).cpu())
            sample = replay_buffer.sample(128).contiguous()
            print(sample)
            data = sample["pixels"]

        t = time.time() - t
        print(
            f"memmap={method==1}, frames per sec: {frames/t: 4.4f} (frames={frames}, t={t})"
        )
        del collector
    exit()
