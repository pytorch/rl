# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Atari game data collection benchmark
====================================

Runs an Atari game with a random policy using a multiprocess async data collector.

Performance results
+-------------------------------+-------------------------------------------------+
| Machine specs                  | 3x A100 GPUs,                                   |
|                               | Intel(R) Xeon(R) Platinum 8275CL CPU @ 3.00GHz  |
|                               |                                                 |
+===============================+===========+
|  Batched transforms         | 1736.0687 fps        |
+-------------------------------+-----------+
| Single env transform      | 2525.5021 fps    |
+-------------------------------+-----------+

"""
import argparse
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
from torchrl.envs.libs.gym import GymEnv

total_frames = 100000

parser = argparse.ArgumentParser()

parser.add_argument(
    "--batched",
    action="store_true",
    help="if True, the transforms will be applied on batches of images.",
)

if __name__ == "__main__":

    def make_env():
        return GymEnv("ALE/Pong-v5")

    # print the raw env output
    print(make_env().fake_tensordict())

    def make_transformed_env(env):
        return TransformedEnv(
            env,
            Compose(
                ToTensorImage(),
                GrayScale(),
                Resize(84, 84),
            ),
        )

    args = parser.parse_args()
    if args.batched:
        parallel_env = make_transformed_env(
            ParallelEnv(16, EnvCreator(lambda: make_env()))
        )
    else:
        parallel_env = ParallelEnv(
            16, EnvCreator(lambda: make_transformed_env(make_env()))
        )
    collector = MultiaSyncDataCollector(
        [
            parallel_env,
            parallel_env,
            parallel_env,
        ],
        RandomPolicy(parallel_env.action_spec),
        total_frames=total_frames,
        frames_per_batch=64,
        devices=["cuda:0", "cuda:1", "cuda:2"],
        passing_devices=["cuda:0", "cuda:1", "cuda:2"],
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
    del collector
    print(f"\n\nframes per sec: {frames/t: 4.4f} (frames={frames}, t={t})\n\n")
    exit()
