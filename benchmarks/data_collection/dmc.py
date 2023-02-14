# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
DeepMind control suite data collection benchmark
================================================

Runs a "cheetah"-"run" dm-control task with a random policy using a multiprocess async data collector.

Image size: torch.Size([240, 320, 3])

Performance results with default configuration:
+-------------------------------+--------------------------------------------------+
| Machine specs                 |  3x A100 GPUs,                                   |
|                               | Intel(R) Xeon(R) Platinum 8275CL CPU @ 3.00GHz   |
|                               |                                                  |
+===============================+==================================================+
|  Batched transforms           | 1885.2913 fps                                    |
+-------------------------------+--------------------------------------------------+
| Single env transform          | 1903.3575 fps                                    |
+-------------------------------+--------------------------------------------------+

"""
import argparse
import time

import torch.cuda
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

parser = argparse.ArgumentParser()

parser.add_argument(
    "--batched",
    action="store_true",
    help="if True, the transforms will be applied on batches of images.",
)
parser.add_argument(
    "--n_envs",
    type=int,
    default=8,
    help="Number of environments to be run in parallel in each collector.",
)
parser.add_argument(
    "--n_workers_collector",
    type=int,
    default=4,
    help="Number sub-collectors in the data collector.",
)
parser.add_argument(
    "--n_frames",
    type=int,
    default=64,
    help="Number of frames in each batch of data collected.",
)

if __name__ == "__main__":

    def make_env():
        return DMControlEnv("cheetah", "run", from_pixels=True)

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
            ParallelEnv(args.n_envs, EnvCreator(make_env))
        )
    else:
        parallel_env = ParallelEnv(
            args.n_envs, EnvCreator(lambda: make_transformed_env(make_env()))
        )
    devices = list(range(torch.cuda.device_count()))[: args.n_workers_collector]
    if len(devices) == 1:
        devices = devices[0]
    elif len(devices) < args.n_workers_collector:
        raise RuntimeError(
            "This benchmark requires at least as many GPUs as the number of collector workers."
        )
    collector = MultiaSyncDataCollector(
        [
            parallel_env,
        ]
        * args.n_workers_collector,
        RandomPolicy(parallel_env.action_spec),
        total_frames=total_frames,
        frames_per_batch=args.n_frames,
        devices=devices,
        passing_devices=devices,
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
