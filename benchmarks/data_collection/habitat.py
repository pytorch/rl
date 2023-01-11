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
|  Batched transforms           | 843.6275 fps                                     |
+-------------------------------+--------------------------------------------------+
|  Single env transform         | 863.7805 fps                                     |
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
from torchrl.envs.libs.habitat import HabitatEnv

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
parser.add_argument(
    "--perf_mode",
    action="store_true",
    help="If True, the env are created in performance mode (lower rendering quality, higher throughput)",
)
if __name__ == "__main__":
    args = parser.parse_args()

    def make_env(device=0):
        return HabitatEnv("HabitatPick-v0" if args.perf_mode else "HabitatRenderPick-v0", from_pixels=True, device=device)

    # print the raw env output
    env = make_env(3)
    r = env.rollout(3)
    env.close()
    del env, r
    gc.collect()

    def make_transformed_env(env):
        return TransformedEnv(
            env,
            Compose(
                ToTensorImage(),
                GrayScale(),
                Resize(84, 84),
            ),
        )

    if args.batched:
        parallel_env = make_transformed_env(
            ParallelEnv(args.n_envs, EnvCreator(make_env))
        )
    else:
        parallel_env = ParallelEnv(
            args.n_envs, EnvCreator(lambda: make_transformed_env(make_env()))
        )
    devices = list(range(torch.cuda.device_count()))[1:(args.n_workers_collector+1)]
    if len(devices) == 1:
        devices = devices[0]
    elif len(devices) < args.n_workers_collector:
        raise RuntimeError(
            "This benchmark requires one more GPU than the number of collector workers."
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
