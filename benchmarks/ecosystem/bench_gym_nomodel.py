# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import time
import warnings
from argparse import ArgumentParser
from pathlib import Path

import gymnasium as gym
import torch.cuda
import torchrl
from git import Repo
from torchrl.collectors.collectors import (
    MultiaSyncDataCollector,
    MultiSyncDataCollector,
    RandomPolicy,
    SyncDataCollector,
)
from torchrl.envs import ParallelEnv
from torchrl.envs.libs.gym import GymEnv
from torchrl.record.loggers.wandb import WandbLogger

print(torch.multiprocessing.get_sharing_strategy())
# torch.multiprocessing.set_sharing_strategy('file_system')

warnings.filterwarnings("ignore", category=UserWarning)

parser = ArgumentParser()
parser.add_argument("--num_workers", default=8, type=int, help="Number of workers.")
parser.add_argument(
    "--frames_per_batch",
    default=1000,
    type=int,
    help="Number of frames collected in a batch. Must be "
    "divisible by the number of workers.",
)
parser.add_argument(
    "--total_frames",
    default=100_000,
    type=int,
    help="Total number of frames collected by the collector. Must be "
    "divisible by the number of frames per batch.",
)
parser.add_argument(
    "--log_every",
    default=10_000,
    type=int,
    help="Number of frames between each log.",
)
parser.add_argument(
    "--env",
    # default="PongNoFrameskip-v4",
    default="ALE/Pong-v5",
    help="Gym environment to be run.",
)


def get_current_branch_and_commit_hash(repo_path):
    repo = Repo(repo_path)
    try:
        current_branch = repo.active_branch.name
    except Exception:
        # assume main
        current_branch = "-main-"
    latest_commit_hash = repo.head.commit.hexsha
    return current_branch, latest_commit_hash


if __name__ == "__main__":
    args = parser.parse_args()
    num_workers = args.num_workers
    frames_per_batch = args.frames_per_batch

    current_branch, latest_commit_hash = get_current_branch_and_commit_hash(
        Path(torchrl.__file__).parent.parent
    )

    print(
        f"Running {num_workers} envs with {frames_per_batch} frames per batch"
        f" (i.e. {frames_per_batch / num_workers} frames per env)."
    )

    # Test asynchronous gym collector
    def test_gym():
        logger = WandbLogger(
            project="benchmark-atari",
            exp_name=f"{current_branch}/{latest_commit_hash[:6]}/gym",
        )
        env = gym.vector.AsyncVectorEnv(
            [lambda: gym.make(args.env) for _ in range(num_workers)]
        )
        env.reset()
        global_step = 0
        times = []
        prev_start = start = time.time()
        print("Timer started.")
        for _ in range(args.total_frames // num_workers):
            obs, *_ = env.step(env.action_space.sample())
            obs * 1 # to make sure data is accessed once
            global_step += num_workers
            if global_step % int(frames_per_batch) == 0:
                times.append(time.time() - start)
                fps = frames_per_batch / times[-1]
                start = time.time()
                if global_step % args.log_every == 0:
                    logger.log_scalar("fps", args.log_every // (start - prev_start))
                    print(f"FPS Gym AsyncVectorEnv at step {global_step}:", fps)
                    prev_start = start
        env.close()
        logger.experiment.finish()
        del logger, env
        print("FPS Gym AsyncVectorEnv mean:", args.total_frames / sum(times))

    # Test asynchronous gym collector
    def test_sb3():
        from stable_baselines3.common.env_util import make_vec_env
        from stable_baselines3.common.vec_env import SubprocVecEnv

        logger = WandbLogger(
            project="benchmark-atari",
            exp_name=f"{current_branch}/{latest_commit_hash[:6]}/sb3",
        )
        env = make_vec_env(args.env, n_envs=num_workers, vec_env_cls=SubprocVecEnv)
        env.reset()
        global_step = 0
        times = []
        prev_start = start = time.time()
        print("Timer started.")
        for _ in range(args.total_frames // num_workers):
            obs, *_ = env.step(env.action_space.sample())
            obs * 1 # to make sure data is accessed once
            global_step += num_workers
            if global_step % int(frames_per_batch) == 0:
                times.append(time.time() - start)
                fps = frames_per_batch / times[-1]
                start = time.time()
                if global_step % args.log_every == 0:
                    logger.log_scalar("fps", args.log_every // (start - prev_start))
                    print(f"FPS Gym AsyncVectorEnv at step {global_step}:", fps)
                    prev_start = start
        env.close()
        logger.experiment.finish()
        del logger, env
        print("FPS Gym AsyncVectorEnv mean:", args.total_frames / sum(times))

    # Test multiprocess TorchRL collector
    def test_torch_rl(collector_class, device):
        logger = WandbLogger(
            project="benchmark-atari",
            exp_name=f"{current_branch}/{latest_commit_hash[:6]}/torchrl/{collector_class.__name__}/{device}",
        )

        # make_env = EnvCreator(lambda: GymEnv(args.env, device=device))
        def make_env():
            return GymEnv(args.env, device=device)

        if collector_class in [MultiSyncDataCollector, MultiaSyncDataCollector]:
            mock_env = make_env()
            collector = collector_class(
                [make_env] * num_workers,
                policy=RandomPolicy(mock_env.action_spec),
                total_frames=args.total_frames,
                frames_per_batch=frames_per_batch,
                device=device,
                storing_device=device,
            )
        elif collector_class in [SyncDataCollector]:
            parallel_env = ParallelEnv(args.num_workers, make_env)
            collector = SyncDataCollector(
                parallel_env,
                policy=RandomPolicy(parallel_env.action_spec),
                total_frames=args.total_frames,
                frames_per_batch=frames_per_batch,
                device=device,
                storing_device=device,
            )
        global_step = 0
        times = []
        prev_start = start = time.time()
        print("Timer started.")
        for data in collector:
            global_step += data.numel()
            data.get("pixels") * 1
            times.append(time.time() - start)
            fps = frames_per_batch / times[-1]
            start = time.time()
            if global_step % args.log_every == 0:
                logger.log_scalar("fps", args.log_every // (start - prev_start))
                print(
                    f"FPS TorchRL with {collector_class.__name__} on {device} at step {global_step}:",
                    fps,
                )
                prev_start = start
        collector.shutdown()
        logger.experiment.finish()
        del logger, collector
        print(
            "FPS TorchRL with",
            collector_class.__name__,
            "on",
            device,
            "mean:",
            args.total_frames / sum(times),
        )

    # Test multiprocess TorchRL collector
    def test_torch_rl_env(device):
        logger = WandbLogger(
            project="benchmark-atari",
            exp_name=f"{current_branch}/{latest_commit_hash[:6]}/torchrl/parallel/{device}",
        )

        # make_env = EnvCreator(lambda: GymEnv(args.env, device=device))
        def make_env():
            return GymEnv(args.env, device=device)

        parallel_env = ParallelEnv(args.num_workers, make_env)
        global_step = 0
        times = []
        prev_start = start = time.time()
        nsteps = frames_per_batch // num_workers
        print("Timer started.")
        while global_step < args.total_frames:
            data = parallel_env.rollout(nsteps, break_when_any_done=False)
            data.get("pixels") * 1
            global_step += data.numel()
            times.append(time.time() - start)
            fps = frames_per_batch / times[-1]
            start = time.time()
            if global_step % args.log_every == 0:
                logger.log_scalar("fps", args.log_every // (start - prev_start))
                print(
                    f"FPS TorchRL with ParallelEnv on {device} at step {global_step}:",
                    fps,
                )
                prev_start = start
        logger.experiment.finish()
        del logger, parallel_env
        print(
            "FPS TorchRL with ParallelEnv on",
            device,
            "mean:",
            args.total_frames / sum(times),
        )

    test_gym()
    test_sb3()

    device_list = ["cpu"]
    if torch.cuda.device_count():
        device_list = ["cuda:0", *device_list]
    for device in device_list:
        test_torch_rl_env(device)

    for collector_class in [
        SyncDataCollector,
        MultiaSyncDataCollector,
        MultiSyncDataCollector,
    ]:
        for device in device_list:
            test_torch_rl(collector_class, device)

    exit()
