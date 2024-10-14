# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import argparse
import time

import pytest
import torch.cuda
import tqdm

from torchrl.collectors import SyncDataCollector
from torchrl.collectors.collectors import (
    MultiaSyncDataCollector,
    MultiSyncDataCollector,
)
from torchrl.data import LazyTensorStorage, ReplayBuffer
from torchrl.data.utils import CloudpickleWrapper
from torchrl.envs import EnvCreator, GymEnv, ParallelEnv, StepCounter, TransformedEnv
from torchrl.envs.libs.dm_control import DMControlEnv
from torchrl.envs.utils import RandomPolicy


def single_collector_setup():
    device = "cuda:0" if torch.cuda.device_count() else "cpu"
    env = TransformedEnv(DMControlEnv("cheetah", "run", device=device), StepCounter(50))
    c = SyncDataCollector(
        env,
        RandomPolicy(env.action_spec),
        total_frames=-1,
        frames_per_batch=100,
        device=device,
    )
    c = iter(c)
    for i, _ in enumerate(c):
        if i == 10:
            break
    return ((c,), {})


def sync_collector_setup():
    device = "cuda:0" if torch.cuda.device_count() else "cpu"
    env = EnvCreator(
        lambda: TransformedEnv(
            DMControlEnv("cheetah", "run", device=device), StepCounter(50)
        )
    )
    c = MultiSyncDataCollector(
        [env, env],
        RandomPolicy(env().action_spec),
        total_frames=-1,
        frames_per_batch=100,
        device=device,
    )
    c = iter(c)
    for i, _ in enumerate(c):
        if i == 10:
            break
    return ((c,), {})


def async_collector_setup():
    device = "cuda:0" if torch.cuda.device_count() else "cpu"
    env = EnvCreator(
        lambda: TransformedEnv(
            DMControlEnv("cheetah", "run", device=device), StepCounter(50)
        )
    )
    c = MultiaSyncDataCollector(
        [env, env],
        RandomPolicy(env().action_spec),
        total_frames=-1,
        frames_per_batch=100,
        device=device,
    )
    c = iter(c)
    for i, _ in enumerate(c):
        if i == 10:
            break
    return ((c,), {})


def single_collector_setup_pixels():
    device = "cuda:0" if torch.cuda.device_count() else "cpu"
    # env = TransformedEnv(
    #     DMControlEnv("cheetah", "run", device=device, from_pixels=True), StepCounter(50)
    # )
    env = TransformedEnv(GymEnv("ALE/Pong-v5"), StepCounter(50))
    c = SyncDataCollector(
        env,
        RandomPolicy(env.action_spec),
        total_frames=-1,
        frames_per_batch=100,
        device=device,
    )
    c = iter(c)
    for i, _ in enumerate(c):
        if i == 10:
            break
    return ((c,), {})


def sync_collector_setup_pixels():
    device = "cuda:0" if torch.cuda.device_count() else "cpu"
    env = EnvCreator(
        lambda: TransformedEnv(
            # DMControlEnv("cheetah", "run", device=device, from_pixels=True),
            GymEnv("ALE/Pong-v5"),
            StepCounter(50),
        )
    )
    c = MultiSyncDataCollector(
        [env, env],
        RandomPolicy(env().action_spec),
        total_frames=-1,
        frames_per_batch=100,
        device=device,
    )
    c = iter(c)
    for i, _ in enumerate(c):
        if i == 10:
            break
    return ((c,), {})


def async_collector_setup_pixels():
    device = "cuda:0" if torch.cuda.device_count() else "cpu"
    env = EnvCreator(
        lambda: TransformedEnv(
            # DMControlEnv("cheetah", "run", device=device, from_pixels=True),
            GymEnv("ALE/Pong-v5"),
            StepCounter(50),
        )
    )
    c = MultiaSyncDataCollector(
        [env, env],
        RandomPolicy(env().action_spec),
        total_frames=-1,
        frames_per_batch=100,
        device=device,
    )
    c = iter(c)
    for i, _ in enumerate(c):
        if i == 10:
            break
    return ((c,), {})


def execute_collector(c):
    # will run for 9 iterations (1 during setup)
    next(c)


def test_single(benchmark):
    (c,), _ = single_collector_setup()
    benchmark(execute_collector, c)


def test_sync(benchmark):
    (c,), _ = sync_collector_setup()
    benchmark(execute_collector, c)


def test_async(benchmark):
    (c,), _ = async_collector_setup()
    benchmark(execute_collector, c)


@pytest.mark.skipif(not torch.cuda.device_count(), reason="no rendering without cuda")
def test_single_pixels(benchmark):
    (c,), _ = single_collector_setup_pixels()
    benchmark(execute_collector, c)


@pytest.mark.skipif(not torch.cuda.device_count(), reason="no rendering without cuda")
def test_sync_pixels(benchmark):
    (c,), _ = sync_collector_setup_pixels()
    benchmark(execute_collector, c)


@pytest.mark.skipif(not torch.cuda.device_count(), reason="no rendering without cuda")
def test_async_pixels(benchmark):
    (c,), _ = async_collector_setup_pixels()
    benchmark(execute_collector, c)


class TestRBGCollector:
    @pytest.mark.parametrize(
        "n_col,n_wokrers_per_col",
        [
            [2, 2],
            [4, 2],
            [8, 2],
            [16, 2],
            [2, 1],
            [4, 1],
            [8, 1],
            [16, 1],
        ],
    )
    def test_multiasync_rb(self, n_col, n_wokrers_per_col):
        make_env = EnvCreator(lambda: GymEnv("ALE/Pong-v5"))
        if n_wokrers_per_col > 1:
            make_env = ParallelEnv(n_wokrers_per_col, make_env)
            env = make_env
            policy = RandomPolicy(env.action_spec)
        else:
            env = make_env()
            policy = RandomPolicy(env.action_spec)

        storage = LazyTensorStorage(10_000)
        rb = ReplayBuffer(storage=storage)
        rb.extend(env.rollout(2, policy).reshape(-1))
        rb.append_transform(CloudpickleWrapper(lambda x: x.reshape(-1)), invert=True)

        fpb = n_wokrers_per_col * 100
        total_frames = n_wokrers_per_col * 100_000
        c = MultiaSyncDataCollector(
            [make_env] * n_col,
            policy,
            frames_per_batch=fpb,
            total_frames=total_frames,
            replay_buffer=rb,
        )
        frames = 0
        pbar = tqdm.tqdm(total=total_frames - (n_col * fpb))
        for i, _ in enumerate(c):
            if i == n_col:
                t0 = time.time()
            if i >= n_col:
                frames += fpb
            if i > n_col:
                fps = frames / (time.time() - t0)
                pbar.update(fpb)
                pbar.set_description(f"fps: {fps: 4.4f}")


if __name__ == "__main__":
    args, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst"] + unknown)
