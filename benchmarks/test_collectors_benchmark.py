# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import argparse

import pytest
import torch.cuda

from torchrl.collectors import SyncDataCollector
from torchrl.collectors.collectors import (
    MultiaSyncDataCollector,
    MultiSyncDataCollector,
)
from torchrl.envs import EnvCreator, GymEnv, StepCounter, TransformedEnv
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


if __name__ == "__main__":
    args, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst"] + unknown)
