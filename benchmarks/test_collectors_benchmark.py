import torch.cuda

from torchrl.collectors import SyncDataCollector
from torchrl.collectors.collectors import (
    MultiaSyncDataCollector,
    MultiSyncDataCollector,
    RandomPolicy,
)
from torchrl.envs import EnvCreator, StepCounter, TransformedEnv
from torchrl.envs.libs.dm_control import DMControlEnv


def single_collector_setup():
    device = "cuda:0" if torch.cuda.device_count() else "cpu"
    env = TransformedEnv(DMControlEnv("cheetah", "run", device=device), StepCounter(50))
    c = SyncDataCollector(
        env,
        RandomPolicy(env.action_spec),
        total_frames=10_000,
        frames_per_batch=100,
        device=device,
    )
    c = iter(c)
    for _ in c:
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
        total_frames=10_000,
        frames_per_batch=100,
        device=device,
    )
    c = iter(c)
    for _ in c:
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
        total_frames=10_000,
        frames_per_batch=100,
        device=device,
    )
    c = iter(c)
    for _ in c:
        break
    return ((c,), {})


def single_collector_setup_pixels():
    device = "cuda:0" if torch.cuda.device_count() else "cpu"
    env = TransformedEnv(
        DMControlEnv("cheetah", "run", device=device, from_pixels=True), StepCounter(50)
    )
    c = SyncDataCollector(
        env,
        RandomPolicy(env.action_spec),
        total_frames=10_000,
        frames_per_batch=100,
        device=device,
    )
    c = iter(c)
    for _ in c:
        break
    return ((c,), {})


def sync_collector_setup_pixels():
    device = "cuda:0" if torch.cuda.device_count() else "cpu"
    env = EnvCreator(
        lambda: TransformedEnv(
            DMControlEnv("cheetah", "run", device=device, from_pixels=True),
            StepCounter(50),
        )
    )
    c = MultiSyncDataCollector(
        [env, env],
        RandomPolicy(env().action_spec),
        total_frames=10_000,
        frames_per_batch=100,
        device=device,
    )
    c = iter(c)
    for _ in c:
        break
    return ((c,), {})


def async_collector_setup_pixels():
    device = "cuda:0" if torch.cuda.device_count() else "cpu"
    env = EnvCreator(
        lambda: TransformedEnv(
            DMControlEnv("cheetah", "run", device=device, from_pixels=True),
            StepCounter(50),
        )
    )
    c = MultiaSyncDataCollector(
        [env, env],
        RandomPolicy(env().action_spec),
        total_frames=10_000,
        frames_per_batch=100,
        device=device,
    )
    c = iter(c)
    for _ in c:
        break
    return ((c,), {})


def execute_collector(c):
    # will run for 9 iterations (1 during setup)
    for _ in c:
        continue


def test_single(benchmark):
    benchmark.pedantic(
        execute_collector, setup=single_collector_setup, iterations=1, rounds=5
    )


def test_sync(benchmark):
    benchmark.pedantic(
        execute_collector, setup=sync_collector_setup, iterations=1, rounds=5
    )


def test_async(benchmark):
    benchmark.pedantic(
        execute_collector, setup=async_collector_setup, iterations=1, rounds=5
    )


def test_single_pixels(benchmark):
    benchmark.pedantic(
        execute_collector, setup=single_collector_setup_pixels, iterations=1, rounds=5
    )


def test_sync_pixels(benchmark):
    benchmark.pedantic(
        execute_collector, setup=sync_collector_setup_pixels, iterations=1, rounds=5
    )


def test_async_pixels(benchmark):
    benchmark.pedantic(
        execute_collector, setup=async_collector_setup_pixels, iterations=1, rounds=5
    )
