import torch.cuda

from torchrl.collectors import SyncDataCollector
from torchrl.collectors.collectors import RandomPolicy, MultiSyncDataCollector
from torchrl.envs import StepCounter, TransformedEnv, EnvCreator
from torchrl.envs.libs.dm_control import DMControlEnv


def single_collector_setup():
    print("setup")
    device = "cuda:0" if torch.cuda.device_count() else "cpu"
    env = TransformedEnv(DMControlEnv("cheetah", "run", device=device), StepCounter(50))
    c = SyncDataCollector(env, RandomPolicy(env.action_spec), total_frames=10_000, frames_per_batch=100, device=device)
    print("done")
    return ((c,), {})

def sync_collector_setup():
    print("setup")
    device = "cuda:0" if torch.cuda.device_count() else "cpu"
    env = EnvCreator(lambda: TransformedEnv(DMControlEnv("cheetah", "run", device=device), StepCounter(50)))
    c = MultiSyncDataCollector([env, env], RandomPolicy(env().action_spec), total_frames=10_000, frames_per_batch=100, device=device)
    return ((c,), {})

def execute_collector(c):
    print("run")
    for _ in c:
        continue
    print("done")
def test_single(benchmark):
    benchmark.pedantic(execute_collector, setup=single_collector_setup, iterations=1, rounds=100)


def test_sync(benchmark):
    benchmark.pedantic(execute_collector, setup=sync_collector_setup, iterations=1, rounds=100)
