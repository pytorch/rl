"""
Example use of a distributed collector
======================================

This example illustrates how a TorchRL collector can be converted into a distributed collector.

This example should create 3 collector instances, 1 local and 2 remote, but 4 instances seem to
be created. Why?
"""

from torch import nn
from tensordict.nn import TensorDictModule
from torchrl.envs.libs.gym import GymEnv
from torchrl.collectors.collectors import SyncDataCollector
from torchrl.collectors.distributed.ray_collector import RayDistributedCollector


if __name__ == "__main__":

    # 1. Create environment
    env_maker = lambda: GymEnv("Pendulum-v1", device="cpu")
    policy = TensorDictModule(nn.Linear(3, 1), in_keys=["observation"], out_keys=["action"])

    # 2. Define distributed collector
    remote_config = {
        "num_cpus": 1,
        "num_gpus": 0.2,
        "memory": 5 * 1024 ** 3,
        "object_store_memory": 2 * 1024 ** 3
    }
    distributed_collector = RayDistributedCollector(
        env_makers=[env_maker],
        policy=policy,
        collector_class=SyncDataCollector,
        collector_kwargs={
            "create_env_fn": env_maker,
            "policy": policy,
            "total_frames": -1,  # automatically set always to -1 ? DistributedCollector already specifies total_frames.
            "max_frames_per_traj": 50,
            "frames_per_batch": 200,
            "init_random_frames": -1,
            "reset_at_each_iter": False,
            "device": "cpu",
            "storing_device": "cpu",
        },
        remote_config=remote_config,
        num_collectors=1,
        total_frames=10000,
        coordination="async",
    )

    # Sample batches until reaching total_frames
    counter = 0
    num_frames = 0
    for batch in distributed_collector:
        counter += 1
        num_frames += batch.shape.numel()
        print(f"batch {counter}, total frames {num_frames}")

