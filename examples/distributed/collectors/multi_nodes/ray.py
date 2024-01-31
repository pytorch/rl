"""
Example use of a distributed collector
======================================

This example illustrates how a TorchRL collector can be converted into a distributed collector.

This example should create 3 collector instances, 1 local and 2 remote, but 4 instances seem to
be created. Why?
"""
from tensordict.nn import TensorDictModule
from torch import nn
from torchrl._utils import logger as torchrl_logger
from torchrl.collectors.distributed.ray import RayCollector
from torchrl.envs.libs.gym import GymEnv


if __name__ == "__main__":

    # 1. Create environment factory
    def env_maker():
        return GymEnv("Pendulum-v1", device="cpu")

    policy = TensorDictModule(
        nn.Linear(3, 1), in_keys=["observation"], out_keys=["action"]
    )

    # 2. Define distributed collector
    remote_config = {
        "num_cpus": 1,
        "num_gpus": 0.2,
        "memory": 5 * 1024**3,
        "object_store_memory": 2 * 1024**3,
    }
    distributed_collector = RayCollector(
        [env_maker],
        policy,
        total_frames=10000,
        frames_per_batch=200,
    )

    # Sample batches until reaching total_frames
    counter = 0
    num_frames = 0
    for batch in distributed_collector:
        counter += 1
        num_frames += batch.shape.numel()
        torchrl_logger.info(f"batch {counter}, total frames {num_frames}")
