"""
Example use of a distributed collector
======================================
This example illustrates how a TorchRL collector can be converted into a distributed collector.
This example should create 3 collector instances, 1 local and 2 remote, but 4 instances seem to
be created. Why?
"""

from tensordict.nn import TensorDictModule
from torch import nn
from torchrl.collectors.collectors import Interruptor, SyncDataCollector
from torchrl.envs.libs.gym import GymEnv


if __name__ == "__main__":
    # 1. Create environment factory
    env_maker = lambda: GymEnv("Pendulum-v1", device="cpu")
    policy = TensorDictModule(
        nn.Linear(3, 1), in_keys=["observation"], out_keys=["action"]
    )

    # 2. Define interruptor object
    interruptor = Interruptor()
    interruptor.start_collection()

    # 2. Define distributed collector
    collector = SyncDataCollector(
        env_maker,
        policy,
        total_frames=2000,
        frames_per_batch=50,
        device="cpu",
        storing_device="cpu",
        interruptor=interruptor,
        split_trajs=False,
    )

    # Sample batches until reaching total_frames
    counter = 0
    num_frames = 0
    interruptor.stop_collection()
    for batch in collector:
        counter += 1
        num_frames += batch.shape.numel()
        import ipdb

        ipdb.set_trace()
        print(f"batch {counter}, total frames {num_frames}")
        assert batch[0, 0]["observation"].sum() != 0
        assert batch[0, 1:]["observation"].sum() == 0
