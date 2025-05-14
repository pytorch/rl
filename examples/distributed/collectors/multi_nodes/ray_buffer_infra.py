"""
Example use of an ever-running, fully async, distributed collector
==================================================================

This example demonstrates how to set up and use a distributed collector
with Ray in a fully asynchronous manner. The collector continuously gathers
data from a gym environment and stores it in a replay buffer, allowing for
concurrent processing and data collection.

Key Components:
1. **Environment Factory**: A simple function that creates instances of the
   `GymEnv` environment. In this example, we use the "Pendulum-v1" environment.
2. **Policy Definition**: A `TensorDictModule` that defines the policy network.
   Here, a simple linear layer is used to map observations to actions.
3. **Replay Buffer**: A `RayReplayBuffer` that stores collected data for later
   use, such as training a reinforcement learning model.
4. **Distributed Collector**: A `RayCollector` that manages the distributed
   collection of data. It is configured with remote resources and interacts
   with the environment and policy to gather data.
5. **Asynchronous Execution**: The collector runs in the background, allowing
   the main program to perform other tasks concurrently. The example includes
   a loop that waits for data to be available in the buffer and samples it.
6. **Graceful Shutdown**: The collector is shut down asynchronously, ensuring
   that all resources are properly released.

This setup is useful for scenarios where you need to collect data from
multiple environments in parallel, leveraging Ray's distributed computing
capabilities to scale efficiently.

"""
import asyncio

from tensordict.nn import TensorDictModule
from torch import nn
from torchrl.collectors.distributed.ray import RayCollector
from torchrl.data.replay_buffers.ray_buffer import RayReplayBuffer
from torchrl.envs.libs.gym import GymEnv


async def main():
    # 1. Create environment factory
    def env_maker():
        return GymEnv("Pendulum-v1", device="cpu")

    policy = TensorDictModule(
        nn.Linear(3, 1), in_keys=["observation"], out_keys=["action"]
    )

    buffer = RayReplayBuffer()

    # 2. Define distributed collector
    remote_config = {
        "num_cpus": 1,
        "num_gpus": 0,
        "memory": 5 * 1024**3,
        "object_store_memory": 2 * 1024**3,
    }
    distributed_collector = RayCollector(
        [env_maker],
        policy,
        total_frames=600,
        frames_per_batch=200,
        remote_configs=remote_config,
        replay_buffer=buffer,
    )

    print("start")
    distributed_collector.start()

    while True:
        while not len(buffer):
            print("waiting")
            await asyncio.sleep(1)  # Use asyncio.sleep instead of time.sleep
        print("sample", buffer.sample(32))
        # break at some point
        break

    await distributed_collector.async_shutdown()


if __name__ == "__main__":
    asyncio.run(main())
