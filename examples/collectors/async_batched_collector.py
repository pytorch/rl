"""AsyncBatchedCollector example.

Demonstrates how to use :class:`~torchrl.collectors.AsyncBatchedCollector` to
run many environments in parallel while automatically batching policy inference
through an :class:`~torchrl.modules.InferenceServer`.

Architecture:
  - An :class:`~torchrl.envs.AsyncEnvPool` runs environments in parallel
    using the chosen backend (``"threading"`` or ``"multiprocessing"``).
  - One lightweight coordinator thread per environment owns a slot in the pool
    and an inference client.
  - An :class:`~torchrl.modules.InferenceServer` batches incoming observations
    and runs a single forward pass.
  - There is no global synchronisation barrier -- fast envs keep stepping
    while slow ones wait for inference.

The user only supplies:
  - A list of environment factories
  - A policy (or policy factory)
"""
import torch.nn as nn
from tensordict.nn import TensorDictModule

from torchrl.collectors import AsyncBatchedCollector
from torchrl.envs import GymEnv


def make_env():
    """Factory that returns a CartPole environment."""
    return GymEnv("CartPole-v1")


def main():
    num_envs = 4
    frames_per_batch = 200
    total_frames = 1_000

    # A simple linear policy (random weights -- just for demonstration)
    policy = TensorDictModule(
        nn.Linear(4, 2), in_keys=["observation"], out_keys=["action"]
    )

    collector = AsyncBatchedCollector(
        create_env_fn=[make_env] * num_envs,
        policy=policy,
        frames_per_batch=frames_per_batch,
        total_frames=total_frames,
        max_batch_size=num_envs,
        device="cpu",
    )

    total_collected = 0
    for i, batch in enumerate(collector):
        n = batch.numel()
        total_collected += n
        print(f"Batch {i}: {batch.shape}  ({n} frames, total={total_collected})")

    collector.shutdown()
    print("Done!")


if __name__ == "__main__":
    main()
