# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Weight Synchronization Schemes - Collector Integration
=======================================================

This example demonstrates how to use weight synchronization schemes with TorchRL
collectors for efficient weight updates across multiple inference workers.

The examples show different synchronization strategies and use cases including
single collectors, multiple collectors, multiple models, and no synchronization.
"""

import torch.nn as nn
from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from torchrl.collectors import MultiSyncDataCollector, SyncDataCollector
from torchrl.envs import GymEnv
from torchrl.weight_update import (
    MultiProcessWeightSyncScheme,
    SharedMemWeightSyncScheme,
)


def example_single_collector_multiprocess():
    """Example 1: Single collector with multiprocess scheme."""
    print("\n" + "=" * 70)
    print("Example 1: Single Collector with Multiprocess Scheme")
    print("=" * 70)

    # Create environment and policy
    env = GymEnv("CartPole-v1")
    policy = TensorDictModule(
        nn.Linear(
            env.observation_spec["observation"].shape[-1], env.action_spec.shape[-1]
        ),
        in_keys=["observation"],
        out_keys=["action"],
    )
    env.close()

    # Create weight sync scheme
    scheme = MultiProcessWeightSyncScheme(strategy="state_dict")

    print("Creating collector with multiprocess weight sync...")
    collector = SyncDataCollector(
        create_env_fn=lambda: GymEnv("CartPole-v1"),
        policy=policy,
        frames_per_batch=64,
        total_frames=200,
        weight_sync_schemes={"policy": scheme},
    )

    # Collect data and update weights periodically
    print("Collecting data...")
    for i, data in enumerate(collector):
        print(f"Iteration {i}: Collected {data.numel()} transitions")

        # Update policy weights every 2 iterations
        if i % 2 == 0:
            new_weights = policy.state_dict()
            collector.update_policy_weights_(new_weights)
            print("  → Updated policy weights")

        if i >= 2:  # Just run a few iterations for demo
            break

    collector.shutdown()
    print("✓ Single collector example completed!\n")


def example_multi_collector_shared_memory():
    """Example 2: Multiple collectors with shared memory."""
    print("\n" + "=" * 70)
    print("Example 2: Multiple Collectors with Shared Memory")
    print("=" * 70)

    # Create environment and policy
    env = GymEnv("CartPole-v1")
    policy = TensorDictModule(
        nn.Linear(
            env.observation_spec["observation"].shape[-1], env.action_spec.shape[-1]
        ),
        in_keys=["observation"],
        out_keys=["action"],
    )
    env.close()

    # Shared memory is more efficient for frequent updates
    scheme = SharedMemWeightSyncScheme(strategy="tensordict", auto_register=True)

    print("Creating multi-collector with shared memory...")
    collector = MultiSyncDataCollector(
        create_env_fn=[
            lambda: GymEnv("CartPole-v1"),
            lambda: GymEnv("CartPole-v1"),
            lambda: GymEnv("CartPole-v1"),
        ],
        policy=policy,
        frames_per_batch=192,
        total_frames=400,
        weight_sync_schemes={"policy": scheme},
    )

    # Workers automatically see weight updates via shared memory
    print("Collecting data...")
    for i, data in enumerate(collector):
        print(f"Iteration {i}: Collected {data.numel()} transitions")

        # Update weights frequently (shared memory makes this very fast)
        collector.update_policy_weights_(TensorDict.from_module(policy))
        print("  → Updated policy weights via shared memory")

        if i >= 1:  # Just run a couple iterations for demo
            break

    collector.shutdown()
    print("✓ Multi-collector with shared memory example completed!\n")


def main():
    """Run all examples."""
    print("\n" + "=" * 70)
    print("Weight Synchronization Schemes - Collector Integration Examples")
    print("=" * 70)

    # Set multiprocessing start method
    import torch.multiprocessing as mp

    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass  # Already set

    # Run examples
    example_single_collector_multiprocess()
    example_multi_collector_shared_memory()

    print("\n" + "=" * 70)
    print("All examples completed successfully!")
    print("=" * 70)
    print("\nKey takeaways:")
    print("  • MultiProcessWeightSyncScheme: Good for general multiprocess scenarios")
    print(
        "  • SharedMemWeightSyncScheme: Fast zero-copy updates for same-machine workers"
    )
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
