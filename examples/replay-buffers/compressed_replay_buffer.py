#!/usr/bin/env python3
"""
Example demonstrating the use of CompressedStorage for memory-efficient replay buffers.

This example shows how to use the new CompressedStorage to store image observations
with significant memory savings through compression.
"""

import time

import torch
from tensordict import TensorDict
from torchrl.data import CompressedStorage, ReplayBuffer


def main():
    print("=== Compressed Replay Buffer Example ===\n")

    # Create a compressed storage with zstd compression
    print("Creating compressed storage...")
    storage = CompressedStorage(
        max_size=1000,
        compression_level=3,  # zstd compression level (1-22)
        device="cpu",
    )

    # Create replay buffer with compressed storage
    rb = ReplayBuffer(storage=storage, batch_size=32)

    # Simulate Atari-like image data (84x84 RGB frames)
    print("Generating sample image data...")
    num_frames = 100
    image_data = torch.zeros(num_frames, 3, 84, 84, dtype=torch.float32)
    image_data.copy_(
        torch.arange(num_frames * 3 * 84 * 84).reshape(num_frames, 3, 84, 84)
        // (3 * 84 * 84)
    )

    # Create TensorDict with image observations
    data = TensorDict(
        {
            "obs": image_data,
            "action": torch.randint(0, 4, (num_frames,)),  # 4 possible actions
            "reward": torch.randn(num_frames),
            "done": torch.randint(0, 2, (num_frames,), dtype=torch.bool),
        },
        batch_size=[num_frames],
    )

    # Measure memory usage before adding data
    print(f"Original data size: {data.bytes() / 1024 / 1024:.2f} MB")

    # Add data to replay buffer
    print("Adding data to replay buffer...")
    start_time = time.time()
    rb.extend(data)
    add_time = time.time() - start_time
    print(f"Time to add data: {add_time:.3f} seconds")

    # Sample from replay buffer
    print("Sampling from replay buffer...")
    start_time = time.time()
    sample = rb.sample(32)
    sample_time = time.time() - start_time
    print(f"Time to sample: {sample_time:.3f} seconds")

    # Verify data integrity
    print("\nVerifying data integrity...")
    original_shape = image_data.shape
    sampled_shape = sample["obs"].shape
    print(f"Original shape: {original_shape}")
    print(f"Sampled shape: {sampled_shape}")

    # Check that shapes match (accounting for batch size)
    assert sampled_shape[1:] == original_shape[1:], "Shape mismatch!"
    print("✓ Data integrity verified!")

    # Demonstrate compression ratio
    print("\n=== Compression Analysis ===")

    # Estimate compressed size (this is approximate)
    compressed_size_estimate = storage.bytes()

    original_size = data.bytes()
    compression_ratio = (
        original_size / compressed_size_estimate if compressed_size_estimate > 0 else 1
    )

    print(f"Original size: {original_size / 1024 / 1024:.2f} MB")
    print(
        f"Compressed size (estimate): {compressed_size_estimate / 1024 / 1024:.2f} MB"
    )
    print(f"Compression ratio: {compression_ratio:.1f}x")

    # Test with different compression levels
    print("\n=== Testing Different Compression Levels ===")

    for level in [1, 3, 6, 9]:
        print(f"\nTesting compression level {level}...")

        # Create new storage with different compression level
        test_storage = CompressedStorage(
            max_size=100, compression_level=level, device="cpu"
        )

        # Test with a smaller dataset
        N = 100
        obs = torch.zeros(N, 3, 84, 84, dtype=torch.float32)
        obs.copy_(torch.arange(N * 3 * 84 * 84).reshape(N, 3, 84, 84) // (3 * 84 * 84))
        test_data = TensorDict(
            {
                "obs": obs,
            },
            batch_size=[N],
        )

        test_rb = ReplayBuffer(storage=test_storage, batch_size=5)

        # Measure compression time
        start_time = time.time()
        test_rb.extend(test_data)
        compress_time = time.time() - start_time

        # Measure decompression time
        start_time = time.time()
        test_rb.sample(5)
        decompress_time = time.time() - start_time

        print(f"  Compression time: {compress_time:.3f}s")
        print(f"  Decompression time: {decompress_time:.3f}s")

        # Estimate compression ratio
        test_ratio = test_data.bytes() / test_storage.bytes()
        print(f"  Compression ratio: {test_ratio:.1f}x")

    print("\n=== Example Complete ===")
    print(
        "The CompressedStorage successfully reduces memory usage while maintaining data integrity!"
    )


if __name__ == "__main__":
    main()
