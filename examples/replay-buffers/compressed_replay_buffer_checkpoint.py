#!/usr/bin/env python3
"""
Example demonstrating the improved CompressedListStorage with memmap functionality.

This example shows how to use the new checkpointing capabilities that leverage
memory-mapped storage for efficient disk I/O and memory usage.
"""

import tempfile
from pathlib import Path

import torch
from tensordict import TensorDict

from torchrl.data import CompressedListStorage, ReplayBuffer


def main():
    """Demonstrate compressed storage with memmap checkpointing."""

    # Create a compressed storage with high compression level
    storage = CompressedListStorage(max_size=1000, compression_level=6)

    # Create some sample data with different shapes and types
    print("Creating sample data...")

    # Add tensor data
    tensor_data = torch.randn(10, 3, 84, 84, dtype=torch.float32)  # Image-like data
    storage.set(0, tensor_data)

    # Add TensorDict data with mixed content
    td_data = TensorDict(
        {
            "obs": torch.randn(5, 4, 84, 84, dtype=torch.float32),
            "action": torch.randint(0, 18, (5,), dtype=torch.long),
            "reward": torch.randn(5, dtype=torch.float32),
            "done": torch.zeros(5, dtype=torch.bool),
            "meta": "some metadata string",
        },
        batch_size=[5],
    )
    storage.set(1, td_data)

    # Add another tensor with different shape
    tensor_data2 = torch.randn(8, 64, dtype=torch.float32)
    storage.set(2, tensor_data2)

    print(f"Storage length: {len(storage)}")
    print(f"Storage contains index 0: {storage.contains(0)}")
    print(f"Storage contains index 3: {storage.contains(3)}")

    # Demonstrate data retrieval
    print("\nRetrieving data...")
    retrieved_tensor = storage.get(0)
    retrieved_td = storage.get(1)
    retrieved_tensor2 = storage.get(2)

    print(f"Retrieved tensor shape: {retrieved_tensor.shape}")
    print(f"Retrieved TensorDict keys: {list(retrieved_td.keys())}")
    print(f"Retrieved tensor2 shape: {retrieved_tensor2.shape}")

    # Verify data integrity
    assert torch.allclose(tensor_data, retrieved_tensor, atol=1e-6)
    assert torch.allclose(td_data["obs"], retrieved_td["obs"], atol=1e-6)
    assert torch.allclose(tensor_data2, retrieved_tensor2, atol=1e-6)
    print("✓ Data integrity verified!")

    # Demonstrate memmap checkpointing
    print("\nDemonstrating memmap checkpointing...")

    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_path = Path(tmpdir) / "compressed_storage_checkpoint"

        # Save to disk using memmap
        print(f"Saving to {checkpoint_path}...")
        storage.dumps(checkpoint_path)

        # Check what files were created
        print("Files created:")
        for file_path in checkpoint_path.rglob("*"):
            if file_path.is_file():
                size_mb = file_path.stat().st_size / (1024 * 1024)
                print(f"  {file_path.relative_to(checkpoint_path)}: {size_mb:.2f} MB")

        # Create new storage and load from checkpoint
        print("\nLoading from checkpoint...")
        new_storage = CompressedListStorage(max_size=1000, compression_level=6)
        new_storage.loads(checkpoint_path)

        # Verify data integrity after checkpointing
        print("Verifying data integrity after checkpointing...")
        new_retrieved_tensor = new_storage.get(0)
        new_retrieved_td = new_storage.get(1)
        new_retrieved_tensor2 = new_storage.get(2)

        assert torch.allclose(tensor_data, new_retrieved_tensor, atol=1e-6)
        assert torch.allclose(td_data["obs"], new_retrieved_td["obs"], atol=1e-6)
        assert torch.allclose(tensor_data2, new_retrieved_tensor2, atol=1e-6)
        print("✓ Data integrity after checkpointing verified!")

        print(f"New storage length: {len(new_storage)}")

    # Demonstrate with ReplayBuffer
    print("\nDemonstrating with ReplayBuffer...")

    rb = ReplayBuffer(storage=CompressedListStorage(max_size=100, compression_level=4))

    # Add some data to the replay buffer
    for _ in range(5):
        data = TensorDict(
            {
                "obs": torch.randn(3, 84, 84, dtype=torch.float32),
                "action": torch.randint(0, 18, (3,), dtype=torch.long),
                "reward": torch.randn(3, dtype=torch.float32),
            },
            batch_size=[3],
        )
        rb.extend(data)

    print(f"ReplayBuffer size: {len(rb)}")

    # Sample from the buffer
    sample = rb.sample(2)
    print(f"Sampled data keys: {list(sample.keys())}")
    print(f"Sampled obs shape: {sample['obs'].shape}")

    # Checkpoint the replay buffer
    with tempfile.TemporaryDirectory() as tmpdir:
        rb_checkpoint_path = Path(tmpdir) / "rb_checkpoint"
        print(f"\nCheckpointing ReplayBuffer to {rb_checkpoint_path}...")
        rb.dumps(rb_checkpoint_path)

        # Create new replay buffer and load
        new_rb = ReplayBuffer(
            storage=CompressedListStorage(max_size=100, compression_level=4)
        )
        new_rb.loads(rb_checkpoint_path)

        print(f"New ReplayBuffer size: {len(new_rb)}")

        # Verify sampling works
        new_sample = new_rb.sample(2)
        assert new_sample["obs"].shape == sample["obs"].shape
        print("✓ ReplayBuffer checkpointing verified!")

    print("\n🎉 All demonstrations completed successfully!")
    print("\nKey benefits of the new memmap implementation:")
    print("1. Efficient disk I/O using memory-mapped storage")
    print("2. Reduced memory usage for large datasets")
    print("3. Fast loading and saving of compressed data")
    print("4. Support for heterogeneous data structures")
    print("5. Seamless integration with ReplayBuffer")


if __name__ == "__main__":
    main()
