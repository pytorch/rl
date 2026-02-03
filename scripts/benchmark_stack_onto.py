#!/usr/bin/env python3
"""Benchmark script for stack_onto_ optimization in TensorStorage.

Compares the performance of:
1. Old path: _flip_list() creates intermediate tensor, then copy to storage
2. New path: _stack_into_storage() stacks directly into storage slice
"""

import time

import torch
from tensordict import TensorDict
from torchrl.data.replay_buffers.storages import TensorStorage


def benchmark_tensor_storage(num_items: int, item_size: int, num_iterations: int = 100):
    """Benchmark writing lists of tensors to storage."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n{'='*60}")
    print(f"Benchmarking TensorStorage with {num_items} items of size {item_size}")
    print(f"Device: {device}, Iterations: {num_iterations}")
    print(f"{'='*60}")

    # Create storage
    storage_size = num_items * 10  # Larger than what we'll write
    init_data = torch.randn(storage_size, item_size, device=device)
    storage = TensorStorage(init_data)

    # Create list of items to write
    items = [torch.randn(item_size, device=device) for _ in range(num_items)]

    # Warmup
    for _ in range(5):
        storage.set(slice(0, num_items), items)

    # Benchmark with slice (should use direct stacking)
    torch.cuda.synchronize() if device == "cuda" else None
    start = time.perf_counter()
    for _ in range(num_iterations):
        storage.set(slice(0, num_items), items)
        torch.cuda.synchronize() if device == "cuda" else None
    slice_time = (time.perf_counter() - start) / num_iterations * 1000

    # Benchmark with tensor indices (should also use direct stacking for contiguous)
    indices = torch.arange(0, num_items, device=device)
    torch.cuda.synchronize() if device == "cuda" else None
    start = time.perf_counter()
    for _ in range(num_iterations):
        storage.set(indices, items)
        torch.cuda.synchronize() if device == "cuda" else None
    tensor_idx_time = (time.perf_counter() - start) / num_iterations * 1000

    # Benchmark with non-contiguous indices (falls back to _flip_list)
    non_contiguous = torch.arange(0, num_items * 2, 2, device=device)[:num_items]
    torch.cuda.synchronize() if device == "cuda" else None
    start = time.perf_counter()
    for _ in range(num_iterations):
        storage.set(non_contiguous, items)
        torch.cuda.synchronize() if device == "cuda" else None
    non_contiguous_time = (time.perf_counter() - start) / num_iterations * 1000

    print(f"  Slice indexing (direct stack):      {slice_time:.3f} ms")
    print(f"  Tensor contiguous (direct stack):   {tensor_idx_time:.3f} ms")
    print(f"  Non-contiguous (fallback):          {non_contiguous_time:.3f} ms")

    if non_contiguous_time > 0:
        speedup_slice = non_contiguous_time / slice_time
        speedup_tensor = non_contiguous_time / tensor_idx_time
        print(f"  Speedup (slice vs fallback):        {speedup_slice:.2f}x")
        print(f"  Speedup (tensor vs fallback):       {speedup_tensor:.2f}x")

    return slice_time, tensor_idx_time, non_contiguous_time


def benchmark_tensordict_storage(
    num_items: int, item_size: int, num_iterations: int = 100
):
    """Benchmark writing lists of TensorDicts to storage."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n{'='*60}")
    print(f"Benchmarking TensorDict Storage with {num_items} items")
    print(f"Device: {device}, Iterations: {num_iterations}")
    print(f"{'='*60}")

    # Create storage
    storage_size = num_items * 10
    init_data = TensorDict(
        {
            "obs": torch.randn(storage_size, item_size, device=device),
            "action": torch.randn(storage_size, 4, device=device),
            "reward": torch.randn(storage_size, 1, device=device),
        },
        batch_size=[storage_size],
    )
    storage = TensorStorage(init_data)

    # Create list of TensorDict items
    items = [
        TensorDict(
            {
                "obs": torch.randn(item_size, device=device),
                "action": torch.randn(4, device=device),
                "reward": torch.randn(1, device=device),
            },
            batch_size=[],
        )
        for _ in range(num_items)
    ]

    # Warmup
    for _ in range(5):
        storage.set(slice(0, num_items), items)

    # Benchmark with slice
    torch.cuda.synchronize() if device == "cuda" else None
    start = time.perf_counter()
    for _ in range(num_iterations):
        storage.set(slice(0, num_items), items)
        torch.cuda.synchronize() if device == "cuda" else None
    slice_time = (time.perf_counter() - start) / num_iterations * 1000

    # Benchmark with non-contiguous indices (fallback)
    non_contiguous = torch.arange(0, num_items * 2, 2, device=device)[:num_items]
    torch.cuda.synchronize() if device == "cuda" else None
    start = time.perf_counter()
    for _ in range(num_iterations):
        storage.set(non_contiguous, items)
        torch.cuda.synchronize() if device == "cuda" else None
    non_contiguous_time = (time.perf_counter() - start) / num_iterations * 1000

    print(f"  Slice indexing (direct stack):      {slice_time:.3f} ms")
    print(f"  Non-contiguous (fallback):          {non_contiguous_time:.3f} ms")

    if non_contiguous_time > 0:
        speedup = non_contiguous_time / slice_time
        print(f"  Speedup:                            {speedup:.2f}x")

    return slice_time, non_contiguous_time


if __name__ == "__main__":
    print("Stack_onto_ Optimization Benchmark")
    print("=" * 60)

    # Test various sizes
    configs = [
        (8, 64),  # Small: 8 items, 64 features (typical rollout)
        (16, 64),  # Medium: 16 items
        (32, 128),  # Larger rollout
        (64, 256),  # Large batch
        (128, 512),  # Very large
    ]

    print("\n" + "=" * 60)
    print("TENSOR STORAGE BENCHMARKS")
    print("=" * 60)

    for num_items, item_size in configs:
        benchmark_tensor_storage(num_items, item_size)

    print("\n" + "=" * 60)
    print("TENSORDICT STORAGE BENCHMARKS")
    print("=" * 60)

    for num_items, item_size in configs[:3]:  # Fewer configs for TensorDict
        benchmark_tensordict_storage(num_items, item_size)

    print("\n" + "=" * 60)
    print("BENCHMARK COMPLETE")
    print("=" * 60)
