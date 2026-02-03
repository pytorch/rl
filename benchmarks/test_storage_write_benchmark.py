# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Benchmarks for TensorStorage write operations.

These benchmarks measure the performance of writing data to replay buffer storage,
particularly comparing lazy stacked tensordicts vs contiguous tensordicts.

The lazy stack path is used when collectors write directly to a replay buffer,
avoiding an intermediate contiguous buffer allocation.
"""
import pytest
import torch

from tensordict import LazyStackedTensorDict, TensorDict
from torchrl.collectors import SyncDataCollector
from torchrl.data import LazyTensorStorage, ReplayBuffer
from torchrl.modules import RandomPolicy
from torchrl.testing import FastImageEnv


# Test configurations: (n_items, img_shape, description)
CONFIGS = [
    (50, (3, 64, 64), "small"),
    (100, (4, 84, 84), "atari"),
    (100, (4, 128, 128), "large_img"),
    (200, (4, 84, 84), "large_batch"),
]


def _create_lazy_stack(n_items, img_shape):
    """Create a lazy stacked tensordict (simulates collector output without RB)."""
    tds = [
        TensorDict(
            {
                "pixels": torch.rand(img_shape),
                "action": torch.rand(4),
                "reward": torch.rand(1),
            },
            batch_size=[],
        )
        for _ in range(n_items)
    ]
    return LazyStackedTensorDict.lazy_stack(tds, dim=0)


def _create_contiguous(n_items, img_shape):
    """Create a contiguous tensordict (simulates stacked collector output)."""
    return TensorDict(
        {
            "pixels": torch.rand(n_items, *img_shape),
            "action": torch.rand(n_items, 4),
            "reward": torch.rand(n_items, 1),
        },
        batch_size=[n_items],
    )


def _create_initialized_storage(n_items, img_shape):
    """Create and initialize a storage with the right shape."""
    storage = LazyTensorStorage(n_items * 2)
    init_data = TensorDict(
        {
            "pixels": torch.zeros(n_items * 2, *img_shape),
            "action": torch.zeros(n_items * 2, 4),
            "reward": torch.zeros(n_items * 2, 1),
        },
        batch_size=[n_items * 2],
    )
    storage.set(slice(0, n_items * 2), init_data)
    return storage


class TestStorageWriteBenchmark:
    """Benchmarks for storage write operations."""

    @pytest.mark.parametrize("n_items,img_shape,desc", CONFIGS)
    def test_storage_write_lazystack(self, benchmark, n_items, img_shape, desc):
        """Benchmark writing a lazy stacked tensordict to storage.

        This measures the performance of the lazy stack write path, which is used
        when collectors have a replay buffer attached and skip the intermediate
        contiguous buffer.
        """
        storage = _create_initialized_storage(n_items, img_shape)
        cursor = slice(0, n_items)

        # Pre-create data for consistent benchmarking
        lazy_data = _create_lazy_stack(n_items, img_shape)

        def write_lazy():
            storage.set(cursor, lazy_data)

        benchmark(write_lazy)

    @pytest.mark.parametrize("n_items,img_shape,desc", CONFIGS)
    def test_storage_write_contiguous(self, benchmark, n_items, img_shape, desc):
        """Benchmark writing a contiguous tensordict to storage.

        This measures the baseline performance of writing pre-stacked contiguous
        data to storage.
        """
        storage = _create_initialized_storage(n_items, img_shape)
        cursor = slice(0, n_items)

        # Pre-create data for consistent benchmarking
        contiguous_data = _create_contiguous(n_items, img_shape)

        def write_contiguous():
            storage.set(cursor, contiguous_data)

        benchmark(write_contiguous)

    @pytest.mark.parametrize("n_items,img_shape,desc", CONFIGS)
    def test_collector_stack_then_write(self, benchmark, n_items, img_shape, desc):
        """Benchmark the full old path: stack individual tds then write.

        This simulates what happens when a collector doesn't have a replay buffer:
        1. Collector stacks individual tensordicts into contiguous buffer
        2. User extends the replay buffer with the contiguous data

        This is the baseline we want to improve upon.
        """
        storage = _create_initialized_storage(n_items, img_shape)
        cursor = slice(0, n_items)

        # Pre-create individual tensordicts (simulating collector's per-step output)
        tds = [
            TensorDict(
                {
                    "pixels": torch.rand(img_shape),
                    "action": torch.rand(4),
                    "reward": torch.rand(1),
                },
                batch_size=[],
            )
            for _ in range(n_items)
        ]

        def stack_then_write():
            # Step 1: Stack to contiguous (this allocates new memory)
            stacked = torch.stack(tds, dim=0)
            # Step 2: Write to storage
            storage.set(cursor, stacked)

        benchmark(stack_then_write)

    @pytest.mark.parametrize("n_items,img_shape,desc", CONFIGS)
    def test_collector_lazystack_then_write(self, benchmark, n_items, img_shape, desc):
        """Benchmark the optimized path: lazy stack then write.

        This simulates what happens when a collector has a replay buffer:
        1. Collector creates lazy stack (no memory allocation)
        2. Storage writes directly from lazy stack components

        This should be faster than test_collector_stack_then_write.
        """
        storage = _create_initialized_storage(n_items, img_shape)
        cursor = slice(0, n_items)

        # Pre-create individual tensordicts (simulating collector's per-step output)
        tds = [
            TensorDict(
                {
                    "pixels": torch.rand(img_shape),
                    "action": torch.rand(4),
                    "reward": torch.rand(1),
                },
                batch_size=[],
            )
            for _ in range(n_items)
        ]

        def lazystack_then_write():
            # Step 1: Create lazy stack (no allocation, just wrapping)
            lazy = LazyStackedTensorDict.lazy_stack(tds, dim=0)
            # Step 2: Write to storage (storage handles the lazy stack)
            storage.set(cursor, lazy)

        benchmark(lazystack_then_write)


# Collector benchmark configurations: (frames_per_batch, img_shape, description)
COLLECTOR_CONFIGS = [
    (100, (4, 84, 84), "atari"),
    (200, (4, 84, 84), "large_batch"),
]


class TestCollectorIntegrationBenchmark:
    """Benchmarks for collector + replay buffer integration."""

    @pytest.mark.parametrize("frames_per_batch,img_shape,desc", COLLECTOR_CONFIGS)
    def test_collector_without_rb(self, benchmark, frames_per_batch, img_shape, desc):
        """Benchmark collector without replay buffer (baseline).

        The collector stacks frames into a contiguous buffer internally.
        User then manually extends the replay buffer.
        """
        env = FastImageEnv(img_shape=img_shape)
        rb = ReplayBuffer(storage=LazyTensorStorage(frames_per_batch * 20))

        collector = SyncDataCollector(
            env,
            RandomPolicy(env.action_spec),
            frames_per_batch=frames_per_batch,
            total_frames=-1,
        )
        collector_iter = iter(collector)

        # Warmup - initialize storage
        data = next(collector_iter)
        rb.extend(data)

        def collect_and_extend():
            data = next(collector_iter)
            rb.extend(data)

        benchmark(collect_and_extend)
        collector.shutdown()

    @pytest.mark.parametrize("frames_per_batch,img_shape,desc", COLLECTOR_CONFIGS)
    def test_collector_with_rb(self, benchmark, frames_per_batch, img_shape, desc):
        """Benchmark collector with replay buffer attached (optimized path).

        The collector uses lazy stacks and the storage writes directly
        without intermediate contiguous allocation.
        """
        env = FastImageEnv(img_shape=img_shape)
        rb = ReplayBuffer(storage=LazyTensorStorage(frames_per_batch * 20))

        collector = SyncDataCollector(
            env,
            RandomPolicy(env.action_spec),
            frames_per_batch=frames_per_batch,
            total_frames=-1,
            replay_buffer=rb,
        )
        collector_iter = iter(collector)

        # Warmup - initialize storage
        next(collector_iter)

        def collect_with_rb():
            next(collector_iter)

        benchmark(collect_with_rb)
        collector.shutdown()

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA not available for GPU benchmark"
    )
    @pytest.mark.parametrize("frames_per_batch,img_shape,desc", COLLECTOR_CONFIGS)
    def test_collector_without_rb_cuda(
        self, benchmark, frames_per_batch, img_shape, desc
    ):
        """Benchmark collector without replay buffer on CUDA (baseline)."""
        device = "cuda:0"
        env = FastImageEnv(img_shape=img_shape, device=device)
        rb = ReplayBuffer(
            storage=LazyTensorStorage(frames_per_batch * 20, device=device)
        )

        collector = SyncDataCollector(
            env,
            RandomPolicy(env.action_spec),
            frames_per_batch=frames_per_batch,
            total_frames=-1,
            device=device,
        )
        collector_iter = iter(collector)

        # Warmup - initialize storage
        data = next(collector_iter)
        rb.extend(data)
        torch.cuda.synchronize()

        def collect_and_extend():
            data = next(collector_iter)
            rb.extend(data)
            torch.cuda.synchronize()

        benchmark(collect_and_extend)
        collector.shutdown()

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA not available for GPU benchmark"
    )
    @pytest.mark.parametrize("frames_per_batch,img_shape,desc", COLLECTOR_CONFIGS)
    def test_collector_with_rb_cuda(self, benchmark, frames_per_batch, img_shape, desc):
        """Benchmark collector with replay buffer attached on CUDA (optimized path)."""
        device = "cuda:0"
        env = FastImageEnv(img_shape=img_shape, device=device)
        rb = ReplayBuffer(
            storage=LazyTensorStorage(frames_per_batch * 20, device=device)
        )

        collector = SyncDataCollector(
            env,
            RandomPolicy(env.action_spec),
            frames_per_batch=frames_per_batch,
            total_frames=-1,
            replay_buffer=rb,
            device=device,
        )
        collector_iter = iter(collector)

        # Warmup - initialize storage
        next(collector_iter)
        torch.cuda.synchronize()

        def collect_with_rb():
            next(collector_iter)
            torch.cuda.synchronize()

        benchmark(collect_with_rb)
        collector.shutdown()


if __name__ == "__main__":
    import argparse

    args, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main([__file__, "--capture", "no", "-v"] + unknown)
