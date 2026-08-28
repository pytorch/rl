"""Benchmarks for StoreStorage vs LazyTensorStorage.

Run with:
    pytest test/test_benchmark_store_storage.py --benchmark-only -v
    pytest test/test_benchmark_store_storage.py --benchmark-only --benchmark-group-by=param:storage_type
"""

import importlib

import pytest
import torch
from tensordict import TensorDict

from torchrl.data import ReplayBuffer
from torchrl.data.replay_buffers import LazyTensorStorage, StoreStorage
from torchrl.data.replay_buffers.samplers import RandomSampler
from torchrl.data.replay_buffers.writers import RoundRobinWriter

_has_redis = importlib.util.find_spec("redis") is not None
pytestmark = pytest.mark.skipif(not _has_redis, reason="redis not available")

BUFFER_SIZE = 10_000
OBS_DIM = 64
ACTION_DIM = 4
BATCH_SIZE = 256
FILL_CHUNK = 500


def _flush_redis():
    import redis

    r = redis.Redis()
    r.flushdb()
    r.close()


def _make_batch(n: int) -> TensorDict:
    return TensorDict(
        {
            "obs": torch.randn(n, OBS_DIM),
            "action": torch.randn(n, ACTION_DIM),
            "reward": torch.randn(n, 1),
        },
        batch_size=[n],
    )


def _make_rb(storage_type: str, prefetch: int = 0) -> ReplayBuffer:
    kwargs = {}
    if prefetch:
        kwargs["prefetch"] = prefetch
    if storage_type == "lazy_tensor":
        storage = LazyTensorStorage(max_size=BUFFER_SIZE)
    else:
        storage = StoreStorage(max_size=BUFFER_SIZE)
    return ReplayBuffer(
        storage=storage,
        sampler=RandomSampler(),
        writer=RoundRobinWriter(),
        batch_size=BATCH_SIZE,
        **kwargs,
    )


def _prefill(rb: ReplayBuffer, n: int = 5000):
    remaining = min(n, BUFFER_SIZE)
    while remaining > 0:
        chunk = min(FILL_CHUNK, remaining)
        rb.extend(_make_batch(chunk))
        remaining -= chunk


@pytest.fixture(autouse=True)
def flush_redis():
    _flush_redis()
    yield
    _flush_redis()


@pytest.mark.parametrize("storage_type", ["lazy_tensor", "store"])
def test_fill(benchmark, storage_type):
    rb = _make_rb(storage_type)

    def _fill():
        _prefill(rb, n=5000)

    benchmark.pedantic(_fill, rounds=3, warmup_rounds=1)


@pytest.mark.parametrize("storage_type", ["lazy_tensor", "store"])
def test_sample(benchmark, storage_type):
    rb = _make_rb(storage_type)
    _prefill(rb)

    benchmark(rb.sample)


@pytest.mark.parametrize("storage_type", ["lazy_tensor", "store"])
def test_sample_prefetch(benchmark, storage_type):
    rb = _make_rb(storage_type, prefetch=4)
    _prefill(rb)

    benchmark(rb.sample)
