# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import argparse
import functools
import os

import pytest
import torch
from tensordict import TensorDict

from torchrl.data import (
    LazyMemmapStorage,
    LazyTensorStorage,
    ListStorage,
    ReplayBuffer,
    TensorDictPrioritizedReplayBuffer,
    TensorDictReplayBuffer,
)
from torchrl.data.replay_buffers import (
    PrioritizedSampler,
    RandomSampler,
    SamplerWithoutReplacement,
    SliceSampler,
)

_TensorDictPrioritizedReplayBuffer = functools.partial(
    TensorDictPrioritizedReplayBuffer, alpha=1, beta=0.9
)
# preserve the name of the class even after partial
_TensorDictPrioritizedReplayBuffer.__name__ = TensorDictPrioritizedReplayBuffer.__name__


class create_rb:
    def __init__(self, rb, storage, sampler, populated, size=1_000_000):
        self.storage = storage
        self.rb = rb
        self.sampler = sampler
        self.populated = populated
        self.size = size

    def __call__(self):

        kwargs = {"batch_size": 256}
        if self.sampler is not None:
            kwargs["sampler"] = self.sampler()
        if self.storage is not None:
            kwargs["storage"] = self.storage(self.size)

        rb = self.rb(**kwargs)
        data = TensorDict(
            {
                "a": torch.zeros(self.size, 5),
                ("b", "c"): torch.zeros(self.size, 3, 32, 32, dtype=torch.uint8),
            },
            batch_size=[self.size],
        )
        if "sampler" in kwargs and isinstance(kwargs["sampler"], SliceSampler):
            data["traj"] = torch.arange(self.size) // 123
        if self.populated:
            rb.extend(data)
            return ((rb,), {})
        else:
            return ((rb, data), {})


def populate(rb, td):
    rb.extend(td)


def sample(rb):
    rb.sample()


def sample_prioritized_sampler(sampler, storage, batch_size):
    sampler.sample(storage, batch_size)
    if sampler.device.type == "cuda":
        torch.cuda.synchronize(sampler.device)


def iterate(rb):
    next(rb)


class StorageView:
    ndim = 1
    shape = None

    def __init__(self, size, device):
        self.size = size
        self.device = torch.device(device)
        self.shape = (size,)

    def __len__(self):
        return self.size


def _skip_or_fail_unavailable(message):
    if os.getenv("TORCHRL_BENCHMARK_DEVICE") in {"CPU", "GPU"}:
        pytest.fail(message)
    pytest.skip(message)


class create_prioritized_sampler:
    def __init__(self, size, device, batch_size, alpha=0.7, beta=0.5):
        self.size = size
        self.device = torch.device(device)
        self.batch_size = batch_size
        self.alpha = alpha
        self.beta = beta

    def __call__(self):
        ext = pytest.importorskip("torchrl._torchrl")
        if not hasattr(ext, "SumSegmentTreeFp32"):
            _skip_or_fail_unavailable("TorchRL was not built with segment tree support")
        if self.device.type == "cuda":
            if not torch.cuda.is_available():
                _skip_or_fail_unavailable("CUDA is not available")
            if not hasattr(ext, "CudaSumSegmentTreeFp32"):
                _skip_or_fail_unavailable(
                    "TorchRL was not built with CUDA segment tree support"
                )
        storage = StorageView(self.size, self.device)
        sampler = PrioritizedSampler(
            max_capacity=self.size,
            alpha=self.alpha,
            beta=self.beta,
            device=self.device,
        )
        index = torch.arange(self.size, device=self.device)
        priority = torch.linspace(0.1, 1.0, self.size, device=self.device)
        sampler.update_priority(index, priority, storage=storage)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        return ((sampler, storage, self.batch_size), {})


def _prioritized_sampler_benchmark_devices():
    device = os.getenv("TORCHRL_BENCHMARK_DEVICE")
    if device == "CPU":
        return ["cpu"]
    if device == "GPU":
        return ["cuda"]
    return ["cpu", "cuda"]


@pytest.mark.parametrize(
    "rb,storage,sampler,size",
    [
        [TensorDictReplayBuffer, ListStorage, RandomSampler, 4000],
        [TensorDictReplayBuffer, LazyMemmapStorage, RandomSampler, 10_000],
        [TensorDictReplayBuffer, LazyTensorStorage, RandomSampler, 10_000],
        [TensorDictReplayBuffer, ListStorage, SamplerWithoutReplacement, 4000],
        [TensorDictReplayBuffer, LazyMemmapStorage, SamplerWithoutReplacement, 10_000],
        [TensorDictReplayBuffer, LazyTensorStorage, SamplerWithoutReplacement, 10_000],
        [
            TensorDictReplayBuffer,
            LazyMemmapStorage,
            functools.partial(SliceSampler, num_slices=8, traj_key="traj"),
            10_000,
        ],
        [
            TensorDictReplayBuffer,
            LazyTensorStorage,
            functools.partial(SliceSampler, num_slices=8, traj_key="traj"),
            10_000,
        ],
        [_TensorDictPrioritizedReplayBuffer, ListStorage, None, 4000],
        [_TensorDictPrioritizedReplayBuffer, LazyMemmapStorage, None, 10_000],
        [_TensorDictPrioritizedReplayBuffer, LazyTensorStorage, None, 10_000],
    ],
)
def test_rb_sample(benchmark, rb, storage, sampler, size):
    (rb,), _ = create_rb(
        rb=rb,
        storage=storage,
        sampler=sampler,
        populated=True,
        size=size,
    )()
    torch.manual_seed(0)
    benchmark(sample, rb)


@pytest.mark.parametrize("device", _prioritized_sampler_benchmark_devices())
@pytest.mark.parametrize("size", [1_000_000, 10_000_000])
def test_prioritized_sampler_sample_scale(benchmark, size, device):
    batch_size = 65_536
    (sampler, storage, batch_size), _ = create_prioritized_sampler(
        size=size, device=device, batch_size=batch_size
    )()
    benchmark(
        sample_prioritized_sampler,
        sampler,
        storage,
        batch_size,
    )


def infinite_iter(obj):
    torch.manual_seed(0)
    while True:
        yield from iter(obj)


@pytest.mark.parametrize(
    "rb,storage,sampler,size",
    [
        [TensorDictReplayBuffer, ListStorage, RandomSampler, 4000],
        [TensorDictReplayBuffer, LazyMemmapStorage, RandomSampler, 10_000],
        [TensorDictReplayBuffer, LazyTensorStorage, RandomSampler, 10_000],
        [TensorDictReplayBuffer, ListStorage, SamplerWithoutReplacement, 4000],
        [TensorDictReplayBuffer, LazyMemmapStorage, SamplerWithoutReplacement, 10_000],
        [TensorDictReplayBuffer, LazyTensorStorage, SamplerWithoutReplacement, 10_000],
        [_TensorDictPrioritizedReplayBuffer, ListStorage, None, 4000],
        [_TensorDictPrioritizedReplayBuffer, LazyMemmapStorage, None, 10_000],
        [_TensorDictPrioritizedReplayBuffer, LazyTensorStorage, None, 10_000],
    ],
)
def test_rb_iterate(benchmark, rb, storage, sampler, size):
    (rb,), _ = create_rb(
        rb=rb,
        storage=storage,
        sampler=sampler,
        populated=True,
        size=size,
    )()
    benchmark(iterate, infinite_iter(rb))


@pytest.mark.parametrize(
    "rb,storage,sampler,size",
    [
        [TensorDictReplayBuffer, ListStorage, RandomSampler, 400],
        [TensorDictReplayBuffer, LazyMemmapStorage, RandomSampler, 400],
        [TensorDictReplayBuffer, LazyTensorStorage, RandomSampler, 400],
        [TensorDictReplayBuffer, ListStorage, SamplerWithoutReplacement, 400],
        [TensorDictReplayBuffer, LazyMemmapStorage, SamplerWithoutReplacement, 400],
        [TensorDictReplayBuffer, LazyTensorStorage, SamplerWithoutReplacement, 400],
        [_TensorDictPrioritizedReplayBuffer, ListStorage, None, 400],
        [_TensorDictPrioritizedReplayBuffer, LazyMemmapStorage, None, 400],
        [_TensorDictPrioritizedReplayBuffer, LazyTensorStorage, None, 400],
    ],
)
def test_rb_populate(benchmark, rb, storage, sampler, size):
    benchmark.pedantic(
        populate,
        setup=create_rb(
            rb=rb,
            storage=storage,
            sampler=sampler,
            populated=False,
            size=size,
        ),
        iterations=1,
        rounds=50,
    )


class create_compiled_tensor_rb:
    def __init__(
        self, rb, storage, sampler, storage_size, data_size, iters, compilable=False
    ):
        self.storage = storage
        self.rb = rb
        self.sampler = sampler
        self.storage_size = storage_size
        self.data_size = data_size
        self.iters = iters
        self.compilable = compilable

    def __call__(self):
        kwargs = {}
        if self.sampler is not None:
            kwargs["sampler"] = self.sampler()
        if self.storage is not None:
            kwargs["storage"] = self.storage(
                self.storage_size, compilable=self.compilable
            )

        rb = self.rb(batch_size=3, compilable=self.compilable, **kwargs)
        data = torch.randn(self.data_size, 1)
        return ((rb, data, self.iters), {})


def extend_and_sample(rb, td, iters):
    for _ in range(iters):
        rb.extend(td)
        rb.sample()


def extend_and_sample_compiled(rb, td, iters):
    @torch.compile
    def fn(td):
        rb.extend(td)
        rb.sample()

    for _ in range(iters):
        fn(td)


@pytest.mark.parametrize(
    "rb,storage,sampler,storage_size,data_size,iters,compiled",
    [
        [ReplayBuffer, LazyTensorStorage, RandomSampler, 10_000, 10_000, 100, True],
        [ReplayBuffer, LazyTensorStorage, RandomSampler, 10_000, 10_000, 100, False],
        [ReplayBuffer, LazyTensorStorage, RandomSampler, 100_000, 10_000, 100, True],
        [ReplayBuffer, LazyTensorStorage, RandomSampler, 100_000, 10_000, 100, False],
        [ReplayBuffer, LazyTensorStorage, RandomSampler, 1_000_000, 10_000, 100, True],
        [ReplayBuffer, LazyTensorStorage, RandomSampler, 1_000_000, 10_000, 100, False],
    ],
)
def test_rb_extend_sample(
    benchmark, rb, storage, sampler, storage_size, data_size, iters, compiled
):
    if compiled:
        torch._dynamo.reset_code_caches()

    benchmark.pedantic(
        extend_and_sample_compiled if compiled else extend_and_sample,
        setup=create_compiled_tensor_rb(
            rb=rb,
            storage=storage,
            sampler=sampler,
            storage_size=storage_size,
            data_size=data_size,
            iters=iters,
            compilable=compiled,
        ),
        iterations=1,
        warmup_rounds=10,
        rounds=50,
    )


if __name__ == "__main__":
    args, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst"] + unknown)
