# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import argparse
import functools

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


def iterate(rb):
    next(rb)


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
