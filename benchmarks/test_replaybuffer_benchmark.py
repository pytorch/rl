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
    TensorDictPrioritizedReplayBuffer,
    TensorDictReplayBuffer,
)
from torchrl.data.replay_buffers import RandomSampler, SamplerWithoutReplacement


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
            functools.partial(TensorDictPrioritizedReplayBuffer, alpha=1.0, beta=0.9),
            ListStorage,
            None,
            4000,
        ],
        [
            functools.partial(TensorDictPrioritizedReplayBuffer, alpha=1.0, beta=0.9),
            LazyMemmapStorage,
            None,
            10_000,
        ],
        [
            functools.partial(TensorDictPrioritizedReplayBuffer, alpha=1.0, beta=0.9),
            LazyTensorStorage,
            None,
            10_000,
        ],
    ],
)
def test_sample_rb(benchmark, rb, storage, sampler, size):
    (rb,), _ = create_rb(
        rb=rb,
        storage=storage,
        sampler=sampler,
        populated=True,
        size=size,
    )()
    benchmark(sample, rb)


def infinite_iter(obj):
    while True:
        obj = iter(obj)
        yield from obj


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
            functools.partial(TensorDictPrioritizedReplayBuffer, alpha=1.0, beta=0.9),
            ListStorage,
            None,
            4000,
        ],
        [
            functools.partial(TensorDictPrioritizedReplayBuffer, alpha=1.0, beta=0.9),
            LazyMemmapStorage,
            None,
            10_000,
        ],
        [
            functools.partial(TensorDictPrioritizedReplayBuffer, alpha=1.0, beta=0.9),
            LazyTensorStorage,
            None,
            10_000,
        ],
    ],
)
def test_iterate_rb(benchmark, rb, storage, sampler, size):
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
        [
            functools.partial(TensorDictPrioritizedReplayBuffer, alpha=1.0, beta=0.9),
            ListStorage,
            None,
            400,
        ],
        [
            functools.partial(TensorDictPrioritizedReplayBuffer, alpha=1.0, beta=0.9),
            LazyMemmapStorage,
            None,
            400,
        ],
        [
            functools.partial(TensorDictPrioritizedReplayBuffer, alpha=1.0, beta=0.9),
            LazyTensorStorage,
            None,
            400,
        ],
    ],
)
def test_populate_rb(benchmark, rb, storage, sampler, size):
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


if __name__ == "__main__":
    args, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst"] + unknown)
