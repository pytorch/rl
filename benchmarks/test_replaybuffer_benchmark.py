# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
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
            print("returning")
            return ((rb,), {})
        else:
            print("returning")
            return ((rb, data), {})


def populate(rb, td):
    for _ in range(1000):
        rb.extend(td)


def sample(rb):
    for _ in range(1000):
        rb.sample()


def iterate(rb):
    for i, _ in enumerate(rb):
        if i == 5:
            break


@pytest.mark.parametrize(
    "rb,storage,sampler,size",
    [
        [TensorDictReplayBuffer, ListStorage, RandomSampler, 4000],
        [TensorDictReplayBuffer, LazyMemmapStorage, RandomSampler, 10_000],
        [TensorDictReplayBuffer, LazyTensorStorage, RandomSampler, 10_000],
        [TensorDictReplayBuffer, ListStorage, SamplerWithoutReplacement, 4000],
        [TensorDictReplayBuffer, LazyMemmapStorage, SamplerWithoutReplacement, 10_000],
        [TensorDictReplayBuffer, LazyTensorStorage, SamplerWithoutReplacement, 10_000],
        [TensorDictPrioritizedReplayBuffer, ListStorage, None, 4000],
        [TensorDictPrioritizedReplayBuffer, LazyMemmapStorage, None, 10_000],
        [TensorDictPrioritizedReplayBuffer, LazyTensorStorage, None, 10_000],
    ],
)
def test_sample_rb(benchmark, rb, storage, sampler, size):
    benchmark.pedantic(
        sample,
        setup=create_rb(
            rb=TensorDictReplayBuffer,
            storage=ListStorage,
            sampler=RandomSampler,
            populated=True,
            size=size,
        ),
        iterations=1,
        rounds=5,
    )


@pytest.mark.parametrize(
    "rb,storage,sampler,size",
    [
        [TensorDictReplayBuffer, ListStorage, RandomSampler, 4000],
        [TensorDictReplayBuffer, LazyMemmapStorage, RandomSampler, 10_000],
        [TensorDictReplayBuffer, LazyTensorStorage, RandomSampler, 10_000],
        [TensorDictReplayBuffer, ListStorage, SamplerWithoutReplacement, 4000],
        [TensorDictReplayBuffer, LazyMemmapStorage, SamplerWithoutReplacement, 10_000],
        [TensorDictReplayBuffer, LazyTensorStorage, SamplerWithoutReplacement, 10_000],
        [TensorDictPrioritizedReplayBuffer, ListStorage, None, 4000],
        [TensorDictPrioritizedReplayBuffer, LazyMemmapStorage, None, 10_000],
        [TensorDictPrioritizedReplayBuffer, LazyTensorStorage, None, 10_000],
    ],
)
def test_iterate_rb(benchmark, rb, storage, sampler, size):
    benchmark.pedantic(
        iterate,
        setup=create_rb(
            rb=TensorDictReplayBuffer,
            storage=ListStorage,
            sampler=RandomSampler,
            populated=True,
            size=size,
        ),
        iterations=1,
        rounds=5,
    )


@pytest.mark.parametrize(
    "rb,storage,sampler,size",
    [
        [TensorDictReplayBuffer, ListStorage, RandomSampler, 400],
        [TensorDictReplayBuffer, LazyMemmapStorage, RandomSampler, 400],
        [TensorDictReplayBuffer, LazyTensorStorage, RandomSampler, 400],
        [TensorDictReplayBuffer, ListStorage, SamplerWithoutReplacement, 400],
        [TensorDictReplayBuffer, LazyMemmapStorage, SamplerWithoutReplacement, 400],
        [TensorDictReplayBuffer, LazyTensorStorage, SamplerWithoutReplacement, 400],
        [TensorDictPrioritizedReplayBuffer, ListStorage, None, 400],
        [TensorDictPrioritizedReplayBuffer, LazyMemmapStorage, None, 400],
        [TensorDictPrioritizedReplayBuffer, LazyTensorStorage, None, 400],
    ],
)
def test_populate_rb(benchmark, rb, storage, sampler, size):
    benchmark.pedantic(
        populate,
        setup=create_rb(
            rb=TensorDictReplayBuffer,
            storage=ListStorage,
            sampler=RandomSampler,
            populated=False,
            size=size,
        ),
        iterations=1,
        rounds=5,
    )
