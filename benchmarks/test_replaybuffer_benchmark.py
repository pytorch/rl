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
    LazyStackStorage,
    LazyTensorStorage,
    ListStorage,
    ReplayBuffer,
    TensorDictPrioritizedReplayBuffer,
    TensorDictReplayBuffer,
)
from torchrl.data.replay_buffers import (
    GeometricTrajectoryWindowSampler,
    PrioritizedSampler,
    PromptGroupSampler,
    RandomSampler,
    RoundRobinWriter,
    SamplerWithoutReplacement,
    Sequence,
    SliceSampler,
)
from torchrl.data.replay_buffers.utils import _boundary_distances_1d
from torchrl.envs.transforms import ActionChunkTransform, CatFrames
from torchrl.modules.llm import TorchRLBufferDataset

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


def _replay_boundary_device():
    device = os.getenv("TORCHRL_BENCHMARK_DEVICE")
    if device == "GPU":
        if not torch.cuda.is_available():
            _skip_or_fail_unavailable("CUDA is not available")
        return torch.device("cuda")
    return torch.device("cpu")


def _make_boundary_storage(size, episode_length, device):
    done = torch.zeros(size, 1, dtype=torch.bool, device=device)
    done[episode_length - 1 :: episode_length] = True
    done[-1] = True
    rb = TensorDictReplayBuffer(
        storage=LazyTensorStorage(size, device=device),
    )
    rb.extend(TensorDict({("next", "done"): done}, [size], device=device))
    return rb


@pytest.mark.parametrize("size", [10_000, 1_000_000])
@pytest.mark.parametrize("episode_length", [32, 256])
@pytest.mark.parametrize("cache_state", ["hot", "write_invalidated"])
def test_sequence_boundary_query_benchmark(
    benchmark, size, episode_length, cache_state
):
    device = _replay_boundary_device()
    rb = _make_boundary_storage(size, episode_length, device)
    anchor = torch.linspace(0, size - 1, 256, device=device).to(torch.long)
    unit = Sequence(length=64, burn_in=40, bootstrap=5)
    unit.expand(anchor, {}, rb.storage)

    if cache_state == "write_invalidated":

        def query():
            rb.storage._bump_mutation_revision()
            return unit.expand(anchor, {}, rb.storage)

    else:

        def query():
            return unit.expand(anchor, {}, rb.storage)

    benchmark(query)


@pytest.mark.parametrize("lanes", [8, 64])
def test_sequence_multidimensional_boundary_query_benchmark(benchmark, lanes):
    device = _replay_boundary_device()
    time = 100_000
    done = torch.zeros(time, lanes, 1, dtype=torch.bool, device=device)
    done[255::256] = True
    done[-1] = True
    rb = TensorDictReplayBuffer(
        storage=LazyTensorStorage(time * lanes, ndim=2, device=device),
        dim_extend=0,
    )
    rb.extend(TensorDict({("next", "done"): done}, [time, lanes], device=device))
    anchors = (
        torch.linspace(0, time - 1, 256, device=device).to(torch.long),
        torch.arange(256, device=device) % lanes,
    )
    unit = Sequence(length=64, burn_in=40, bootstrap=5)
    unit.expand(anchors, {}, rb.storage)

    benchmark(unit.expand, anchors, {}, rb.storage)


@pytest.mark.parametrize("size", [10_000, 1_000_000])
@pytest.mark.parametrize("episode_length", [32, 256])
@pytest.mark.parametrize("cache_values", [False, True])
def test_slice_sampler_boundary_query_benchmark(
    benchmark, size, episode_length, cache_values
):
    device = _replay_boundary_device()
    rb = _make_boundary_storage(size, episode_length, device)
    sampler = SliceSampler(
        num_slices=256,
        cache_values=cache_values,
        end_key=("next", "done"),
    )
    benchmark(sampler._get_stop_and_length, rb.storage)


@pytest.mark.parametrize("compiled", [False, True])
def test_replay_boundary_kernel_benchmark(benchmark, compiled):
    device = _replay_boundary_device()
    size = 1_000_000
    episode_length = 256
    stop = torch.arange(
        episode_length - 1, size, episode_length, device=device, dtype=torch.long
    )
    if stop[-1] != size - 1:
        stop = torch.cat([stop, stop.new_tensor([size - 1])])
    start = torch.remainder(torch.roll(stop, 1) + 1, size)
    anchor = torch.linspace(0, size - 1, 256, device=device).to(torch.long)
    query = _boundary_distances_1d
    if compiled:
        query = torch.compile(query, fullgraph=True)
        query(anchor, start, stop, size)
    benchmark(query, anchor, start, stop, size)


def test_replay_buffer_direct_client_identity(benchmark):
    replay_buffer = ReplayBuffer(storage=ListStorage(1))
    client = benchmark(replay_buffer.client)
    assert client is replay_buffer


def sample_prioritized_sampler(sampler, storage, batch_size):
    sampler.sample(storage, batch_size)
    if sampler.device.type == "cuda":
        torch.cuda.synchronize(sampler.device)


def sample_prioritized_replay_buffer(rb):
    sample = rb.sample()
    sampler_device = rb._sampler.device
    if sampler_device.type == "cuda":
        torch.cuda.synchronize(sampler_device)
    elif sample.device is not None and sample.device.type == "cuda":
        torch.cuda.synchronize(sample.device)


def iterate(rb):
    next(rb)


def consume_buffer_dataset(dataset):
    return list(dataset)


class StorageView:
    ndim = 1
    shape = None

    def __init__(self, size, device):
        self.size = size
        self.device = torch.device(device)
        self.shape = (size,)

    def __len__(self):
        return self.size


def test_torchrl_buffer_dataset(benchmark):
    replay_buffer = ReplayBuffer(storage=LazyTensorStorage(1024))
    replay_buffer.extend(
        TensorDict(
            {
                "input_ids": torch.randint(0, 1024, (1024, 128)),
                ("metadata", "score"): torch.randn(1024),
            },
            batch_size=[1024],
        )
    )
    dataset = TorchRLBufferDataset(replay_buffer, batch_size=256)

    samples = benchmark(consume_buffer_dataset, dataset)

    assert len(samples) == 256


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


class create_prioritized_replay_buffer:
    def __init__(
        self,
        size,
        storage_type,
        storage_device,
        sampler_device,
        batch_size,
        alpha=0.7,
        beta=0.5,
    ):
        self.size = size
        self.storage_type = storage_type
        self.storage_device = torch.device(storage_device)
        self.sampler_device = torch.device(sampler_device)
        self.batch_size = batch_size
        self.alpha = alpha
        self.beta = beta

    def __call__(self):
        ext = pytest.importorskip("torchrl._torchrl")
        if not hasattr(ext, "SumSegmentTreeFp32"):
            _skip_or_fail_unavailable("TorchRL was not built with segment tree support")
        if "cuda" in {self.storage_device.type, self.sampler_device.type}:
            if not torch.cuda.is_available():
                _skip_or_fail_unavailable("CUDA is not available")
            if not hasattr(ext, "CudaSumSegmentTreeFp32"):
                _skip_or_fail_unavailable(
                    "TorchRL was not built with CUDA segment tree support"
                )
        if self.storage_type == "memmap":
            storage = LazyMemmapStorage(self.size)
            data_device = torch.device("cpu")
        elif self.storage_type == "tensor":
            storage = LazyTensorStorage(self.size, device=self.storage_device)
            data_device = self.storage_device
        else:
            raise RuntimeError(f"Unknown storage_type {self.storage_type}.")
        rb = TensorDictPrioritizedReplayBuffer(
            alpha=self.alpha,
            beta=self.beta,
            storage=storage,
            sampler_device=self.sampler_device,
            batch_size=self.batch_size,
            priority_key="td_error",
        )
        data = TensorDict(
            {
                "obs": torch.arange(
                    self.size, dtype=torch.float32, device=data_device
                ).unsqueeze(-1),
                "td_error": torch.linspace(0.1, 1.0, self.size, device=data_device),
            },
            batch_size=[self.size],
            device=data_device,
        )
        rb.extend(data)
        if self.storage_device.type == "cuda":
            torch.cuda.synchronize(self.storage_device)
        if (
            self.sampler_device.type == "cuda"
            and self.sampler_device != self.storage_device
        ):
            torch.cuda.synchronize(self.sampler_device)
        return ((rb,), {})


def _prioritized_sampler_benchmark_devices():
    device = os.getenv("TORCHRL_BENCHMARK_DEVICE")
    if device == "CPU":
        return ["cpu"]
    if device == "GPU":
        return ["cuda"]
    return ["cpu", "cuda"]


def _prioritized_replay_buffer_benchmark_configs():
    device = os.getenv("TORCHRL_BENCHMARK_DEVICE")
    if device == "CPU":
        return [
            pytest.param("memmap", "cpu", "cpu", id="memmap_cpu_storage_cpu_sampler")
        ]
    if device == "GPU":
        return [
            pytest.param("tensor", "cuda", "cuda", id="cuda_storage_cuda_sampler"),
            pytest.param("memmap", "cpu", "cuda", id="memmap_cpu_storage_cuda_sampler"),
            pytest.param("tensor", "cuda", "cpu", id="cuda_storage_cpu_sampler"),
        ]
    return [
        pytest.param("memmap", "cpu", "cpu", id="memmap_cpu_storage_cpu_sampler"),
        pytest.param("tensor", "cuda", "cuda", id="cuda_storage_cuda_sampler"),
        pytest.param("memmap", "cpu", "cuda", id="memmap_cpu_storage_cuda_sampler"),
        pytest.param("tensor", "cuda", "cpu", id="cuda_storage_cpu_sampler"),
    ]


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


@pytest.mark.parametrize("size", [1_000, 100_000])
def test_prompt_group_sampler_cached_sample(benchmark, size):
    rb = ReplayBuffer(
        storage=LazyStackStorage(size),
        sampler=PromptGroupSampler(num_groups=8, group_key="prompt"),
        batch_size=64,
    )
    rb.extend(
        TensorDict(
            {
                "prompt": torch.arange(size) % 64,
                "value": torch.arange(size),
            },
            batch_size=[size],
        )
    )
    rb.sample()
    benchmark(sample, rb)


@pytest.mark.parametrize("size", [1_000, 100_000])
def test_geometric_trajectory_window_sampler_cached_sample(benchmark, size):
    trajectory_length = 128
    trajectory = torch.arange(size) // trajectory_length
    step = torch.arange(size) % trajectory_length
    rb = TensorDictReplayBuffer(
        storage=LazyTensorStorage(size),
        sampler=GeometricTrajectoryWindowSampler(
            history=8,
            continuation_probability=0.9,
            trajectory_key="trajectory",
            step_key="step",
        ),
        batch_size=64,
    )
    rb.extend(
        TensorDict(
            {"trajectory": trajectory, "step": step, "value": torch.arange(size)},
            batch_size=[size],
        )
    )
    rb.sample()
    benchmark(sample, rb)


class TestPrioritizedReplayBufferBenchmark:
    @pytest.mark.parametrize("device", _prioritized_sampler_benchmark_devices())
    @pytest.mark.parametrize("size", [1_000_000, 10_000_000])
    def test_sampler_sample_scale(self, benchmark, size, device):
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

    @pytest.mark.parametrize(
        "storage_type,storage_device,sampler_device",
        _prioritized_replay_buffer_benchmark_configs(),
    )
    @pytest.mark.parametrize("size", [1_000_000])
    def test_sample_mixed_devices(
        self, benchmark, size, storage_type, storage_device, sampler_device
    ):
        batch_size = 65_536
        (rb,), _ = create_prioritized_replay_buffer(
            size=size,
            storage_type=storage_type,
            storage_device=storage_device,
            sampler_device=sampler_device,
            batch_size=batch_size,
        )()
        benchmark(sample_prioritized_replay_buffer, rb)


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


class create_wraparound_rb:
    """Builds a full generation-tracking buffer so every timed extend reuses slots and bumps generations."""

    def __init__(self, size=10_000, batch=1_000):
        self.size = size
        self.batch = batch

    def __call__(self):
        rb = ReplayBuffer(
            storage=LazyTensorStorage(self.size),
            writer=RoundRobinWriter(track_generations=True),
        )
        data = TensorDict({"a": torch.zeros(self.batch, 5)}, batch_size=[self.batch])
        while rb.write_count < self.size:
            rb.extend(data)
        return ((rb, data), {})


def extend_wraparound(rb, data):
    for _ in range(10):
        rb.extend(data)


def test_rb_extend_generation_stamping(benchmark):
    benchmark.pedantic(
        extend_wraparound,
        setup=create_wraparound_rb(),
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


class TestWindowingTransformsBenchmark:
    """Offline (sample-path) sliding-window transforms: CatFrames.unfolding
    and the ActionChunkTransform recipe built on top of it."""

    @pytest.mark.parametrize("done_key", ["done", None], ids=["done_aware", "no_done"])
    def test_action_chunk_transform(self, benchmark, done_key):
        t = ActionChunkTransform(chunk_size=8, done_key=done_key)
        td = TensorDict(
            {
                "action": torch.randn(64, 32, 7),
                ("next", "done"): torch.zeros(64, 32, 1, dtype=torch.bool),
            },
            batch_size=[64],
        )
        benchmark(t, td)

    def test_catframes_offline(self, benchmark):
        t = CatFrames(N=4, dim=-3, in_keys=["pixels"], out_keys=["pixels_cat"])
        td = TensorDict(
            {
                "pixels": torch.randn(8, 32, 3, 32, 32),
                ("next", "done"): torch.zeros(8, 32, 1, dtype=torch.bool),
            },
            batch_size=[8, 32],
        ).refine_names(None, "time")
        benchmark(t, td)


if __name__ == "__main__":
    args, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst"] + unknown)
