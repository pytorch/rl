# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import argparse
import functools
import gc
import inspect
import time

import psutil
import pytest
import torch.cuda
import tqdm
from bench_collectors import PixelMockEnv, PolicyFactory
from tensordict import TensorDict, TensorDictBase
from tensordict.nn import TensorDictModule

from torchrl.collectors import (
    AsyncBatchedCollector,
    Collector,
    MultiAsyncCollector,
    MultiSyncCollector,
)
from torchrl.data import (
    Binary,
    Categorical,
    Composite,
    LazyTensorStorage,
    ReplayBuffer,
    Unbounded,
)
from torchrl.data.utils import CloudpickleWrapper
from torchrl.envs import (
    EnvBase,
    EnvCreator,
    GymEnv,
    ParallelEnv,
    StepCounter,
    TransformedEnv,
)
from torchrl.envs.libs.dm_control import DMControlEnv
from torchrl.modules import inference_server, MLP, RandomPolicy
from torchrl.modules.inference_server import (
    InferenceDeviceConfig,
    InferenceServerConfig,
)
from torchrl.testing.mocking_classes import MockBatchedLockedEnv


class _ResetLatencyPixelEnv(PixelMockEnv):
    """Add reset stalls to the shared fake-pixel benchmark workload."""

    def __init__(self, *, reset_latency_s=0.0, **kwargs):
        super().__init__(**kwargs)
        self.reset_latency_s = reset_latency_s

    def _reset(self, tensordict=None, **kwargs):
        time.sleep(self.reset_latency_s)
        return super()._reset(tensordict, **kwargs)


@pytest.mark.parametrize("regime", ["uniform", "slow-reset"])
@pytest.mark.parametrize(
    "mode",
    [
        "parallel",
        "async-queue",
        "async-shm",
        "async-shm-integrated",
        "async-shm-grouped",
        "async-process-slots",
        "async-process-slots-integrated",
        "async-process-slots-chunked",
        pytest.param("async-shm-static", marks=pytest.mark.gpu),
        pytest.param("async-process-slots-static", marks=pytest.mark.gpu),
    ],
)
def test_async_collection_pixels(benchmark, mode, regime):
    """Fixed end-to-end series; see ASYNC_BENCHMARKS.md before changing inputs."""
    _benchmark_async_collection_pixels(benchmark, mode, regime)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("mode", ["async-shm", "async-process-slots"])
def test_async_collection_pixels_64_envs(benchmark, mode):
    """Compare transport paths with a matched 64-env CUDA pixel-policy workload."""
    _benchmark_async_collection_pixels(benchmark, mode, "uniform", num_envs=64)


def _benchmark_async_collection_pixels(benchmark, mode, regime, *, num_envs=None):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    has_static = (
        "static_batch_size" in inspect.signature(InferenceServerConfig).parameters
    )
    has_grouping = (
        "envs_per_worker" in inspect.signature(AsyncBatchedCollector).parameters
    )
    use_process = mode.startswith("async-process-slots")
    if use_process and not hasattr(inference_server, "ProcessSlotTransport"):
        pytest.skip("Direct process slots are not available on this revision")
    integrated = mode in ("async-shm-integrated", "async-process-slots-integrated")
    has_chunking = (
        "transition_chunk_size" in inspect.signature(AsyncBatchedCollector).parameters
    )
    if mode == "async-process-slots-chunked" and not has_chunking:
        pytest.skip("Chunked worker results are not available on this revision")
    use_static = mode in ("async-shm-static", "async-process-slots-static") or (
        integrated and device != "cpu" and has_static
    )
    group_size = (
        4
        if mode == "async-shm-grouped"
        or (mode == "async-shm-integrated" and has_grouping)
        else 1
    )
    # The direct-transport curve adopts chunked worker results once the
    # collector accepts them; one transition per message before that.
    chunk_size = (
        64
        if mode == "async-process-slots-chunked"
        or (mode == "async-process-slots-integrated" and has_chunking)
        else 1
    )
    if use_static:
        if device == "cpu":
            pytest.skip("CUDA graphs require CUDA")
        if not has_static:
            pytest.skip("Static inference batches are not available on this revision")
    if group_size > 1 and not has_grouping:
        pytest.skip("Grouped environment workers are not available on this revision")

    # Keep these constants stable across merges. CPU and GPU are separate series.
    if num_envs is None:
        num_envs = 32 if device != "cpu" else 8
    frames_per_batch = 256
    batches_per_round = 4
    rounds = 5
    warmup_rounds = 2
    gc.collect()
    if device != "cpu" and not use_process:
        torch.cuda.empty_cache()
    torch.manual_seed(0)
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    factories = [
        functools.partial(
            _ResetLatencyPixelEnv,
            step_latency_s=0.001,
            reset_latency_s=0.2 if regime == "slow-reset" and index == 0 else 0.0,
            max_steps=16 if regime == "slow-reset" else 200,
        )
        for index in range(num_envs)
    ]
    policy_factory = PolicyFactory(
        hidden_features=1024 if device != "cpu" else 256, hidden_layers=2
    )
    collector = None
    try:
        if mode == "parallel":
            collector = Collector(
                ParallelEnv(num_envs, factories, num_threads=1, num_sub_threads=1),
                policy_factory(),
                frames_per_batch=frames_per_batch,
                total_frames=-1,
                policy_device=device,
                env_device="cpu",
                storing_device="cpu",
                trust_policy=True,
                use_buffers=False,
                auto_register_policy_transforms=False,
            )
        else:
            config = {"static_batch_size": num_envs} if use_static else {}
            grouped = {"envs_per_worker": group_size} if group_size > 1 else {}
            process_options = {}
            if use_process:
                probe = factories[0]()
                try:
                    request_spec = probe.fake_tensordict().select("pixels", strict=True)
                    response_spec = probe.rand_action().set(
                        "policy_version",
                        torch.zeros(probe.batch_size, dtype=torch.long),
                    )
                finally:
                    probe.close()
                process_options["transport"] = inference_server.ProcessSlotTransport(
                    request_spec=request_spec,
                    response_spec=response_spec,
                    num_slots=num_envs,
                )
                config["service_backend"] = "process"
                # Fixed series pin one transition per message; the collector
                # default is "auto".
                process_options["transition_chunk_size"] = chunk_size
            else:
                # The fixed thread-server series keep the driver-mediated transport
                # explicitly, so a change of the collector default cannot move them.
                process_options["transport"] = "driver"
            collector = AsyncBatchedCollector(
                factories,
                policy_factory=policy_factory,
                frames_per_batch=frames_per_batch,
                total_frames=-1,
                env_backend="multiprocessing",
                env_exchange=(
                    "queue" if mode == "async-queue" or use_process else "shm"
                ),
                server_config=InferenceServerConfig(
                    max_batch_size=num_envs, min_batch_size=1, timeout=0.001, **config
                ),
                device_config=InferenceDeviceConfig(
                    policy_device=device, output_device="cpu", storing_device="cpu"
                ),
                **grouped,
                **process_options,
            )
        iterator = iter(collector)
        latencies = []

        def collect_round():
            frames = 0
            for _ in range(batches_per_round):
                start = time.perf_counter()
                batch = next(iterator)
                latencies.append(time.perf_counter() - start)
                frames += batch.numel()
            # A faster run must not silently collect fewer transitions.
            assert frames == frames_per_batch * batches_per_round

        for _ in range(warmup_rounds):
            collect_round()
        latencies.clear()
        if device != "cpu" and not use_process:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        if isinstance(collector, AsyncBatchedCollector):
            collector.server_stats(reset=True)
        benchmark.pedantic(collect_round, rounds=rounds, iterations=1)
        latency_ms = torch.tensor(latencies[: rounds * batches_per_round]) * 1000
        process = psutil.Process()
        benchmark.extra_info.update(
            execution=(
                f"{'graph' if use_static else 'eager'}; {group_size} envs/worker"
                + ("; process acting" if use_process else "")
                + (f"; {chunk_size} transitions/message" if chunk_size > 1 else "")
            ),
            num_envs=num_envs,
            frames_per_batch=frames_per_batch,
            transitions=frames_per_batch * batches_per_round,
            warmup_rounds=warmup_rounds,
            measured_rounds=rounds,
            batch_latency_p50_ms=latency_ms.quantile(0.5).item(),
            batch_latency_p95_ms=latency_ms.quantile(0.95).item(),
            process_tree_rss_bytes=sum(
                child.memory_info().rss
                for child in [process, *process.children(recursive=True)]
            ),
        )
        if device != "cpu" and not use_process:
            benchmark.extra_info[
                "cuda_peak_allocated_bytes"
            ] = torch.cuda.max_memory_allocated()
        if isinstance(collector, AsyncBatchedCollector):
            benchmark.extra_info["server_stats"] = collector.server_stats()
    finally:
        if collector is not None:
            collector.shutdown()
        torch.set_num_threads(old_threads)


class _PayloadEnv(EnvBase):
    """Zero-work environment with a configurable observation payload."""

    def __init__(self, payload_size: int = 65536):
        super().__init__()
        self.observation_spec = Composite(
            observation=Unbounded((payload_size,)), shape=self.batch_size
        )
        self.action_spec = Binary(1, shape=(1,))
        self.reward_spec = Unbounded((1,))
        self.done_spec = Categorical(2, shape=(1,), dtype=torch.bool)
        self.register_buffer("observation", torch.zeros(payload_size))

    def _set_seed(self, seed: int | None) -> None:
        return None

    def _reset(self, tensordict: TensorDictBase | None = None) -> TensorDictBase:
        return TensorDict(
            {
                "observation": self.observation,
                "done": torch.zeros(1, dtype=torch.bool),
                "terminated": torch.zeros(1, dtype=torch.bool),
            },
            batch_size=self.batch_size,
        )

    def _step(self, tensordict: TensorDictBase) -> TensorDictBase:
        return TensorDict(
            {
                "observation": self.observation,
                "reward": torch.zeros(1),
                "done": torch.zeros(1, dtype=torch.bool),
                "terminated": torch.zeros(1, dtype=torch.bool),
            },
            batch_size=self.batch_size,
        )


def single_collector_setup():
    device = "cuda:0" if torch.cuda.device_count() else "cpu"
    env = TransformedEnv(DMControlEnv("cheetah", "run", device=device), StepCounter(50))
    c = Collector(
        env,
        RandomPolicy(env.action_spec),
        total_frames=-1,
        frames_per_batch=100,
        device=device,
    )
    c = iter(c)
    for i, _ in enumerate(c):
        if i == 10:
            break
    return ((c,), {})


def sync_collector_setup():
    device = "cuda:0" if torch.cuda.device_count() else "cpu"
    env = EnvCreator(
        lambda: TransformedEnv(
            DMControlEnv("cheetah", "run", device=device), StepCounter(50)
        )
    )
    c = MultiSyncCollector(
        [env, env],
        RandomPolicy(env().action_spec),
        total_frames=-1,
        frames_per_batch=100,
        device=device,
    )
    c = iter(c)
    for i, _ in enumerate(c):
        if i == 10:
            break
    return ((c,), {})


def sync_collector_setup_preempt():
    """Sync collector with preemption: exercises the per-step interruptor polling
    in the workers and the masking path in the main process."""
    device = "cuda:0" if torch.cuda.device_count() else "cpu"
    env = EnvCreator(
        lambda: TransformedEnv(
            DMControlEnv("cheetah", "run", device=device), StepCounter(50)
        )
    )
    c = MultiSyncCollector(
        [env, env],
        RandomPolicy(env().action_spec),
        total_frames=-1,
        frames_per_batch=100,
        device=device,
        preemptive_threshold=0.9,
        cat_results="stack",
    )
    c = iter(c)
    for i, _ in enumerate(c):
        if i == 10:
            break
    return ((c,), {})


def async_collector_setup():
    device = "cuda:0" if torch.cuda.device_count() else "cpu"
    env = EnvCreator(
        lambda: TransformedEnv(
            DMControlEnv("cheetah", "run", device=device), StepCounter(50)
        )
    )
    c = MultiAsyncCollector(
        [env, env],
        RandomPolicy(env().action_spec),
        total_frames=-1,
        frames_per_batch=100,
        device=device,
    )
    collector = c
    c = iter(collector)
    for i, _ in enumerate(c):
        if i == 10:
            break
    return ((c, collector), {})


def parallel_env_compact_obs_setup(payload_size):
    env = ParallelEnv(4, functools.partial(_PayloadEnv, payload_size))
    collector = Collector(
        env,
        RandomPolicy(env.action_spec),
        total_frames=-1,
        frames_per_batch=128,
        compact_obs=True,
    )
    collector_iterator = iter(collector)
    for _ in range(10):
        next(collector_iterator)
    return ((collector_iterator, collector), {})


def single_collector_setup_pixels():
    device = "cuda:0" if torch.cuda.device_count() else "cpu"
    # env = TransformedEnv(
    #     DMControlEnv("cheetah", "run", device=device, from_pixels=True), StepCounter(50)
    # )
    env = TransformedEnv(GymEnv("ALE/Pong-v5"), StepCounter(50))
    c = Collector(
        env,
        RandomPolicy(env.action_spec),
        total_frames=-1,
        frames_per_batch=100,
        device=device,
    )
    c = iter(c)
    for i, _ in enumerate(c):
        if i == 10:
            break
    return ((c,), {})


def sync_collector_setup_pixels():
    device = "cuda:0" if torch.cuda.device_count() else "cpu"
    env = EnvCreator(
        lambda: TransformedEnv(
            # DMControlEnv("cheetah", "run", device=device, from_pixels=True),
            GymEnv("ALE/Pong-v5"),
            StepCounter(50),
        )
    )
    c = MultiSyncCollector(
        [env, env],
        RandomPolicy(env().action_spec),
        total_frames=-1,
        frames_per_batch=100,
        device=device,
    )
    c = iter(c)
    for i, _ in enumerate(c):
        if i == 10:
            break
    return ((c,), {})


def async_collector_setup_pixels():
    device = "cuda:0" if torch.cuda.device_count() else "cpu"
    env = EnvCreator(
        lambda: TransformedEnv(
            # DMControlEnv("cheetah", "run", device=device, from_pixels=True),
            GymEnv("ALE/Pong-v5"),
            StepCounter(50),
        )
    )
    c = MultiAsyncCollector(
        [env, env],
        RandomPolicy(env().action_spec),
        total_frames=-1,
        frames_per_batch=100,
        device=device,
    )
    c = iter(c)
    for i, _ in enumerate(c):
        if i == 10:
            break
    return ((c,), {})


def execute_collector(c):
    # will run for 9 iterations (1 during setup)
    next(c)


def execute_collector_with_rb(c, rb):
    """Execute collector iteration and verify data was stored in replay buffer."""
    next(c)


# --- Benchmarks for collector with replay buffer (lazy stack optimization) ---


def single_collector_with_rb_setup():
    """Setup single collector with replay buffer - tests lazy stack optimization."""
    device = "cuda:0" if torch.cuda.device_count() else "cpu"
    env = TransformedEnv(DMControlEnv("cheetah", "run", device=device), StepCounter(50))
    rb = ReplayBuffer(storage=LazyTensorStorage(10000))
    c = Collector(
        env,
        RandomPolicy(env.action_spec),
        total_frames=-1,
        frames_per_batch=100,
        device=device,
        replay_buffer=rb,
    )
    c = iter(c)
    # Warmup
    for i, _ in enumerate(c):
        if i == 10:
            break
    return ((c, rb), {})


def single_collector_with_rb_setup_pixels():
    """Setup single collector with replay buffer for pixel observations."""
    device = "cuda:0" if torch.cuda.device_count() else "cpu"
    env = TransformedEnv(GymEnv("ALE/Pong-v5"), StepCounter(50))
    rb = ReplayBuffer(storage=LazyTensorStorage(10000))
    c = Collector(
        env,
        RandomPolicy(env.action_spec),
        total_frames=-1,
        frames_per_batch=100,
        device=device,
        replay_buffer=rb,
    )
    c = iter(c)
    # Warmup
    for i, _ in enumerate(c):
        if i == 10:
            break
    return ((c, rb), {})


def sync_collector_with_rb_setup():
    """Setup a multi-process collector whose workers write whole rollouts."""
    device = "cuda:0" if torch.cuda.device_count() else "cpu"
    env = EnvCreator(
        lambda: TransformedEnv(
            DMControlEnv("cheetah", "run", device=device), StepCounter(50)
        )
    )
    rb = ReplayBuffer(storage=LazyTensorStorage(10000))
    c = MultiSyncCollector(
        [env, env],
        RandomPolicy(env().action_spec),
        total_frames=-1,
        frames_per_batch=100,
        device=device,
        replay_buffer=rb,
    )
    c = iter(c)
    for i, _ in enumerate(c):
        if i == 10:
            break
    return ((c, rb), {})


def sync_payload_collector_with_rb_setup():
    """Setup replay writes where rollout materialization dominates env work."""
    env = EnvCreator(_PayloadEnv)
    rb = ReplayBuffer(storage=LazyTensorStorage(512))
    c = MultiSyncCollector(
        [env, env],
        RandomPolicy(env().action_spec),
        total_frames=-1,
        frames_per_batch=64,
        replay_buffer=rb,
    )
    collector = c
    c = iter(collector)
    for _ in range(10):
        next(c)
    return ((c, rb, collector), {})


def test_single_with_rb(benchmark):
    """Benchmark single collector with replay buffer (lazy stack path)."""
    (c, rb), _ = single_collector_with_rb_setup()
    benchmark(execute_collector_with_rb, c, rb)


def test_sync_with_rb(benchmark):
    """Benchmark the runner-managed replay-buffer write path."""
    (c, rb), _ = sync_collector_with_rb_setup()
    benchmark(execute_collector_with_rb, c, rb)


def test_sync_payload_with_rb(benchmark):
    """Benchmark clone avoidance for large runner-managed rollouts."""
    (c, rb, collector), _ = sync_payload_collector_with_rb_setup()
    try:
        benchmark.pedantic(
            execute_collector_with_rb,
            args=(c, rb),
            iterations=5,
            rounds=10,
        )
    finally:
        collector.shutdown()


@pytest.mark.skipif(not torch.cuda.device_count(), reason="no rendering without cuda")
def test_single_with_rb_pixels(benchmark):
    """Benchmark single collector with replay buffer for pixel observations."""
    (c, rb), _ = single_collector_with_rb_setup_pixels()
    benchmark(execute_collector_with_rb, c, rb)


def test_single(benchmark):
    (c,), _ = single_collector_setup()
    benchmark(execute_collector, c)


def test_sync(benchmark):
    (c,), _ = sync_collector_setup()
    benchmark(execute_collector, c)


def test_sync_preempt(benchmark):
    (c,), _ = sync_collector_setup_preempt()
    benchmark(execute_collector, c)


def test_async(benchmark):
    (c, collector), _ = async_collector_setup()
    try:
        benchmark.pedantic(
            execute_collector,
            args=(c,),
            iterations=10,
            rounds=10,
        )
    finally:
        collector.shutdown()


@pytest.mark.parametrize("payload_size", [65536, 262144], ids=["256KiB", "1MiB"])
def test_parallel_env_compact_obs(benchmark, payload_size):
    """Benchmark copies for a collector over a large-payload ParallelEnv."""
    (c, collector), _ = parallel_env_compact_obs_setup(payload_size)
    try:
        benchmark.pedantic(
            execute_collector,
            args=(c,),
            iterations=2,
            rounds=10,
        )
    finally:
        collector.shutdown()


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_compiled_policy_large_observation(benchmark):
    device = torch.device("cuda:0")
    batch_size = torch.Size([4096])
    observation_dim = 1024
    payload_spec = Unbounded((*batch_size, observation_dim), device=device)
    env = MockBatchedLockedEnv(device=device, batch_size=batch_size)
    env.observation_spec = Composite(observation=payload_spec, shape=batch_size)
    env.state_spec = env.observation_spec.clone()
    env.reward_spec = payload_spec.clone()
    with torch.random.fork_rng(devices=[device]):
        torch.manual_seed(0)
        policy = TensorDictModule(
            MLP(
                in_features=observation_dim,
                out_features=1,
                num_cells=[64, 64],
                activation_class=torch.nn.Tanh,
                activate_last_layer=True,
                device=device,
            ),
            in_keys=["observation"],
            out_keys=["action"],
        )
    frames_per_batch = batch_size.numel() * 8
    collector = Collector(
        env,
        policy,
        frames_per_batch=frames_per_batch,
        total_frames=-1,
        device=device,
        max_frames_per_traj=-1,
        compile_policy={"mode": "reduce-overhead", "warmup": 1},
    )
    previous_matmul_precision = torch.get_float32_matmul_precision()
    batches_per_round = 10

    def collect_batches():
        for _ in range(batches_per_round):
            next(collector_iterator)
        torch.cuda.synchronize(device)

    def reset_collector():
        collector.reset()
        torch.cuda.synchronize(device)

    try:
        torch.set_float32_matmul_precision("high")
        collector_iterator = iter(collector)
        for _ in range(20):
            next(collector_iterator)
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        benchmark.extra_info.update(
            batches_per_round=batches_per_round,
            frames_per_batch=frames_per_batch,
            num_envs=batch_size.numel(),
            observation_dim=observation_dim,
            transitions_per_round=frames_per_batch * batches_per_round,
        )
        benchmark.pedantic(
            collect_batches,
            setup=reset_collector,
            iterations=1,
            rounds=10,
        )
        benchmark.extra_info[
            "cuda_peak_allocated_bytes"
        ] = torch.cuda.max_memory_allocated(device)
    finally:
        try:
            collector.shutdown()
        finally:
            torch.set_float32_matmul_precision(previous_matmul_precision)


@pytest.mark.skipif(not torch.cuda.device_count(), reason="no rendering without cuda")
def test_single_pixels(benchmark):
    (c,), _ = single_collector_setup_pixels()
    benchmark(execute_collector, c)


@pytest.mark.skipif(not torch.cuda.device_count(), reason="no rendering without cuda")
def test_sync_pixels(benchmark):
    (c,), _ = sync_collector_setup_pixels()
    benchmark(execute_collector, c)


@pytest.mark.skipif(not torch.cuda.device_count(), reason="no rendering without cuda")
def test_async_pixels(benchmark):
    (c,), _ = async_collector_setup_pixels()
    benchmark(execute_collector, c)


class TestRBGCollector:
    @pytest.mark.parametrize(
        "n_col,n_wokrers_per_col",
        [
            [2, 2],
            [4, 2],
            [8, 2],
            [16, 2],
            [2, 1],
            [4, 1],
            [8, 1],
            [16, 1],
        ],
    )
    def test_multiasync_rb(self, n_col, n_wokrers_per_col):
        make_env = EnvCreator(lambda: GymEnv("ALE/Pong-v5"))
        if n_wokrers_per_col > 1:
            make_env = ParallelEnv(n_wokrers_per_col, make_env)
            env = make_env
            policy = RandomPolicy(env.action_spec)
        else:
            env = make_env()
            policy = RandomPolicy(env.action_spec)

        storage = LazyTensorStorage(10_000)
        rb = ReplayBuffer(storage=storage)
        rb.extend(env.rollout(2, policy).reshape(-1))
        rb.append_transform(CloudpickleWrapper(lambda x: x.reshape(-1)), invert=True)

        fpb = n_wokrers_per_col * 100
        total_frames = n_wokrers_per_col * 100_000
        c = MultiAsyncCollector(
            [make_env] * n_col,
            policy,
            frames_per_batch=fpb,
            total_frames=total_frames,
            replay_buffer=rb,
        )
        frames = 0
        pbar = tqdm.tqdm(total=total_frames - (n_col * fpb))
        for i, _ in enumerate(c):
            if i == n_col:
                t0 = time.time()
            if i >= n_col:
                frames += fpb
            if i > n_col:
                fps = frames / (time.time() - t0)
                pbar.update(fpb)
                pbar.set_description(f"fps: {fps: 4.4f}")


if __name__ == "__main__":
    args, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst"] + unknown)
