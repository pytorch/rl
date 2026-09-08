"""Benchmark async policy-server collection against synchronous ParallelEnv.

The default path is portable and uses a fake pixel environment with
configurable environment and policy latency. Optional MuJoCo/Gymnasium pixel
rendering can be enabled with ``--env gym-mujoco --from-pixels`` and, on Linux
CUDA hosts, ``--mujoco-gl egl``.

Examples:
    python benchmarks/bench_collectors.py --total-frames 2000
    python benchmarks/bench_collectors.py --num-envs 1,2,4,8 --policy-delay-ms 20
    python benchmarks/bench_collectors.py --backends async-env-mp --replay-mode iterator
    python benchmarks/bench_collectors.py --backends async-env-mp --replay-mode background
    # Replay throughput of direct process slots, per-transition vs chunked results
    python benchmarks/bench_collectors.py --num-envs 8 --backends async-process-slot \
        --replay-mode background --total-frames 4096 --transition-chunk-size 1
    python benchmarks/bench_collectors.py --num-envs 8 --backends async-process-slot \
        --replay-mode background --total-frames 4096 --transition-chunk-size 64
    python benchmarks/bench_collectors.py --num-envs 64 \
        --backends async-env-mp --env-exchange shm \
        --env-step-latency-ms 50 --policy-hidden-features 9216 \
        --policy-hidden-layers 2 --policy-device cuda:0
    MUJOCO_GL=egl python benchmarks/bench_collectors.py \
        --env gym-mujoco --from-pixels --mujoco-gl egl --policy-device cuda:0
"""

from __future__ import annotations

import argparse
import csv
import functools as ft
import importlib.util
import json
import os
import time as time_module
from dataclasses import asdict, dataclass, field
from pathlib import Path

import psutil
import torch
import torch.nn as nn
from tensordict import TensorDict
from tensordict.nn import TensorDictModule

from torchrl.collectors import AsyncBatchedCollector, Collector
from torchrl.data import (
    Bounded,
    Categorical,
    Composite,
    LazyTensorStorage,
    ReplayBuffer,
    Unbounded,
)
from torchrl.envs import (
    EnvBase,
    GymEnv,
    ParallelEnv,
    Resize,
    StepCounter,
    ToTensorImage,
    TransformedEnv,
)
from torchrl.modules import ConvNet, MLP
from torchrl.modules.inference_server import (
    InferenceDeviceConfig,
    InferenceServerConfig,
    ProcessSlotTransport,
)

OBS_SHAPE = (3, 84, 84)
ACTION_DIM = 6
MAX_STEPS = 200


@dataclass
class BenchmarkResult:
    collector: str
    backend: str
    batch_rule: str
    env: str
    num_envs: int
    frames_per_batch: int
    total_frames: int
    policy_device: str
    output_device: str
    env_step_latency_ms: float
    policy_delay_ms: float
    status: str
    envs_per_worker: int = 1
    warmup_batches: int = 0
    p50_batch_ms: float | None = 0.0
    p95_batch_ms: float | None = 0.0
    host_rss_mib: float = 0.0
    cuda_used_mib: float = 0.0
    frames: int = 0
    elapsed_s: float = 0.0
    frames_per_s: float = 0.0
    decisions_per_s: float = 0.0
    # CPU time (user + system) of the driver process per collected frame over
    # the measured window. With process-backed environments and inference this
    # is the driver's own transition-path cost; thread backends include the
    # coordinator and inference-server threads.
    driver_cpu_ms_per_frame: float = 0.0
    failure: str = ""
    policy_stats: dict[str, float | int] = field(default_factory=dict)


class PixelMockEnv(EnvBase):
    """Fake pixel env producing configurable-latency continuous-control data."""

    def __init__(
        self,
        *,
        max_steps: int = MAX_STEPS,
        step_latency_s: float = 0.001,
        action_dim: int = ACTION_DIM,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.max_steps = max_steps
        self.step_latency_s = step_latency_s
        self.action_dim = action_dim
        self.observation_spec = Composite(
            pixels=Unbounded(
                shape=(*self.batch_size, *OBS_SHAPE),
                dtype=torch.uint8,
                device=self.device,
            ),
            shape=self.batch_size,
        )
        self.action_spec = Bounded(
            low=-1.0,
            high=1.0,
            shape=(*self.batch_size, action_dim),
            dtype=torch.float32,
            device=self.device,
        )
        self.reward_spec = Unbounded(
            shape=(*self.batch_size, 1),
            dtype=torch.float32,
            device=self.device,
        )
        self.done_spec = Categorical(
            n=2,
            dtype=torch.bool,
            shape=(*self.batch_size, 1),
            device=self.device,
        )
        self._step_count = 0

    def _set_seed(self, seed: int) -> None:
        torch.manual_seed(seed)

    def _make_pixels(self) -> torch.Tensor:
        return torch.randint(
            0,
            256,
            (*self.batch_size, *OBS_SHAPE),
            dtype=torch.uint8,
            device=self.device,
        )

    def _reset(self, tensordict=None, **kwargs) -> TensorDict:
        self._step_count = 0
        return TensorDict(
            {
                "pixels": self._make_pixels(),
                "done": torch.zeros(
                    *self.batch_size, 1, dtype=torch.bool, device=self.device
                ),
                "terminated": torch.zeros(
                    *self.batch_size, 1, dtype=torch.bool, device=self.device
                ),
            },
            batch_size=self.batch_size,
            device=self.device,
        )

    def _step(self, tensordict: TensorDict) -> TensorDict:
        if self.step_latency_s > 0:
            time_module.sleep(self.step_latency_s)
        self._step_count += 1
        done = self._step_count >= self.max_steps
        return TensorDict(
            {
                "pixels": self._make_pixels(),
                "reward": torch.randn(*self.batch_size, 1, device=self.device),
                "done": torch.full(
                    (*self.batch_size, 1), done, dtype=torch.bool, device=self.device
                ),
                "terminated": torch.full(
                    (*self.batch_size, 1), done, dtype=torch.bool, device=self.device
                ),
            },
            batch_size=self.batch_size,
            device=self.device,
        )


@dataclass
class EnvFactory:
    env: str = "fake-pixels"
    gym_id: str = "HalfCheetah-v4"
    from_pixels: bool = False
    max_steps: int = MAX_STEPS
    step_latency_s: float = 0.001
    action_dim: int = ACTION_DIM

    def __call__(self) -> EnvBase:
        if self.env == "fake-pixels":
            return PixelMockEnv(
                max_steps=self.max_steps,
                step_latency_s=self.step_latency_s,
                action_dim=self.action_dim,
            )
        if self.env == "gym-mujoco":
            env = GymEnv(
                self.gym_id,
                from_pixels=self.from_pixels,
                pixels_only=False,
            )
            env = TransformedEnv(env, StepCounter(max_steps=self.max_steps))
            if self.from_pixels:
                env.append_transform(ToTensorImage())
                env.append_transform(Resize(OBS_SHAPE[-2], OBS_SHAPE[-1]))
            return env
        raise ValueError(f"Unknown env {self.env!r}.")


class LatencyHead(nn.Module):
    """Pixel policy head with configurable VLA-like latency."""

    def __init__(self, net: nn.Module, *, delay_s: float = 0.0) -> None:
        super().__init__()
        self.net = net
        self.delay_s = delay_s

    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        if torch.is_floating_point(pixels):
            pixels = pixels.to(torch.float32)
        else:
            pixels = pixels.to(torch.float32) / 255.0
        action = torch.tanh(self.net(pixels))
        if self.delay_s > 0:
            if action.is_cuda:
                torch.cuda.synchronize(action.device)
            time_module.sleep(self.delay_s)
        return action


@dataclass
class PolicyFactory:
    action_dim: int = ACTION_DIM
    delay_s: float = 0.0
    from_pixels: bool = True
    hidden_features: int = 256
    hidden_layers: int = 1

    def __call__(self) -> TensorDictModule:
        torch.manual_seed(0)
        if not self.from_pixels:
            mlp = MLP(
                activation_class=nn.ReLU,
                out_features=self.action_dim,
                num_cells=[256, 256],
            )
            return TensorDictModule(
                LatencyHead(mlp, delay_s=self.delay_s),
                in_keys=["observation"],
                out_keys=["action"],
            )
        cnn = ConvNet(
            activation_class=nn.ReLU,
            num_cells=[16, 32, 32],
            kernel_sizes=[8, 4, 3],
            strides=[4, 2, 1],
        )
        with torch.no_grad():
            cnn_out = cnn(torch.ones(OBS_SHAPE, dtype=torch.float32))
        mlp = MLP(
            in_features=cnn_out.shape[-1],
            activation_class=nn.ReLU,
            out_features=self.action_dim,
            num_cells=[self.hidden_features] * self.hidden_layers,
        )
        return TensorDictModule(
            LatencyHead(nn.Sequential(cnn, mlp), delay_s=self.delay_s),
            in_keys=["pixels"],
            out_keys=["action"],
        )


def _make_process_slot_transport(
    env_factory: EnvFactory, num_envs: int
) -> ProcessSlotTransport:
    env = env_factory()
    try:
        input_key = "pixels" if env_factory.from_pixels else "observation"
        request_spec = env.fake_tensordict().select(input_key, strict=True)
        response_spec = env.rand_action().set(
            "policy_version",
            torch.zeros(env.batch_size, dtype=torch.long),
        )
    finally:
        env.close()
    return ProcessSlotTransport(
        request_spec=request_spec,
        response_spec=response_spec,
        num_slots=num_envs,
    )


def _parse_int_list(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item]


def _parse_backend_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _auto_policy_device() -> str:
    if torch.cuda.is_available():
        return "cuda:0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _check_optional_env(factory: EnvFactory) -> tuple[bool, str, int]:
    if factory.env == "fake-pixels":
        return True, "", factory.action_dim
    if factory.env == "gym-mujoco":
        if (
            importlib.util.find_spec("gymnasium") is None
            and importlib.util.find_spec("gym") is None
        ):
            return False, "gymnasium/gym is not installed", factory.action_dim
        if importlib.util.find_spec("mujoco") is None:
            return False, "mujoco is not installed", factory.action_dim
    try:
        env = factory()
        action_shape = env.action_spec.shape
        action_dim = int(action_shape[-1]) if action_shape else factory.action_dim
        env.close()
    except Exception as err:
        return False, repr(err), factory.action_dim
    return True, "", action_dim


def bench(
    *,
    name: str,
    backend: str,
    batch_rule: str,
    factory,
    env_name: str,
    num_envs: int,
    frames_per_batch: int,
    total_frames: int,
    policy_device: str,
    output_device: str,
    env_step_latency_ms: float,
    policy_delay_ms: float,
    warmup_batches: int,
    envs_per_worker: int = 1,
    replay_mode: str = "none",
) -> BenchmarkResult:
    collector = None
    total = 0
    t0 = None
    policy_stats: dict[str, float | int] = {}
    try:
        collector = factory()
        if replay_mode != "none":
            # Both modes store the same number of transitions in the same storage.
            collector.total_frames = total_frames + warmup_batches * frames_per_batch
            collector.replay_buffer = ReplayBuffer(
                storage=LazyTensorStorage(collector.total_frames)
            )
        iterator = iter(collector)
        for _ in range(warmup_batches):
            next(iterator)
        if hasattr(collector, "server_stats"):
            collector.server_stats(reset=True)
        latencies = []
        initial_frames = collector._frames
        process = psutil.Process()
        cpu_start = process.cpu_times()
        t0 = previous = time_module.perf_counter()
        if replay_mode == "background":
            iterator.close()
            collector.start()
            collector._replay_thread.join(timeout=300)
            if collector._replay_thread.is_alive():
                raise TimeoutError("Background replay benchmark exceeded 300 seconds.")
            total = collector.replay_buffer.write_count - initial_frames
            if collector._replay_error is not None:
                raise RuntimeError(
                    "Background replay benchmark failed."
                ) from collector._replay_error
        else:
            for batch in iterator:
                now = time_module.perf_counter()
                latencies.append((now - previous) * 1000)
                previous = now
                if batch is None:
                    total = collector.replay_buffer.write_count - initial_frames
                else:
                    total += batch.numel()
                if total >= total_frames:
                    break
        elapsed = time_module.perf_counter() - t0
        cpu_end = process.cpu_times()
        driver_cpu_s = (cpu_end.user - cpu_start.user) + (
            cpu_end.system - cpu_start.system
        )
        if hasattr(collector, "server_stats"):
            policy_stats = collector.server_stats()
        host_rss = process.memory_info().rss
        for child in process.children(recursive=True):
            try:
                host_rss += child.memory_info().rss
            except psutil.NoSuchProcess:
                pass
        cuda_used = 0
        if torch.device(policy_device).type == "cuda":
            free, capacity = torch.cuda.mem_get_info(policy_device)
            cuda_used = capacity - free
        # Background collection has no consumer-side batch latency to report.
        percentiles = (
            torch.tensor(latencies).quantile(torch.tensor([0.5, 0.95]))
            if latencies
            else None
        )
        fps = total / elapsed if elapsed > 0 else 0.0
        return BenchmarkResult(
            collector=name,
            backend=backend,
            batch_rule=batch_rule,
            env=env_name,
            num_envs=num_envs,
            frames_per_batch=frames_per_batch,
            total_frames=total_frames,
            policy_device=policy_device,
            output_device=output_device,
            env_step_latency_ms=env_step_latency_ms,
            policy_delay_ms=policy_delay_ms,
            status="ok",
            envs_per_worker=envs_per_worker,
            warmup_batches=warmup_batches,
            p50_batch_ms=percentiles[0].item() if percentiles is not None else None,
            p95_batch_ms=percentiles[1].item() if percentiles is not None else None,
            host_rss_mib=host_rss / 2**20,
            cuda_used_mib=cuda_used / 2**20,
            frames=total,
            elapsed_s=elapsed,
            frames_per_s=fps,
            decisions_per_s=fps,
            driver_cpu_ms_per_frame=driver_cpu_s * 1000 / total if total else 0.0,
            policy_stats=policy_stats,
        )
    except Exception as err:
        elapsed = time_module.perf_counter() - t0 if t0 is not None else 0.0
        return BenchmarkResult(
            collector=name,
            backend=backend,
            batch_rule=batch_rule,
            env=env_name,
            num_envs=num_envs,
            frames_per_batch=frames_per_batch,
            total_frames=total_frames,
            policy_device=policy_device,
            output_device=output_device,
            env_step_latency_ms=env_step_latency_ms,
            policy_delay_ms=policy_delay_ms,
            status="error",
            frames=total,
            elapsed_s=elapsed,
            frames_per_s=total / elapsed if elapsed > 0 else 0.0,
            decisions_per_s=total / elapsed if elapsed > 0 else 0.0,
            failure=repr(err),
            policy_stats=policy_stats,
        )
    finally:
        if collector is not None:
            try:
                collector.shutdown()
            except Exception:
                pass


def _write_results(
    results: list[BenchmarkResult], jsonl_path: str | None, csv_path: str | None
) -> None:
    rows = [asdict(result) for result in results]
    if jsonl_path:
        path = Path(jsonl_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as file:
            for row in rows:
                file.write(json.dumps(row, sort_keys=True) + "\n")
    if csv_path:
        path = Path(csv_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        flat_rows = []
        for row in rows:
            policy_stats = row.pop("policy_stats")
            for key, value in policy_stats.items():
                row[f"policy_{key}"] = value
            flat_rows.append(row)
        fieldnames = sorted({key for row in flat_rows for key in row})
        with path.open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(flat_rows)


def _print_summary(results: list[BenchmarkResult]) -> None:
    print("\nCollector throughput summary")
    print("=" * 132)
    print(
        f"{'collector':<28} {'backend':<16} {'batch rule':<18} "
        f"{'envs':>5} {'status':<8} {'fps':>10} {'drv_ms/f':>9} {'avg_bs':>8} "
        f"{'p95_q_ms':>10} {'p95_fwd_ms':>11} failure"
    )
    print("-" * 132)
    for result in results:
        stats = result.policy_stats
        print(
            f"{result.collector:<28} {result.backend:<16} "
            f"{result.batch_rule:<18} {result.num_envs:>5} "
            f"{result.status:<8} {result.frames_per_s:>10.1f} "
            f"{result.driver_cpu_ms_per_frame:>9.3f} "
            f"{float(stats.get('avg_batch_size', 0.0)):>8.2f} "
            f"{float(stats.get('p95_queue_ms', 0.0)):>10.2f} "
            f"{float(stats.get('p95_forward_ms', 0.0)):>11.2f} "
            f"{result.failure[:60]}"
        )
    print("=" * 132)


def _resolve_batching_rule(rule: str, num_envs: int) -> tuple[int, int, float, str]:
    if rule == "no-batch":
        return 1, 1, 0.001, "max_bs=1"
    if rule == "auto":
        return num_envs, 1, 0.001, f"max_bs={num_envs}, min_bs=1"
    if rule == "min4":
        min_batch_size = min(4, num_envs)
        return (
            num_envs,
            min_batch_size,
            0.01,
            (f"max_bs={num_envs}, min_bs={min_batch_size}"),
        )
    if rule == "full":
        return num_envs, num_envs, 0.01, f"max_bs={num_envs}, min_bs={num_envs}"
    raise ValueError(
        f"Unknown batching rule {rule!r}. "
        "Expected one of no-batch, auto, min4, full."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--env", default="fake-pixels", choices=["fake-pixels", "gym-mujoco"]
    )
    parser.add_argument("--gym-id", default="HalfCheetah-v4")
    parser.add_argument("--from-pixels", action="store_true")
    parser.add_argument(
        "--mujoco-gl",
        default=None,
        help="Set MUJOCO_GL/PYOPENGL_PLATFORM, e.g. egl",
    )
    parser.add_argument("--num-envs", default="1,2,4")
    parser.add_argument("--envs-per-worker", type=int, default=1)
    parser.add_argument(
        "--backends",
        default=("parallel,async-thread,async-env-mp,async-process,async-process-slot"),
        help=(
            "Comma-separated: parallel, async-thread, async-env-mp, "
            "async-process, async-process-slot"
        ),
    )
    parser.add_argument(
        "--batching-rules",
        default="auto",
        help=(
            "Comma-separated async batching rules: no-batch, auto, min4, full. "
            "ParallelEnv is run once with its synchronous barrier."
        ),
    )
    parser.add_argument("--frames-per-batch", type=int, default=200)
    parser.add_argument("--total-frames", type=int, default=1000)
    parser.add_argument("--warmup-batches", type=int, default=1)
    parser.add_argument("--env-step-latency-ms", type=float, default=1.0)
    parser.add_argument("--policy-delay-ms", type=float, default=0.0)
    parser.add_argument(
        "--policy-hidden-features",
        type=int,
        default=256,
        help="Width of each hidden policy layer.",
    )
    parser.add_argument(
        "--policy-hidden-layers",
        type=int,
        default=1,
        help="Number of hidden policy layers.",
    )
    parser.add_argument(
        "--env-exchange",
        default="queue",
        choices=["queue", "shm", "auto"],
        help="AsyncEnvPool exchange used by multiprocessing env backends.",
    )
    parser.add_argument(
        "--replay-mode",
        choices=["none", "iterator", "background"],
        default="none",
        help=(
            "Replay write mode for async-env-mp and async-process-slot; iterator "
            "and background use identical storage."
        ),
    )
    parser.add_argument(
        "--transition-chunk-size",
        type=int,
        default=1,
        help=(
            "Transitions each environment process accumulates per message with "
            "async-process-slot; 1 sends every transition on its own."
        ),
    )
    parser.add_argument("--policy-device", default="auto")
    parser.add_argument("--output-device", default="cpu")
    parser.add_argument("--jsonl", default="bench_collectors_results.jsonl")
    parser.add_argument("--csv", default="bench_collectors_results.csv")
    args = parser.parse_args()

    if args.mujoco_gl:
        os.environ["MUJOCO_GL"] = args.mujoco_gl
        os.environ["PYOPENGL_PLATFORM"] = args.mujoco_gl

    policy_device = (
        _auto_policy_device() if args.policy_device == "auto" else args.policy_device
    )
    output_device = args.output_device
    num_env_values = _parse_int_list(args.num_envs)
    backends = _parse_backend_list(args.backends)
    batching_rules = _parse_backend_list(args.batching_rules)

    env_factory = EnvFactory(
        env=args.env,
        gym_id=args.gym_id,
        from_pixels=args.from_pixels or args.env == "fake-pixels",
        step_latency_s=args.env_step_latency_ms / 1000.0,
    )
    available, reason, action_dim = _check_optional_env(env_factory)
    env_factory.action_dim = action_dim
    policy_factory = PolicyFactory(
        action_dim=action_dim,
        delay_s=args.policy_delay_ms / 1000.0,
        from_pixels=env_factory.from_pixels,
        hidden_features=args.policy_hidden_features,
        hidden_layers=args.policy_hidden_layers,
    )

    results: list[BenchmarkResult] = []
    if not available:
        for num_envs in num_env_values:
            for backend in backends:
                results.append(
                    BenchmarkResult(
                        collector=backend,
                        backend=backend,
                        batch_rule="",
                        env=args.env,
                        num_envs=num_envs,
                        frames_per_batch=args.frames_per_batch,
                        total_frames=args.total_frames,
                        policy_device=policy_device,
                        output_device=output_device,
                        env_step_latency_ms=args.env_step_latency_ms,
                        policy_delay_ms=args.policy_delay_ms,
                        status="skipped",
                        failure=reason,
                    )
                )
        _print_summary(results)
        _write_results(results, args.jsonl, args.csv)
        return

    for num_envs in num_env_values:
        for backend in backends:
            if backend == "parallel":
                results.append(
                    bench(
                        name="Collector + ParallelEnv",
                        backend=backend,
                        batch_rule="sync barrier",
                        factory=lambda num_envs=num_envs: Collector(
                            create_env_fn=ParallelEnv(num_envs, env_factory),
                            policy=policy_factory(),
                            frames_per_batch=args.frames_per_batch,
                            total_frames=-1,
                            policy_device=policy_device,
                            env_device=output_device,
                            storing_device=output_device,
                            trust_policy=True,
                            use_buffers=False,
                            auto_register_policy_transforms=False,
                        ),
                        env_name=args.env,
                        num_envs=num_envs,
                        frames_per_batch=args.frames_per_batch,
                        total_frames=args.total_frames,
                        policy_device=policy_device,
                        output_device=output_device,
                        env_step_latency_ms=args.env_step_latency_ms,
                        policy_delay_ms=args.policy_delay_ms,
                        warmup_batches=args.warmup_batches,
                    )
                )
            elif backend == "async-thread":
                for rule in batching_rules:
                    (
                        max_batch_size,
                        min_batch_size,
                        timeout,
                        label,
                    ) = _resolve_batching_rule(rule, num_envs)
                    results.append(
                        bench(
                            name="AsyncBatched thread env",
                            backend=backend,
                            batch_rule=label,
                            factory=lambda num_envs=num_envs, max_batch_size=max_batch_size, min_batch_size=min_batch_size, timeout=timeout: AsyncBatchedCollector(
                                create_env_fn=[env_factory] * num_envs,
                                policy=policy_factory(),
                                frames_per_batch=args.frames_per_batch,
                                total_frames=-1,
                                env_backend="threading",
                                server_config=InferenceServerConfig(
                                    service_backend="thread",
                                    max_batch_size=max_batch_size,
                                    min_batch_size=min_batch_size,
                                    timeout=timeout,
                                ),
                                device_config=InferenceDeviceConfig(
                                    policy_device=policy_device,
                                    output_device=output_device,
                                ),
                            ),
                            env_name=args.env,
                            num_envs=num_envs,
                            frames_per_batch=args.frames_per_batch,
                            total_frames=args.total_frames,
                            policy_device=policy_device,
                            output_device=output_device,
                            env_step_latency_ms=args.env_step_latency_ms,
                            policy_delay_ms=args.policy_delay_ms,
                            warmup_batches=args.warmup_batches,
                        )
                    )
            elif backend == "async-env-mp":
                for rule in batching_rules:
                    (
                        max_batch_size,
                        min_batch_size,
                        timeout,
                        label,
                    ) = _resolve_batching_rule(rule, num_envs)
                    results.append(
                        bench(
                            name=f"AsyncBatched mp env ({args.env_exchange}, replay={args.replay_mode})",
                            backend=f"{backend}-{args.env_exchange}-replay-{args.replay_mode}",
                            replay_mode=args.replay_mode,
                            batch_rule=label,
                            factory=lambda num_envs=num_envs, max_batch_size=max_batch_size, min_batch_size=min_batch_size, timeout=timeout: AsyncBatchedCollector(
                                create_env_fn=[env_factory] * num_envs,
                                policy=policy_factory(),
                                frames_per_batch=args.frames_per_batch,
                                total_frames=-1,
                                env_backend="multiprocessing",
                                env_exchange=args.env_exchange,
                                envs_per_worker=args.envs_per_worker,
                                server_config=InferenceServerConfig(
                                    service_backend="thread",
                                    max_batch_size=max_batch_size,
                                    min_batch_size=min_batch_size,
                                    timeout=timeout,
                                ),
                                device_config=InferenceDeviceConfig(
                                    policy_device=policy_device,
                                    output_device=output_device,
                                ),
                            ),
                            env_name=args.env,
                            num_envs=num_envs,
                            frames_per_batch=args.frames_per_batch,
                            total_frames=args.total_frames,
                            policy_device=policy_device,
                            output_device=output_device,
                            env_step_latency_ms=args.env_step_latency_ms,
                            policy_delay_ms=args.policy_delay_ms,
                            warmup_batches=args.warmup_batches,
                            envs_per_worker=args.envs_per_worker,
                        )
                    )
            elif backend == "async-process":
                for rule in batching_rules:
                    (
                        max_batch_size,
                        min_batch_size,
                        timeout,
                        label,
                    ) = _resolve_batching_rule(rule, num_envs)
                    results.append(
                        bench(
                            name=(
                                "AsyncBatched process server " f"({args.env_exchange})"
                            ),
                            backend=f"{backend}-{args.env_exchange}",
                            batch_rule=label,
                            factory=lambda num_envs=num_envs, max_batch_size=max_batch_size, min_batch_size=min_batch_size, timeout=timeout: AsyncBatchedCollector(
                                create_env_fn=[env_factory] * num_envs,
                                policy_factory=policy_factory,
                                frames_per_batch=args.frames_per_batch,
                                total_frames=-1,
                                env_backend="multiprocessing",
                                env_exchange=args.env_exchange,
                                server_config=InferenceServerConfig(
                                    service_backend="process",
                                    max_batch_size=max_batch_size,
                                    min_batch_size=min_batch_size,
                                    timeout=timeout,
                                ),
                                device_config=InferenceDeviceConfig(
                                    policy_device=policy_device,
                                    output_device=output_device,
                                ),
                            ),
                            env_name=args.env,
                            num_envs=num_envs,
                            frames_per_batch=args.frames_per_batch,
                            total_frames=args.total_frames,
                            policy_device=policy_device,
                            output_device=output_device,
                            env_step_latency_ms=args.env_step_latency_ms,
                            policy_delay_ms=args.policy_delay_ms,
                            warmup_batches=args.warmup_batches,
                        )
                    )
            elif backend == "async-process-slot":
                for rule in batching_rules:
                    (
                        max_batch_size,
                        min_batch_size,
                        timeout,
                        label,
                    ) = _resolve_batching_rule(rule, num_envs)
                    results.append(
                        bench(
                            name=(
                                "AsyncBatched direct process slots (chunk="
                                f"{args.transition_chunk_size}, replay={args.replay_mode})"
                            ),
                            backend=(
                                f"{backend}-chunk-{args.transition_chunk_size}"
                                f"-replay-{args.replay_mode}"
                            ),
                            replay_mode=args.replay_mode,
                            batch_rule=label,
                            factory=ft.partial(
                                AsyncBatchedCollector,
                                create_env_fn=[env_factory] * num_envs,
                                policy_factory=policy_factory,
                                transport=_make_process_slot_transport(
                                    env_factory, num_envs
                                ),
                                frames_per_batch=args.frames_per_batch,
                                total_frames=-1,
                                env_backend="multiprocessing",
                                transition_chunk_size=args.transition_chunk_size,
                                server_config=InferenceServerConfig(
                                    service_backend="process",
                                    max_batch_size=max_batch_size,
                                    min_batch_size=min_batch_size,
                                    timeout=timeout,
                                ),
                                device_config=InferenceDeviceConfig(
                                    policy_device=policy_device,
                                    output_device=output_device,
                                ),
                            ),
                            env_name=args.env,
                            num_envs=num_envs,
                            frames_per_batch=args.frames_per_batch,
                            total_frames=args.total_frames,
                            policy_device=policy_device,
                            output_device=output_device,
                            env_step_latency_ms=args.env_step_latency_ms,
                            policy_delay_ms=args.policy_delay_ms,
                            warmup_batches=args.warmup_batches,
                        )
                    )
            else:
                results.append(
                    BenchmarkResult(
                        collector=backend,
                        backend=backend,
                        batch_rule="",
                        env=args.env,
                        num_envs=num_envs,
                        frames_per_batch=args.frames_per_batch,
                        total_frames=args.total_frames,
                        policy_device=policy_device,
                        output_device=output_device,
                        env_step_latency_ms=args.env_step_latency_ms,
                        policy_delay_ms=args.policy_delay_ms,
                        status="skipped",
                        failure=f"unknown backend {backend!r}",
                    )
                )

    _print_summary(results)
    _write_results(results, args.jsonl, args.csv)


if __name__ == "__main__":
    main()
