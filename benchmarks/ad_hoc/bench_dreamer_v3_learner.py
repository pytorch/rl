# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Benchmark complete DreamerV3 learner updates after warmup.

The workload uses the maintained DMC Walker configuration and includes all
model, actor, value, and replay-value losses, backward, the optimizer step,
and the slow-value-target update. The optional replay workload uses the same
native replay construction, prefetch, and conditional writeback as training.

Example::

    python benchmarks/ad_hoc/bench_dreamer_v3_learner.py
"""

from __future__ import annotations

import argparse
import functools as ft
import json
import runpy
import statistics
import sys
import time
from pathlib import Path

import torch
from omegaconf import OmegaConf
from tensordict import TensorDict


def _load_example(repo_root: Path) -> dict:
    example_dir = repo_root / "sota-implementations/dreamer_v3"
    sys.path.insert(0, str(example_dir))
    return runpy.run_path(
        example_dir / "train.py",
        run_name="dreamer_v3_learner_benchmark",
    )


def _load_config(repo_root: Path):
    example_dir = repo_root / "sota-implementations/dreamer_v3"
    base = OmegaConf.load(example_dir / "config.yaml")
    walker = OmegaConf.load(example_dir / "config_dmc_walker.yaml")
    del walker.defaults
    return OmegaConf.merge(base, walker)


def _make_data(
    cfg,
    device: torch.device,
    *,
    batch: int,
    steps: int,
    obs_dim: int,
    action_dim: int,
) -> TensorDict:
    state_dim = cfg.networks.num_categoricals * cfg.networks.num_classes
    return TensorDict(
        {
            "state": torch.zeros(batch, steps, state_dim, device=device),
            "belief": torch.zeros(
                batch, steps, cfg.networks.rnn_hidden_dim, device=device
            ),
            "action": torch.randn(batch, steps, action_dim, device=device),
            "is_init": torch.zeros(batch, steps, 1, dtype=torch.bool, device=device),
            "next": {
                "observation": torch.randn(batch, steps, obs_dim, device=device),
                "reward": torch.randn(batch, steps, 1, device=device),
                "done": torch.zeros(batch, steps, 1, dtype=torch.bool, device=device),
                "terminated": torch.zeros(
                    batch, steps, 1, dtype=torch.bool, device=device
                ),
                "truncated": torch.zeros(
                    batch, steps, 1, dtype=torch.bool, device=device
                ),
            },
        },
        [batch, steps],
        device=device,
    )


class _ReplayLearnerStep:
    def __init__(
        self,
        example: dict,
        cfg,
        learner_update,
        *,
        device: torch.device,
        replay_device: torch.device,
        obs_dim: int,
        action_dim: int,
    ):
        self.cfg = cfg
        self.device = device
        self.learner_update = learner_update
        self.replay_context_update = example["replay_context_update"]
        sequence_records = cfg.replay_buffer.seq_len + 1
        num_streams = 4
        records_per_stream = sequence_records * cfg.replay_buffer.batch_size
        cfg.replay_buffer.buffer_size = records_per_stream * num_streams
        cfg.replay_buffer.online = False
        self.replay_buffer = example["_build_replay"](
            cfg, num_streams, replay_device, device
        )
        replay_data = _make_data(
            cfg,
            replay_device,
            batch=num_streams,
            steps=records_per_stream,
            obs_dim=obs_dim,
            action_dim=action_dim,
        )
        replay_data["is_init"][:, 0] = True
        self.replay_buffer.extend(replay_data)

    def __call__(self) -> None:
        sequence_records = self.cfg.replay_buffer.seq_len + 1
        replay_sample = self.replay_buffer.sample().reshape(
            self.cfg.replay_buffer.batch_size, sequence_records
        )
        sample_info = replay_sample.select("index", "index_generation")
        sample = replay_sample.exclude("index", "index_generation")
        sample = sample.to(self.device, non_blocking=True)[:, :-1]
        if self.learner_update.replay_must_be_idle:
            self.replay_buffer.synchronize()
        _, state, belief = self.learner_update(sample)
        if self.cfg.optimization.cudagraph_train_step:
            state = state.clone()
            belief = belief.clone()
        index, generation, patch = self.replay_context_update(
            sample_info, state, belief
        )
        self.replay_buffer.submit_update_if_present(
            index=index, generation=generation, patch=patch
        )

    def synchronize(self) -> None:
        self.replay_buffer.synchronize()
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def close(self) -> None:
        self.replay_buffer.shutdown()


def _measure(step, synchronize, *, warmup: int, iterations: int):
    for _ in range(warmup):
        step()
    synchronize()
    torch.cuda.reset_peak_memory_stats()
    host_latencies = []
    started = time.perf_counter()
    for _ in range(iterations):
        before_update = time.perf_counter()
        step()
        host_latencies.append((time.perf_counter() - before_update) * 1000)
    synchronize()
    return (time.perf_counter() - started) * 1000 / iterations, host_latencies


def _profile_update(step, synchronize) -> dict:
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
    ) as profile:
        step()
        synchronize()

    events = profile.events()
    kernel_events = [
        event
        for event in events
        if event.device_type == torch.autograd.DeviceType.CUDA
        and not event.name.startswith(("Memcpy", "Memset"))
    ]
    launch_events = [
        event
        for event in events
        if event.name.startswith("cudaLaunchKernel") or event.name == "cudaGraphLaunch"
    ]
    return {
        "profile_kernel_count": len(kernel_events),
        "profile_gpu_time_ms": sum(event.device_time_total for event in kernel_events)
        / 1000,
        "profile_cpu_launch_time_ms": sum(
            event.self_cpu_time_total for event in launch_events
        )
        / 1000,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--unroll", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument(
        "--replay-device",
        choices=("cpu", "cuda"),
        help="Include native replay sampling and writeback on this device.",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=(
            "eager",
            "compiled_scan",
            "cuda_graph",
            "compiled_train_step",
            "compiled_train_step_cuda_graph",
        ),
        default=(
            "cuda_graph",
            "compiled_train_step_cuda_graph",
        ),
    )
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires CUDA.")

    repo_root = Path(__file__).parents[2]
    example = _load_example(repo_root)
    template_cfg = _load_config(repo_root)
    device = torch.device("cuda:0")
    torch.set_float32_matmul_precision("high")

    real_env = example["make_env"](template_cfg, template_cfg.env.seed)
    obs_dim = real_env.observation_spec["observation"].shape[0]
    action_dim = real_env.action_spec.shape[0]
    real_env.close()
    torch.manual_seed(1)
    data = _make_data(
        template_cfg,
        device,
        batch=args.batch,
        steps=args.steps,
        obs_dim=obs_dim,
        action_dim=action_dim,
    )

    for variant in args.variants:
        cfg = _load_config(repo_root)
        cfg.replay_buffer.batch_size = args.batch
        cfg.replay_buffer.seq_len = args.steps
        cfg.optimization.compile_rssm = (
            "scan" if variant in ("compiled_scan", "cuda_graph") else None
        )
        cfg.optimization.rssm_scan_unroll = args.unroll
        cfg.optimization.compile_train_step = variant in (
            "compiled_train_step",
            "compiled_train_step_cuda_graph",
        )
        cfg.optimization.cudagraph_train_step = variant in (
            "cuda_graph",
            "compiled_train_step_cuda_graph",
        )
        torch.manual_seed(0)
        learner = example["_build_learner"](
            cfg,
            device,
            obs_dim,
            action_dim,
        )
        learner_update = example["_LearnerUpdate"](
            cfg,
            device,
            learner,
            cudagraph_warmup=args.warmup,
        )
        if cfg.optimization.separate_policy_rng:
            torch.manual_seed(
                example["stream_seed"](
                    cfg.env.seed,
                    0,
                    example["LEARNER_RNG_STREAM"],
                )
            )
        replay_step = None
        if args.replay_device is None:
            step = ft.partial(learner_update, data.clone())
            synchronize = ft.partial(torch.cuda.synchronize, device)
            workload = "complete_learner_update"
        else:
            replay_step = _ReplayLearnerStep(
                example,
                cfg,
                learner_update,
                device=device,
                replay_device=torch.device(args.replay_device),
                obs_dim=obs_dim,
                action_dim=action_dim,
            )
            step = replay_step
            synchronize = replay_step.synchronize
            workload = "complete_learner_update_with_replay"
        try:
            mean_ms, host_latencies = _measure(
                step,
                synchronize,
                warmup=args.warmup,
                iterations=args.iterations,
            )
            peak_memory = torch.cuda.max_memory_allocated(device) / 2**20
            profile_metrics = _profile_update(step, synchronize)
        finally:
            if replay_step is not None:
                replay_step.close()
        result = {
            "variant": variant,
            "workload": workload,
            "device": torch.cuda.get_device_name(device),
            "replay_device": args.replay_device,
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "batch": args.batch,
            "steps": args.steps,
            "unroll": args.unroll,
            "warmup": args.warmup,
            "iterations": args.iterations,
            "mixed_precision": cfg.optimization.mixed_precision,
            "warmup_updates": args.warmup,
            "measured_updates": args.iterations,
            "mean_update_ms": mean_ms,
            "transitions_per_second": args.batch * args.steps * 1000 / mean_ms,
            "p50_host_update_ms": statistics.median(host_latencies),
            "p95_host_update_ms": statistics.quantiles(
                host_latencies, n=20, method="inclusive"
            )[18],
            "peak_cuda_allocated_mib": peak_memory,
            **profile_metrics,
        }
        print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
