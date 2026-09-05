# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Benchmark complete DreamerV3 learner updates after warmup.

The workload uses the maintained DMC Walker configuration and includes all
model, actor, value, and replay-value losses, backward, the optimizer step,
and the slow-value-target update. Replay sampling and environment collection
are intentionally outside the learner timing.

Example::

    python benchmarks/ad_hoc/bench_dreamer_v3_learner.py
"""

from __future__ import annotations

import argparse
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
            },
        },
        [batch, steps],
        device=device,
    )


def _measure(
    learner_update,
    data: TensorDict,
    *,
    device: torch.device,
    warmup: int,
    iterations: int,
) -> list[float]:
    for _ in range(warmup):
        learner_update(data)
    torch.cuda.synchronize(device)
    samples = []
    for _ in range(iterations):
        started = time.perf_counter()
        learner_update(data)
        torch.cuda.synchronize(device)
        samples.append((time.perf_counter() - started) * 1000)
    return samples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--unroll", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=("compiled_scan", "cuda_graph"),
        default=("compiled_scan", "cuda_graph"),
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
        cfg.optimization.compile_rssm = "scan"
        cfg.optimization.rssm_scan_unroll = args.unroll
        cfg.optimization.cudagraph_train_step = variant == "cuda_graph"
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
        samples = _measure(
            learner_update,
            data.clone(),
            device=device,
            warmup=args.warmup,
            iterations=args.iterations,
        )
        median_ms = statistics.median(samples)
        result = {
            "variant": variant,
            "workload": "complete_learner_update",
            "device": torch.cuda.get_device_name(device),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "batch": args.batch,
            "steps": args.steps,
            "unroll": args.unroll,
            "warmup": args.warmup,
            "iterations": args.iterations,
            "mixed_precision": cfg.optimization.mixed_precision,
            "median_ms": median_ms,
            "transitions_per_second": args.batch * args.steps * 1000 / median_ms,
            "min_ms": min(samples),
            "max_ms": max(samples),
        }
        print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
