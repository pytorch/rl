# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Benchmark eager and static CUDA-graph DreamerV3 policy serving.

The default workload builds a 170.8M-parameter recurrent DreamerV3 acting
policy and measures batches of 1, 16, and 64 requests. Both server forward
latency and end-to-end batch latency include synchronized CPU results; capture
and warm-up are excluded.

Example::

    python benchmarks/ad_hoc/bench_inference_server_cudagraph.py
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
from tensordict.nn.probabilistic import InteractionType, set_interaction_type

from torchrl.modules.inference_server import (
    InferenceServer,
    PolicyClientModule,
    ThreadingTransport,
)


def _load_agent(repo_root: Path) -> dict:
    example_dir = repo_root / "sota-implementations/dreamer_v3"
    sys.path.insert(0, str(example_dir))
    return runpy.run_path(
        example_dir / "dreamer_v3_agent.py",
        run_name="dreamer_v3_inference_benchmark",
    )


def _make_policy(
    repo_root: Path,
    *,
    device: torch.device,
    observation_dim: int,
    action_dim: int,
    rnn_hidden_dim: int,
):
    example_dir = repo_root / "sota-implementations/dreamer_v3"
    cfg = OmegaConf.load(example_dir / "config.yaml")
    cfg.networks.rnn_hidden_dim = rnn_hidden_dim
    cfg.networks.num_categoricals = 32
    cfg.networks.num_classes = 64
    cfg.networks.hidden_dim = 1024
    cfg.networks.actor_layers = 3
    agent = _load_agent(repo_root)
    world_model, *_ = agent["build_world_model"](
        cfg=cfg,
        obs_dim=observation_dim,
        action_dim=action_dim,
    )
    actor = agent["build_actor"](cfg=cfg, action_dim=action_dim)
    policy = agent["build_real_world_actor"](
        world_model=world_model,
        actor_model=actor,
        mixed_precision=True,
    )
    return policy.to(device), cfg


def _make_request(
    cfg,
    *,
    observation_dim: int,
    action_dim: int,
) -> TensorDict:
    state_dim = cfg.networks.num_categoricals * cfg.networks.num_classes
    return TensorDict(
        {
            "observation": torch.randn(observation_dim),
            "state": torch.zeros(state_dim),
            "belief": torch.zeros(cfg.networks.rnn_hidden_dim),
            "previous_action": torch.zeros(action_dim),
            "is_init": torch.zeros(1, dtype=torch.bool),
        }
    )


def _run_batch(client: PolicyClientModule, request: TensorDict, batch_size: int):
    futures = [client.submit(request.clone()) for _ in range(batch_size)]
    return [future.result() for future in futures]


def _measure(
    policy,
    request: TensorDict,
    *,
    batch_size: int,
    static_batch_size: int | None,
    warmup: int,
    iterations: int,
) -> tuple[dict[str, float | int], list[float]]:
    transport = ThreadingTransport()
    server = InferenceServer(
        policy,
        transport,
        max_batch_size=static_batch_size or batch_size,
        static_batch_size=static_batch_size,
        min_batch_size=batch_size,
        timeout=1.0,
        request_spec=request if static_batch_size is not None else None,
        policy_device="cuda:0",
        output_device="cpu",
        stats_window_size=iterations,
    )
    client = PolicyClientModule(transport)
    roundtrip_ms = []
    with set_interaction_type(InteractionType.RANDOM):
        with server:
            for _ in range(warmup):
                _run_batch(client, request, batch_size)
            server.stats(reset=True)
            for _ in range(iterations):
                started = time.perf_counter()
                _run_batch(client, request, batch_size)
                roundtrip_ms.append((time.perf_counter() - started) * 1000)
            stats = server.stats()
    return stats, roundtrip_ms


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=(1, 16, 64))
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--observation-dim", type=int, default=1024)
    parser.add_argument("--action-dim", type=int, default=20)
    parser.add_argument("--rnn-hidden-dim", type=int, default=11840)
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=("eager", "cuda_graph"),
        default=("eager", "cuda_graph"),
    )
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires CUDA.")

    repo_root = Path(__file__).parents[2]
    device = torch.device("cuda:0")
    static_batch_size = max(args.batch_sizes)
    torch.manual_seed(0)
    torch.set_float32_matmul_precision("high")

    for variant in args.variants:
        policy, cfg = _make_policy(
            repo_root,
            device=device,
            observation_dim=args.observation_dim,
            action_dim=args.action_dim,
            rnn_hidden_dim=args.rnn_hidden_dim,
        )
        parameter_count = sum(parameter.numel() for parameter in policy.parameters())
        request = _make_request(
            cfg,
            observation_dim=args.observation_dim,
            action_dim=args.action_dim,
        )
        for batch_size in args.batch_sizes:
            stats, roundtrip_ms = _measure(
                policy,
                request,
                batch_size=batch_size,
                static_batch_size=(
                    static_batch_size if variant == "cuda_graph" else None
                ),
                warmup=args.warmup,
                iterations=args.iterations,
            )
            result = {
                "variant": variant,
                "device": torch.cuda.get_device_name(device),
                "parameters": parameter_count,
                "batch_size": batch_size,
                "static_batch_size": (
                    static_batch_size if variant == "cuda_graph" else None
                ),
                "iterations": args.iterations,
                "p50_forward_ms": stats["p50_forward_ms"],
                "p95_forward_ms": stats["p95_forward_ms"],
                "p50_roundtrip_ms": statistics.median(roundtrip_ms),
                "p95_roundtrip_ms": sorted(roundtrip_ms)[
                    round(0.95 * (len(roundtrip_ms) - 1))
                ],
            }
            print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
