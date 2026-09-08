# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Throughput benchmark for the process-hosted inference server.

Measures requests per second served by a
:class:`~torchrl.modules.inference_server.ProcessInferenceServer` behind a
:class:`~torchrl.modules.inference_server.ProcessSlotTransport`, the topology
used by :class:`~torchrl.collectors.AsyncBatchedCollector` with process
environment workers. ``--num-clients`` workers (one process each by default)
keep one request in flight at all times, so every server pass sees a full
batch of ``--num-clients`` requests. The per-pass server overhead (collation,
host-device transfers, synchronization and response scattering) dominates when
the policy is small, which is what this script isolates.

The default workload is an MLP policy with 64 clients, i.e. a batch of 64 per
pass. Pass ``--static-batch-size 64`` to serve the policy through a CUDA graph.
Pinned staging, non-blocking transfers and CUDA graphs require CUDA; the script
runs on CPU as a smoke test only.

To compare two revisions, run the same command from a checkout of each and
compare ``requests_per_s``::

    python benchmarks/bench_inference_server.py --device cuda:0
    python benchmarks/bench_inference_server.py --device cuda:0 \
        --static-batch-size 64

Each run prints one JSON line with the throughput and the server's forward
latency percentiles after a warm-up window.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import threading
import time
from dataclasses import dataclass

import torch
from tensordict import TensorDict
from tensordict.nn import TensorDictModule

from torchrl.modules import MLP
from torchrl.modules.inference_server import (
    ProcessInferenceServer,
    ProcessSlotTransport,
)


@dataclass
class MLPPolicyFactory:
    """Picklable policy factory constructed inside the server process."""

    observation_dim: int
    action_dim: int
    hidden_features: int
    hidden_layers: int

    def __call__(self) -> TensorDictModule:
        torch.manual_seed(0)
        return TensorDictModule(
            MLP(
                in_features=self.observation_dim,
                out_features=self.action_dim,
                num_cells=[self.hidden_features] * self.hidden_layers,
            ),
            in_keys=["observation"],
            out_keys=["action"],
        )


def _client_loop(client, observation_dim: int, stop_event) -> None:
    """Keep one request in flight until asked to stop."""
    torch.set_num_threads(1)
    request = TensorDict({"observation": torch.randn(observation_dim)})
    while not stop_event.is_set():
        request["observation"].add_(1.0)
        client(request)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--num-clients", type=int, default=64)
    parser.add_argument("--observation-dim", type=int, default=1024)
    parser.add_argument("--action-dim", type=int, default=20)
    parser.add_argument("--hidden-features", type=int, default=1024)
    parser.add_argument("--hidden-layers", type=int, default=3)
    parser.add_argument(
        "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--static-batch-size",
        type=int,
        default=None,
        help="serve through a CUDA graph with this static batch size "
        "(at least --num-clients).",
    )
    parser.add_argument(
        "--client-backend",
        choices=("process", "thread"),
        default="process",
        help="run clients in spawned processes (as env workers do) or threads.",
    )
    parser.add_argument("--warmup-s", type=float, default=3.0)
    parser.add_argument("--duration-s", type=float, default=10.0)
    args = parser.parse_args()

    ctx = mp.get_context("spawn")
    request_spec = TensorDict({"observation": torch.zeros(args.observation_dim)})
    response_spec = TensorDict(
        {
            "action": torch.zeros(args.action_dim),
            "policy_version": torch.zeros((), dtype=torch.long),
        }
    )
    transport = ProcessSlotTransport(
        request_spec, response_spec, num_slots=args.num_clients, ctx=ctx
    )
    server = ProcessInferenceServer(
        policy_factory=MLPPolicyFactory(
            observation_dim=args.observation_dim,
            action_dim=args.action_dim,
            hidden_features=args.hidden_features,
            hidden_layers=args.hidden_layers,
        ),
        transport=transport,
        request_spec=request_spec,
        max_batch_size=args.num_clients,
        static_batch_size=args.static_batch_size,
        policy_device=args.device,
        output_device="cpu",
        stats_window_size=100_000,
        mp_context=ctx,
    )
    clients = server.clients(args.num_clients)
    stop_event = ctx.Event() if args.client_backend == "process" else threading.Event()
    if args.client_backend == "process":
        workers = [
            ctx.Process(
                target=_client_loop,
                args=(client, args.observation_dim, stop_event),
                daemon=True,
            )
            for client in clients
        ]
    else:
        workers = [
            threading.Thread(
                target=_client_loop,
                args=(client, args.observation_dim, stop_event),
                daemon=True,
            )
            for client in clients
        ]
    with server:
        for worker in workers:
            worker.start()
        time.sleep(args.warmup_s)
        server.stats(reset=True)
        started = time.perf_counter()
        time.sleep(args.duration_s)
        stats = server.stats()
        elapsed = time.perf_counter() - started
        stop_event.set()
        for worker in workers:
            worker.join(timeout=30.0)
    result = {
        "device": args.device,
        "num_clients": args.num_clients,
        "static_batch_size": args.static_batch_size,
        "client_backend": args.client_backend,
        "observation_dim": args.observation_dim,
        "action_dim": args.action_dim,
        "hidden_features": args.hidden_features,
        "hidden_layers": args.hidden_layers,
        "duration_s": round(elapsed, 3),
        "requests": stats["requests"],
        "requests_per_s": round(stats["requests"] / elapsed, 1),
        "batches_per_s": round(stats["batches"] / elapsed, 1),
        "avg_batch_size": round(stats["avg_batch_size"], 2),
        "p50_forward_ms": round(stats["p50_forward_ms"], 3),
        "p95_forward_ms": round(stats["p95_forward_ms"], 3),
        "p50_queue_ms": round(stats["p50_queue_ms"], 3),
    }
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
