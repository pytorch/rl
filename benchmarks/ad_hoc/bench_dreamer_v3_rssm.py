# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Benchmark DreamerV3 RSSM recurrent and sequence backends after warmup.

The default dimensions match the DMC Walker reproduction configuration. The
benchmark measures synchronized forward/backward updates for compiled scan and
CUDA-graph-captured loop/scan variants. Compilation and capture warmup are
intentionally excluded.

Example::

    python benchmarks/ad_hoc/bench_dreamer_v3_rssm.py
"""

from __future__ import annotations

import argparse
import functools as ft
import json
import statistics
import time
from collections.abc import Callable
from typing import Literal

import torch
from tensordict import TensorDict
from tensordict.nn import CudaGraphModule, TensorDictModule

from torchrl.modules.models import RSSMPosteriorV3, RSSMPriorV3, RSSMRolloutV3


def _make_rollout(
    device: torch.device,
    recurrent_backend: Literal["reference", "triton"],
) -> RSSMRolloutV3:
    prior = RSSMPriorV3(
        action_shape=torch.Size([6]),
        hidden_dim=64,
        rnn_hidden_dim=512,
        num_categoricals=32,
        num_classes=4,
        action_dim=6,
        recurrent_model="block_gru",
        recurrent_backend=recurrent_backend,
        num_blocks=8,
        num_layers=1,
        prior_num_layers=2,
        unimix=0.01,
    )
    posterior = RSSMPosteriorV3(
        hidden_dim=64,
        num_categoricals=32,
        num_classes=4,
        rnn_hidden_dim=512,
        obs_embed_dim=64,
        use_rms_norm=True,
        num_layers=1,
        unimix=0.01,
    )
    return RSSMRolloutV3(
        TensorDictModule(
            prior,
            in_keys=["state", "belief", "action"],
            out_keys=[
                ("next", "prior_logits"),
                ("next", "state"),
                ("next", "belief"),
            ],
        ),
        TensorDictModule(
            posterior,
            in_keys=[("next", "belief"), ("next", "encoded_latents")],
            out_keys=[("next", "posterior_logits"), ("next", "state")],
        ),
        reset_key="is_init",
    ).to(device)


def _make_data(device: torch.device, batch: int, steps: int) -> TensorDict:
    return TensorDict(
        {
            "state": torch.zeros(batch, steps, 128, device=device),
            "belief": torch.zeros(batch, steps, 512, device=device),
            "action": torch.randn(batch, steps, 6, device=device),
            "is_init": torch.zeros(batch, steps, 1, dtype=torch.bool, device=device),
            "next": {"encoded_latents": torch.randn(batch, steps, 64, device=device)},
        },
        [batch, steps],
        device=device,
    )


def _train_step(rollout: RSSMRolloutV3, value: TensorDict) -> torch.Tensor:
    rollout.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        output = rollout(value)
        loss = (
            output["next", "posterior_logits"].float().square().mean()
            + output["next", "prior_logits"].float().square().mean()
            + 1e-4 * output["next", "belief"].float().square().mean()
        )
    loss.backward()
    return loss.detach()


def _measure(
    step: Callable[[], object],
    *,
    device: torch.device,
    warmup: int,
    iterations: int,
) -> list[float]:
    for _ in range(warmup):
        step()
    torch.cuda.synchronize(device)
    samples = []
    for _ in range(iterations):
        started = time.perf_counter()
        step()
        torch.cuda.synchronize(device)
        samples.append((time.perf_counter() - started) * 1000)
    return samples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--unroll", type=int, default=8)
    parser.add_argument(
        "--recurrent-backends",
        nargs="+",
        choices=("reference", "triton"),
        default=("reference", "triton"),
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=50)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires CUDA.")

    torch.manual_seed(0)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda:0")
    data = _make_data(device, args.batch, args.steps)
    for recurrent_backend in args.recurrent_backends:
        for sequence_backend in ("loop", "scan"):
            rollout = _make_rollout(device, recurrent_backend)
            if sequence_backend == "scan":
                rollout.compile_rollout("scan", unroll=args.unroll)
            train_step = ft.partial(_train_step, rollout)

            variants: dict[str, Callable[[], object]] = {}
            if sequence_backend == "scan":
                variants["compiled_scan"] = ft.partial(train_step, data)
            graphed_step = CudaGraphModule(
                train_step, warmup=args.warmup, device=device
            )
            variants[f"cuda_graph_{sequence_backend}"] = ft.partial(graphed_step, data)

            for name, step in variants.items():
                samples = _measure(
                    step,
                    device=device,
                    warmup=args.warmup,
                    iterations=args.iterations,
                )
                median_ms = statistics.median(samples)
                result = {
                    "variant": name,
                    "recurrent_backend": recurrent_backend,
                    "batch": args.batch,
                    "steps": args.steps,
                    "unroll": args.unroll,
                    "median_ms": median_ms,
                    "transitions_per_second": args.batch
                    * args.steps
                    * 1000
                    / median_ms,
                    "min_ms": min(samples),
                    "max_ms": max(samples),
                }
                print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
