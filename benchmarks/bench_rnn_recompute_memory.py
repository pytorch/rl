# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Compare peak GPU memory for ``recurrent_recompute='none'`` vs ``'full'``.

Runs forward + backward on :class:`~torchrl.modules.LSTMModule` /
:class:`~torchrl.modules.GRUModule` with each ``recurrent_backend`` that
supports the ``recurrent_recompute`` knob (``"scan"`` and ``"triton"``), and
reports the peak allocated memory delta the knob produces at the Isaac
training shape ``[B=4096, T=32, H=256]`` by default.

Example::

    python benchmarks/bench_rnn_recompute_memory.py \
        --rnn lstm --backend triton --batch 4096 --seq-len 32 --hidden 256

The script is no-op on CPU/MPS systems (memory metrics are CUDA-only).
"""
from __future__ import annotations

import argparse
from typing import Literal

import torch
from tensordict import TensorDict

from torchrl import cuda_memory_stats, reset_cuda_peak_stats
from torchrl.modules import GRUModule, LSTMModule


RNNType = Literal["lstm", "gru"]
Backend = Literal["scan", "triton"]


def _build_module(
    rnn_type: RNNType,
    backend: Backend,
    recompute: str,
    *,
    input_size: int,
    hidden_size: int,
    num_layers: int,
    device: torch.device,
) -> LSTMModule | GRUModule:
    kwargs: dict = {
        "input_size": input_size,
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "recurrent_backend": backend,
        "recurrent_recompute": recompute,
        "default_recurrent_mode": True,
        "device": device,
    }
    if rnn_type == "lstm":
        return LSTMModule(
            in_keys=["obs", "hidden0", "hidden1"],
            out_keys=["feat", ("next", "hidden0"), ("next", "hidden1")],
            **kwargs,
        )
    return GRUModule(
        in_keys=["obs", "hidden"],
        out_keys=["feat", ("next", "hidden")],
        **kwargs,
    )


def _build_inputs(
    rnn_type: RNNType,
    *,
    batch: int,
    seq_len: int,
    input_size: int,
    hidden_size: int,
    num_layers: int,
    device: torch.device,
) -> TensorDict:
    obs = torch.randn(batch, seq_len, input_size, device=device, requires_grad=True)
    is_init = torch.zeros(batch, seq_len, 1, dtype=torch.bool, device=device)
    is_init[:, 0] = True
    if rnn_type == "lstm":
        hidden0 = torch.zeros(batch, seq_len, num_layers, hidden_size, device=device)
        hidden1 = torch.zeros_like(hidden0)
        return TensorDict(
            {"obs": obs, "hidden0": hidden0, "hidden1": hidden1, "is_init": is_init},
            [batch, seq_len],
        )
    hidden = torch.zeros(batch, seq_len, num_layers, hidden_size, device=device)
    return TensorDict(
        {"obs": obs, "hidden": hidden, "is_init": is_init}, [batch, seq_len]
    )


def _run_one(
    rnn_type: RNNType,
    backend: Backend,
    recompute: str,
    *,
    batch: int,
    seq_len: int,
    input_size: int,
    hidden_size: int,
    num_layers: int,
    device: torch.device,
) -> dict[str, float]:
    module = _build_module(
        rnn_type,
        backend,
        recompute,
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        device=device,
    )
    data = _build_inputs(
        rnn_type,
        batch=batch,
        seq_len=seq_len,
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        device=device,
    )
    # Warmup: trigger any lazy compile so the measured peak excludes one-shot
    # compile workspaces.
    out = module(data.clone())
    out["feat"].pow(2).sum().backward()
    for param in module.parameters():
        if param.grad is not None:
            param.grad = None
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    reset_cuda_peak_stats(device)

    out = module(data.clone())
    loss = out["feat"].pow(2).sum()
    loss.backward()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return cuda_memory_stats(device)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rnn", choices=["lstm", "gru"], default="lstm")
    parser.add_argument(
        "--backend",
        choices=["scan", "triton"],
        default="triton",
    )
    parser.add_argument("--batch", type=int, default=4096)
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--input-size", type=int, default=32)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type != "cuda":
        print(
            "[bench_rnn_recompute_memory] No CUDA device — memory stats are zero "
            "on CPU/MPS. Skipping."
        )
        return

    print(
        f"shape: batch={args.batch} seq_len={args.seq_len} "
        f"hidden={args.hidden} num_layers={args.num_layers}\n"
        f"rnn: {args.rnn}  backend: {args.backend}\n"
    )

    results: dict[str, dict[str, float]] = {}
    for recompute in ("none", "full"):
        results[recompute] = _run_one(
            args.rnn,
            args.backend,
            recompute,
            batch=args.batch,
            seq_len=args.seq_len,
            input_size=args.input_size,
            hidden_size=args.hidden,
            num_layers=args.num_layers,
            device=device,
        )
    none_peak = results["none"]["max_allocated_gb"]
    full_peak = results["full"]["max_allocated_gb"]
    if none_peak == 0.0:
        ratio_str = "n/a"
    else:
        ratio_str = f"{full_peak / none_peak:.2%}"
    print(f"{'recompute':10}  {'max_alloc_gb':>14}  {'max_reserved_gb':>16}")
    for recompute, stats in results.items():
        print(
            f"{recompute:10}  {stats['max_allocated_gb']:>14.3f}  "
            f"{stats['max_reserved_gb']:>16.3f}"
        )
    print(f"\nfull / none peak allocated ratio: {ratio_str}")


if __name__ == "__main__":
    main()
