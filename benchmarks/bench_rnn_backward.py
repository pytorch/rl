# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Backward-pass benchmark comparing the cuDNN, scan and Triton RNN backends.

Sweeps a grid of batch sizes, horizons (sequence lengths) and cell counts
(hidden sizes) and reports forward time, backward time and peak allocated
memory for each :class:`~torchrl.modules.GRUModule` /
:class:`~torchrl.modules.LSTMModule` ``recurrent_backend``:

* ``cudnn`` -> the ``"pad"`` backend (cuDNN-flattened RNN);
* ``scan``  -> the ``torch._higher_order_ops.scan`` backend;
* ``triton`` -> the fused Triton kernel backend.

All backends run with ``recurrent_recompute="none"`` so the comparison isolates
the backward kernel itself (cuDNN's ``"pad"`` backend does not expose the
recompute knob). Use ``--recompute full`` to instead measure scan/Triton with
backward recomputation enabled (cuDNN is skipped in that mode).

Example::

    python benchmarks/bench_rnn_backward.py --rnn gru \
        --batches 256,1024,4096 --seq-lens 16,32,64 --hiddens 128,256,512

The script is a no-op on CPU/MPS (timings and memory require CUDA).
"""
from __future__ import annotations

import argparse
import statistics
from typing import Literal

import torch
from tensordict import TensorDict

from torchrl import cuda_memory_stats, reset_cuda_peak_stats
from torchrl.modules import GRUModule, LSTMModule

RNNType = Literal["lstm", "gru"]
# User-facing backend name -> recurrent_backend value.
_BACKENDS: dict[str, str] = {"cudnn": "pad", "scan": "scan", "triton": "triton"}


def _build_module(
    rnn_type: RNNType,
    recurrent_backend: str,
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
        "recurrent_backend": recurrent_backend,
        "default_recurrent_mode": True,
        "device": device,
    }
    # The cuDNN ("pad") backend rejects a non-"none" recompute value.
    if recurrent_backend != "pad":
        kwargs["recurrent_recompute"] = recompute
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


def _time_ms(fn, *, iters: int, device: torch.device) -> float:
    """Median wall time (ms) of ``fn`` over ``iters`` CUDA-synchronized runs."""
    samples: list[float] = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(iters):
        torch.cuda.synchronize(device)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize(device)
        samples.append(start.elapsed_time(end))
    return statistics.median(samples)


def _bench_one(
    rnn_type: RNNType,
    recurrent_backend: str,
    recompute: str,
    *,
    batch: int,
    seq_len: int,
    input_size: int,
    hidden_size: int,
    num_layers: int,
    device: torch.device,
    warmup: int,
    iters: int,
) -> dict[str, float]:
    module = _build_module(
        rnn_type,
        recurrent_backend,
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

    def forward():
        return module(data.clone())

    # Warmup (also triggers any lazy autotune / compile workspaces).
    for _ in range(max(warmup, 1)):
        out = forward()
        out["feat"].pow(2).sum().backward()
        for p in module.parameters():
            p.grad = None
    torch.cuda.synchronize(device)

    fwd_ms = _time_ms(lambda: forward()["feat"], iters=iters, device=device)

    def fwd_bwd():
        out = forward()
        out["feat"].pow(2).sum().backward()
        for p in module.parameters():
            p.grad = None

    total_ms = _time_ms(fwd_bwd, iters=iters, device=device)

    reset_cuda_peak_stats(device)
    out = forward()
    out["feat"].pow(2).sum().backward()
    torch.cuda.synchronize(device)
    mem = cuda_memory_stats(device)

    return {
        "fwd_ms": fwd_ms,
        "bwd_ms": max(total_ms - fwd_ms, 0.0),
        "total_ms": total_ms,
        "peak_gb": mem["max_allocated_gb"],
    }


def _parse_int_list(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rnn", choices=["lstm", "gru"], default="gru")
    parser.add_argument(
        "--backends",
        default="cudnn,scan,triton",
        help="Comma list among cudnn,scan,triton.",
    )
    parser.add_argument("--batches", default="256,1024,4096", type=_parse_int_list)
    parser.add_argument("--seq-lens", default="16,32,64", type=_parse_int_list)
    parser.add_argument("--hiddens", default="128,256,512", type=_parse_int_list)
    parser.add_argument("--input-size", type=int, default=32)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--recompute", choices=["none", "full"], default="none")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type != "cuda":
        print("[bench_rnn_backward] CUDA required for timing/memory. Skipping.")
        return

    backends = [b.strip() for b in args.backends.split(",") if b.strip()]
    if args.recompute == "full":
        backends = [b for b in backends if b != "cudnn"]
        print("[bench_rnn_backward] recompute=full -> skipping cuDNN (no recompute).")

    print(
        f"rnn={args.rnn} layers={args.num_layers} input_size={args.input_size} "
        f"recompute={args.recompute} warmup={args.warmup} iters={args.iters}\n"
        f"device={torch.cuda.get_device_name(device)}\n"
    )
    header = (
        f"{'batch':>6} {'T':>4} {'H':>5} {'backend':>8} "
        f"{'fwd_ms':>9} {'bwd_ms':>9} {'total_ms':>9} {'peak_gb':>8}"
    )
    print(header)
    print("-" * len(header))
    for batch in args.batches:
        for seq_len in args.seq_lens:
            for hidden in args.hiddens:
                for name in backends:
                    recurrent_backend = _BACKENDS[name]
                    try:
                        r = _bench_one(
                            args.rnn,
                            recurrent_backend,
                            args.recompute,
                            batch=batch,
                            seq_len=seq_len,
                            input_size=args.input_size,
                            hidden_size=hidden,
                            num_layers=args.num_layers,
                            device=device,
                            warmup=args.warmup,
                            iters=args.iters,
                        )
                        print(
                            f"{batch:>6} {seq_len:>4} {hidden:>5} {name:>8} "
                            f"{r['fwd_ms']:>9.3f} {r['bwd_ms']:>9.3f} "
                            f"{r['total_ms']:>9.3f} {r['peak_gb']:>8.3f}"
                        )
                    except Exception as exc:  # noqa: BLE001
                        print(
                            f"{batch:>6} {seq_len:>4} {hidden:>5} {name:>8} "
                            f"  ERROR: {type(exc).__name__}: {str(exc)[:80]}"
                        )
                    finally:
                        if device.type == "cuda":
                            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
