# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Forward/backward benchmark for recurrent backends.

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

Representative DreamerV3 sweep::

    python benchmarks/bench_rnn_backward.py --rnn block_gru \
        --backends scan,triton --batches 16 --seq-lens 64,512 \
        --hiddens 512 --input-size 512 --projection-size 512 --blocks 8 \
        --dtype bfloat16 --compile-modes default,reduce-overhead,max-autotune

The block-GRU grid compares its reference, specialized scan, and fused Triton
backends across batch size, horizon, hidden width, and block count. Timings are
CUDA-synchronized means with 95% confidence intervals. ``--compile-modes``
optionally compares uncompiled execution with the default, reduce-overhead, and
max-autotune ``torch.compile`` modes using full graphs and static shapes. The
script is a no-op on CPU/MPS.
"""
from __future__ import annotations

import argparse
import statistics
import sys
from typing import Literal

import torch
from tensordict import TensorDict

from torchrl import cuda_memory_stats, reset_cuda_peak_stats
from torchrl.modules import DreamerV3BlockGRU, GRUModule, LSTMModule

RNNType = Literal["lstm", "gru", "block_gru"]
CompileMode = Literal["none", "default", "reduce-overhead", "max-autotune"]
# User-facing backend name -> recurrent_backend value.
_BACKENDS: dict[str, str] = {
    "cudnn": "pad",
    "reference": "reference",
    "scan": "scan",
    "triton": "triton",
}


def _build_module(
    rnn_type: RNNType,
    recurrent_backend: str,
    recompute: str,
    *,
    input_size: int,
    hidden_size: int,
    num_layers: int,
    projection_size: int,
    num_blocks: int,
    device: torch.device,
    dtype: torch.dtype,
) -> LSTMModule | GRUModule | DreamerV3BlockGRU:
    if rnn_type == "block_gru":
        return DreamerV3BlockGRU(
            input_size,
            hidden_size,
            projection_size=projection_size,
            num_blocks=num_blocks,
            num_layers=num_layers,
            recurrent_backend=recurrent_backend,
            device=device,
        )
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
        module = LSTMModule(
            in_keys=["obs", "hidden0", "hidden1"],
            out_keys=["feat", ("next", "hidden0"), ("next", "hidden1")],
            **kwargs,
        )
    else:
        module = GRUModule(
            in_keys=["obs", "hidden"],
            out_keys=["feat", ("next", "hidden")],
            **kwargs,
        )
    return module.to(dtype)


def _build_inputs(
    rnn_type: RNNType,
    *,
    batch: int,
    seq_len: int,
    input_size: int,
    hidden_size: int,
    num_layers: int,
    device: torch.device,
    dtype: torch.dtype,
) -> TensorDict | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    obs = torch.randn(
        batch,
        seq_len,
        input_size,
        device=device,
        dtype=dtype,
        requires_grad=True,
    )
    is_init = torch.zeros(batch, seq_len, 1, dtype=torch.bool, device=device)
    is_init[:, 0] = True
    if rnn_type == "block_gru":
        hidden = torch.zeros(
            batch,
            hidden_size,
            device=device,
            dtype=dtype,
            requires_grad=True,
        )
        return obs, hidden, is_init
    if rnn_type == "lstm":
        hidden0 = torch.zeros(
            batch,
            seq_len,
            num_layers,
            hidden_size,
            device=device,
            dtype=dtype,
        )
        hidden1 = torch.zeros_like(hidden0)
        return TensorDict(
            {"obs": obs, "hidden0": hidden0, "hidden1": hidden1, "is_init": is_init},
            [batch, seq_len],
        )
    hidden = torch.zeros(
        batch,
        seq_len,
        num_layers,
        hidden_size,
        device=device,
        dtype=dtype,
    )
    return TensorDict(
        {"obs": obs, "hidden": hidden, "is_init": is_init}, [batch, seq_len]
    )


def _time_ms(fn, *, iters: int, device: torch.device) -> tuple[float, float]:
    """Return mean wall time and its 95% confidence half-width in milliseconds."""
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
    mean = statistics.mean(samples)
    if len(samples) < 2:
        return mean, 0.0
    confidence = 1.96 * statistics.stdev(samples) / len(samples) ** 0.5
    return mean, confidence


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
    projection_size: int,
    num_blocks: int,
    device: torch.device,
    dtype: torch.dtype,
    compile_mode: CompileMode,
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
        projection_size=projection_size,
        num_blocks=num_blocks,
        device=device,
        dtype=dtype,
    )
    if compile_mode != "none":
        module = torch.compile(
            module,
            fullgraph=True,
            dynamic=False,
            mode=compile_mode,
        )
    data = _build_inputs(
        rnn_type,
        batch=batch,
        seq_len=seq_len,
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        device=device,
        dtype=dtype,
    )

    def forward():
        if rnn_type == "block_gru":
            value, hidden, is_init = data
            return module(value, hidden, is_init)
        return module(data.clone())

    def loss(output):
        if rnn_type == "block_gru":
            return output[0].float().square().mean()
        return output["feat"].float().square().mean()

    def clear_grads():
        for parameter in module.parameters():
            parameter.grad = None
        if rnn_type == "block_gru":
            data[0].grad = None
            data[1].grad = None

    # Warmup (also triggers any lazy autotune / compile workspaces).
    for _ in range(max(warmup, 1)):
        out = forward()
        loss(out).backward()
        clear_grads()
    torch.cuda.synchronize(device)

    fwd_ms, fwd_ci_ms = _time_ms(forward, iters=iters, device=device)

    def fwd_bwd():
        out = forward()
        loss(out).backward()
        clear_grads()

    total_ms, total_ci_ms = _time_ms(fwd_bwd, iters=iters, device=device)

    reset_cuda_peak_stats(device)
    out = forward()
    loss(out).backward()
    torch.cuda.synchronize(device)
    mem = cuda_memory_stats(device)

    return {
        "fwd_ms": fwd_ms,
        "fwd_ci_ms": fwd_ci_ms,
        "bwd_ms": max(total_ms - fwd_ms, 0.0),
        "bwd_ci_ms": fwd_ci_ms + total_ci_ms,
        "total_ms": total_ms,
        "total_ci_ms": total_ci_ms,
        "peak_gb": mem["max_allocated_gb"],
    }


def _parse_int_list(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x.strip()]


def main() -> None:
    """Run the recurrent-backend benchmark."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--rnn", choices=["lstm", "gru", "block_gru"], default="gru")
    parser.add_argument(
        "--backends",
        default=None,
        help="Comma list among cudnn,reference,scan,triton.",
    )
    parser.add_argument("--batches", default="256,1024,4096", type=_parse_int_list)
    parser.add_argument("--seq-lens", default="16,32,64", type=_parse_int_list)
    parser.add_argument("--hiddens", default="128,256,512", type=_parse_int_list)
    parser.add_argument("--input-size", type=int, default=32)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--projection-size", type=int, default=128)
    parser.add_argument("--blocks", default="1,8", type=_parse_int_list)
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")
    parser.add_argument(
        "--compile-modes",
        default="none",
        help="Comma list among none,default,reduce-overhead,max-autotune.",
    )
    parser.add_argument("--recompute", choices=["none", "full"], default="none")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()

    backend_arg = args.backends
    if backend_arg is None:
        backend_arg = (
            "reference,scan,triton" if args.rnn == "block_gru" else "cudnn,scan,triton"
        )
    backends = [b.strip() for b in backend_arg.split(",") if b.strip()]
    invalid_backends = set(backends) - set(_BACKENDS)
    if invalid_backends:
        parser.error(
            "--backends contains invalid values: "
            + ", ".join(sorted(invalid_backends))
            + ". Valid values: "
            + ",".join(_BACKENDS)
        )
    compile_modes = [
        mode.strip() for mode in args.compile_modes.split(",") if mode.strip()
    ]
    valid_compile_modes = {"none", "default", "reduce-overhead", "max-autotune"}
    invalid_compile_modes = set(compile_modes) - valid_compile_modes
    if invalid_compile_modes:
        parser.error(
            "--compile-modes contains invalid values: "
            + ", ".join(sorted(invalid_compile_modes))
        )
    if args.rnn == "block_gru" and args.recompute == "full":
        parser.error("--recompute full is not supported for --rnn block_gru.")

    device = torch.device(args.device)
    if device.type != "cuda":
        sys.stdout.write(
            "[bench_rnn_backward] CUDA required for timing/memory. Skipping.\n"
        )
        return
    if device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    # Timing events and kernel launches must target the benchmarked device.
    torch.cuda.set_device(device)

    dtype = torch.float32 if args.dtype == "float32" else torch.bfloat16
    if args.recompute == "full":
        backends = [b for b in backends if b != "cudnn"]
        sys.stdout.write(
            "[bench_rnn_backward] recompute=full -> skipping cuDNN " "(no recompute).\n"
        )

    sys.stdout.write(
        f"rnn={args.rnn} layers={args.num_layers} input_size={args.input_size} "
        f"projection_size={args.projection_size} dtype={args.dtype} "
        f"recompute={args.recompute} warmup={args.warmup} iters={args.iters}\n"
        f"compile_modes={','.join(compile_modes)}\n"
        f"device={torch.cuda.get_device_name(device)}\n\n"
    )
    header = (
        f"{'batch':>6} {'T':>4} {'H':>5} {'blocks':>6} {'backend':>9} "
        f"{'compile':>15} "
        f"{'fwd_ms (95% CI)':>21} {'bwd_ms (95% CI)':>21} "
        f"{'total_ms (95% CI)':>23} {'peak_gb':>8}"
    )
    sys.stdout.write(f"{header}\n{'-' * len(header)}\n")
    for batch in args.batches:
        for seq_len in args.seq_lens:
            for hidden in args.hiddens:
                blocks = args.blocks if args.rnn == "block_gru" else [1]
                for num_blocks in blocks:
                    for name in backends:
                        for compile_mode in compile_modes:
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
                                    projection_size=args.projection_size,
                                    num_blocks=num_blocks,
                                    device=device,
                                    dtype=dtype,
                                    compile_mode=compile_mode,
                                    warmup=args.warmup,
                                    iters=args.iters,
                                )
                                sys.stdout.write(
                                    f"{batch:>6} {seq_len:>4} {hidden:>5} "
                                    f"{num_blocks:>6} {name:>9} {compile_mode:>15} "
                                    f"{r['fwd_ms']:>9.3f} +/- {r['fwd_ci_ms']:<7.3f} "
                                    f"{r['bwd_ms']:>9.3f} +/- {r['bwd_ci_ms']:<7.3f} "
                                    f"{r['total_ms']:>9.3f} +/- {r['total_ci_ms']:<7.3f} "
                                    f"{r['peak_gb']:>8.3f}\n"
                                )
                            except Exception as exc:  # noqa: BLE001
                                sys.stdout.write(
                                    f"{batch:>6} {seq_len:>4} {hidden:>5} "
                                    f"{num_blocks:>6} {name:>9} "
                                    f"{compile_mode:>15} ERROR: "
                                    f"{type(exc).__name__}: {str(exc)[:80]}\n"
                                )
                            finally:
                                if device.type == "cuda":
                                    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
