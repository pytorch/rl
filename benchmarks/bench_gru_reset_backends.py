# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import argparse
import itertools
import statistics
import time
from collections.abc import Callable
from typing import Literal

import torch
from tensordict import TensorDict

from torchrl.modules import GRUModule, LSTMModule


RNNType = Literal["gru", "lstm"]
CompileMode = Literal["none", "default", "reduce-overhead", "max-autotune"]
Mode = Literal[
    "cudnn_pad_td",
    "scan_eager_td",
    "scan_compile_td",
    "scan_eager_direct",
    "scan_compile_direct",
    "triton_td",
]


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _make_is_init(
    batch: int,
    steps: int,
    reset_prob: float,
    device: torch.device,
    generator: torch.Generator,
) -> torch.Tensor:
    is_init = torch.zeros(batch, steps, 1, dtype=torch.bool, device=device)
    if reset_prob:
        is_init[:, 1:] = torch.rand(
            batch, steps - 1, 1, device=device, generator=generator
        ).lt(reset_prob)
    return is_init


def _make_td(
    rnn_type: RNNType,
    obs: torch.Tensor,
    hidden: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
    is_init: torch.Tensor,
) -> TensorDict:
    if rnn_type == "gru":
        return TensorDict(
            {"obs": obs, "hidden": hidden, "is_init": is_init}, obs.shape[:2]
        )
    hidden0, hidden1 = hidden
    return TensorDict(
        {"obs": obs, "hidden0": hidden0, "hidden1": hidden1, "is_init": is_init},
        obs.shape[:2],
    )


def _bench(
    fn: Callable[[], object],
    device: torch.device,
    warmup: int,
    iters: int,
) -> tuple[float, float]:
    with torch.inference_mode():
        for _ in range(warmup):
            fn()
        _sync(device)
        times = []
        for _ in range(iters):
            start = time.perf_counter()
            fn()
            _sync(device)
            times.append((time.perf_counter() - start) * 1000)
    return statistics.median(times), min(times)


def _compile_fn(
    fn: Callable[[], object],
    compile_mode: CompileMode,
    fullgraph: bool,
    dynamic: bool | None,
) -> Callable[[], object]:
    if compile_mode == "none":
        return fn
    kwargs = {}
    if compile_mode != "default":
        kwargs["mode"] = compile_mode
    kwargs["fullgraph"] = fullgraph
    if dynamic is not None:
        kwargs["dynamic"] = dynamic
    return torch.compile(fn, **kwargs)


def _first_call_ms(fn: Callable[[], object], device: torch.device) -> float:
    with torch.inference_mode():
        _sync(device)
        start = time.perf_counter()
        fn()
        _sync(device)
    return (time.perf_counter() - start) * 1000


def _make_modules(
    rnn_type: RNNType,
    input_size: int,
    hidden_size: int,
    num_layers: int,
    dropout: float,
    device: torch.device,
) -> tuple[
    GRUModule | LSTMModule,
    GRUModule | LSTMModule | None,
    GRUModule | LSTMModule | None,
    GRUModule | LSTMModule | None,
]:
    # scan backend cannot run with dropout > 0. cuDNN's nn.LSTM/nn.GRU only
    # applies the configured dropout when num_layers > 1. The triton backend
    # uses cudnn_pad's state_dict, so all modules see the same parameters.
    scan_supported = dropout == 0.0
    triton_supported = device.type == "cuda"
    # Whether dropout is exercised at runtime (only kicks in with multi-layer
    # stacks where there's a between-layer dropout point).
    effective_dropout = dropout if num_layers > 1 else 0.0
    train_mode = effective_dropout > 0
    if rnn_type == "lstm":
        cudnn_pad = LSTMModule(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            in_keys=["obs", "hidden0", "hidden1"],
            out_keys=["feat", ("next", "hidden0"), ("next", "hidden1")],
            default_recurrent_mode=True,
            device=device,
        )
        cudnn_pad.train(train_mode)
        scan_eager = None
        scan_compile = None
        if scan_supported:
            scan_eager = LSTMModule(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout,
                in_keys=["obs", "hidden0", "hidden1"],
                out_keys=["feat", ("next", "hidden0"), ("next", "hidden1")],
                recurrent_backend="scan",
                default_recurrent_mode=True,
                device=device,
            )
            scan_eager.load_state_dict(cudnn_pad.state_dict())
            scan_eager.train(train_mode)
            scan_compile = LSTMModule(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout,
                in_keys=["obs", "hidden0", "hidden1"],
                out_keys=["feat", ("next", "hidden0"), ("next", "hidden1")],
                recurrent_backend="scan",
                default_recurrent_mode=True,
                python_based=True,
                device=device,
            )
            scan_compile.load_state_dict(cudnn_pad.state_dict())
            scan_compile.train(train_mode)
        triton_mod = None
        if triton_supported:
            try:
                triton_mod = LSTMModule(
                    input_size=input_size,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    dropout=dropout,
                    in_keys=["obs", "hidden0", "hidden1"],
                    out_keys=["feat", ("next", "hidden0"), ("next", "hidden1")],
                    recurrent_backend="triton",
                    default_recurrent_mode=True,
                    device=device,
                )
                triton_mod.load_state_dict(cudnn_pad.state_dict())
                triton_mod.train(train_mode)
            except RuntimeError:
                triton_mod = None
        return cudnn_pad, scan_eager, scan_compile, triton_mod

    cudnn_pad = GRUModule(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        in_keys=["obs", "hidden"],
        out_keys=["feat", ("next", "hidden")],
        default_recurrent_mode=True,
        device=device,
    )
    cudnn_pad.train(train_mode)
    scan_eager = None
    scan_compile = None
    if scan_supported:
        scan_eager = GRUModule(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            in_keys=["obs", "hidden"],
            out_keys=["feat", ("next", "hidden")],
            recurrent_backend="scan",
            default_recurrent_mode=True,
            device=device,
        )
        scan_eager.load_state_dict(cudnn_pad.state_dict())
        scan_eager.train(train_mode)
        scan_compile = GRUModule(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            in_keys=["obs", "hidden"],
            out_keys=["feat", ("next", "hidden")],
            recurrent_backend="scan",
            default_recurrent_mode=True,
            python_based=True,
            device=device,
        )
        scan_compile.load_state_dict(cudnn_pad.state_dict())
        scan_compile.train(train_mode)
    triton_mod = None
    if triton_supported:
        try:
            triton_mod = GRUModule(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout,
                in_keys=["obs", "hidden"],
                out_keys=["feat", ("next", "hidden")],
                recurrent_backend="triton",
                default_recurrent_mode=True,
                device=device,
            )
            triton_mod.load_state_dict(cudnn_pad.state_dict())
        except RuntimeError:
            triton_mod = None
    return cudnn_pad, scan_eager, scan_compile, triton_mod


def _make_fn(
    rnn_type: RNNType,
    mode: Mode,
    cudnn_pad: GRUModule | LSTMModule,
    scan_eager: GRUModule | LSTMModule | None,
    scan_compile: GRUModule | LSTMModule | None,
    triton_mod: GRUModule | LSTMModule | None,
    obs: torch.Tensor,
    hidden: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
    is_init: torch.Tensor,
):
    batch, steps = obs.shape[:2]
    if mode == "cudnn_pad_td":

        def fn():
            return cudnn_pad(_make_td(rnn_type, obs, hidden, is_init))

        return fn
    if mode == "scan_eager_td":

        def fn():
            return scan_eager(_make_td(rnn_type, obs, hidden, is_init))

        return fn
    if mode == "triton_td":
        if triton_mod is None:
            raise ValueError("triton backend not available on this device / num_layers")

        def fn():
            return triton_mod(_make_td(rnn_type, obs, hidden, is_init))

        return fn
    if mode == "scan_compile_td" and rnn_type == "gru":

        def fn(obs, hidden, is_init):
            out = scan_compile(_make_td(rnn_type, obs, hidden, is_init))
            return out["feat"], out["next", "hidden"]

        compiled = torch.compile(fn)

        def compiled_fn():
            return compiled(obs, hidden, is_init)

        return compiled_fn
    if mode == "scan_compile_td" and rnn_type == "lstm":

        def fn(obs, hidden0, hidden1, is_init):
            out = scan_compile(_make_td(rnn_type, obs, (hidden0, hidden1), is_init))
            return out["feat"], out["next", "hidden0"], out["next", "hidden1"]

        compiled = torch.compile(fn)

        def compiled_fn():
            hidden0, hidden1 = hidden
            return compiled(obs, hidden0, hidden1, is_init)

        return compiled_fn
    if mode == "scan_eager_direct" and rnn_type == "gru":

        def fn():
            return scan_compile._gru(
                obs,
                batch,
                steps,
                obs.device,
                obs.dtype,
                hidden,
                None,
                is_init=is_init.squeeze(-1),
            )

        return fn
    if mode == "scan_eager_direct" and rnn_type == "lstm":

        def fn():
            hidden0, hidden1 = hidden
            return scan_compile._lstm(
                obs,
                batch,
                steps,
                obs.device,
                obs.dtype,
                hidden0,
                hidden1,
                None,
                is_init=is_init.squeeze(-1),
            )

        return fn
    if mode == "scan_compile_direct" and rnn_type == "gru":

        def fn(obs, hidden, is_init):
            return scan_compile._gru(
                obs,
                batch,
                steps,
                obs.device,
                obs.dtype,
                hidden,
                None,
                is_init=is_init.squeeze(-1),
            )

        compiled = torch.compile(fn, fullgraph=True)

        def compiled_fn():
            return compiled(obs, hidden, is_init)

        return compiled_fn
    if mode == "scan_compile_direct" and rnn_type == "lstm":

        def fn(obs, hidden0, hidden1, is_init):
            return scan_compile._lstm(
                obs,
                batch,
                steps,
                obs.device,
                obs.dtype,
                hidden0,
                hidden1,
                None,
                is_init=is_init.squeeze(-1),
            )

        compiled = torch.compile(fn, fullgraph=True)

        def compiled_fn():
            hidden0, hidden1 = hidden
            return compiled(obs, hidden0, hidden1, is_init)

        return compiled_fn
    raise ValueError(f"Unknown mode {mode}.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--input-size", type=int, default=64)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--num-layers", type=int, nargs="+", default=[1])
    parser.add_argument(
        "--dropouts",
        type=float,
        nargs="+",
        default=[0.0],
        help=(
            "Dropout probabilities to sweep. Only effective with num_layers > 1 "
            "(dropout is between stacked layers). Scan modes are skipped when "
            "dropout > 0 since they raise NotImplementedError."
        ),
    )
    parser.add_argument(
        "--rnn-types", nargs="+", default=["gru"], choices=["gru", "lstm"]
    )
    parser.add_argument("--batches", type=int, nargs="+", default=[128, 512, 2048])
    parser.add_argument("--lengths", type=int, nargs="+", default=[16, 64, 256])
    parser.add_argument("--reset-probs", type=float, nargs="+", default=[0, 0.01, 0.1])
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument(
        "--modes",
        nargs="+",
        default=[
            "cudnn_pad_td",
            "scan_eager_td",
            "scan_compile_td",
            "scan_eager_direct",
            "scan_compile_direct",
            "triton_td",
        ],
        choices=[
            "cudnn_pad_td",
            "scan_eager_td",
            "scan_compile_td",
            "scan_eager_direct",
            "scan_compile_direct",
            "triton_td",
        ],
    )
    parser.add_argument(
        "--compile",
        choices=["none", "default", "reduce-overhead", "max-autotune"],
        default="none",
        help=(
            "Optionally wrap selected benchmark modes in torch.compile. "
            "'default' calls torch.compile(fn); the other values are passed as "
            "torch.compile(..., mode=...)."
        ),
    )
    parser.add_argument(
        "--compile-modes",
        nargs="+",
        default=["triton_td"],
        choices=[
            "all",
            "cudnn_pad_td",
            "scan_eager_td",
            "scan_compile_td",
            "scan_eager_direct",
            "scan_compile_direct",
            "triton_td",
        ],
        help=(
            "Modes to wrap with --compile. Use 'all' to compile every selected "
            "mode. The legacy scan_compile_* modes are already compiled before "
            "this wrapper is applied."
        ),
    )
    parser.add_argument(
        "--compile-fullgraph",
        action="store_true",
        help="Pass fullgraph=True to torch.compile for modes selected by --compile.",
    )
    parser.add_argument(
        "--compile-dynamic",
        choices=["auto", "true", "false"],
        default="auto",
        help=(
            "Pass dynamic=True/False to torch.compile for selected modes. "
            "'auto' leaves the argument unset."
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if "scan_compile_td" in args.modes or args.compile != "none":
        torch._dynamo.config.capture_scalar_outputs = True

    compile_dynamic = {
        "auto": None,
        "true": True,
        "false": False,
    }[args.compile_dynamic]
    compile_modes = set(args.compile_modes)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    generator = torch.Generator(device=device).manual_seed(args.seed)

    print(
        "device,rnn_type,batch,steps,num_layers,dropout,reset_prob,mode,"
        "compile,compile_fullgraph,compile_dynamic,first_call_ms,"
        "median_ms,min_ms,frames_per_s,actual_reset_frac"
    )
    for rnn_type, batch, steps, num_layers, dropout, reset_prob in itertools.product(
        args.rnn_types,
        args.batches,
        args.lengths,
        args.num_layers,
        args.dropouts,
        args.reset_probs,
    ):
        torch.manual_seed(args.seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(args.seed)
        cudnn_pad, scan_eager, scan_compile, triton_mod = _make_modules(
            rnn_type,
            args.input_size,
            args.hidden_size,
            num_layers,
            dropout,
            device,
        )
        obs = torch.randn(
            batch, steps, args.input_size, device=device, generator=generator
        )
        hidden0 = torch.zeros(
            batch,
            steps,
            num_layers,
            args.hidden_size,
            device=device,
        )
        if rnn_type == "gru":
            hidden = hidden0
        else:
            hidden = (hidden0, torch.zeros_like(hidden0))
        is_init = _make_is_init(batch, steps, reset_prob, device, generator)
        actual_reset_frac = is_init[:, 1:].float().mean().item() if steps > 1 else 0
        for mode in args.modes:
            if mode == "triton_td" and triton_mod is None:
                continue
            if mode.startswith("scan_") and (
                scan_eager is None or scan_compile is None
            ):
                continue
            fn = _make_fn(
                rnn_type,
                mode,
                cudnn_pad,
                scan_eager,
                scan_compile,
                triton_mod,
                obs,
                hidden,
                is_init,
            )
            mode_compile = (
                args.compile
                if args.compile != "none"
                and ("all" in compile_modes or mode in compile_modes)
                else "none"
            )
            fn = _compile_fn(
                fn,
                mode_compile,
                args.compile_fullgraph,
                compile_dynamic,
            )
            first_call_ms = _first_call_ms(fn, device)
            median_ms, min_ms = _bench(fn, device, args.warmup, args.iters)
            frames_per_s = batch * steps / (median_ms / 1000)
            print(
                f"{device},{rnn_type},{batch},{steps},{num_layers},{dropout},"
                f"{reset_prob},{mode},"
                f"{mode_compile},{args.compile_fullgraph},{args.compile_dynamic},"
                f"{first_call_ms:.4f},"
                f"{median_ms:.4f},{min_ms:.4f},{frames_per_s:.2f},"
                f"{actual_reset_frac:.6f}"
            )


if __name__ == "__main__":
    main()
