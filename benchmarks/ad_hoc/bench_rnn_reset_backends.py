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
Mode = Literal[
    "cudnn_pad_td",
    "scan_eager_td",
    "scan_compile_td",
    "scan_eager_direct",
    "scan_compile_direct",
]

CompileStrategy = Literal[
    "default",
    "inductor",
    "reduce-overhead",
    "max-autotune",
    "aot-eager",
    "eager",
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


def _compile(
    fn: Callable,
    strategy: CompileStrategy,
    *,
    fullgraph: bool,
) -> Callable:
    torch._dynamo.reset()
    if strategy == "default":
        return torch.compile(fn, fullgraph=fullgraph)
    if strategy == "inductor":
        return torch.compile(fn, backend="inductor", fullgraph=fullgraph)
    if strategy == "reduce-overhead":
        return torch.compile(fn, mode="reduce-overhead", fullgraph=fullgraph)
    if strategy == "max-autotune":
        return torch.compile(fn, mode="max-autotune", fullgraph=fullgraph)
    if strategy == "aot-eager":
        return torch.compile(fn, backend="aot_eager", fullgraph=fullgraph)
    if strategy == "eager":
        return torch.compile(fn, backend="eager", fullgraph=fullgraph)
    raise ValueError(f"Unknown compile strategy {strategy}.")


def _make_modules(
    rnn_type: RNNType,
    input_size: int,
    hidden_size: int,
    num_layers: int,
    device: torch.device,
) -> tuple[GRUModule | LSTMModule, GRUModule | LSTMModule, GRUModule | LSTMModule]:
    if rnn_type == "lstm":
        cudnn_pad = LSTMModule(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            in_keys=["obs", "hidden0", "hidden1"],
            out_keys=["feat", ("next", "hidden0"), ("next", "hidden1")],
            default_recurrent_mode=True,
            device=device,
        ).eval()
        scan_eager = LSTMModule(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            in_keys=["obs", "hidden0", "hidden1"],
            out_keys=["feat", ("next", "hidden0"), ("next", "hidden1")],
            recurrent_backend="scan",
            default_recurrent_mode=True,
            device=device,
        ).eval()
        scan_compile = LSTMModule(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            in_keys=["obs", "hidden0", "hidden1"],
            out_keys=["feat", ("next", "hidden0"), ("next", "hidden1")],
            recurrent_backend="scan",
            default_recurrent_mode=True,
            python_based=True,
            device=device,
        ).eval()
        scan_eager.load_state_dict(cudnn_pad.state_dict())
        scan_compile.load_state_dict(cudnn_pad.state_dict())
        return cudnn_pad, scan_eager, scan_compile

    cudnn_pad = GRUModule(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        in_keys=["obs", "hidden"],
        out_keys=["feat", ("next", "hidden")],
        default_recurrent_mode=True,
        device=device,
    ).eval()
    scan_eager = GRUModule(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        in_keys=["obs", "hidden"],
        out_keys=["feat", ("next", "hidden")],
        recurrent_backend="scan",
        default_recurrent_mode=True,
        device=device,
    ).eval()
    scan_compile = GRUModule(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        in_keys=["obs", "hidden"],
        out_keys=["feat", ("next", "hidden")],
        recurrent_backend="scan",
        default_recurrent_mode=True,
        python_based=True,
        device=device,
    ).eval()
    scan_eager.load_state_dict(cudnn_pad.state_dict())
    scan_compile.load_state_dict(cudnn_pad.state_dict())
    return cudnn_pad, scan_eager, scan_compile


def _make_fn(
    rnn_type: RNNType,
    mode: Mode,
    cudnn_pad: GRUModule | LSTMModule,
    scan_eager: GRUModule | LSTMModule,
    scan_compile: GRUModule | LSTMModule,
    obs: torch.Tensor,
    hidden: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
    is_init: torch.Tensor,
    compile_strategy: CompileStrategy,
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
    if mode == "scan_compile_td" and rnn_type == "gru":

        def fn(obs, hidden, is_init):
            out = scan_compile(_make_td(rnn_type, obs, hidden, is_init))
            return out["feat"], out["next", "hidden"]

        compiled = _compile(fn, compile_strategy, fullgraph=False)

        def compiled_fn():
            return compiled(obs, hidden, is_init)

        return compiled_fn
    if mode == "scan_compile_td" and rnn_type == "lstm":

        def fn(obs, hidden0, hidden1, is_init):
            out = scan_compile(_make_td(rnn_type, obs, (hidden0, hidden1), is_init))
            return out["feat"], out["next", "hidden0"], out["next", "hidden1"]

        compiled = _compile(fn, compile_strategy, fullgraph=False)

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

        compiled = _compile(fn, compile_strategy, fullgraph=True)

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

        compiled = _compile(fn, compile_strategy, fullgraph=True)

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
    parser.add_argument("--num-layers", type=int, default=1)
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
        ],
        choices=[
            "cudnn_pad_td",
            "scan_eager_td",
            "scan_compile_td",
            "scan_eager_direct",
            "scan_compile_direct",
        ],
    )
    parser.add_argument(
        "--compile-strategies",
        nargs="+",
        default=["default"],
        choices=[
            "default",
            "inductor",
            "reduce-overhead",
            "max-autotune",
            "aot-eager",
            "eager",
        ],
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if any(mode.startswith("scan_compile") for mode in args.modes):
        torch._dynamo.config.capture_scalar_outputs = True

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    generator = torch.Generator(device=device).manual_seed(args.seed)

    print(
        "device,rnn_type,batch,steps,reset_prob,mode,compile_strategy,median_ms,"
        "min_ms,frames_per_s,actual_reset_frac,status,error"
    )
    for rnn_type, batch, steps, reset_prob in itertools.product(
        args.rnn_types, args.batches, args.lengths, args.reset_probs
    ):
        torch.manual_seed(args.seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(args.seed)
        cudnn_pad, scan_eager, scan_compile = _make_modules(
            rnn_type, args.input_size, args.hidden_size, args.num_layers, device
        )
        obs = torch.randn(
            batch, steps, args.input_size, device=device, generator=generator
        )
        hidden0 = torch.zeros(
            batch,
            steps,
            args.num_layers,
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
            compile_strategies = (
                args.compile_strategies if mode.startswith("scan_compile") else ["none"]
            )
            for compile_strategy in compile_strategies:
                try:
                    fn = _make_fn(
                        rnn_type,
                        mode,
                        cudnn_pad,
                        scan_eager,
                        scan_compile,
                        obs,
                        hidden,
                        is_init,
                        compile_strategy,
                    )
                    median_ms, min_ms = _bench(fn, device, args.warmup, args.iters)
                    frames_per_s = batch * steps / (median_ms / 1000)
                    status = "ok"
                    error = ""
                except Exception as err:
                    median_ms = min_ms = frames_per_s = float("nan")
                    status = "error"
                    error = type(err).__name__ + ": " + str(err).splitlines()[0]
                    error = error.replace(",", ";")
                print(
                    f"{device},{rnn_type},{batch},{steps},{reset_prob},{mode},"
                    f"{compile_strategy},{median_ms:.4f},{min_ms:.4f},"
                    f"{frames_per_s:.2f},{actual_reset_frac:.6f},{status},{error}"
                )


if __name__ == "__main__":
    main()
