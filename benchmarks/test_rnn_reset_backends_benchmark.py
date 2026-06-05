# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import argparse
import gc
import os
from collections.abc import Callable, Iterator
from typing import Literal

import pytest
import torch
from tensordict import TensorDict

from torchrl._utils import compile_with_warmup
from torchrl.modules import GRUModule, LSTMModule


RNNType = Literal["gru", "lstm"]
Backend = Literal["cudnn", "scan", "triton"]
RNNShape = tuple[int, int, int, int]

_DEFAULT_RNN_SHAPE: RNNShape = (65536, 128, 32, 256)

# Eager calls ``compile_with_warmup`` makes before compiling the module.
_COMPILE_WARMUP = 1


def _shape_id(shape: RNNShape) -> str:
    batch, steps, input_size, hidden_size = shape
    return f"b{batch}-t{steps}-i{input_size}-h{hidden_size}"


def _parse_shape(shape: str) -> RNNShape:
    try:
        batch, steps, input_size, hidden_size = (int(part) for part in shape.split(","))
    except ValueError as err:
        raise ValueError(
            "RNN benchmark shapes must be comma-separated "
            "batch,steps,input_size,hidden_size values."
        ) from err
    if batch <= 0 or steps <= 0 or input_size <= 0 or hidden_size <= 0:
        raise ValueError(f"RNN benchmark shape values must be positive: {shape}.")
    return batch, steps, input_size, hidden_size


def _rnn_shapes() -> tuple[RNNShape, ...]:
    shapes = os.environ.get("TORCHRL_RNN_RESET_BENCHMARK_SHAPES")
    if shapes:
        return tuple(_parse_shape(shape) for shape in shapes.split(";"))
    return (_DEFAULT_RNN_SHAPE,)


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _make_tensordict(
    rnn_type: RNNType,
    device: torch.device,
    generator: torch.Generator,
    shape: RNNShape,
) -> TensorDict:
    batch, steps, input_size, hidden_size = shape
    obs = torch.randn(batch, steps, input_size, device=device, generator=generator)
    hidden = torch.zeros(batch, steps, 1, hidden_size, device=device)
    is_init = torch.rand(batch, steps, 1, device=device, generator=generator).lt(0.03)
    is_init[:, 0] = False
    if rnn_type == "gru":
        return TensorDict(
            {"obs": obs, "hidden": hidden, "is_init": is_init}, [batch, steps]
        )
    return TensorDict(
        {
            "obs": obs,
            "hidden0": hidden,
            "hidden1": torch.zeros_like(hidden),
            "is_init": is_init,
        },
        [batch, steps],
    )


def _make_module(
    rnn_type: RNNType, backend: Backend, device: torch.device, shape: RNNShape
) -> GRUModule | LSTMModule:
    _, _, input_size, hidden_size = shape
    recurrent_backend = "pad" if backend == "cudnn" else backend
    if rnn_type == "gru":
        return GRUModule(
            input_size=input_size,
            hidden_size=hidden_size,
            in_keys=["obs", "hidden"],
            out_keys=["feat", ("next", "hidden")],
            recurrent_backend=recurrent_backend,
            default_recurrent_mode=True,
            device=device,
        )
    return LSTMModule(
        input_size=input_size,
        hidden_size=hidden_size,
        in_keys=["obs", "hidden0", "hidden1"],
        out_keys=["feat", ("next", "hidden0"), ("next", "hidden1")],
        recurrent_backend=recurrent_backend,
        default_recurrent_mode=True,
        device=device,
    )


def _call(
    module: Callable[[TensorDict], TensorDict],
    tensordict: TensorDict,
    device: torch.device,
) -> TensorDict:
    # Synchronize inside the timed region: CUDA launches are async, and the
    # tiled Triton path issues one launch per time step, so without an in-loop
    # sync pytest-benchmark would time enqueue cost (CPU running ahead of the
    # GPU) rather than kernel execution -- producing a spuriously low Min and
    # huge StdDev. Syncing here makes every round reflect GPU completion.
    with torch.inference_mode():
        out = module(tensordict)
    _sync(device)
    return out


def _mib(num_bytes: int) -> float:
    return num_bytes / 1024**2


def _reset_cuda_memory_stats(device: torch.device) -> dict[str, int] | None:
    if device.type != "cuda":
        return None
    _sync(device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    return {
        "cuda_allocated_before_bytes": torch.cuda.memory_allocated(device),
        "cuda_reserved_before_bytes": torch.cuda.memory_reserved(device),
    }


def _collect_cuda_memory_stats(
    device: torch.device,
    before: dict[str, int] | None,
) -> dict[str, int | float] | None:
    if before is None:
        return None
    _sync(device)
    allocated_after = torch.cuda.memory_allocated(device)
    reserved_after = torch.cuda.memory_reserved(device)
    max_allocated = torch.cuda.max_memory_allocated(device)
    max_reserved = torch.cuda.max_memory_reserved(device)
    allocated_before = before["cuda_allocated_before_bytes"]
    reserved_before = before["cuda_reserved_before_bytes"]
    stats = {
        **before,
        "cuda_allocated_after_bytes": allocated_after,
        "cuda_reserved_after_bytes": reserved_after,
        "cuda_max_allocated_bytes": max_allocated,
        "cuda_max_reserved_bytes": max_reserved,
        "cuda_peak_allocated_delta_bytes": max(0, max_allocated - allocated_before),
        "cuda_peak_reserved_delta_bytes": max(0, max_reserved - reserved_before),
    }
    stats.update(
        {key.replace("_bytes", "_mib"): _mib(value) for key, value in stats.items()}
    )
    return stats


@pytest.fixture(autouse=True)
def _reset_compile_cache() -> Iterator[None]:
    _reset_caches()
    yield
    _sync(torch.device("cuda:0" if torch.cuda.device_count() else "cpu"))
    _reset_caches()


def _reset_caches() -> None:
    torch.compiler.reset()
    if hasattr(torch, "_dynamo"):
        torch._dynamo.reset()
        reset_code_caches = getattr(torch._dynamo, "reset_code_caches", None)
        if reset_code_caches is not None:
            reset_code_caches()
    gc.collect()
    if torch.cuda.device_count():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


@pytest.mark.parametrize("rnn_type", ["gru", "lstm"])
@pytest.mark.parametrize("reset_seed", [0])
@pytest.mark.parametrize(
    ("backend", "compile"),
    [
        ("cudnn", False),
        ("cudnn", True),
        ("scan", False),
        ("scan", True),
        pytest.param("triton", False, marks=pytest.mark.gpu),
        pytest.param("triton", True, marks=pytest.mark.gpu),
    ],
)
@pytest.mark.parametrize("rnn_shape", _rnn_shapes(), ids=_shape_id)
def test_rnn_rollout_with_intermediate_resets(
    benchmark,
    record_cuda_memory_stats,
    rnn_type: RNNType,
    reset_seed: int,
    backend: Backend,
    compile: bool,
    rnn_shape: RNNShape,
) -> None:
    device = torch.device("cuda:0" if torch.cuda.device_count() else "cpu")
    if backend == "triton" and device.type != "cuda":
        pytest.skip("triton recurrent backend requires CUDA")
    generator = torch.Generator(device=device).manual_seed(reset_seed)
    tensordict = _make_tensordict(rnn_type, device, generator, rnn_shape)
    try:
        module = _make_module(rnn_type, backend, device, rnn_shape)
    except RuntimeError as err:
        if backend == "triton":
            pytest.skip(f"triton recurrent backend unavailable: {err}")
        raise
    if compile:
        module = compile_with_warmup(module, warmup=_COMPILE_WARMUP)
    # ``compile_with_warmup`` runs ``_COMPILE_WARMUP`` eager calls before it
    # compiles, so prep with ``warmup + 3`` calls: the warmup calls, the call
    # that triggers compilation, plus a couple of warm steady-state calls. The
    # eager path only needs a single prep call.
    prep_iters = _COMPILE_WARMUP + 3 if compile else 1
    for _ in range(prep_iters):
        _call(module, tensordict, device)
    memory_before = _reset_cuda_memory_stats(device)
    benchmark(_call, module, tensordict, device)
    _sync(device)
    record_cuda_memory_stats(
        benchmark, _collect_cuda_memory_stats(device, memory_before)
    )


if __name__ == "__main__":
    _, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst"] + unknown)
