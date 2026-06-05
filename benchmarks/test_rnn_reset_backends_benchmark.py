# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import argparse
from typing import Literal

import pytest
import torch
from tensordict import TensorDict

from torchrl.modules import GRUModule, LSTMModule


RNNType = Literal["gru", "lstm"]
Backend = Literal["cudnn", "scan", "triton"]


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _make_tensordict(
    rnn_type: RNNType,
    device: torch.device,
    generator: torch.Generator,
) -> TensorDict:
    batch, steps, input_size, hidden_size = 128, 128, 32, 256
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
    rnn_type: RNNType, backend: Backend, device: torch.device
) -> GRUModule | LSTMModule:
    recurrent_backend = "pad" if backend == "cudnn" else backend
    if rnn_type == "gru":
        return GRUModule(
            input_size=32,
            hidden_size=256,
            in_keys=["obs", "hidden"],
            out_keys=["feat", ("next", "hidden")],
            recurrent_backend=recurrent_backend,
            default_recurrent_mode=True,
            device=device,
        )
    return LSTMModule(
        input_size=32,
        hidden_size=256,
        in_keys=["obs", "hidden0", "hidden1"],
        out_keys=["feat", ("next", "hidden0"), ("next", "hidden1")],
        recurrent_backend=recurrent_backend,
        default_recurrent_mode=True,
        device=device,
    )


def _call(module: torch.nn.Module, tensordict: TensorDict) -> TensorDict:
    with torch.inference_mode():
        return module(tensordict)


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
def test_rnn_rollout_with_intermediate_resets(
    benchmark,
    rnn_type: RNNType,
    reset_seed: int,
    backend: Backend,
    compile: bool,
) -> None:
    device = torch.device("cuda:0" if torch.cuda.device_count() else "cpu")
    if backend == "triton" and device.type != "cuda":
        pytest.skip("triton recurrent backend requires CUDA")
    generator = torch.Generator(device=device).manual_seed(reset_seed)
    tensordict = _make_tensordict(rnn_type, device, generator)
    try:
        module = _make_module(rnn_type, backend, device)
    except RuntimeError as err:
        if backend == "triton":
            pytest.skip(f"triton recurrent backend unavailable: {err}")
        raise
    prev_capture = torch._dynamo.config.capture_scalar_outputs
    torch._dynamo.config.capture_scalar_outputs = compile
    try:
        if compile:
            module = torch.compile(module)
        for _ in range(3 if compile else 1):
            try:
                _call(module, tensordict)
                _sync(device)
            except Exception as err:
                if compile:
                    pytest.xfail(
                        f"torch.compile currently fails for public {rnn_type} "
                        f"{backend} TensorDict rollout: {err}"
                    )
                raise
        benchmark(_call, module, tensordict)
        _sync(device)
    finally:
        torch._dynamo.config.capture_scalar_outputs = prev_capture


if __name__ == "__main__":
    _, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst"] + unknown)
