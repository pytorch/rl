# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import argparse
from collections.abc import Callable
from typing import Literal

import pytest
import torch
from tensordict import TensorDict

from torchrl.modules import GRUModule, LSTMModule
from torchrl.objectives.value.functional import _split_and_pad_sequence
from torchrl.objectives.value.utils import _get_num_per_traj_init


RNNType = Literal["gru", "lstm"]
Backend = Literal["cudnn", "scan", "triton"]


def _device() -> torch.device:
    if torch.cuda.device_count():
        return torch.device("cuda:0")
    return torch.device("cpu")


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _make_modules(
    rnn_type: RNNType,
    input_size: int,
    hidden_size: int,
    num_layers: int,
    device: torch.device,
) -> tuple[
    GRUModule | LSTMModule,
    GRUModule | LSTMModule,
    GRUModule | LSTMModule,
    GRUModule | LSTMModule | None,
]:
    if rnn_type == "lstm":
        cudnn_pad = LSTMModule(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            in_keys=["obs", "hidden0", "hidden1"],
            out_keys=["feat", ("next", "hidden0"), ("next", "hidden1")],
            default_recurrent_mode=True,
            device=device,
        )
        scan_eager = LSTMModule(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            in_keys=["obs", "hidden0", "hidden1"],
            out_keys=["feat", ("next", "hidden0"), ("next", "hidden1")],
            recurrent_backend="scan",
            default_recurrent_mode=True,
            device=device,
        )
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
        )
        triton_mod = None
        if device.type == "cuda":
            try:
                triton_mod = LSTMModule(
                    input_size=input_size,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    in_keys=["obs", "hidden0", "hidden1"],
                    out_keys=["feat", ("next", "hidden0"), ("next", "hidden1")],
                    recurrent_backend="triton",
                    default_recurrent_mode=True,
                    device=device,
                )
            except RuntimeError:
                triton_mod = None
    else:
        cudnn_pad = GRUModule(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            in_keys=["obs", "hidden"],
            out_keys=["feat", ("next", "hidden")],
            default_recurrent_mode=True,
            device=device,
        )
        scan_eager = GRUModule(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            in_keys=["obs", "hidden"],
            out_keys=["feat", ("next", "hidden")],
            recurrent_backend="scan",
            default_recurrent_mode=True,
            device=device,
        )
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
        )
        triton_mod = None
        if device.type == "cuda":
            try:
                triton_mod = GRUModule(
                    input_size=input_size,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    in_keys=["obs", "hidden"],
                    out_keys=["feat", ("next", "hidden")],
                    recurrent_backend="triton",
                    default_recurrent_mode=True,
                    device=device,
                )
            except RuntimeError:
                triton_mod = None

    scan_eager.load_state_dict(cudnn_pad.state_dict())
    scan_compile.load_state_dict(cudnn_pad.state_dict())
    if triton_mod is not None:
        triton_mod.load_state_dict(cudnn_pad.state_dict())
    return cudnn_pad, scan_eager, scan_compile, triton_mod


def _make_intermediate_resets(
    batch: int,
    steps: int,
    reset_prob: float,
    device: torch.device,
    generator: torch.Generator,
) -> torch.Tensor:
    is_init = torch.rand(
        batch,
        steps,
        1,
        device=device,
        generator=generator,
    ).lt(reset_prob)
    # Keep the first step as a normal continuation so the benchmark measures
    # intermediate reset handling rather than a full-sequence initial reset.
    is_init[:, 0] = False
    return is_init


def _execute(
    fn: Callable[[TensorDict], object],
    tensordict: TensorDict,
    device: torch.device,
) -> object:
    with torch.inference_mode():
        result = fn(tensordict)
    _sync(device)
    return result


def _make_tensordict(
    rnn_type: RNNType,
    obs: torch.Tensor,
    hidden: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
    is_init: torch.Tensor,
) -> TensorDict:
    if rnn_type == "gru":
        return TensorDict(
            {
                "obs": obs,
                "hidden": hidden,
                "is_init": is_init,
            },
            obs.shape[:2],
        )
    hidden0, hidden1 = hidden
    return TensorDict(
        {
            "obs": obs,
            "hidden0": hidden0,
            "hidden1": hidden1,
            "is_init": is_init,
        },
        obs.shape[:2],
    )


def _make_backend_tensordict(
    rnn_type: RNNType,
    backend: Backend,
    obs: torch.Tensor,
    hidden: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
    is_init: torch.Tensor,
) -> tuple[TensorDict, torch.Tensor | None]:
    tensordict = _make_tensordict(rnn_type, obs, hidden, is_init)
    if backend != "cudnn":
        return tensordict, None

    splits = _get_num_per_traj_init(is_init.squeeze(-1))
    # The pad/cuDNN backend consumes padded trajectory chunks. Create that
    # TensorDict once so the compiled callable still receives a TensorDict
    # input, while the timed region targets the recurrent backend rather than
    # data-dependent split/pad bookkeeping.
    return _split_and_pad_sequence(tensordict, splits), splits


def _make_backend_fn(
    rnn_type: RNNType,
    backend: Backend,
    cudnn_pad,
    scan_mod,
    triton_mod,
    splits: torch.Tensor | None,
) -> Callable[[TensorDict], object]:
    if backend == "cudnn":
        if splits is None:
            raise RuntimeError("cudnn backend requires precomputed trajectory splits")
        if rnn_type == "gru":

            def fn(tensordict: TensorDict) -> object:
                obs = tensordict["obs"]
                return cudnn_pad._gru(
                    obs,
                    tensordict.shape[0],
                    tensordict.shape[1],
                    obs.device,
                    obs.dtype,
                    tensordict["hidden"],
                    splits=splits,
                    is_init=None,
                    backend="pad",
                )

            return fn

        def fn(tensordict: TensorDict) -> object:
            obs = tensordict["obs"]
            return cudnn_pad._lstm(
                obs,
                tensordict.shape[0],
                tensordict.shape[1],
                obs.device,
                obs.dtype,
                tensordict["hidden0"],
                tensordict["hidden1"],
                splits=splits,
                is_init=None,
                backend="pad",
            )

        return fn

    if backend == "scan":
        if rnn_type == "gru":

            def fn(tensordict: TensorDict) -> object:
                obs = tensordict["obs"]
                return scan_mod._gru(
                    obs,
                    tensordict.shape[0],
                    tensordict.shape[1],
                    obs.device,
                    obs.dtype,
                    tensordict["hidden"],
                    None,
                    is_init=tensordict["is_init"].squeeze(-1),
                    backend="scan",
                )

            return fn

        def fn(tensordict: TensorDict) -> object:
            obs = tensordict["obs"]
            return scan_mod._lstm(
                obs,
                tensordict.shape[0],
                tensordict.shape[1],
                obs.device,
                obs.dtype,
                tensordict["hidden0"],
                tensordict["hidden1"],
                None,
                is_init=tensordict["is_init"].squeeze(-1),
                backend="scan",
            )

        return fn

    if triton_mod is None:
        raise RuntimeError("triton recurrent backend is unavailable")
    if rnn_type == "gru":

        def fn(tensordict: TensorDict) -> object:
            obs = tensordict["obs"]
            return triton_mod._gru(
                obs,
                tensordict.shape[0],
                tensordict.shape[1],
                obs.device,
                obs.dtype,
                tensordict["hidden"],
                None,
                is_init=tensordict["is_init"].squeeze(-1),
                backend="triton",
            )

        return fn

    def fn(tensordict: TensorDict) -> object:
        obs = tensordict["obs"]
        return triton_mod._lstm(
            obs,
            tensordict.shape[0],
            tensordict.shape[1],
            obs.device,
            obs.dtype,
            tensordict["hidden0"],
            tensordict["hidden1"],
            None,
            is_init=tensordict["is_init"].squeeze(-1),
            backend="triton",
        )

    return fn


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
    device = _device()
    if backend == "triton" and device.type != "cuda":
        pytest.skip("triton recurrent backend requires CUDA")

    batch = 128
    steps = 128
    input_size = 32
    hidden_size = 256
    num_layers = 1
    reset_prob = 0.03
    generator = torch.Generator(device=device).manual_seed(reset_seed)
    cudnn_pad, scan_eager, scan_compile, triton_mod = _make_modules(
        rnn_type,
        input_size,
        hidden_size,
        num_layers,
        device,
    )
    obs = torch.randn(batch, steps, input_size, device=device, generator=generator)
    hidden0 = torch.zeros(batch, steps, num_layers, hidden_size, device=device)
    hidden = hidden0 if rnn_type == "gru" else (hidden0, torch.zeros_like(hidden0))
    is_init = _make_intermediate_resets(
        batch,
        steps,
        reset_prob,
        device,
        generator,
    )

    if backend == "triton" and triton_mod is None:
        pytest.skip("triton recurrent backend unavailable for this configuration")

    # Build a direct recurrent rollout whose callable accepts a TensorDict, so
    # compile-mode benchmarks include TensorDict input guards. The cudnn path
    # still receives precomputed intermediate-reset segments via ``splits``.
    scan_mod = scan_compile if compile else scan_eager
    tensordict, splits = _make_backend_tensordict(
        rnn_type,
        backend,
        obs,
        hidden,
        is_init,
    )
    fn = _make_backend_fn(
        rnn_type,
        backend,
        cudnn_pad,
        scan_mod,
        triton_mod,
        splits,
    )
    if compile:
        fn = torch.compile(fn, fullgraph=backend == "scan")

    # Trigger setup and compilation before timing steady-state rollout
    # throughput. Compiled runs get three warmup executions to stabilize the
    # compiled path before pytest-benchmark starts timing. The benchmark result
    # is uploaded by the continuous benchmark workflow and stored on gh-pages.
    for _ in range(3 if compile else 1):
        _execute(fn, tensordict, device)
    benchmark(_execute, fn, tensordict, device)


if __name__ == "__main__":
    _, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst"] + unknown)
