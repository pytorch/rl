# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import gc
import time

import pytest
from tensordict import set_capture_non_tensor_stack
from torchrl.envs import ParallelEnv, SerialEnv
from torchrl.testing.mocking_classes import EnvWithMetadata


def _rollout(env, n_steps: int, break_when_any_done: bool) -> None:
    env.rollout(n_steps, break_when_any_done=break_when_any_done)


@pytest.mark.parametrize("break_when_any_done", [True, False])
@pytest.mark.parametrize(
    "kind,use_buffers",
    [
        pytest.param("single", None, id="single"),
        pytest.param("serial", False, id="serial-no-buffers"),
        pytest.param("serial", True, id="serial-buffers"),
        pytest.param("parallel", False, id="parallel-no-buffers"),
        pytest.param("parallel", True, id="parallel-buffers"),
    ],
)
@pytest.mark.parametrize("n_steps", [1000])
def test_non_tensor_env_rollout_speed(
    benchmark,
    break_when_any_done: bool,
    kind: str,
    use_buffers: bool | None,
    n_steps: int,
):
    """Benchmarks a single rollout, after a warmup rollout, for non-tensor stacking envs.

    Mirrors `test/test_env.py::TestNonTensorEnv`'s option matrix (single/serial/parallel,
    break_when_any_done, use_buffers).
    """
    with set_capture_non_tensor_stack(False):
        if kind == "single":
            env = EnvWithMetadata()
        elif kind == "serial":
            env = SerialEnv(2, EnvWithMetadata, use_buffers=use_buffers)
        elif kind == "parallel":
            env = ParallelEnv(2, EnvWithMetadata, use_buffers=use_buffers)
        else:
            raise RuntimeError(f"Unknown kind={kind}")

        env.set_seed(0)
        env.reset()

        try:
            # Warmup run (not timed)
            _rollout(env, n_steps=n_steps, break_when_any_done=break_when_any_done)

            # Timed run(s)
            benchmark(
                _rollout, env, n_steps=n_steps, break_when_any_done=break_when_any_done
            )
        finally:
            env.close(raise_if_closed=False)
            del env
            # Give multiprocessing envs a brief chance to terminate cleanly.
            time.sleep(0.05)
            gc.collect()
