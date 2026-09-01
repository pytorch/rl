# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import argparse
import time
from functools import partial

import pytest
import torch

from tensordict import set_capture_non_tensor_stack, TensorDict
from tensordict.nn import TensorDictModule
from torchrl.envs import (
    AsyncEnvPool,
    FixedBatchedInference,
    ParallelEnv,
    SerialEnv,
    step_mdp,
    StepCounter,
    TransformedEnv,
)
from torchrl.envs.libs.dm_control import DMControlEnv
from torchrl.envs.libs.libero import _has_libero, LiberoEnv
from torchrl.envs.transforms.functional import cat_frames
from torchrl.testing.mocking_classes import CountingEnv


def make_simple_env():
    device = "cuda:0" if torch.cuda.device_count() else "cpu"
    env = DMControlEnv("cheetah", "run", device=device)
    env.rollout(3)
    return ((env,), {})


def make_transformed_env():
    device = "cuda:0" if torch.cuda.device_count() else "cpu"
    env = TransformedEnv(DMControlEnv("cheetah", "run", device=device), StepCounter(50))
    env.rollout(3)
    return ((env,), {})


def make_serial_env():
    device = "cuda:0" if torch.cuda.device_count() else "cpu"
    env = SerialEnv(3, lambda: DMControlEnv("cheetah", "run", device=device))
    env.rollout(3)
    return ((env,), {})


def make_parallel_env():
    device = "cuda:0" if torch.cuda.device_count() else "cpu"
    env = ParallelEnv(3, lambda: DMControlEnv("cheetah", "run", device=device))
    env.rollout(3)
    return ((env,), {})


def make_libero_env():
    env = LiberoEnv(
        "libero_spatial",
        task_id=0,
        camera_height=256,
        camera_width=256,
        settle_steps=0,
        max_episode_steps=None,
    )
    env.rollout(3)
    return ((env,), {})


def make_nested_td():
    return TensorDict(
        {
            ("agent", "action"): 0,
            ("agent", "done"): 0,
            ("agent", "obs"): 0,
            ("agent", "other"): 0,
            ("next", "agent", "action"): 1,
            ("next", "agent", "reward"): 1,
            ("next", "agent", "done"): 1,
            ("next", "agent", "obs"): 1,
        },
        [],
    )


def make_flat_td():
    return TensorDict(
        {
            "action": 0,
            "done": 0,
            "obs": 0,
            "other": 0,
            ("next", "action"): 1,
            ("next", "reward"): 1,
            ("next", "done"): 1,
            ("next", "obs"): 1,
        },
        [],
    )


def execute_env(env):
    env.rollout(1000, break_when_any_done=False)


def test_simple(benchmark):
    (c,), _ = make_simple_env()
    benchmark(execute_env, c)


def test_transformed(benchmark):
    (c,), _ = make_transformed_env()
    benchmark(execute_env, c)


def test_serial(benchmark):
    (c,), _ = make_serial_env()
    benchmark(execute_env, c)


def test_parallel(benchmark):
    (c,), _ = make_parallel_env()
    benchmark(execute_env, c)


@pytest.mark.skipif(not _has_libero, reason="libero not found")
def test_libero(benchmark):
    # raw simulation + render throughput of the LIBERO adapter (steps/s)
    (c,), _ = make_libero_env()
    benchmark(lambda: c.rollout(100, break_when_any_done=False))


@pytest.mark.parametrize("nested", [True, False])
@pytest.mark.parametrize("keep_other", [True, False])
@pytest.mark.parametrize("exclude_reward", [True, False])
@pytest.mark.parametrize("exclude_done", [True, False])
@pytest.mark.parametrize("exclude_action", [True, False])
def test_step_mdp_speed(
    benchmark, nested, keep_other, exclude_reward, exclude_done, exclude_action
):
    if nested:
        td = make_nested_td()
        reward_key = ("agent", "reward")
        done_key = ("agent", "done")
        action_key = ("agent", "action")
    else:
        td = make_flat_td()
        reward_key = "reward"
        done_key = "done"
        action_key = "action"

    benchmark(
        step_mdp,
        td,
        action_keys=action_key,
        reward_keys=reward_key,
        done_keys=done_key,
        keep_other=keep_other,
        exclude_reward=exclude_reward,
        exclude_done=exclude_done,
        exclude_action=exclude_action,
    )


@pytest.mark.parametrize("padding", ["same", "constant"])
@pytest.mark.parametrize("N", [4, 16])
def test_cat_frames_functional(benchmark, padding, N):
    device = "cuda:0" if torch.cuda.device_count() else "cpu"
    # batch of trajectories: (batch, time, channels)
    tensor = torch.randn(32, 200, 8, device=device)
    benchmark(
        cat_frames,
        tensor,
        N,
        dim=-1,
        padding=padding,
        time_dim=-2,
    )


# AsyncEnvPool throughput benchmarks (north-star series for #4061).
#
# These series track the async data plane on the continuous benchmark
# dashboard and are expected to fall as the exchange internals are optimized.
# Series names are load-bearing for trend continuity - do not rename them, and
# do not change the workload constants below; add new series instead.
#
# - test_async_env_pool_dispatch: free envs, so the round time is
#   ASYNC_POOL_DISPATCH_TRANSITIONS x the consumer-side dispatch cost
#   (recv -> action write -> send) per transition. The most sensitive tracker.
# - test_async_env_pool_fast_step_slow_reset: many envs with millisecond steps
#   and long occasional resets (the game-engine regime). Sized so that the
#   consumer is the bottleneck today: the round time falls toward the
#   env-supply ceiling as dispatch gets cheaper.

ASYNC_POOL_DISPATCH_ENVS = 8
ASYNC_POOL_DISPATCH_TRANSITIONS = 1024
ASYNC_POOL_REGIME_ENVS = 32
ASYNC_POOL_REGIME_TRANSITIONS = 2048
ASYNC_POOL_REGIME_STEP_LATENCY = 1e-3
ASYNC_POOL_REGIME_RESET_LATENCY = 0.2
ASYNC_POOL_REGIME_EPISODE_STEPS = 200


class DelayedCountingEnv(CountingEnv):
    """A CountingEnv with configurable synchronous step and reset latencies."""

    def __init__(
        self, *, step_latency: float = 0.0, reset_latency: float = 0.0, **kwargs
    ):
        super().__init__(**kwargs)
        self.step_latency = step_latency
        self.reset_latency = reset_latency

    def _step(self, tensordict):
        if self.step_latency:
            time.sleep(self.step_latency)
        return super()._step(tensordict)

    def _reset(self, tensordict, **kwargs):
        if self.reset_latency:
            time.sleep(self.reset_latency)
        return super()._reset(tensordict, **kwargs)


def _make_async_pool(num_envs, exchange, step_latency, reset_latency, max_steps):
    pool = AsyncEnvPool(
        [
            partial(
                DelayedCountingEnv,
                max_steps=max_steps,
                step_latency=step_latency,
                reset_latency=reset_latency,
            )
        ]
        * num_envs,
        backend="multiprocessing",
        exchange=exchange,
    )
    # Prime the steady state: after this, every round starts with a recv.
    tensordict = pool.reset()
    tensordict["action"] = torch.ones(num_envs, 1)
    pool.async_step_and_maybe_reset_send(tensordict)
    return pool


def _async_pool_harvest(pool, num_transitions, max_get):
    harvested = 0
    while harvested < num_transitions:
        _, td_next = pool.async_step_and_maybe_reset_recv(
            min_get=1, max_get=max_get, timeout=1e-3
        )
        num_ready = td_next.shape[0]
        td_next["action"] = torch.ones(num_ready, 1)
        pool.async_step_and_maybe_reset_send(td_next)
        harvested += num_ready


@pytest.mark.parametrize("exchange", ["queue", "shm"])
def test_async_env_pool_dispatch(benchmark, exchange):
    """Consumer dispatch cost of AsyncEnvPool with free envs."""
    with set_capture_non_tensor_stack(False):
        pool = _make_async_pool(
            ASYNC_POOL_DISPATCH_ENVS,
            exchange,
            step_latency=0.0,
            reset_latency=0.0,
            max_steps=10_000_000,
        )
        try:
            benchmark.extra_info["num_envs"] = ASYNC_POOL_DISPATCH_ENVS
            benchmark.extra_info["transitions"] = ASYNC_POOL_DISPATCH_TRANSITIONS
            benchmark.pedantic(
                _async_pool_harvest,
                args=(
                    pool,
                    ASYNC_POOL_DISPATCH_TRANSITIONS,
                    ASYNC_POOL_DISPATCH_ENVS,
                ),
                rounds=5,
                warmup_rounds=1,
                iterations=1,
            )
        finally:
            pool._maybe_shutdown()


# Note: the queue exchange is deliberately not parametrized here. At 32
# workers the queue path hangs during setup/reset with the spawn start method
# (macOS-confirmed; tracked for a fix in the AsyncEnvPool v2 sequence). The
# queue baseline is covered by test_async_env_pool_dispatch at 8 workers.
@pytest.mark.parametrize("exchange", ["shm"])
def test_async_env_pool_fast_step_slow_reset(benchmark, exchange):
    """AsyncEnvPool throughput with millisecond steps and long resets.

    32 envs with 1 ms steps and a 200 ms reset every 200 steps supply roughly
    16k transitions/s; resets are absorbed worker-side by
    ``step_and_maybe_reset``. The round time is dispatch-bound until the
    consumer path gets cheaper than the env supply.
    """
    with set_capture_non_tensor_stack(False):
        pool = _make_async_pool(
            ASYNC_POOL_REGIME_ENVS,
            exchange,
            step_latency=ASYNC_POOL_REGIME_STEP_LATENCY,
            reset_latency=ASYNC_POOL_REGIME_RESET_LATENCY,
            max_steps=ASYNC_POOL_REGIME_EPISODE_STEPS,
        )
        try:
            benchmark.extra_info["num_envs"] = ASYNC_POOL_REGIME_ENVS
            benchmark.extra_info["transitions"] = ASYNC_POOL_REGIME_TRANSITIONS
            benchmark.pedantic(
                _async_pool_harvest,
                args=(
                    pool,
                    ASYNC_POOL_REGIME_TRANSITIONS,
                    ASYNC_POOL_REGIME_ENVS,
                ),
                rounds=5,
                warmup_rounds=1,
                iterations=1,
            )
        finally:
            pool._maybe_shutdown()


def test_fixed_batched_inference_overhead(benchmark):
    """Per-call overhead of the fixed-shape inference helper (CPU path).

    A trivial policy over small observations, so the round time is dominated
    by staging (key selection, padded copy, mask) rather than compute.
    """
    policy = TensorDictModule(
        torch.nn.Linear(4, 2), in_keys=["obs"], out_keys=["action"]
    )
    helper = FixedBatchedInference(policy, "cpu", bucket_sizes=[64])
    batch = TensorDict({"obs": torch.randn(48, 4)}, batch_size=[48])
    benchmark(helper, batch)


if __name__ == "__main__":
    args, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst"] + unknown)
