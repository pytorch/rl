# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import argparse
import os
import statistics
import time
from functools import partial

import pytest
import torch

from tensordict import set_capture_non_tensor_stack, TensorDict
from torchrl.envs import (
    AsyncEnvPool,
    LastAction,
    ParallelEnv,
    SerialEnv,
    step_mdp,
    StepCounter,
    TransformedEnv,
)
from torchrl.envs.libs.dm_control import DMControlEnv
from torchrl.envs.libs.libero import _has_libero, LiberoEnv
from torchrl.envs.transforms.functional import cat_frames
from torchrl.testing.mocking_classes import ContinuousActionVecMockEnv, CountingEnv


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
# - test_async_env_pool_per_env_dispatch: the same free-env workload through
#   the per-env receive path used by AsyncBatchedCollector.
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
ASYNC_POOL_AFFINITY_MIN_CPUS = 16
ASYNC_POOL_AFFINITY_ENVS = 32
ASYNC_POOL_AFFINITY_TRANSITIONS = 2048
ASYNC_POOL_AFFINITY_STEP_LATENCY = 1e-3

_AFFINITY_CPUS = (
    tuple(sorted(os.sched_getaffinity(0))) if hasattr(os, "sched_getaffinity") else ()
)
ASYNC_POOL_GROUPING_ENVS = 64
ASYNC_POOL_GROUPING_TRANSITIONS = 512
ASYNC_POOL_GROUPING_STEP_LATENCY = 0.05


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


def _make_async_pool(
    num_envs,
    exchange,
    step_latency,
    reset_latency,
    max_steps,
    worker_affinity=None,
    envs_per_worker=1,
):
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
        worker_affinity=worker_affinity,
        envs_per_worker=envs_per_worker,
    )
    # Prime the steady state: after this, every round starts with a recv.
    tensordict = pool.reset()
    tensordict["action"] = torch.ones(num_envs, 1)
    pool.async_step_and_maybe_reset_send(tensordict)
    return pool


def _async_pool_harvest(pool, num_transitions, max_get):
    harvested = 0
    while harvested < num_transitions:
        # timeout=None: block for the first result, then drain whatever else
        # is ready without waiting. A finite timeout bounds the whole call
        # (#4184) and raises TimeoutError whenever recv lands while all envs
        # are mid-step, which is the normal state of this loop.
        _, td_next = pool.async_step_and_maybe_reset_recv(
            min_get=1, max_get=min(max_get, num_transitions - harvested)
        )
        num_ready = td_next.shape[0]
        td_next["action"] = torch.ones(num_ready, 1)
        pool.async_step_and_maybe_reset_send(td_next)
        harvested += num_ready


def _make_async_pool_per_env(num_envs, exchange):
    pool = AsyncEnvPool(
        [partial(DelayedCountingEnv, max_steps=10_000_000)] * num_envs,
        backend="multiprocessing",
        exchange=exchange,
    )
    for env_index in range(num_envs):
        pool.async_reset_send(env_index=env_index)
    for env_index in range(num_envs):
        tensordict = pool.async_reset_recv(env_index=env_index)
        tensordict["action"] = torch.ones(1)
        pool.async_step_and_maybe_reset_send(tensordict, env_index=env_index)
    return pool


def _async_pool_harvest_per_env(pool, num_transitions):
    for index in range(num_transitions):
        env_index = index % pool.num_envs
        _, tensordict = pool.async_step_and_maybe_reset_recv(env_index=env_index)
        tensordict["action"] = torch.ones(1)
        pool.async_step_and_maybe_reset_send(tensordict, env_index=env_index)


def _async_pool_step_latencies(pool, num_transitions):
    # Drain every in-flight warm-up step, then send a fresh synchronized round
    # so each observed latency has a precise dispatch timestamp.
    _, td_next = pool.async_step_and_maybe_reset_recv(min_get=pool.num_envs)
    td_next["action"] = torch.ones(pool.num_envs, 1)
    sent_at = time.perf_counter()
    pool.async_step_and_maybe_reset_send(td_next)
    send_times = [sent_at] * pool.num_envs

    latencies = []
    while len(latencies) < num_transitions:
        _, td_next = pool.async_step_and_maybe_reset_recv(
            min_get=1, max_get=min(pool.num_envs, num_transitions - len(latencies))
        )
        observed_at = time.perf_counter()
        env_indices = td_next["env_index"]
        if isinstance(env_indices, torch.Tensor):
            env_indices = env_indices.reshape(-1).tolist()
        else:
            env_indices = [int(env_index) for env_index in env_indices]
        latencies.extend(
            observed_at - send_times[env_index] for env_index in env_indices
        )

        td_next["action"] = torch.ones(td_next.shape[0], 1)
        sent_at = time.perf_counter()
        pool.async_step_and_maybe_reset_send(td_next)
        for env_index in env_indices:
            send_times[env_index] = sent_at
    return latencies[:num_transitions]


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


@pytest.mark.parametrize("exchange", ["queue", "shm"])
def test_async_env_pool_per_env_dispatch(benchmark, exchange):
    """Per-env dispatch cost on the path used by AsyncBatchedCollector."""
    with set_capture_non_tensor_stack(False):
        pool = _make_async_pool_per_env(ASYNC_POOL_DISPATCH_ENVS, exchange)
        try:
            benchmark.extra_info["num_envs"] = ASYNC_POOL_DISPATCH_ENVS
            benchmark.extra_info["transitions"] = ASYNC_POOL_DISPATCH_TRANSITIONS
            benchmark.pedantic(
                _async_pool_harvest_per_env,
                args=(pool, ASYNC_POOL_DISPATCH_TRANSITIONS),
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


@pytest.mark.skipif(
    len(_AFFINITY_CPUS) < ASYNC_POOL_AFFINITY_MIN_CPUS,
    reason="CPU-affinity jitter benchmark requires a many-core Linux host",
)
@pytest.mark.parametrize("pinned", [False, True], ids=["default", "pinned"])
def test_async_env_pool_step_latency_jitter(benchmark, pinned):
    """Track pool wall time and last-round latency with worker-only affinity.

    ``pytest-benchmark`` measures each complete 2048-transition round. The
    latency summary in ``extra_info`` describes only the samples returned by
    the last of five measured rounds; the driver remains unpinned in both
    variants.
    """
    num_envs = min(ASYNC_POOL_AFFINITY_ENVS, len(_AFFINITY_CPUS))
    worker_affinity = [(cpu,) for cpu in _AFFINITY_CPUS[:num_envs]] if pinned else None
    with set_capture_non_tensor_stack(False):
        pool = _make_async_pool(
            num_envs,
            exchange="shm",
            step_latency=ASYNC_POOL_AFFINITY_STEP_LATENCY,
            reset_latency=0.0,
            max_steps=10_000_000,
            worker_affinity=worker_affinity,
        )
        try:
            samples = benchmark.pedantic(
                _async_pool_step_latencies,
                args=(pool, ASYNC_POOL_AFFINITY_TRANSITIONS),
                rounds=5,
                warmup_rounds=1,
                iterations=1,
            )
            samples_ms = sorted(sample * 1e3 for sample in samples)
            benchmark.extra_info.update(
                {
                    "affinity": "pinned" if pinned else "default",
                    "num_envs": num_envs,
                    "transitions": ASYNC_POOL_AFFINITY_TRANSITIONS,
                    "step_latency_mean_ms": statistics.fmean(samples_ms),
                    "step_latency_stddev_ms": statistics.pstdev(samples_ms),
                    "step_latency_p50_ms": samples_ms[len(samples_ms) // 2],
                    "step_latency_p99_ms": samples_ms[
                        min(len(samples_ms) - 1, int(len(samples_ms) * 0.99))
                    ],
                }
            )
        finally:
            pool._maybe_shutdown()


@pytest.mark.parametrize(
    "envs_per_worker",
    [1, 4],
    ids=["64-processes", "16-processes-of-4"],
)
def test_async_env_pool_multi_env_workers(benchmark, envs_per_worker):
    """Compare process layouts for many sleeping environments."""
    with set_capture_non_tensor_stack(False):
        pool = _make_async_pool(
            ASYNC_POOL_GROUPING_ENVS,
            "shm",
            step_latency=ASYNC_POOL_GROUPING_STEP_LATENCY,
            reset_latency=0.0,
            max_steps=10_000_000,
            envs_per_worker=envs_per_worker,
        )
        try:
            benchmark.extra_info["num_envs"] = ASYNC_POOL_GROUPING_ENVS
            benchmark.extra_info["num_processes"] = pool.num_workers
            benchmark.extra_info["envs_per_worker"] = envs_per_worker
            benchmark.extra_info["transitions"] = ASYNC_POOL_GROUPING_TRANSITIONS
            benchmark.pedantic(
                _async_pool_harvest,
                args=(
                    pool,
                    ASYNC_POOL_GROUPING_TRANSITIONS,
                    ASYNC_POOL_GROUPING_ENVS,
                ),
                rounds=5,
                warmup_rounds=1,
                iterations=1,
            )
        finally:
            pool._maybe_shutdown()


def _make_last_action_env():
    env = TransformedEnv(ContinuousActionVecMockEnv(), LastAction())
    env.rollout(3, break_when_any_done=False)
    return env


def test_last_action_reset(benchmark):
    env = _make_last_action_env()
    benchmark(env.reset)


def test_last_action_step(benchmark):
    env = _make_last_action_env()
    td = env.reset()
    td = env.rand_action(td)

    def _step():
        return env.step(td.clone())

    benchmark(_step)


if __name__ == "__main__":
    args, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst"] + unknown)
