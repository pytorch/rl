# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import argparse

import pytest
import torch

from tensordict import TensorDict
from torchrl.envs import ParallelEnv, SerialEnv, step_mdp, StepCounter, TransformedEnv
from torchrl.envs.libs.dm_control import DMControlEnv


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


if __name__ == "__main__":
    args, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst"] + unknown)
