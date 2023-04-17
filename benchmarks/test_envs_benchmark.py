# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import torch

from torchrl.envs import ParallelEnv, SerialEnv, StepCounter, TransformedEnv
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


def execute_env(env):
    env.rollout(1000, break_when_any_done=False)


def test_simple(benchmark):
    benchmark.pedantic(execute_env, setup=make_simple_env, iterations=1, rounds=5)


def test_transformed(benchmark):
    benchmark.pedantic(execute_env, setup=make_transformed_env, iterations=1, rounds=5)


def test_serial(benchmark):
    benchmark.pedantic(execute_env, setup=make_serial_env, iterations=1, rounds=5)


def test_parallel(benchmark):
    benchmark.pedantic(execute_env, setup=make_parallel_env, iterations=1, rounds=5)
