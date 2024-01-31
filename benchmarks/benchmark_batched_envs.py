# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Benchmarking different types of batched environments
====================================================
Compares runtime for different environments which allow performing operations in a batch.
- SerialEnv executes the operations sequentially
- ParallelEnv uses multiprocess parallelism
- MultiThreadedEnv uses multithreaded parallelism and is based on envpool library.

Run as "python benchmarks/benchmark_batched_envs.py"
Requires pandas ("pip install pandas").

"""

import pandas as pd
from torchrl._utils import logger as torchrl_logger

pd.set_option("display.max_columns", 100)
pd.set_option("display.width", 1000)
import torch
from torch.utils.benchmark import Timer
from torchrl.envs import MultiThreadedEnv, ParallelEnv, SerialEnv
from torchrl.envs.libs.gym import GymEnv

N_STEPS = 1000


def create_multithreaded(num_workers, device):
    env = MultiThreadedEnv(num_workers=num_workers, env_name="Pendulum-v1")
    # GPU doesn't lead to any speedup for MultiThreadedEnv, as the underlying library (envpool) works only on CPU
    env = env.to(device=torch.device(device))
    env.rollout(policy=None, max_steps=5)  # Warm-up
    return env


def factory():
    return GymEnv("Pendulum-v1")


def create_serial(num_workers, device):
    env = SerialEnv(num_workers=num_workers, create_env_fn=factory)
    env = env.to(device=torch.device(device))
    env.rollout(policy=None, max_steps=5)  # Warm-up
    return env


def create_parallel(num_workers, device):
    env = ParallelEnv(num_workers=num_workers, create_env_fn=factory)
    env = env.to(device=torch.device(device))
    env.rollout(policy=None, max_steps=5)  # Warm-up
    return env


def run_env(env):
    env.rollout(policy=None, max_steps=N_STEPS)


if __name__ == "__main__":
    res = {}
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")
    for device in devices:
        for num_workers in [1, 4, 16]:
            torchrl_logger.info(f"With num_workers={num_workers}, {device}")
            torchrl_logger.info("Multithreaded...")
            env_multithreaded = create_multithreaded(num_workers, device)
            res_multithreaded = Timer(
                stmt="run_env(env)",
                setup="from __main__ import run_env",
                globals={"env": env_multithreaded},
            )
            time_multithreaded = res_multithreaded.blocked_autorange().mean

            torchrl_logger.info("Serial...")
            env_serial = create_serial(num_workers, device)
            res_serial = Timer(
                stmt="run_env(env)",
                setup="from __main__ import run_env",
                globals={"env": env_serial},
            )
            time_serial = res_serial.blocked_autorange().mean

            torchrl_logger.info("Parallel...")
            env_parallel = create_parallel(num_workers, device)
            res_parallel = Timer(
                stmt="run_env(env)",
                setup="from __main__ import run_env",
                globals={"env": env_parallel},
            )
            time_parallel = res_parallel.blocked_autorange().mean

            res[f"num_workers_{num_workers}_{device}"] = {
                "Serial, s": time_serial,
                "Parallel, s": time_parallel,
                "Multithreaded, s": time_multithreaded,
            }
    df = pd.DataFrame(res).round(3)
    gain = 1 - df.loc["Multithreaded, s"] / df.loc["Parallel, s"]
    df.loc["Gain, %", :] = (gain * 100).round(1)
    df.to_csv("multithreaded_benchmark.csv")
