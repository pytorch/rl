# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Profile a TorchRL collector running an Isaac Lab environment.

Isaac Lab requires ``AppLauncher`` to be initialised before importing torch.
This example therefore sets up the launcher and the TorchRL profiling env var
before importing torch / torchrl.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

os.environ.setdefault("TORCHRL_PROFILING", "1")

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(
    description="Profile a TorchRL Collector with an Isaac Lab environment."
)
parser.add_argument("--task", default="Isaac-Ant-v0", help="Isaac Lab gym task id.")
parser.add_argument(
    "--frames-per-batch",
    default=8192,
    type=int,
    help="Frames requested from the collector per rollout.",
)
parser.add_argument(
    "--profile-rollouts",
    default=5,
    type=int,
    help="Number of collector rollouts observed by torch.profiler.",
)
parser.add_argument(
    "--warmup-rollouts",
    default=1,
    type=int,
    help="Rollouts skipped by the profiler schedule before active recording.",
)
parser.add_argument(
    "--output-dir",
    default=Path("isaaclab_collector_profiles"),
    type=Path,
    help="Directory where Chrome trace JSON files are written.",
)
parser.add_argument("--seed", default=0, type=int, help="Random seed.")
parser.add_argument(
    "--env-device",
    default="cuda:0",
    help="Device passed to the TorchRL IsaacLabWrapper.",
)
parser.add_argument(
    "--activities",
    nargs="+",
    default=["cpu", "cuda"],
    choices=["cpu", "cuda"],
    help="Profiler activities.",
)
parser.add_argument(
    "--record-shapes",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Record tensor shapes in the profiler trace.",
)
parser.add_argument(
    "--profile-memory",
    action="store_true",
    help="Record memory events in the profiler trace.",
)
parser.add_argument(
    "--with-stack",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Record Python stack traces in the profiler trace.",
)
parser.add_argument(
    "--log-level",
    default="INFO",
    choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    help="Python logging level.",
)
AppLauncher.add_app_launcher_args(parser)

launch_args = sys.argv[1:]
headless_requested = any(
    arg == "--headless" or arg.startswith("--headless=") for arg in launch_args
)
if not headless_requested:
    launch_args = [*launch_args, "--headless"]
args_cli, _ = parser.parse_known_args(launch_args)
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import isaaclab_tasks  # noqa: F401
import torch

from isaaclab_tasks.manager_based.classic.ant.ant_env_cfg import AntEnvCfg
from torchrl._utils import logger as torchrl_logger
from torchrl.collectors import Collector
from torchrl.envs.libs.isaac_lab import IsaacLabWrapper


def make_env(task: str, device: str):
    if task == "Isaac-Ant-v0":
        env = gym.make(task, cfg=AntEnvCfg())
    else:
        env = gym.make(task)
    return IsaacLabWrapper(env, device=torch.device(device))


def main() -> None:
    logging.basicConfig(level=getattr(logging, args_cli.log_level))
    torch.manual_seed(args_cli.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args_cli.seed)

    total_rollouts = args_cli.warmup_rollouts + args_cli.profile_rollouts
    args_cli.output_dir.mkdir(parents=True, exist_ok=True)
    trace_path = args_cli.output_dir / "collector_worker_{worker_idx}.json"
    worker_0_trace_path = args_cli.output_dir / "collector_worker_0.json"

    torchrl_logger.info(f"Creating Isaac Lab env {args_cli.task}.")
    env = make_env(args_cli.task, args_cli.env_device)
    env.set_seed(args_cli.seed)

    collector = Collector(
        env,
        env.rand_action,
        frames_per_batch=args_cli.frames_per_batch,
        total_frames=-1,
        no_cuda_sync=True,
        trust_policy=True,
    )
    collector.enable_profile(
        workers=[0],
        num_rollouts=args_cli.profile_rollouts,
        warmup_rollouts=args_cli.warmup_rollouts,
        save_path=trace_path,
        activities=args_cli.activities,
        record_shapes=args_cli.record_shapes,
        profile_memory=args_cli.profile_memory,
        with_stack=args_cli.with_stack,
    )

    try:
        for idx, batch in enumerate(collector):
            torchrl_logger.info(
                f"Rollout {idx}: batch_size={tuple(batch.batch_size)}, "
                f"frames={batch.numel()}."
            )
            if idx + 1 >= total_rollouts:
                break
    finally:
        collector.disable_profile()
        collector.shutdown(close_env=False)
        env.close()
        simulation_app.close()

    torchrl_logger.info(f"Trace written to {worker_0_trace_path}.")


if __name__ == "__main__":
    main()
