# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations


def make_isaac_env(env_name: str = "Isaac-Ant-v0"):
    """Helper function to create an IsaacLab env."""
    import torch

    torch.manual_seed(0)
    import argparse

    # This code block ensures that the Isaac app is started in headless mode
    from isaaclab.app import AppLauncher
    from torchrl import logger as torchrl_logger

    parser = argparse.ArgumentParser(description="Train an RL agent with TorchRL.")
    AppLauncher.add_app_launcher_args(parser)
    args_cli, hydra_args = parser.parse_known_args(["--headless"])
    AppLauncher(args_cli)

    # Imports and env
    import isaaclab_tasks  # noqa: F401
    from isaaclab_tasks.manager_based.classic.ant.ant_env_cfg import AntEnvCfg
    from torchrl.envs.libs.isaac_lab import IsaacLabEnv

    torchrl_logger.info("Making IsaacLab env...")
    env = IsaacLabEnv(env_name, cfg=AntEnvCfg())
    return env
