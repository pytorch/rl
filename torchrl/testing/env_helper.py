# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations


def _isaac_app_launcher_init() -> None:
    """Initialise Isaac Lab's ``AppLauncher`` in headless mode.

    Isaac Lab requires ``AppLauncher`` to run before ``import torch`` (it
    configures Omniverse in a way that is incompatible with a pre-existing
    torch CUDA context).  This helper intentionally contains no torch imports
    and is suitable as an ``init_fn`` for the Evaluator's process / Ray
    backends.
    """
    import argparse

    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description="TorchRL Isaac Lab env launcher.")
    AppLauncher.add_app_launcher_args(parser)
    args_cli, _ = parser.parse_known_args(["--headless"])
    AppLauncher(args_cli)


def make_isaac_env(
    env_name: str = "Isaac-Ant-v0",
    *,
    device=None,
    init_app: bool = True,
    native_autoreset: bool = False,
):
    """Helper function to create an IsaacLab env.

    Args:
        env_name: gym registry id of the Isaac Lab task.
        device: optional torch device for the wrapped env.  Passed through to
            :class:`~torchrl.envs.libs.isaac_lab.IsaacLabWrapper`.  When
            ``None`` the wrapper falls back to ``cuda:0``.
        init_app: if ``True`` (default) the Omniverse ``AppLauncher`` is
            started in headless mode before the env is created.  Set to
            ``False`` when the caller has already initialised ``AppLauncher``
            (e.g. via an ``init_fn`` in a child process).
        native_autoreset: if ``True``, keep Isaac Lab's native autoreset
            observations in TorchRL's step-and-reset path.
    """
    if init_app:
        _isaac_app_launcher_init()

    import torch

    torch.manual_seed(0)

    import gymnasium as gym
    import isaaclab_tasks  # noqa: F401
    from isaaclab_tasks.manager_based.classic.ant.ant_env_cfg import AntEnvCfg
    from torchrl import logger as torchrl_logger
    from torchrl.envs.libs.isaac_lab import IsaacLabWrapper

    torchrl_logger.info("Making IsaacLab env...")
    env = gym.make(env_name, cfg=AntEnvCfg())
    torchrl_logger.info("Wrapping IsaacLab env...")
    env = IsaacLabWrapper(env, device=device, native_autoreset=native_autoreset)
    return env


def make_isaac_policy(env=None, device=None):
    """Build a minimal deterministic MLP policy for Isaac Lab Ant.

    Returns a :class:`tensordict.nn.TensorDictModule` mapping the ``"policy"``
    observation to ``"action"``.  Kept intentionally small so tests focus on
    backend wiring rather than policy correctness.

    ``env`` may be ``None``: in that case a fixed-shape placeholder is
    returned, which is what the Evaluator's process-backend probe (inside
    :class:`~torchrl.collectors.MultiSyncCollector`) calls from the main
    process to detect that this factory produces an ``nn.Module``.  The
    Isaac-Ant-v0 observation / action shapes (60 / 8) are hard-coded there
    to avoid having to start Isaac just for the probe.
    """
    import torch
    from tensordict.nn import TensorDictModule
    from torchrl.modules import MLP

    if env is None:
        obs_size, action_size = 60, 8  # Isaac-Ant-v0 shapes; probe-only.
        # Match what a real Isaac env would produce (cuda:0) so that the
        # Evaluator's main-process weight-sync probe and the worker-created
        # policy agree on device.  Fall back to CPU if CUDA is unavailable.
        if device is None:
            target_device = (
                torch.device("cuda:0")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        else:
            target_device = torch.device(device)
    else:
        obs_size = env.observation_spec["policy"].shape[-1]
        action_size = env.action_spec.shape[-1]
        target_device = torch.device(device if device is not None else env.device)

    mlp = MLP(in_features=obs_size, out_features=action_size, num_cells=[64, 64])
    module = TensorDictModule(mlp, in_keys=["policy"], out_keys=["action"]).to(
        target_device
    )
    return module
