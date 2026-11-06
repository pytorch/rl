# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

_ISAACLAB_LAUNCHES = []


def _isaaclab_launches_per_task() -> bool:
    """Return whether Isaac Lab launches its simulation runtime per task.

    Isaac Lab 3.x exposes ``isaaclab.app.launch_simulation``, which starts only
    the runtimes a task config needs (none for the default Newton physics).
    Isaac Lab 2.x always starts Isaac Sim through ``AppLauncher``.
    """
    import isaaclab.app

    return hasattr(isaaclab.app, "launch_simulation")


def _isaac_app_launcher_init() -> None:
    """Initialise Isaac Lab's ``AppLauncher`` in headless mode.

    Isaac Lab 2.x requires ``AppLauncher`` to run before ``import torch`` (it
    configures Omniverse in a way that is incompatible with a pre-existing
    torch CUDA context).  This helper intentionally contains no torch imports
    and is suitable as an ``init_fn`` for the Evaluator's process / Ray
    backends.  On Isaac Lab 3.x it does nothing: :func:`make_isaac_env`
    launches the simulation runtime of each task.
    """
    import argparse

    if _isaaclab_launches_per_task():
        return

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
    num_envs: int | None = None,
):
    """Helper function to create an IsaacLab env.

    Args:
        env_name: gym registry id of the Isaac Lab task.  Isaac Lab 3.x
            registers tasks without the ``-v0`` suffix; a ``-v0`` id that is
            not registered is looked up without it.
        device: optional torch device for the wrapped env.  Passed through to
            :class:`~torchrl.envs.libs.isaac_lab.IsaacLabWrapper`.  When
            ``None`` the wrapper falls back to ``cuda:0``.
        init_app: if ``True`` (default) the Omniverse ``AppLauncher`` is
            started in headless mode before the env is created.  Set to
            ``False`` when the caller has already initialised ``AppLauncher``
            (e.g. via an ``init_fn`` in a child process).
        native_autoreset: if ``True``, keep Isaac Lab's native autoreset
            observations in TorchRL's step-and-reset path.
        num_envs: optional override for the number of sub-environments. When
            ``None`` (the default), Isaac Lab's per-task default is kept.
    """
    if init_app:
        _isaac_app_launcher_init()

    import importlib

    import torch

    torch.manual_seed(0)

    import gymnasium
    import isaaclab_tasks  # noqa: F401
    from torchrl import logger as torchrl_logger
    from torchrl.envs.libs.isaac_lab import IsaacLabWrapper

    if env_name not in gymnasium.registry and env_name.endswith("-v0"):
        env_name = env_name.removesuffix("-v0")
    torchrl_logger.info("Making IsaacLab env %s...", env_name)
    if _isaaclab_launches_per_task():
        env_cfg = _launch_isaaclab_task(env_name, num_envs)
    else:
        spec = gymnasium.spec(env_name)
        entry = spec.kwargs["env_cfg_entry_point"]
        if isinstance(entry, str):
            module_path, class_name = entry.rsplit(":", 1)
            env_cfg = getattr(importlib.import_module(module_path), class_name)()
        else:
            env_cfg = entry()
        if num_envs is not None:
            env_cfg.scene.num_envs = num_envs
    env = gymnasium.make(env_name, cfg=env_cfg)
    torchrl_logger.info("Wrapping IsaacLab env...")
    env = IsaacLabWrapper(env, device=device, native_autoreset=native_autoreset)
    return env


def _launch_isaaclab_task(env_name: str, num_envs: int | None):
    """Resolve an Isaac Lab 3.x task config and launch its simulation runtime.

    The task keeps its default physics preset (Newton on the core tasks), so no
    Isaac Sim installation is needed.  The launch stays open until the process
    exits, as Isaac Lab's own scripts keep it open around the environment.
    """
    import argparse

    from isaaclab.app import add_launcher_args, launch_simulation
    from isaaclab_tasks.utils import resolve_task_config

    env_cfg, _ = resolve_task_config(env_name, None, overrides=[])
    if num_envs is not None:
        env_cfg.scene.num_envs = num_envs
    parser = argparse.ArgumentParser(description="TorchRL Isaac Lab env launcher.")
    add_launcher_args(parser)
    args_cli, _ = parser.parse_known_args([])
    launch = launch_simulation(env_cfg, args_cli)
    launch.__enter__()
    _ISAACLAB_LAUNCHES.append(launch)
    return env_cfg


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
