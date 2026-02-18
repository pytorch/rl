# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import functools
import importlib.util

import torch
from torchrl.envs.libs.gym import _GymAsyncMeta, GymEnv, GymWrapper
from torchrl.envs.utils import _classproperty

_has_isaaclab = importlib.util.find_spec("isaaclab") is not None


def _raise_isaaclab_import_error():
    raise ImportError(
        "IsaacLab could not be loaded. Consider installing it and importing/launching "
        "IsaacLab before creating an environment. Refer to TorchRL's knowledge base in "
        "the documentation to debug IsaacLab installation."
    )


def _wrap_import_error(fun):
    @functools.wraps(fun)
    def new_fun(*args, **kwargs):
        if not _has_isaaclab:
            _raise_isaaclab_import_error()
        return fun(*args, **kwargs)

    return new_fun


@_wrap_import_error
def _get_available_envs():
    for env in GymEnv.available_envs:
        if env.startswith("Isaac"):
            yield env


class _IsaacLabMeta(_GymAsyncMeta):
    """Metaclass for IsaacLabEnv that returns a lazy ParallelEnv when num_workers > 1."""

    def __call__(cls, *args, num_workers: int | None = None, **kwargs):
        # Extract num_workers from explicit kwarg or kwargs dict
        if num_workers is None:
            num_workers = kwargs.pop("num_workers", 1)
        else:
            kwargs.pop("num_workers", None)

        num_workers = int(num_workers) if num_workers is not None else 1
        if getattr(cls, "__name__", None) == "IsaacLabEnv" and num_workers > 1:
            from torchrl.envs import ParallelEnv

            env_name = args[0] if len(args) >= 1 else kwargs.get("env_name")
            env_kwargs = {k: v for k, v in kwargs.items() if k != "env_name"}
            make_env = functools.partial(cls, env_name, num_workers=1, **env_kwargs)
            return ParallelEnv(num_workers, make_env)

        return super().__call__(*args, **kwargs)


class _IsaacLabMixin:
    def seed(self, seed: int | None):
        self._set_seed(seed)

    def _output_transform(self, step_outputs_tuple):  # noqa: F811
        # IsaacLab will modify the `terminated` and `truncated` tensors in-place.
        # We clone them here to make sure data doesn't inadvertently get modified.
        # The variable naming follows torchrl's convention here.
        observations, reward, terminated, truncated, info = step_outputs_tuple
        done = terminated | truncated
        reward = reward.unsqueeze(-1)  # to get to (num_envs, 1)
        return (
            observations,
            reward,
            terminated.clone(),
            truncated.clone(),
            done.clone(),
            info,
        )


class IsaacLabWrapper(_IsaacLabMixin, GymWrapper):
    """A wrapper for IsaacLab environments.

    Args:
        env (scripts_isaaclab.envs.ManagerBasedRLEnv or equivalent): the environment instance to wrap.
        categorical_action_encoding (bool, optional): if ``True``, categorical
            specs will be converted to the TorchRL equivalent (:class:`torchrl.data.Categorical`),
            otherwise a one-hot encoding will be used (:class:`torchrl.data.OneHot`).
            Defaults to ``False``.
        allow_done_after_reset (bool, optional): if ``True``, it is tolerated
            for envs to be ``done`` just after :meth:`reset` is called.
            Defaults to ``True``.

    For other arguments, see the :class:`torchrl.envs.GymWrapper` documentation.

    Refer to `the Isaac Lab doc for installation instructions <https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/pip_installation.html>`_.

    Example:
        >>> # This code block ensures that the Isaac app is started in headless mode
        >>> from scripts_isaaclab.app import AppLauncher
        >>> import argparse

        >>> parser = argparse.ArgumentParser(description="Train an RL agent with TorchRL.")
        >>> AppLauncher.add_app_launcher_args(parser)
        >>> args_cli, hydra_args = parser.parse_known_args(["--headless"])
        >>> app_launcher = AppLauncher(args_cli)

        >>> # Imports and env
        >>> import gymnasium as gym
        >>> import isaaclab_tasks  # noqa: F401
        >>> from isaaclab_tasks.manager_based.classic.ant.ant_env_cfg import AntEnvCfg
        >>> from torchrl.envs.libs.isaac_lab import IsaacLabWrapper

        >>> env = gym.make("Isaac-Ant-v0", cfg=AntEnvCfg())
        >>> env = IsaacLabWrapper(env)

    """

    def __init__(
        self,
        env: isaaclab.envs.ManagerBasedRLEnv,  # noqa: F821
        *,
        categorical_action_encoding: bool = False,
        allow_done_after_reset: bool = True,
        convert_actions_to_numpy: bool = False,
        device: torch.device | None = None,
        **kwargs,
    ):
        if device is None:
            device = torch.device("cuda:0")
        super().__init__(
            env,
            device=device,
            categorical_action_encoding=categorical_action_encoding,
            allow_done_after_reset=allow_done_after_reset,
            convert_actions_to_numpy=convert_actions_to_numpy,
            **kwargs,
        )


class IsaacLabEnv(_IsaacLabMixin, GymEnv, metaclass=_IsaacLabMeta):
    """IsaacLab environment wrapper built from environment ID.

    This class behaves like :class:`~torchrl.envs.GymEnv` but applies IsaacLab-specific
    defaults and output processing.

    Args:
        env_name (str): environment ID registered in gymnasium.

    Keyword Args:
        num_workers (int, optional): if provided and greater than 1, a lazy
            :class:`torchrl.envs.ParallelEnv` will be instantiated with
            ``num_workers`` copies of ``IsaacLabEnv``. Defaults to ``1``.
        allow_done_after_reset (bool, optional): defaults to ``True`` for IsaacLab
            compatibility.
        convert_actions_to_numpy (bool, optional): defaults to ``False`` so actions
            stay as tensors.
        device (torch.device, optional): defaults to ``torch.device("cuda:0")``.

    For other keyword arguments, see :class:`~torchrl.envs.GymEnv`.
    """

    @_classproperty
    def available_envs(cls):
        if not _has_isaaclab:
            return []
        return list(_get_available_envs())

    @_wrap_import_error
    def __init__(self, env_name: str, **kwargs):
        kwargs.setdefault("backend", "gymnasium")
        kwargs.setdefault("allow_done_after_reset", True)
        kwargs.setdefault("convert_actions_to_numpy", False)
        device = kwargs.pop("device", None)
        if device is None:
            device = torch.device("cuda:0")
        kwargs["device"] = device
        super().__init__(env_name=env_name, **kwargs)
