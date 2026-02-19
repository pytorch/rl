# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from omegaconf import MISSING
from torchrl.envs.libs.gym import set_gym_backend
from torchrl.envs.transforms.transforms import DoubleToFloat
from torchrl.trainers.algorithms.configs.common import ConfigBase


@dataclass
class EnvLibsConfig(ConfigBase):
    """Base configuration class for environment libs."""

    _partial_: bool = False

    def __post_init__(self) -> None:
        """Post-initialization hook for environment libs configurations."""


@dataclass
class GymEnvConfig(EnvLibsConfig):
    """Configuration for GymEnv environment."""

    env_name: str = MISSING
    categorical_action_encoding: bool = False
    from_pixels: bool = False
    pixels_only: bool = True
    frame_skip: int = 1
    device: str = "cpu"
    batch_size: list[int] | None = None
    allow_done_after_reset: bool = False
    convert_actions_to_numpy: bool = True
    missing_obs_value: Any = None
    disable_env_checker: bool | None = None
    render_mode: str | None = None
    num_envs: int = 0
    backend: str = "gymnasium"
    _target_: str = "torchrl.trainers.algorithms.configs.envs_libs.make_gym_env"

    def __post_init__(self) -> None:
        """Post-initialization hook for GymEnv configuration."""
        super().__post_init__()


def make_gym_env(
    env_name: str,
    backend: str = "gymnasium",
    from_pixels: bool = False,
    double_to_float: bool = False,
    **kwargs,
):
    """Create a Gym/Gymnasium environment.

    Args:
        env_name: Name of the environment to create.
        backend: Backend to use (gym or gymnasium).
        from_pixels: Whether to use pixel observations.
        double_to_float: Whether to convert double to float.

    Returns:
        The created environment instance.
    """
    from torchrl.envs.libs.gym import GymEnv

    if backend is not None:
        with set_gym_backend(backend):
            env = GymEnv(env_name, from_pixels=from_pixels, **kwargs)
    else:
        env = GymEnv(env_name, from_pixels=from_pixels, **kwargs)

    if double_to_float:
        env = env.append_transform(DoubleToFloat(in_keys=["observation"]))

    return env


@dataclass
class MOGymEnvConfig(EnvLibsConfig):
    """Configuration for MOGymEnv environment."""

    env_name: str = MISSING
    categorical_action_encoding: bool = False
    from_pixels: bool = False
    pixels_only: bool = True
    frame_skip: int | None = None
    device: str = "cpu"
    batch_size: list[int] | None = None
    allow_done_after_reset: bool = False
    convert_actions_to_numpy: bool = True
    missing_obs_value: Any = None
    backend: str | None = None
    disable_env_checker: bool | None = None
    render_mode: str | None = None
    num_envs: int = 0
    _target_: str = "torchrl.envs.libs.gym.MOGymEnv"

    def __post_init__(self) -> None:
        """Post-initialization hook for MOGymEnv configuration."""
        super().__post_init__()


@dataclass
class BraxEnvConfig(EnvLibsConfig):
    """Configuration for BraxEnv environment."""

    env_name: str = MISSING
    categorical_action_encoding: bool = False
    cache_clear_frequency: int | None = None
    from_pixels: bool = False
    frame_skip: int | None = None
    device: str = "cpu"
    batch_size: list[int] | None = None
    allow_done_after_reset: bool = False
    requires_grad: bool = False
    _target_: str = "torchrl.envs.libs.brax.BraxEnv"

    def __post_init__(self) -> None:
        """Post-initialization hook for BraxEnv configuration."""
        super().__post_init__()


@dataclass
class DMControlEnvConfig(EnvLibsConfig):
    """Configuration for DMControlEnv environment."""

    env_name: str = MISSING
    task_name: str = MISSING
    from_pixels: bool = False
    pixels_only: bool = True
    frame_skip: int | None = None
    device: str = "cpu"
    batch_size: list[int] | None = None
    allow_done_after_reset: bool = False
    _target_: str = "torchrl.envs.libs.dm_control.DMControlEnv"

    def __post_init__(self) -> None:
        """Post-initialization hook for DMControlEnv configuration."""
        super().__post_init__()


@dataclass
class HabitatEnvConfig(EnvLibsConfig):
    """Configuration for HabitatEnv environment."""

    env_name: str = MISSING
    from_pixels: bool = False
    pixels_only: bool = True
    frame_skip: int | None = None
    device: str = "cpu"
    batch_size: list[int] | None = None
    allow_done_after_reset: bool = False
    _target_: str = "torchrl.envs.libs.habitat.HabitatEnv"

    def __post_init__(self) -> None:
        """Post-initialization hook for HabitatEnv configuration."""
        super().__post_init__()


@dataclass
class IsaacGymEnvConfig(EnvLibsConfig):
    """Configuration for IsaacGymEnv environment."""

    env_name: str = MISSING
    from_pixels: bool = False
    pixels_only: bool = True
    frame_skip: int | None = None
    device: str = "cpu"
    batch_size: list[int] | None = None
    allow_done_after_reset: bool = False
    _target_: str = "torchrl.envs.libs.isaacgym.IsaacGymEnv"

    def __post_init__(self) -> None:
        """Post-initialization hook for IsaacGymEnv configuration."""
        super().__post_init__()


@dataclass
class JumanjiEnvConfig(EnvLibsConfig):
    """Configuration for JumanjiEnv environment."""

    env_name: str = MISSING
    from_pixels: bool = False
    pixels_only: bool = True
    frame_skip: int | None = None
    device: str = "cpu"
    batch_size: list[int] | None = None
    allow_done_after_reset: bool = False
    _target_: str = "torchrl.envs.libs.jumanji.JumanjiEnv"

    def __post_init__(self) -> None:
        """Post-initialization hook for JumanjiEnv configuration."""
        super().__post_init__()


@dataclass
class MeltingpotEnvConfig(EnvLibsConfig):
    """Configuration for MeltingpotEnv environment."""

    env_name: str = MISSING
    from_pixels: bool = False
    pixels_only: bool = True
    frame_skip: int | None = None
    device: str = "cpu"
    batch_size: list[int] | None = None
    allow_done_after_reset: bool = False
    _target_: str = "torchrl.envs.libs.meltingpot.MeltingpotEnv"

    def __post_init__(self) -> None:
        """Post-initialization hook for MeltingpotEnv configuration."""
        super().__post_init__()


@dataclass
class OpenEnvEnvConfig(EnvLibsConfig):
    """Configuration for OpenEnvEnv environment."""

    env_name: str = MISSING
    auto_action: bool = True
    env_kwargs: dict[str, Any] = field(default_factory=dict)
    action_cls: Any | None = None
    observation_cls: Any | None = None
    return_observation_dict: bool = False
    sync: bool = True
    device: str = "cpu"
    batch_size: list[int] | None = None
    allow_done_after_reset: bool = False
    _target_: str = "torchrl.envs.libs.openenv.OpenEnvEnv"

    def __post_init__(self) -> None:
        """Post-initialization hook for OpenEnvEnv configuration."""
        super().__post_init__()


@dataclass
class OpenMLEnvConfig(EnvLibsConfig):
    """Configuration for OpenMLEnv environment."""

    env_name: str = MISSING
    from_pixels: bool = False
    pixels_only: bool = True
    frame_skip: int | None = None
    device: str = "cpu"
    batch_size: list[int] | None = None
    allow_done_after_reset: bool = False
    _target_: str = "torchrl.envs.libs.openml.OpenMLEnv"

    def __post_init__(self) -> None:
        """Post-initialization hook for OpenMLEnv configuration."""
        super().__post_init__()


@dataclass
class OpenSpielEnvConfig(EnvLibsConfig):
    """Configuration for OpenSpielEnv environment."""

    env_name: str = MISSING
    from_pixels: bool = False
    pixels_only: bool = True
    frame_skip: int | None = None
    device: str = "cpu"
    batch_size: list[int] | None = None
    allow_done_after_reset: bool = False
    _target_: str = "torchrl.envs.libs.openspiel.OpenSpielEnv"

    def __post_init__(self) -> None:
        """Post-initialization hook for OpenSpielEnv configuration."""
        super().__post_init__()


@dataclass
class PettingZooEnvConfig(EnvLibsConfig):
    """Configuration for PettingZooEnv environment."""

    env_name: str = MISSING
    from_pixels: bool = False
    pixels_only: bool = True
    frame_skip: int | None = None
    device: str = "cpu"
    batch_size: list[int] | None = None
    allow_done_after_reset: bool = False
    _target_: str = "torchrl.envs.libs.pettingzoo.PettingZooEnv"

    def __post_init__(self) -> None:
        """Post-initialization hook for PettingZooEnv configuration."""
        super().__post_init__()


@dataclass
class RoboHiveEnvConfig(EnvLibsConfig):
    """Configuration for RoboHiveEnv environment."""

    env_name: str = MISSING
    from_pixels: bool = False
    pixels_only: bool = True
    frame_skip: int | None = None
    device: str = "cpu"
    batch_size: list[int] | None = None
    allow_done_after_reset: bool = False
    _target_: str = "torchrl.envs.libs.robohive.RoboHiveEnv"

    def __post_init__(self) -> None:
        """Post-initialization hook for RoboHiveEnv configuration."""
        super().__post_init__()


@dataclass
class SMACv2EnvConfig(EnvLibsConfig):
    """Configuration for SMACv2Env environment."""

    env_name: str = MISSING
    from_pixels: bool = False
    pixels_only: bool = True
    frame_skip: int | None = None
    device: str = "cpu"
    batch_size: list[int] | None = None
    allow_done_after_reset: bool = False
    _target_: str = "torchrl.envs.libs.smacv2.SMACv2Env"

    def __post_init__(self) -> None:
        """Post-initialization hook for SMACv2Env configuration."""
        super().__post_init__()


@dataclass
class UnityMLAgentsEnvConfig(EnvLibsConfig):
    """Configuration for UnityMLAgentsEnv environment."""

    env_name: str = MISSING
    from_pixels: bool = False
    pixels_only: bool = True
    frame_skip: int | None = None
    device: str = "cpu"
    batch_size: list[int] | None = None
    allow_done_after_reset: bool = False
    _target_: str = "torchrl.envs.libs.unity_mlagents.UnityMLAgentsEnv"

    def __post_init__(self) -> None:
        """Post-initialization hook for UnityMLAgentsEnv configuration."""
        super().__post_init__()


@dataclass
class VmasEnvConfig(EnvLibsConfig):
    """Configuration for VmasEnv environment."""

    env_name: str = MISSING
    from_pixels: bool = False
    pixels_only: bool = True
    frame_skip: int | None = None
    device: str = "cpu"
    batch_size: list[int] | None = None
    allow_done_after_reset: bool = False
    _target_: str = "torchrl.envs.libs.vmas.VmasEnv"

    def __post_init__(self) -> None:
        """Post-initialization hook for VmasEnv configuration."""
        super().__post_init__()


@dataclass
class MultiThreadedEnvConfig(EnvLibsConfig):
    """Configuration for MultiThreadedEnv environment."""

    env_name: str = MISSING
    from_pixels: bool = False
    pixels_only: bool = True
    frame_skip: int | None = None
    device: str = "cpu"
    batch_size: list[int] | None = None
    allow_done_after_reset: bool = False
    _target_: str = "torchrl.envs.libs.envpool.MultiThreadedEnv"

    def __post_init__(self) -> None:
        """Post-initialization hook for MultiThreadedEnv configuration."""
        super().__post_init__()
