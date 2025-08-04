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
class EnvConfig(ConfigBase):
    """Base configuration class for environments."""

    _partial_: bool = False

    def __post_init__(self) -> None:
        """Post-initialization hook for environment configurations."""
        self._partial_ = False


@dataclass
class GymEnvConfig(EnvConfig):
    """Configuration for Gym/Gymnasium environments."""

    env_name: str | None = None
    backend: str = "gymnasium"  # Changed from Literal to str
    from_pixels: bool = False
    double_to_float: bool = False
    _target_: str = "torchrl.trainers.algorithms.configs.envs.make_env"

    def __post_init__(self) -> None:
        """Post-initialization hook for Gym environment configurations."""
        super().__post_init__()


@dataclass
class BatchedEnvConfig(EnvConfig):
    """Configuration for batched environments."""

    create_env_fn: Any = MISSING
    num_workers: int = 1
    create_env_kwargs: dict = field(default_factory=dict)
    batched_env_type: str = "parallel"
    device: str | None = None
    # batched_env_type: Literal["parallel", "serial", "async"] = "parallel"
    _target_: str = "torchrl.trainers.algorithms.configs.envs.make_batched_env"

    def __post_init__(self) -> None:
        """Post-initialization hook for batched environment configurations."""
        super().__post_init__()
        self.create_env_fn._partial_ = True


@dataclass
class TransformedEnvConfig(EnvConfig):
    """Configuration for transformed environments."""

    base_env: Any = MISSING
    transform: Any = None
    cache_specs: bool = True
    auto_unwrap: bool | None = None
    _target_: str = "torchrl.trainers.algorithms.configs.envs.make_transformed_env"

    def __post_init__(self) -> None:
        """Post-initialization hook for transformed environment configurations."""
        super().__post_init__()
        if self.base_env is not None:
            self.base_env._partial_ = True
        if self.transform is not None:
            self.transform._partial_ = True


def make_env(
    env_name: str,
    backend: str = "gymnasium",
    from_pixels: bool = False,
    double_to_float: bool = False,
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
            env = GymEnv(env_name, from_pixels=from_pixels)
    else:
        env = GymEnv(env_name, from_pixels=from_pixels)

    if double_to_float:
        env = env.append_transform(DoubleToFloat(in_keys=["observation"]))

    return env


def make_batched_env(create_env_fn, num_workers, batched_env_type="parallel", device=None, **kwargs):
    """Create a batched environment.

    Args:
        create_env_fn: Function to create individual environments.
        num_workers: Number of worker environments.
        batched_env_type: Type of batched environment (parallel, serial, async).
        device: Device to place the batched environment on.
        **kwargs: Additional keyword arguments.

    Returns:
        The created batched environment instance.
    """
    from torchrl.envs import AsyncEnvPool, ParallelEnv, SerialEnv
    from omegaconf import OmegaConf

    if create_env_fn is None:
        raise ValueError("create_env_fn must be provided")

    if num_workers is None:
        raise ValueError("num_workers must be provided")

    # Instantiate the create_env_fn if it's a config
    if hasattr(create_env_fn, '_target_'):
        create_env_fn = OmegaConf.to_object(create_env_fn)

    # Add device to kwargs if provided
    if device is not None:
        kwargs["device"] = device

    if batched_env_type == "parallel":
        return ParallelEnv(num_workers, create_env_fn, **kwargs)
    elif batched_env_type == "serial":
        return SerialEnv(num_workers, create_env_fn, **kwargs)
    elif batched_env_type == "async":
        return AsyncEnvPool([create_env_fn] * num_workers, **kwargs)
    else:
        raise ValueError(f"Unknown batched_env_type: {batched_env_type}")


def make_transformed_env(base_env, transform=None, cache_specs=True, auto_unwrap=None):
    """Create a transformed environment.

    Args:
        base_env: Base environment to transform.
        transform: Transform to apply to the environment.
        cache_specs: Whether to cache the specs.
        auto_unwrap: Whether to auto-unwrap transforms.

    Returns:
        The created transformed environment instance.
    """
    from torchrl.envs import TransformedEnv
    from omegaconf import OmegaConf

    # Instantiate the base environment if it's a config
    if hasattr(base_env, '_target_'):
        base_env = OmegaConf.to_object(base_env)
    
    # If base_env is a callable (like a partial function), call it to get the actual env
    if callable(base_env):
        base_env = base_env()
    
    # Instantiate the transform if it's a config
    if transform is not None and hasattr(transform, '_target_'):
        transform = OmegaConf.to_object(transform)

    return TransformedEnv(
        env=base_env,
        transform=transform,
        cache_specs=cache_specs,
        auto_unwrap=auto_unwrap,
    )
