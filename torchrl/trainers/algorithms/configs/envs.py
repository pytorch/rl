# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from omegaconf import MISSING

from torchrl.envs.common import EnvBase
from torchrl.trainers.algorithms.configs.common import ConfigBase


@dataclass
class EnvConfig(ConfigBase):
    """Base configuration class for environments."""

    _partial_: bool = False

    def __post_init__(self) -> None:
        """Post-initialization hook for environment configurations."""
        self._partial_ = False


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
        if hasattr(self.create_env_fn, "_partial_"):
            self.create_env_fn._partial_ = True


@dataclass
class TransformedEnvConfig(EnvConfig):
    """Configuration for transformed environments."""

    base_env: Any = MISSING
    transform: Any = None
    cache_specs: bool = True
    auto_unwrap: bool | None = None
    _target_: str = "torchrl.envs.TransformedEnv"


def make_batched_env(
    create_env_fn, num_workers, batched_env_type="parallel", device=None, **kwargs
):
    """Create a batched environment.

    Args:
        create_env_fn: Function to create individual environments or environment instance.
        num_workers: Number of worker environments.
        batched_env_type: Type of batched environment (parallel, serial, async).
        device: Device to place the batched environment on.
        **kwargs: Additional keyword arguments.

    Returns:
        The created batched environment instance.
    """
    from omegaconf import OmegaConf
    from torchrl.envs import AsyncEnvPool, ParallelEnv, SerialEnv

    if create_env_fn is None:
        raise ValueError("create_env_fn must be provided")

    if num_workers is None:
        raise ValueError("num_workers must be provided")

    # If create_env_fn is a config object, create a lambda that instantiates it each time
    if isinstance(create_env_fn, EnvBase):
        # Already an instance (either instantiated config or actual env), wrap in lambda
        env_instance = create_env_fn
        env_fn = lambda env_instance=env_instance: env_instance
    else:
        env_fn = create_env_fn
    assert callable(env_fn), env_fn

    # Add device to kwargs if provided
    if device is not None:
        kwargs["device"] = device

    if batched_env_type == "parallel":
        return ParallelEnv(num_workers, env_fn, **kwargs)
    elif batched_env_type == "serial":
        return SerialEnv(num_workers, env_fn, **kwargs)
    elif batched_env_type == "async":
        return AsyncEnvPool([env_fn] * num_workers, **kwargs)
    else:
        raise ValueError(f"Unknown batched_env_type: {batched_env_type}")
