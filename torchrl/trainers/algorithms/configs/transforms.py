# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from omegaconf import MISSING
from torchrl.trainers.algorithms.configs.common import ConfigBase


@dataclass
class TransformConfig(ConfigBase):
    """Base configuration class for transforms."""

    _target_: str = MISSING

    def __post_init__(self) -> None:
        """Post-initialization hook for transform configurations."""
        pass


@dataclass
class NoopResetEnvConfig(TransformConfig):
    """Configuration for NoopResetEnv transform."""

    noops: int = 30
    random: bool = True
    _target_: str = "torchrl.trainers.algorithms.configs.transforms.make_noop_reset_env"

    def __post_init__(self) -> None:
        """Post-initialization hook for NoopResetEnv configuration."""
        super().__post_init__()


@dataclass
class ComposeConfig(TransformConfig):
    """Configuration for Compose transform."""

    transforms: list[TransformConfig] | None = None
    _target_: str = "torchrl.trainers.algorithms.configs.transforms.make_compose"

    def __post_init__(self) -> None:
        """Post-initialization hook for Compose configuration."""
        super().__post_init__()
        if self.transforms is None:
            self.transforms = []


def make_noop_reset_env(noops: int = 30, random: bool = True):
    """Create a NoopResetEnv transform.

    Args:
        noops: Upper-bound on the number of actions performed after reset.
        random: If False, the number of random ops will always be equal to the noops value.
               If True, the number of random actions will be randomly selected between 0 and noops.

    Returns:
        The created NoopResetEnv transform instance.
    """
    from torchrl.envs.transforms.transforms import NoopResetEnv

    return NoopResetEnv(noops=noops, random=random)


def make_compose(transforms: list[TransformConfig] | None = None):
    """Create a Compose transform.

    Args:
        transforms: List of transform configurations to compose.

    Returns:
        The created Compose transform instance.
    """
    from torchrl.envs.transforms.transforms import Compose

    if transforms is None:
        transforms = []

    # For now, we'll just return an empty Compose
    # In a full implementation with hydra, we would instantiate each transform
    return Compose() 