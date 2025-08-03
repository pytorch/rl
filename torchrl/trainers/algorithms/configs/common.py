# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class ConfigBase(ABC):
    """Abstract base class for all configuration classes.

    This class serves as the foundation for all configuration classes in the
    configurable configuration system, providing a common interface and structure.
    """

    @abstractmethod
    def __post_init__(self) -> None:
        """Post-initialization hook for configuration classes."""
        pass


# Main configuration class that can be instantiated from YAML
@dataclass
class Config:
    """Main configuration class that can be instantiated from YAML."""

    trainer: Any = None
    env: Any = None
    network: Any = None
    model: Any = None
    loss: Any = None
    replay_buffer: Any = None
    sampler: Any = None
    storage: Any = None
    writer: Any = None
    collector: Any = None
    optimizer: Any = None
    logger: Any = None
    networks: Any = None
    models: Any = None
