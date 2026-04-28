# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from omegaconf import DictConfig, ListConfig
from tensordict import NestedKey


@dataclass
class ConfigBase(ABC):
    """Abstract base class for all configuration classes.

    This class serves as the foundation for all configuration classes in the
    configurable configuration system, providing a common interface and structure.
    """

    @abstractmethod
    def __post_init__(self) -> None:
        """Post-initialization hook for configuration classes."""


@dataclass
class Config:
    """A flexible config that allows arbitrary fields."""

    def __init__(self, **kwargs):
        self._config = DictConfig(kwargs)

    def __getattr__(self, name):
        return getattr(self._config, name)

    def __setattr__(self, name, value):
        if name == "_config":
            super().__setattr__(name, value)
        else:
            setattr(self._config, name, value)


def _normalize_hydra_key(key: Any) -> NestedKey:
    """Converts Hydra list of strings to a NestedKey."""
    if isinstance(key, ListConfig):
        key = tuple(key)
    return key


def _normalize_hydra_keys(keys: Sequence[Any] | None) -> Sequence[NestedKey] | None:
    return keys if keys is None else [_normalize_hydra_key(key) for key in keys]
