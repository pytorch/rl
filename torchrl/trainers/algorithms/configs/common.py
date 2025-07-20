# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from abc import ABC

from dataclasses import dataclass
from typing import Any


@dataclass
class ConfigBase(ABC):
    pass


# Main configuration class that can be instantiated from YAML
@dataclass
class Config:
    """Main configuration class that can be instantiated from YAML."""

    trainer: Any = None
    env: Any = None
