# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

# Import classes that don't have external dependencies
from .atari_dqn import AtariDQNExperienceReplay
from .common import BaseDatasetExperienceReplay

# Conditional imports for classes with external dependencies
try:
    from .d4rl import D4RLExperienceReplay
except ImportError:
    pass

try:
    from .minari_data import MinariExperienceReplay
except ImportError:
    pass

__all__ = [
    "AtariDQNExperienceReplay",
    "BaseDatasetExperienceReplay",
    "D4RLExperienceReplay",
    "MinariExperienceReplay",
]
