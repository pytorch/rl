# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

# Import classes that don't have external dependencies
from .atari_dqn import AtariDQNExperienceReplay
from .common import BaseDatasetExperienceReplay
from .lerobot import lerobot_columns_to_tensordict, LeRobotExperienceReplay
from .utils import load_dataset

# Conditional imports for classes with external dependencies
try:
    from .d4rl import D4RLExperienceReplay
except ImportError:
    pass

try:
    from .minari_data import MinariExperienceReplay
except ImportError:
    pass

try:
    from .gen_dgrl import GenDGRLExperienceReplay
except ImportError:
    pass

try:
    from .openml import OpenMLExperienceReplay
except ImportError:
    pass

try:
    from .openx import OpenXExperienceReplay
except ImportError:
    pass

try:
    from .roboset import RobosetExperienceReplay
except ImportError:
    pass

try:
    from .vd4rl import VD4RLExperienceReplay
except ImportError:
    pass

__all__ = [
    "AtariDQNExperienceReplay",
    "BaseDatasetExperienceReplay",
    "load_dataset",
    "D4RLExperienceReplay",
    "MinariExperienceReplay",
    "GenDGRLExperienceReplay",
    "LeRobotExperienceReplay",
    "lerobot_columns_to_tensordict",
    "OpenMLExperienceReplay",
    "OpenXExperienceReplay",
    "RobosetExperienceReplay",
    "VD4RLExperienceReplay",
]
