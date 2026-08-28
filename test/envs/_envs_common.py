# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import importlib.util
import os
from collections import defaultdict
from sys import platform

import yaml
from packaging import version

from torchrl.envs.libs.gym import _has_gym

_has_ale = importlib.util.find_spec("ale_py") is not None
_has_mujoco = importlib.util.find_spec("mujoco") is not None

gym_version = None
if _has_gym:
    try:
        import gymnasium as gym
    except ModuleNotFoundError:
        import gym

    gym_version = version.parse(gym.__version__)

try:
    this_dir = os.path.dirname(os.path.realpath(__file__))
    with open(os.path.join(this_dir, "..", "configs", "atari.yaml")) as file:
        atari_confs = yaml.load(file, Loader=yaml.FullLoader)
    _atari_found = True
except FileNotFoundError:
    _atari_found = False
    atari_confs = defaultdict(str)

IS_OSX = platform == "darwin"
IS_WIN = platform == "win32"
if IS_WIN:
    mp_ctx = "spawn"
else:
    mp_ctx = "fork"

_has_chess = importlib.util.find_spec("chess") is not None
_has_tv = importlib.util.find_spec("torchvision") is not None
_has_cairosvg = importlib.util.find_spec("cairosvg") is not None
_has_transformers = importlib.util.find_spec("transformers") is not None
_has_gymnasium = importlib.util.find_spec("gymnasium") is not None
