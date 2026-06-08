# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import importlib
import importlib.util
from sys import platform

import pytest
import torch
from packaging import version

from torchrl.envs.libs.gym import _has_gym, gym_backend, set_gym_backend

pytestmark = [
    pytest.mark.filterwarnings("error"),
    pytest.mark.filterwarnings(
        "ignore:Got multiple backends for torchrl.data.replay_buffers.storages"
    ),
    pytest.mark.filterwarnings("ignore:unclosed file"),
]

_has_ray = importlib.util.find_spec("ray") is not None
_has_ale = importlib.util.find_spec("ale_py") is not None
_has_atari_py = False
if importlib.util.find_spec("atari_py") is not None:
    try:
        import atari_py

        _has_atari_py = hasattr(atari_py, "get_game_path")
    except Exception:
        _has_atari_py = False
_has_mujoco = (
    importlib.util.find_spec("mujoco") is not None
    or importlib.util.find_spec("mujoco_py") is not None
)

TORCH_VERSION = version.parse(version.parse(torch.__version__).base_version)

_has_d4rl = importlib.util.find_spec("d4rl") is not None

_has_mo = importlib.util.find_spec("mo_gymnasium") is not None

_has_sklearn = importlib.util.find_spec("sklearn") is not None

_has_gym_robotics = importlib.util.find_spec("gymnasium_robotics") is not None

_has_minari = importlib.util.find_spec("minari") is not None

_has_gymnasium = importlib.util.find_spec("gymnasium") is not None

_has_isaaclab = importlib.util.find_spec("isaaclab") is not None

_has_gym_regular = importlib.util.find_spec("gym") is not None
if _has_gymnasium:
    set_gym_backend("gymnasium").set()
    import gymnasium

    assert gym_backend() is gymnasium
elif _has_gym:
    set_gym_backend("gym").set()
    import gym

    assert gym_backend() is gym

_has_meltingpot = importlib.util.find_spec("meltingpot") is not None

_has_minigrid = importlib.util.find_spec("minigrid") is not None

_has_procgen = importlib.util.find_spec("procgen") is not None


@pytest.fixture(scope="session", autouse=True)
def maybe_init_minigrid():
    if _has_minigrid and _has_gymnasium:
        import minigrid

        minigrid.register_minigrid_envs()


IS_OSX = platform == "darwin"
RTOL = 1e-1
ATOL = 1e-1
