# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Versioned gym environment name helpers for TorchRL tests."""

from __future__ import annotations

import sys

from torchrl._utils import implement_for
from torchrl.envs.libs.gym import _has_gym, gym_backend

__all__ = [
    "BREAKOUT_VERSIONED",
    "CARTPOLE_VERSIONED",
    "CLIFFWALKING_VERSIONED",
    "HALFCHEETAH_VERSIONED",
    "PENDULUM_VERSIONED",
    "PONG_VERSIONED",
]

PYTHON_3_9 = sys.version_info.major == 3 and sys.version_info.minor <= 9

# Module-level variables that will be set by _set_gym_environments
_CARTPOLE_VERSIONED = None
_HALFCHEETAH_VERSIONED = None
_PENDULUM_VERSIONED = None
_PONG_VERSIONED = None
_BREAKOUT_VERSIONED = None
_CLIFFWALKING_VERSIONED = None


def CARTPOLE_VERSIONED():
    """Return the versioned CartPole environment name for the current gym backend."""
    if gym_backend() is not None:
        _set_gym_environments()
        return _CARTPOLE_VERSIONED


def HALFCHEETAH_VERSIONED():
    """Return the versioned HalfCheetah environment name for the current gym backend."""
    if gym_backend() is not None:
        _set_gym_environments()
        return _HALFCHEETAH_VERSIONED


def PONG_VERSIONED():
    """Return the versioned Pong environment name for the current gym backend."""
    # Gymnasium says that the ale_py behavior changes from 1.0
    # but with python 3.12 it is already the case with 0.29.1
    try:
        import ale_py  # noqa: F401
    except ImportError:
        pass

    if gym_backend() is not None:
        _set_gym_environments()
        return _PONG_VERSIONED


def CLIFFWALKING_VERSIONED():
    """Return the versioned CliffWalking environment name for the current gym backend."""
    if gym_backend() is not None:
        _set_gym_environments()
        return _CLIFFWALKING_VERSIONED


def BREAKOUT_VERSIONED():
    """Return the versioned Breakout environment name for the current gym backend."""
    # Gymnasium says that the ale_py behavior changes from 1.0
    # but with python 3.12 it is already the case with 0.29.1
    try:
        import ale_py  # noqa: F401
    except ImportError:
        pass

    if gym_backend() is not None:
        _set_gym_environments()
        return _BREAKOUT_VERSIONED


def PENDULUM_VERSIONED():
    """Return the versioned Pendulum environment name for the current gym backend."""
    if gym_backend() is not None:
        _set_gym_environments()
        return _PENDULUM_VERSIONED


def _set_gym_environments():
    global _CARTPOLE_VERSIONED, _HALFCHEETAH_VERSIONED, _PENDULUM_VERSIONED, _PONG_VERSIONED, _BREAKOUT_VERSIONED, _CLIFFWALKING_VERSIONED

    _CARTPOLE_VERSIONED = None
    _HALFCHEETAH_VERSIONED = None
    _PENDULUM_VERSIONED = None
    _PONG_VERSIONED = None
    _BREAKOUT_VERSIONED = None
    _CLIFFWALKING_VERSIONED = None


@implement_for("gym", None, "0.21.0")
def _set_gym_environments():  # noqa: F811
    global _CARTPOLE_VERSIONED, _HALFCHEETAH_VERSIONED, _PENDULUM_VERSIONED, _PONG_VERSIONED, _BREAKOUT_VERSIONED, _CLIFFWALKING_VERSIONED

    _CARTPOLE_VERSIONED = "CartPole-v0"
    _HALFCHEETAH_VERSIONED = "HalfCheetah-v2"
    _PENDULUM_VERSIONED = "Pendulum-v0"
    _PONG_VERSIONED = "Pong-v4"
    _BREAKOUT_VERSIONED = "Breakout-v4"
    _CLIFFWALKING_VERSIONED = "CliffWalking-v0"


@implement_for("gym", "0.21.0", "0.26.0")
def _set_gym_environments():  # noqa: F811
    global _CARTPOLE_VERSIONED, _HALFCHEETAH_VERSIONED, _PENDULUM_VERSIONED, _PONG_VERSIONED, _BREAKOUT_VERSIONED, _CLIFFWALKING_VERSIONED

    _CARTPOLE_VERSIONED = "CartPole-v1"
    # Use v3 for gym < 0.26 (uses mujoco-py); v4 requires gym 0.26+ with new mujoco bindings
    _HALFCHEETAH_VERSIONED = "HalfCheetah-v3"
    _PENDULUM_VERSIONED = "Pendulum-v1"
    _PONG_VERSIONED = "ALE/Pong-v5"
    _BREAKOUT_VERSIONED = "ALE/Breakout-v5"
    _CLIFFWALKING_VERSIONED = "CliffWalking-v0"


@implement_for("gym", "0.26.0", None)
def _set_gym_environments():  # noqa: F811
    global _CARTPOLE_VERSIONED, _HALFCHEETAH_VERSIONED, _PENDULUM_VERSIONED, _PONG_VERSIONED, _BREAKOUT_VERSIONED, _CLIFFWALKING_VERSIONED

    _CARTPOLE_VERSIONED = "CartPole-v1"
    _HALFCHEETAH_VERSIONED = "HalfCheetah-v4"
    _PENDULUM_VERSIONED = "Pendulum-v1"
    _PONG_VERSIONED = "ALE/Pong-v5"
    _BREAKOUT_VERSIONED = "ALE/Breakout-v5"
    _CLIFFWALKING_VERSIONED = "CliffWalking-v0"


@implement_for("gymnasium", None, "1.0.0")
def _set_gym_environments():  # noqa: F811
    global _CARTPOLE_VERSIONED, _HALFCHEETAH_VERSIONED, _PENDULUM_VERSIONED, _PONG_VERSIONED, _BREAKOUT_VERSIONED, _CLIFFWALKING_VERSIONED

    _CARTPOLE_VERSIONED = "CartPole-v1"
    _HALFCHEETAH_VERSIONED = "HalfCheetah-v4"
    _PENDULUM_VERSIONED = "Pendulum-v1"
    _PONG_VERSIONED = "ALE/Pong-v5"
    _BREAKOUT_VERSIONED = "ALE/Breakout-v5"
    _CLIFFWALKING_VERSIONED = "CliffWalking-v0"


@implement_for("gymnasium", "1.0.0", "1.1.0")
def _set_gym_environments():  # noqa: F811
    raise ImportError


@implement_for("gymnasium", "1.1.0")
def _set_gym_environments():  # noqa: F811
    global _CARTPOLE_VERSIONED, _HALFCHEETAH_VERSIONED, _PENDULUM_VERSIONED, _PONG_VERSIONED, _BREAKOUT_VERSIONED, _CLIFFWALKING_VERSIONED

    _CARTPOLE_VERSIONED = "CartPole-v1"
    _HALFCHEETAH_VERSIONED = "HalfCheetah-v5"
    _PENDULUM_VERSIONED = "Pendulum-v1"
    _PONG_VERSIONED = "ALE/Pong-v5"
    _BREAKOUT_VERSIONED = "ALE/Breakout-v5"
    _CLIFFWALKING_VERSIONED = "CliffWalking-v1" if not PYTHON_3_9 else "CliffWalking-v0"


if _has_gym:
    _set_gym_environments()
