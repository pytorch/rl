# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""MuJoCo-backed custom envs with selectable physics backend.

Most subclasses accept an XML asset (path or URL) through
:class:`MujocoEnv`, which dispatches the simulation to one of three engines:

* ``mujoco-torch`` (default) -- native torch, batched, ``torch.compile``-friendly.
* ``mjx`` -- JAX-vectorized via :func:`jax.vmap` + :func:`jax.jit`,
  bridged to torch through DLPack.
* ``mujoco`` -- official C-bindings, batched by Python loop.

Subclasses describe the *task*: reward, termination, optional
observation map. The locomotion subclasses (:class:`HumanoidEnv`,
:class:`AntEnv`, :class:`Walker2dEnv`, :class:`HopperEnv`) mirror the
Gymnasium ``-v4`` reward / termination spec. :class:`SatelliteEnv`
implements an attitude-control task with 4- or 6-CMG clusters and a
manipulability-based singularity penalty. :class:`CubeBowlEnv` is a
compact Menagerie-backed manipulation task for macro-control examples.
"""

from torchrl.envs.custom.mujoco._humanoid_primitives import HumanoidAction
from torchrl.envs.custom.mujoco._satellite_primitives import (
    SatelliteAction,
    SatelliteAttitudeTransform,
)
from torchrl.envs.custom.mujoco.ant import AntEnv
from torchrl.envs.custom.mujoco.base import MujocoEnv
from torchrl.envs.custom.mujoco.cube_bowl import CubeBowlEnv
from torchrl.envs.custom.mujoco.hopper import HopperEnv
from torchrl.envs.custom.mujoco.humanoid import HumanoidEnv
from torchrl.envs.custom.mujoco.satellite import SatelliteEnv
from torchrl.envs.custom.mujoco.walker import Walker2dEnv

__all__ = [
    "AntEnv",
    "CubeBowlEnv",
    "HopperEnv",
    "HumanoidAction",
    "HumanoidEnv",
    "MujocoEnv",
    "SatelliteAction",
    "SatelliteAttitudeTransform",
    "SatelliteEnv",
    "Walker2dEnv",
]
