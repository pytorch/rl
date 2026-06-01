# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Small MuJoCo viewer helper for the macro-control examples."""

from __future__ import annotations

import importlib
import importlib.util
import time

from collections.abc import Callable
from types import MethodType
from typing import Any

import torch

_has_mujoco_viewer = (
    importlib.util.find_spec("mujoco") is not None
    and importlib.util.find_spec("mujoco.viewer") is not None
)


class ViewerClosed(RuntimeError):
    """Raised internally when the passive MuJoCo viewer is closed."""


class MujocoViewerLoop:
    """Launch a passive MuJoCo viewer and sync it after every env step.

    ``MultiAction`` executes a macro by calling the parent env once per
    low-level action in the sequence. Wrapping the C-bindings backend's ``step``
    method here lets the viewer refresh at each low-level action while example
    code still uses normal TorchRL APIs such as ``rollout`` and
    ``step_and_maybe_reset``.
    """

    def __init__(
        self,
        env: Any,
        *,
        realtime: bool = True,
        speed: float = 1.0,
    ) -> None:
        self.env = env
        self.backend = getattr(env, "_backend", None)
        if self.backend is None or not all(
            hasattr(self.backend, attr) for attr in ("_m", "_d")
        ):
            raise RuntimeError(
                "MuJoCo viewer examples require backend='mujoco', which exposes "
                "a live mujoco.MjModel and mujoco.MjData."
            )
        if speed <= 0:
            raise ValueError("speed must be strictly positive.")
        self.realtime = bool(realtime)
        self.speed = float(speed)
        self.viewer = None
        self._step: Callable[[torch.Tensor, int], None] | None = None

    def __enter__(self) -> MujocoViewerLoop:
        if not _has_mujoco_viewer:
            raise ImportError(
                "The MuJoCo viewer requires the `mujoco.viewer` module. Install "
                "the `mujoco` package and run with a viewer-capable Python."
            )
        mujoco_viewer = importlib.import_module("mujoco.viewer")
        self.viewer = mujoco_viewer.launch_passive(self.backend._m, self.backend._d)
        self._step = self.backend.step

        def step_and_sync(backend, ctrl: torch.Tensor, frame_skip: int) -> None:
            del backend
            if self.viewer is None or not self.viewer.is_running():
                raise ViewerClosed
            self._step(ctrl, frame_skip)
            self.viewer.sync()
            if self.realtime:
                time.sleep(frame_skip * self.backend.timestep / self.speed)

        self.backend.step = MethodType(step_and_sync, self.backend)
        self.viewer.sync()
        return self

    def __exit__(self, exc_type, exc, traceback) -> bool:
        if self._step is not None:
            self.backend.step = self._step
        if self.viewer is not None:
            self.viewer.close()
        return exc_type is ViewerClosed

    def is_running(self) -> bool:
        return self.viewer is not None and self.viewer.is_running()
