# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Small MuJoCo viewer helper for the macro-control examples."""

from __future__ import annotations

import importlib
import importlib.util
import os
import platform
import shutil
import sys
import time

from collections.abc import Callable
from pathlib import Path
from types import MethodType
from typing import Any

import torch
from torchrl.record import CSVLogger, PixelRenderTransform, VideoRecorder

_has_mujoco_viewer = (
    importlib.util.find_spec("mujoco") is not None
    and importlib.util.find_spec("mujoco.viewer") is not None
)
_MJPYTHON_REEXEC_ENV = "TORCHRL_MUJOCO_MACRO_MJPYTHON_REEXEC"
_DEFAULT_VIDEO_DIR = Path("mujoco_macro_videos")


class ViewerClosed(RuntimeError):
    """Raised internally when the passive MuJoCo viewer is closed."""


def ensure_mjpython_for_passive_viewer() -> None:
    """Relaunch the current script with ``mjpython`` when macOS requires it."""
    if platform.system() != "Darwin":
        return
    if Path(sys.executable).name == "mjpython":
        return
    if os.environ.get(_MJPYTHON_REEXEC_ENV) == "1":
        return

    mjpython = shutil.which("mjpython")
    if mjpython is None:
        raise RuntimeError(
            "MuJoCo's passive viewer requires mjpython on macOS, but mjpython "
            "was not found on PATH. Try `uv run --no-sync mjpython "
            "examples/mujoco_macros/<script>.py`."
        )

    env = os.environ.copy()
    env[_MJPYTHON_REEXEC_ENV] = "1"
    python_lib = Path(sys.base_prefix) / "lib"
    if python_lib.exists():
        current_dyld = env.get("DYLD_LIBRARY_PATH", "")
        entries = [str(python_lib)]
        if current_dyld:
            entries.append(current_dyld)
        env["DYLD_LIBRARY_PATH"] = ":".join(entries)

    cmd = [mjpython, *sys.argv]
    sys.stdout.write("Relaunching MuJoCo viewer example under mjpython on macOS.\n")
    sys.stdout.flush()
    os.execvpe(mjpython, cmd, env)


def add_rollout_video_args(parser: Any) -> None:
    """Add common finite-rollout and MP4-recording options."""
    parser.add_argument(
        "--max-rollouts",
        type=int,
        default=None,
        help=(
            "Maximum number of reset-and-replay rollouts before exiting. "
            "Defaults to None, which keeps replaying until the viewer closes."
        ),
    )
    parser.add_argument(
        "--video-dir",
        type=Path,
        default=None,
        help=(
            "Directory where MP4 rollouts are written through VideoRecorder. "
            f"If omitted with --max-rollouts, defaults to ./{_DEFAULT_VIDEO_DIR}."
        ),
    )
    parser.add_argument("--video-fps", type=int, default=30)
    parser.add_argument("--video-width", type=int, default=640)
    parser.add_argument("--video-height", type=int, default=480)


def maybe_add_video_recorder(
    env: Any,
    args: Any,
    *,
    tag: str,
) -> tuple[Any, VideoRecorder | None, CSVLogger | None]:
    """Append ``PixelRenderTransform`` / ``VideoRecorder`` when requested."""
    video_dir = args.video_dir
    if video_dir is None and args.max_rollouts is not None:
        video_dir = _DEFAULT_VIDEO_DIR
    if video_dir is None:
        return env, None, None

    logger = CSVLogger(
        exp_name=tag,
        log_dir=str(video_dir),
        video_format="mp4",
        video_fps=args.video_fps,
    )
    env = env.append_transform(
        PixelRenderTransform(
            width=args.video_width,
            height=args.video_height,
        )
    )
    recorder = VideoRecorder(
        logger=logger,
        tag=tag,
        in_keys=["pixels"],
        skip=1,
        make_grid=False,
        fps=args.video_fps,
    )
    env = env.append_transform(recorder)
    return env, recorder, logger


def video_path(logger: CSVLogger, tag: str, step: int) -> Path:
    """Return the CSVLogger MP4 path for a given tag and step."""
    return Path(logger.experiment.log_dir) / "videos" / f"{tag}_{step}.mp4"


def dump_video(
    recorder: VideoRecorder | None,
    logger: CSVLogger | None,
    *,
    tag: str,
    step: int,
) -> None:
    """Dump a rollout video and report its path if recording is enabled."""
    if recorder is None or logger is None:
        return
    recorder.dump(step=step)
    sys.stdout.write(f"Saved video to {video_path(logger, tag, step)}\n")
    sys.stdout.flush()


class MujocoViewerLoop:
    """Launch a passive MuJoCo viewer and sync it after every env step.

    ``MultiAction`` executes a macro by calling the parent env once per
    low-level action in the sequence. Wrapping the C-bindings backend's ``step``
    method here lets the viewer refresh at each low-level action while example
    code still uses normal TorchRL APIs such as ``rollout`` and
    ``step_and_maybe_reset``. Reset calls are synced too, so a transformed env
    with ``PixelRenderTransform`` / ``VideoRecorder`` can write video frames
    while the same MuJoCo state is transmitted live to the passive viewer.
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
        self._reset: Callable[[torch.Tensor, torch.Tensor], None] | None = None
        self._reset_mask: Callable[
            [torch.Tensor, torch.Tensor, torch.Tensor], None
        ] | None = None

    def __enter__(self) -> MujocoViewerLoop:
        ensure_mjpython_for_passive_viewer()
        if not _has_mujoco_viewer:
            raise ImportError(
                "The MuJoCo viewer requires the `mujoco.viewer` module. Install "
                "the `mujoco` package and run with a viewer-capable Python."
            )
        mujoco_viewer = importlib.import_module("mujoco.viewer")
        self.viewer = mujoco_viewer.launch_passive(self.backend._m, self.backend._d)
        self._step = self.backend.step
        self._reset = self.backend.reset
        self._reset_mask = self.backend.reset_mask

        def sync_viewer() -> None:
            if self.viewer is None or not self.viewer.is_running():
                raise ViewerClosed
            self.viewer.sync()

        def step_and_sync(backend, ctrl: torch.Tensor, frame_skip: int) -> None:
            del backend
            self._step(ctrl, frame_skip)
            sync_viewer()
            if self.realtime:
                time.sleep(frame_skip * self.backend.timestep / self.speed)

        def reset_and_sync(backend, qpos: torch.Tensor, qvel: torch.Tensor) -> None:
            del backend
            self._reset(qpos, qvel)
            sync_viewer()

        def reset_mask_and_sync(
            backend,
            mask: torch.Tensor,
            qpos: torch.Tensor,
            qvel: torch.Tensor,
        ) -> None:
            del backend
            self._reset_mask(mask, qpos, qvel)
            sync_viewer()

        self.backend.step = MethodType(step_and_sync, self.backend)
        self.backend.reset = MethodType(reset_and_sync, self.backend)
        self.backend.reset_mask = MethodType(reset_mask_and_sync, self.backend)
        sync_viewer()
        return self

    def __exit__(self, exc_type, exc, traceback) -> bool:
        if self._step is not None:
            self.backend.step = self._step
        if self._reset is not None:
            self.backend.reset = self._reset
        if self._reset_mask is not None:
            self.backend.reset_mask = self._reset_mask
        if self.viewer is not None:
            self.viewer.close()
        return exc_type is ViewerClosed

    def is_running(self) -> bool:
        return self.viewer is not None and self.viewer.is_running()
