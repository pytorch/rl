# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import importlib.util
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch

from torchrl.record.loggers.common import _write_video

_has_pil = importlib.util.find_spec("PIL") is not None

_PIL_ERROR = (
    "Writing image frames or GIF artifacts requires Pillow. When running TorchRL "
    "from this repository with uv, use `uv run --extra rendering <command>` so "
    "Pillow is installed in the command environment. Otherwise install it with "
    "`pip install pillow`."
)

__all__ = [
    "compose_frame_grid",
    "encode_gif",
    "encode_video",
    "normalize_frame",
    "normalize_frame_output",
    "write_png",
]


def normalize_frame(frame: Any) -> np.ndarray:
    """Converts a tensor-like image into an ``H x W x 3`` uint8 array."""
    if (
        hasattr(frame, "data")
        and not torch.is_tensor(frame)
        and not isinstance(frame, np.ndarray)
    ):
        frame = frame.data
    if torch.is_tensor(frame):
        frame = frame.detach().cpu().numpy()
    frame = np.asarray(frame)
    if frame.ndim == 0:
        raise ValueError("A rendered frame must have at least two dimensions.")
    while frame.ndim > 3:
        frame = frame[0]
    if frame.ndim == 2:
        frame = frame[..., None]
    if frame.ndim != 3:
        raise ValueError(f"Expected a 2D or 3D frame, got shape {frame.shape}.")
    if frame.shape[0] in (1, 3) and frame.shape[-1] != 3:
        frame = np.moveaxis(frame, 0, -1)
    if frame.shape[-1] == 1:
        frame = np.repeat(frame, 3, axis=-1)
    elif frame.shape[-1] == 4:
        frame = frame[..., :3]
    elif frame.shape[-1] != 3:
        raise ValueError(f"Expected 1, 3, or 4 frame channels, got {frame.shape[-1]}.")
    if frame.dtype == np.uint8:
        return np.ascontiguousarray(frame)
    if np.issubdtype(frame.dtype, np.floating):
        max_value = float(np.nanmax(frame)) if frame.size else 0.0
        if max_value <= 1.0:
            frame = frame * 255.0
    frame = np.nan_to_num(frame, nan=0.0, posinf=255.0, neginf=0.0)
    frame = np.clip(frame, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(frame)


def normalize_frame_output(output: Any) -> dict[str, np.ndarray]:
    """Normalizes renderer output into named RGB frames."""
    if output is None:
        return {}
    if isinstance(output, Mapping):
        return {
            str(name): normalize_frame(frame)
            for name, frame in output.items()
            if frame is not None
        }
    if isinstance(output, (list, tuple)):
        return {
            f"camera_{index}": normalize_frame(frame)
            for index, frame in enumerate(output)
            if frame is not None
        }
    return {"default": normalize_frame(output)}


def compose_frame_grid(
    frames: Sequence[np.ndarray],
    layout: Literal["single", "grid", "horizontal", "vertical"] = "grid",
) -> np.ndarray:
    """Composes multiple frames into one RGB image.

    Args:
        frames: Frames to compose.
        layout: ``"grid"`` (and ``"single"``) tile the frames into a
            near-square grid, ``"horizontal"`` composes one row, and
            ``"vertical"`` composes one column.
    """
    if not frames:
        raise ValueError("Cannot compose an empty frame grid.")
    normalized = [normalize_frame(frame) for frame in frames]
    height = max(frame.shape[0] for frame in normalized)
    width = max(frame.shape[1] for frame in normalized)
    count = len(normalized)
    if layout == "horizontal":
        cols = count
    elif layout == "vertical":
        cols = 1
    else:
        cols = math.ceil(math.sqrt(count))
    rows = math.ceil(count / cols)
    grid = np.zeros((rows * height, cols * width, 3), dtype=np.uint8)
    for index, frame in enumerate(normalized):
        row = index // cols
        col = index % cols
        padded = _pad_frame(frame, height, width)
        grid[
            row * height : (row + 1) * height, col * width : (col + 1) * width
        ] = padded
    return grid


def encode_video(
    frames: Sequence[np.ndarray],
    path: str | Path,
    fps: float,
    *,
    video_codec: str | None = None,
) -> Path:
    """Encodes RGB frames as an MP4 using TorchRL's torchcodec writer."""
    path = Path(path)
    if not frames:
        raise ValueError("Cannot encode a video with no frames.")
    array = np.stack([normalize_frame(frame) for frame in frames], axis=0)
    tensor = torch.as_tensor(array, dtype=torch.uint8)
    kwargs: dict[str, Any] = {"fps": fps}
    if video_codec is not None:
        kwargs["video_codec"] = video_codec
    _write_video(str(path), tensor, **kwargs)
    return path


def encode_gif(frames: Sequence[np.ndarray], path: str | Path, fps: float) -> Path:
    """Encodes RGB frames as an animated GIF using Pillow."""
    if not _has_pil:
        raise ModuleNotFoundError(_PIL_ERROR)
    from PIL import Image

    path = Path(path)
    if not frames:
        raise ValueError("Cannot encode a GIF with no frames.")
    images = [Image.fromarray(normalize_frame(frame)) for frame in frames]
    duration = max(int(round(1000.0 / fps)), 1)
    images[0].save(
        path,
        save_all=True,
        append_images=images[1:],
        duration=duration,
        loop=0,
    )
    return path


def write_png(frame: Any, path: str | Path) -> Path:
    """Writes one RGB frame as a PNG file using Pillow."""
    if not _has_pil:
        raise ModuleNotFoundError(_PIL_ERROR)
    from PIL import Image

    path = Path(path)
    Image.fromarray(normalize_frame(frame)).save(path)
    return path


def _pad_frame(frame: np.ndarray, height: int, width: int) -> np.ndarray:
    if frame.shape[0] == height and frame.shape[1] == width:
        return frame
    padded = np.zeros((height, width, 3), dtype=np.uint8)
    padded[: frame.shape[0], : frame.shape[1]] = frame
    return padded
