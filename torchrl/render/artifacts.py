# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import json
import platform
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

import torchrl
from tensordict import TensorDictBase
from torchrl.render.checkpoint import checkpoint_hash
from torchrl.render.config import FrameBundle, key_to_string, RenderConfig, RenderResult
from torchrl.render.notebook import write_render_notebook
from torchrl.render.video import compose_frame_grid, encode_gif, encode_video, write_png

__all__ = ["write_render_artifact"]


def write_render_artifact(result: RenderResult, config: RenderConfig) -> RenderResult:
    """Writes the configured render artifact and sidecar metadata."""
    out = _default_out(config)
    _ensure_writable(out, config)
    asset_dir = _asset_dir(out, config)
    asset_dir.mkdir(parents=True, exist_ok=True)
    result.metadata.update(_runtime_metadata(config, asset_dir))
    frame_paths: list[Path] = []
    artifact_path: Path | None = out
    if config.format == "frames":
        artifact_path = out
        frame_paths.extend(_write_frames(result.frames, out, config))
    elif config.format == "mp4":
        frame_paths.extend(_write_videos(result.frames, out, config))
    elif config.format == "gif":
        frame_paths.extend(_write_gif(result.frames, out, config))
    elif config.format == "npz":
        _write_npz(result, out, config)
    elif config.format == "jsonl":
        _write_jsonl(result, out)
    elif config.format == "ipynb":
        result.metadata["asset_dir"] = _relative_asset_dir(asset_dir, out.parent)
        _write_notebook_assets(result, config, asset_dir)
        if _has_any_frames(result.frames):
            try:
                frame_paths.extend(_write_videos(result.frames, asset_dir, config))
            except Exception as err:
                warning = f"Could not write notebook video preview: {err}"
                result.warnings.append(warning)
                result.metadata.setdefault("warnings", []).append(warning)
        result.frame_paths = frame_paths
        _write_json(asset_dir / "metadata.json", result.metadata)
        if config.metadata is not None:
            _write_json(Path(config.metadata), result.metadata)
        write_render_notebook(result, config, out)
    else:
        raise ValueError(f"Unsupported render format {config.format!r}.")
    if config.save_frames and config.format != "frames":
        frame_dir = (
            Path(config.frame_dir)
            if config.frame_dir is not None
            else asset_dir / "frames"
        )
        frame_paths.extend(_write_frames(result.frames, frame_dir, config))
    if (config.save_rollout or config.save_tensordicts) and config.format != "ipynb":
        _write_rollout_assets(result, asset_dir)
        _write_json(asset_dir / "config.json", config.to_dict())
    if config.format != "ipynb":
        result.frame_paths = frame_paths
        _write_json(_metadata_path(out, config), result.metadata)
    result.artifact_path = artifact_path
    return result


def _default_out(config: RenderConfig) -> Path:
    if config.out is not None:
        return Path(config.out)
    defaults = {
        "mp4": "render.mp4",
        "gif": "render.gif",
        "frames": "render_frames",
        "npz": "render_rollouts.npz",
        "jsonl": "render_events.jsonl",
        "ipynb": "render_report.ipynb",
    }
    base = Path(config.artifact_dir or ".")
    return base / defaults[config.format]


def _asset_dir(out: Path, config: RenderConfig) -> Path:
    if config.artifact_dir is not None and config.format != "frames":
        return Path(config.artifact_dir)
    if config.format == "frames":
        return out
    return out.with_suffix("") if out.suffix else out / "assets"


def _metadata_path(out: Path, config: RenderConfig) -> Path:
    if config.metadata is not None:
        return Path(config.metadata)
    if out.suffix:
        return out.with_suffix(out.suffix + ".metadata.json")
    return out / "metadata.json"


def _ensure_writable(path: Path, config: RenderConfig) -> None:
    if path.exists() and not config.overwrite:
        raise FileExistsError(
            f"Refusing to overwrite existing path {path!s}; pass --overwrite."
        )
    parent = path.parent if path.suffix else path
    parent.mkdir(parents=True, exist_ok=True)


def _write_frames(
    frames: list[list[FrameBundle]], out: Path, config: RenderConfig
) -> list[Path]:
    out.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for traj_index, trajectory in enumerate(frames):
        for bundle in trajectory:
            for camera, frame in bundle.frames.items():
                path = (
                    out / f"traj_{traj_index:03d}" / camera / f"{bundle.step:06d}.png"
                )
                path.parent.mkdir(parents=True, exist_ok=True)
                paths.append(write_png(frame, path))
    return paths


def _write_videos(
    frames: list[list[FrameBundle]], out: Path, config: RenderConfig
) -> list[Path]:
    def encode(stream, path):
        return encode_video(stream, path, config.fps, video_codec=config.video_codec)

    return _write_stream_artifacts(frames, out, config, suffix=".mp4", encode=encode)


def _write_gif(
    frames: list[list[FrameBundle]], out: Path, config: RenderConfig
) -> list[Path]:
    def encode(stream, path):
        return encode_gif(stream, path, config.fps)

    return _write_stream_artifacts(frames, out, config, suffix=".gif", encode=encode)


def _write_stream_artifacts(
    frames: list[list[FrameBundle]],
    out: Path,
    config: RenderConfig,
    *,
    suffix: str,
    encode,
) -> list[Path]:
    streams = _streams(frames)
    if not streams:
        raise RuntimeError(
            f"No frames were captured, so no {suffix.lstrip('.')} can be written."
        )
    single_file = out.suffix.lower() == suffix and config.camera_layout != "separate"
    if single_file and len(streams) == 1:
        key = next(iter(streams))
        return [encode(streams[key], out)]
    if single_file:
        return [encode(_compose_streams(streams, config.camera_layout), out)]
    if out.suffix.lower() == suffix:
        base_dir, stem = out.parent, out.stem + "_"
    else:
        out.mkdir(parents=True, exist_ok=True)
        base_dir, stem = out, ""
    paths = []
    for (traj_index, camera), stream in streams.items():
        path = base_dir / f"{stem}traj_{traj_index:03d}_{camera}{suffix}"
        paths.append(encode(stream, path))
    return paths


def _write_npz(result: RenderResult, out: Path, config: RenderConfig) -> None:
    arrays: dict[str, Any] = {
        "metadata": np.asarray(json.dumps(result.metadata, sort_keys=True)),
        "config": np.asarray(json.dumps(config.to_dict(), sort_keys=True)),
    }
    for index, trajectory in enumerate(result.trajectories):
        for key, value in _tensor_items(trajectory):
            arrays[f"traj_{index:03d}/{key}"] = value.detach().cpu().numpy()
    np.savez_compressed(out, **arrays)


def _write_jsonl(result: RenderResult, out: Path) -> None:
    with out.open("w", encoding="utf-8") as file:
        file.write(json.dumps({"type": "metadata", "metadata": result.metadata}) + "\n")
        for item in result.metadata.get("trajectories", []):
            file.write(json.dumps({"type": "trajectory", **item}) + "\n")
        for traj_index, trajectory in enumerate(result.frames):
            for bundle in trajectory:
                file.write(
                    json.dumps(
                        {
                            "type": "frame",
                            "trajectory_index": traj_index,
                            "step": bundle.step,
                            "cameras": sorted(bundle.frames),
                            "metadata": bundle.metadata,
                        }
                    )
                    + "\n"
                )


def _write_notebook_assets(
    result: RenderResult, config: RenderConfig, asset_dir: Path
) -> None:
    if config.save_rollout or config.save_tensordicts or config.format == "ipynb":
        _write_rollout_assets(result, asset_dir)
    _write_json(asset_dir / "config.json", config.to_dict())


def _write_rollout_assets(result: RenderResult, asset_dir: Path) -> None:
    rollouts_dir = asset_dir / "rollouts"
    rollouts_dir.mkdir(parents=True, exist_ok=True)
    for index, trajectory in enumerate(result.trajectories):
        torch.save(trajectory, rollouts_dir / f"traj_{index:03d}.pt")


def _runtime_metadata(config: RenderConfig, asset_dir: Path) -> dict[str, Any]:
    try:
        torchrl_version = torchrl.__version__
    except Exception:
        torchrl_version = None
    checkpoint = {"path": str(config.ckpt), "sha256": None}
    if config.ckpt.is_file():
        checkpoint["sha256"] = checkpoint_hash(config.ckpt)
    return {
        "config": config.to_dict(),
        "command": " ".join(sys.argv),
        "torchrl_version": torchrl_version,
        "torch_version": torch.__version__,
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "checkpoint": checkpoint,
        "policy": config.to_dict()["policy"],
        "env": config.to_dict()["env"],
        "seed": config.seed,
        "device": str(config.device),
        "keys": {
            "obs": key_to_string(config.obs_key),
            "action": key_to_string(config.action_key),
            "done": key_to_string(config.done_key),
            "reward": key_to_string(config.reward_key),
            "pixels": key_to_string(config.pixel_key),
        },
        "asset_dir": str(asset_dir),
    }


def _streams(
    frames: list[list[FrameBundle]],
) -> dict[tuple[int, str], list[np.ndarray]]:
    streams: dict[tuple[int, str], list[np.ndarray]] = {}
    for traj_index, trajectory in enumerate(frames):
        for bundle in trajectory:
            for camera, frame in bundle.frames.items():
                streams.setdefault((traj_index, camera), []).append(frame)
    return streams


def _compose_streams(
    streams: dict[tuple[int, str], list[np.ndarray]],
    layout: str = "grid",
) -> list[np.ndarray]:
    keys = sorted(streams)
    max_len = max(len(streams[key]) for key in keys)
    composed = []
    for frame_index in range(max_len):
        frame_list = []
        for key in keys:
            stream = streams[key]
            frame_list.append(stream[min(frame_index, len(stream) - 1)])
        composed.append(compose_frame_grid(frame_list, layout))
    return composed


def _tensor_items(tensordict: TensorDictBase, prefix: str = ""):
    for key, value in tensordict.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, TensorDictBase):
            yield from _tensor_items(value, name)
        elif torch.is_tensor(value):
            yield name, value


def _has_any_frames(frames: list[list[FrameBundle]]) -> bool:
    return any(bundle.frames for trajectory in frames for bundle in trajectory)


def _relative_asset_dir(asset_dir: Path, base_dir: Path) -> str:
    try:
        return str(asset_dir.relative_to(base_dir))
    except ValueError:
        return str(asset_dir)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
