# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

from torchrl.render import render_policy

from torchrl.render.config import parse_nested_key, RenderConfig, RenderFormat

_has_yaml = importlib.util.find_spec("yaml") is not None
_has_tomllib = importlib.util.find_spec("tomllib") is not None

_YAML_ERROR = (
    "YAML config and kwargs files require PyYAML. When running TorchRL from this "
    "repository with uv, use `uv run --extra rendering <command>` or install "
    "PyYAML separately."
)
_TOML_ERROR = (
    "TOML config files require Python >= 3.11 tomllib support in rlrender MVP."
)

__all__ = ["build_parser", "config_from_args", "main"]


def build_parser() -> argparse.ArgumentParser:
    """Builds the rlrender command-line parser."""
    parser = argparse.ArgumentParser(description="Render TorchRL policy rollouts.")
    parser.add_argument(
        "--config", help="JSON/YAML/TOML config file containing CLI options."
    )
    parser.add_argument("--ckpt", help="Local policy checkpoint path.")
    parser.add_argument(
        "--policy", help="Policy factory import spec, e.g. project.policy:make_policy."
    )
    parser.add_argument(
        "--env", help="Environment factory import spec, e.g. project.env:make_env."
    )
    parser.add_argument(
        "--num-trajs", type=int, help="Number of trajectories to render."
    )
    parser.add_argument(
        "--format",
        choices=["ipynb", "mp4", "gif", "frames", "npz", "jsonl"],
        help="Output artifact format. Defaults from --out suffix, otherwise mp4.",
    )
    parser.add_argument("--out", help="Output artifact path.")
    parser.add_argument("--max-steps", type=int, help="Maximum steps per trajectory.")
    parser.add_argument(
        "--fps", type=float, help="Frames per second for video artifacts."
    )
    parser.add_argument("--seed", type=int, help="Environment seed.")
    parser.add_argument("--device", help="Default device.")
    parser.add_argument("--policy-device", help="Policy device.")
    parser.add_argument("--env-device", help="Environment device.")
    parser.add_argument("--render-backend", choices=["auto", "pixels", "env", "null"])
    parser.add_argument(
        "--notebook-render-backend",
        choices=["auto", "static", "mujoco_wasm", "mujoco-wasm"],
        help=(
            "Notebook-only render helper. Use mujoco_wasm to generate a browser "
            "MuJoCo viewer sidecar for saved qpos rollouts."
        ),
    )
    parser.add_argument(
        "--notebook-rollout-mode",
        choices=["saved", "live", "both"],
        help=(
            "Notebook-only rollout mode. 'saved' collects rollouts before writing "
            "the notebook, 'live' collects them when notebook cells run, and "
            "'both' does both."
        ),
    )
    parser.add_argument(
        "--env-backend",
        choices=[
            "auto",
            "torchrl",
            "gym",
            "gymnasium",
            "mujoco",
            "dm_control",
            "isaaclab",
        ],
    )
    parser.add_argument(
        "--env-kwargs", help="Inline JSON or path to JSON/YAML/TOML env kwargs."
    )
    parser.add_argument(
        "--policy-kwargs", help="Inline JSON or path to JSON/YAML/TOML policy kwargs."
    )
    parser.add_argument("--checkpoint-key", help="Checkpoint payload key for loading.")
    parser.add_argument(
        "--state-dict-key", help="Explicit state-dict key for automatic loading."
    )
    parser.add_argument(
        "--strict-load", dest="strict_load", action="store_true", default=None
    )
    parser.add_argument("--no-strict-load", dest="strict_load", action="store_false")
    parser.add_argument(
        "--auto-load-policy", dest="auto_load_policy", action="store_true", default=None
    )
    parser.add_argument(
        "--no-auto-load-policy", dest="auto_load_policy", action="store_false"
    )
    parser.add_argument("--eval", dest="policy_eval", action="store_true", default=None)
    parser.add_argument("--train", dest="policy_eval", action="store_false")
    parser.add_argument("--obs-key", help="Observation key for tensor-only policies.")
    parser.add_argument(
        "--action-key", help="Action key to write for tensor-only policies."
    )
    parser.add_argument(
        "--done-key", help="Done key used to detect episode completion."
    )
    parser.add_argument("--reward-key", help="Reward key used for metadata returns.")
    parser.add_argument(
        "--pixel-key", help="Pixel key used by the TensorDict pixel backend."
    )
    parser.add_argument(
        "--from-pixels", dest="from_pixels", action="store_true", default=None
    )
    parser.add_argument("--no-from-pixels", dest="from_pixels", action="store_false")
    parser.add_argument(
        "--pixels-only", dest="pixels_only", action="store_true", default=None
    )
    parser.add_argument("--no-pixels-only", dest="pixels_only", action="store_false")
    parser.add_argument(
        "--render-mode", help="Environment render mode, e.g. rgb_array."
    )
    deterministic = parser.add_mutually_exclusive_group()
    deterministic.add_argument(
        "--deterministic", dest="deterministic", action="store_true", default=None
    )
    deterministic.add_argument(
        "--stochastic", dest="deterministic", action="store_false"
    )
    parser.add_argument(
        "--exploration-mode",
        choices=["deterministic", "mode", "mean", "random"],
        help="TorchRL exploration mode for rollout.",
    )
    parser.add_argument(
        "--camera", help="Comma-separated camera names for metadata/backends."
    )
    parser.add_argument(
        "--camera-layout",
        choices=["single", "grid", "horizontal", "vertical", "separate"],
    )
    parser.add_argument(
        "--save-rollout", dest="save_rollout", action="store_true", default=None
    )
    parser.add_argument("--no-save-rollout", dest="save_rollout", action="store_false")
    parser.add_argument(
        "--save-tensordicts", dest="save_tensordicts", action="store_true", default=None
    )
    parser.add_argument(
        "--no-save-tensordicts", dest="save_tensordicts", action="store_false"
    )
    parser.add_argument(
        "--save-frames", dest="save_frames", action="store_true", default=None
    )
    parser.add_argument("--no-save-frames", dest="save_frames", action="store_false")
    parser.add_argument("--frame-dir", help="Directory for saved frames.")
    parser.add_argument("--artifact-dir", help="Directory for sidecar assets.")
    parser.add_argument("--metadata", help="Metadata JSON path.")
    parser.add_argument(
        "--overwrite", dest="overwrite", action="store_true", default=None
    )
    parser.add_argument("--no-overwrite", dest="overwrite", action="store_false")
    parser.add_argument("--video-codec", help="Codec name forwarded to torchcodec.")
    parser.add_argument(
        "--quality", type=int, help="Reserved image/video quality setting."
    )
    parser.add_argument("--width", type=int, help="Reserved render width.")
    parser.add_argument("--height", type=int, help="Reserved render height.")
    parser.add_argument(
        "--mujoco-model-path",
        help="MJCF/XML model copied into MuJoCo WASM notebook artifacts.",
    )
    parser.add_argument(
        "--mujoco-asset-paths",
        action="append",
        help=(
            "Additional MuJoCo asset file or directory for WASM notebooks. "
            "May be passed more than once."
        ),
    )
    parser.add_argument(
        "--mujoco-qpos-key",
        help="TensorDict key containing qpos trajectories for WASM playback.",
    )
    parser.add_argument(
        "--notebook-viewer-port",
        type=int,
        help="Localhost port used by generated MuJoCo WASM notebook viewer.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print config without rendering.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate config without rendering.",
    )
    parser.add_argument(
        "--print-config", action="store_true", help="Print normalized config JSON."
    )
    return parser


def config_from_args(args: argparse.Namespace) -> RenderConfig:
    """Constructs a :class:`~torchrl.render.RenderConfig` from parsed CLI args."""
    data: dict[str, Any] = {}
    if args.config:
        data.update(_load_mapping(args.config))
    cli_data = vars(args).copy()
    for ignored in ("config", "print_config"):
        cli_data.pop(ignored, None)
    for key, value in cli_data.items():
        if value is not None:
            data[key.replace("-", "_")] = value
    if "env_kwargs" in data and isinstance(data["env_kwargs"], str):
        data["env_kwargs"] = _load_mapping_or_inline(data["env_kwargs"])
    if "policy_kwargs" in data and isinstance(data["policy_kwargs"], str):
        data["policy_kwargs"] = _load_mapping_or_inline(data["policy_kwargs"])
    for key in (
        "obs_key",
        "action_key",
        "done_key",
        "reward_key",
        "pixel_key",
        "mujoco_qpos_key",
    ):
        if key in data:
            data[key] = parse_nested_key(data[key])
    if "format" not in data or data["format"] is None:
        data["format"] = _infer_format(data.get("out"))
    for required in ("ckpt", "policy", "env"):
        if required not in data or data[required] is None:
            raise ValueError(
                f"Missing required rlrender option --{required.replace('_', '-')}."
            )
    if data.get("format") == "ipynb" and "save_rollout" not in data:
        data["save_rollout"] = data.get("notebook_rollout_mode", "saved") != "live"
    return RenderConfig(**data)


def main(argv: list[str] | None = None) -> int:
    """Entry point for ``rlrender`` and ``torchrl-render``."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config = config_from_args(args)
        if args.print_config or args.dry_run:
            sys.stdout.write(config.to_json(indent=2, sort_keys=True) + "\n")
        if args.dry_run or args.validate_only:
            return 0
        result = render_policy(config)
        if result.artifact_path is not None:
            sys.stdout.write(str(result.artifact_path) + "\n")
        return 0
    except Exception as err:
        parser.exit(2, f"rlrender: error: {err}\n")


def _infer_format(out: Any) -> RenderFormat:
    if out is None:
        return "mp4"
    suffix = Path(out).suffix.lower()
    mapping: dict[str, RenderFormat] = {
        ".ipynb": "ipynb",
        ".mp4": "mp4",
        ".gif": "gif",
        ".npz": "npz",
        ".jsonl": "jsonl",
    }
    return mapping.get(suffix, "frames" if suffix == "" else "mp4")


def _load_mapping_or_inline(value: str) -> dict[str, Any]:
    stripped = value.strip()
    if stripped.startswith("{"):
        payload = json.loads(stripped)
    else:
        payload = _load_mapping(value)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a mapping, got {type(payload).__name__}.")
    return payload


def _load_mapping(path: str | Path) -> dict[str, Any]:
    path = Path(path).expanduser()
    suffix = path.suffix.lower()
    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
    elif suffix in (".yaml", ".yml"):
        if not _has_yaml:
            raise ModuleNotFoundError(_YAML_ERROR)
        yaml = importlib.import_module("yaml")
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    elif suffix == ".toml":
        if not _has_tomllib:
            raise ModuleNotFoundError(_TOML_ERROR)
        tomllib = importlib.import_module("tomllib")
        payload = tomllib.loads(path.read_text(encoding="utf-8"))
    else:
        raise ValueError(
            f"Unsupported config file suffix {suffix!r}; use JSON, YAML, or TOML."
        )
    if not isinstance(payload, dict):
        raise TypeError(
            f"Expected {path!s} to contain a mapping, got {type(payload).__name__}."
        )
    return payload


if __name__ == "__main__":
    raise SystemExit(main())
