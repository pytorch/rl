# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import dataclasses
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import torch
from tensordict import NestedKey, TensorDictBase

RenderFormat = Literal["ipynb", "mp4", "gif", "frames", "npz", "jsonl"]
CameraLayout = Literal["single", "grid", "horizontal", "vertical", "separate"]
RenderBackendName = Literal["auto", "pixels", "env", "null"]
NotebookRenderBackendName = Literal["auto", "static", "mujoco_wasm", "mujoco-wasm"]
NotebookRolloutMode = Literal["saved", "live", "both"]
EnvBackendName = Literal[
    "auto", "torchrl", "gym", "gymnasium", "mujoco", "dm_control", "isaaclab"
]
ExplorationMode = Literal["deterministic", "mode", "mean", "random"]

__all__ = [
    "CameraLayout",
    "EnvBackendName",
    "ExplorationMode",
    "FrameBundle",
    "NotebookRenderBackendName",
    "NotebookRolloutMode",
    "RenderBackendName",
    "RenderConfig",
    "RenderEnvSpec",
    "RenderFormat",
    "RenderPolicySpec",
    "RenderResult",
    "key_to_string",
    "parse_nested_key",
]


@dataclass
class RenderConfig:
    """Configuration for rendering policy rollouts.

    Args:
        ckpt: Local checkpoint path passed to the policy factory.
        policy: Policy factory or ``"module:object"`` import specification.
        env: Environment factory or ``"module:object"`` import specification.
        num_trajs: Number of trajectories to render.
        format: Artifact format to write.
        notebook_rollout_mode: For ``format="ipynb"``, whether rollouts are
            collected before notebook creation (``"saved"``), inside notebook
            cells (``"live"``), or both (``"both"``).

    Examples:
        >>> from torchrl.render import RenderConfig
        >>> cfg = RenderConfig(
        ...     ckpt="policy.pt",
        ...     policy="project.policy:make_policy",
        ...     env="project.env:make_env",
        ...     max_steps=10,
        ... )
        >>> cfg.num_trajs
        1
    """

    ckpt: str | Path
    policy: str | Callable[..., Any]
    env: str | Callable[..., Any]
    num_trajs: int = 1
    format: RenderFormat = "mp4"
    out: str | Path | None = None
    max_steps: int | None = None
    fps: float = 30.0
    camera: list[str] = field(default_factory=lambda: ["default"])
    camera_layout: CameraLayout = "single"
    deterministic: bool = True
    exploration_mode: ExplorationMode | None = None
    seed: int | None = 0
    device: torch.device | str = "cpu"
    policy_device: torch.device | str | None = None
    env_device: torch.device | str | None = None
    render_backend: RenderBackendName = "auto"
    notebook_render_backend: NotebookRenderBackendName = "auto"
    notebook_rollout_mode: NotebookRolloutMode = "saved"
    env_backend: EnvBackendName = "auto"
    env_kwargs: dict[str, Any] = field(default_factory=dict)
    policy_kwargs: dict[str, Any] = field(default_factory=dict)
    checkpoint_key: str | None = None
    state_dict_key: str | None = None
    strict_load: bool = True
    auto_load_policy: bool = True
    policy_eval: bool = True
    obs_key: NestedKey = "observation"
    action_key: NestedKey = "action"
    done_key: NestedKey = "done"
    reward_key: NestedKey = "reward"
    pixel_key: NestedKey = "pixels"
    from_pixels: bool = False
    pixels_only: bool = False
    render_mode: str | None = None
    save_rollout: bool = False
    save_tensordicts: bool = False
    save_frames: bool = False
    frame_dir: str | Path | None = None
    artifact_dir: str | Path | None = None
    metadata: str | Path | None = None
    overwrite: bool = False
    video_codec: str | None = None
    mujoco_model_path: str | Path | None = None
    mujoco_asset_paths: list[str | Path] = field(default_factory=list)
    mujoco_qpos_key: NestedKey | None = None
    notebook_viewer_port: int = 5178
    dry_run: bool = False
    validate_only: bool = False

    def __post_init__(self) -> None:
        self.ckpt = Path(self.ckpt)
        if self.out is not None:
            self.out = Path(self.out)
        if self.frame_dir is not None:
            self.frame_dir = Path(self.frame_dir)
        if self.artifact_dir is not None:
            self.artifact_dir = Path(self.artifact_dir)
        if self.metadata is not None:
            self.metadata = Path(self.metadata)
        _validate_choice(
            "format", self.format, {"ipynb", "mp4", "gif", "frames", "npz", "jsonl"}
        )
        _validate_choice(
            "camera_layout",
            self.camera_layout,
            {"single", "grid", "horizontal", "vertical", "separate"},
        )
        _validate_choice(
            "render_backend", self.render_backend, {"auto", "pixels", "env", "null"}
        )
        _validate_choice(
            "env_backend",
            self.env_backend,
            {"auto", "torchrl", "gym", "gymnasium", "mujoco", "dm_control", "isaaclab"},
        )
        if self.exploration_mode is not None:
            _validate_choice(
                "exploration_mode",
                self.exploration_mode,
                {"deterministic", "mode", "mean", "random"},
            )
        if self.notebook_render_backend == "mujoco-wasm":
            self.notebook_render_backend = "mujoco_wasm"
        _validate_choice(
            "notebook_render_backend",
            self.notebook_render_backend,
            {"auto", "static", "mujoco_wasm"},
        )
        _validate_choice(
            "notebook_rollout_mode",
            self.notebook_rollout_mode,
            {"saved", "live", "both"},
        )
        if self.notebook_rollout_mode != "saved" and self.format != "ipynb":
            raise ValueError(
                "notebook_rollout_mode='live' or 'both' is only valid for "
                "format='ipynb'."
            )
        if self.mujoco_model_path is not None:
            self.mujoco_model_path = Path(self.mujoco_model_path)
        if isinstance(self.mujoco_asset_paths, str):
            self.mujoco_asset_paths = _split_csv(self.mujoco_asset_paths)
        self.mujoco_asset_paths = [Path(path) for path in self.mujoco_asset_paths]
        self.device = torch.device(self.device)
        if self.policy_device is not None:
            self.policy_device = torch.device(self.policy_device)
        if self.env_device is not None:
            self.env_device = torch.device(self.env_device)
        self.obs_key = parse_nested_key(self.obs_key)
        self.action_key = parse_nested_key(self.action_key)
        self.done_key = parse_nested_key(self.done_key)
        self.reward_key = parse_nested_key(self.reward_key)
        self.pixel_key = parse_nested_key(self.pixel_key)
        if self.mujoco_qpos_key is not None:
            self.mujoco_qpos_key = parse_nested_key(self.mujoco_qpos_key)
        if isinstance(self.camera, str):
            self.camera = _split_csv(self.camera)
        if not self.camera:
            self.camera = ["default"]
        if self.num_trajs < 1:
            raise ValueError(f"num_trajs must be >= 1, got {self.num_trajs}.")
        if self.max_steps is not None and self.max_steps < 1:
            raise ValueError(f"max_steps must be >= 1, got {self.max_steps}.")
        if self.fps <= 0:
            raise ValueError(f"fps must be positive, got {self.fps}.")
        if self.notebook_viewer_port < 1:
            raise ValueError(
                f"notebook_viewer_port must be positive, got {self.notebook_viewer_port}."
            )

    def to_dict(self) -> dict[str, Any]:
        """Returns a JSON-serializable dictionary representation."""
        data = dataclasses.asdict(self)
        for key in ("ckpt", "out", "frame_dir", "artifact_dir", "metadata"):
            if data[key] is not None:
                data[key] = str(data[key])
        if data["mujoco_model_path"] is not None:
            data["mujoco_model_path"] = str(data["mujoco_model_path"])
        data["mujoco_asset_paths"] = [str(path) for path in data["mujoco_asset_paths"]]
        for key in ("device", "policy_device", "env_device"):
            if data[key] is not None:
                data[key] = str(data[key])
        for key in ("policy", "env"):
            if not isinstance(data[key], str):
                data[key] = _callable_name(data[key])
        for key in (
            "obs_key",
            "action_key",
            "done_key",
            "reward_key",
            "pixel_key",
            "mujoco_qpos_key",
        ):
            if data[key] is not None:
                data[key] = key_to_string(data[key])
        return data

    def to_json(self, **kwargs: Any) -> str:
        """Returns this configuration as formatted JSON."""
        return json.dumps(self.to_dict(), **kwargs)


@dataclass
class RenderEnvSpec:
    """Context object passed to environment factories.

    Args:
        device: Device requested for the environment.
        seed: Optional environment seed.
        max_steps: Optional rollout step limit.
        from_pixels: Whether the factory should request pixel observations.
        pixels_only: Whether only pixel observations should be returned.
        camera: Requested camera names.
        render_mode: Optional render mode, such as ``"rgb_array"``.
        env_kwargs: Extra keyword arguments supplied by the user.
        config: Full render configuration.
        checkpoint: Checkpoint payload loaded from ``config.ckpt``, when
            available. Factories can read checkpointed environment metadata
            from it (see :func:`~torchrl.render.save_render_checkpoint`).

    Examples:
        >>> from torchrl.render import RenderConfig, RenderEnvSpec
        >>> cfg = RenderConfig("policy.pt", "p:make", "e:make", max_steps=2)
        >>> spec = RenderEnvSpec.from_config(cfg)
        >>> spec.max_steps
        2
    """

    device: torch.device
    seed: int | None
    max_steps: int | None
    from_pixels: bool
    pixels_only: bool
    camera: list[str]
    render_mode: str | None
    env_kwargs: dict[str, Any]
    config: RenderConfig
    checkpoint: Any | None = None

    @classmethod
    def from_config(
        cls, config: RenderConfig, checkpoint: Any | None = None
    ) -> RenderEnvSpec:
        """Builds an environment spec from a render config."""
        return cls(
            device=torch.device(config.env_device or config.device),
            seed=config.seed,
            max_steps=config.max_steps,
            from_pixels=config.from_pixels,
            pixels_only=config.pixels_only,
            camera=list(config.camera),
            render_mode=config.render_mode,
            env_kwargs=dict(config.env_kwargs),
            config=config,
            checkpoint=checkpoint,
        )


@dataclass
class RenderPolicySpec:
    """Context object passed to policy factories.

    Args:
        ckpt_path: Local checkpoint path.
        checkpoint: Loaded checkpoint payload, if loading succeeded.
        checkpoint_hash: SHA256 checkpoint hash.
        device: Device requested for policy inference.
        env_specs: Optional environment specs exposed by the environment.
        policy_kwargs: Extra keyword arguments supplied by the user.
        config: Full render configuration.

    Examples:
        >>> from pathlib import Path
        >>> from torchrl.render import RenderConfig, RenderPolicySpec
        >>> cfg = RenderConfig("policy.pt", "p:make", "e:make", max_steps=2)
        >>> spec = RenderPolicySpec(Path("policy.pt"), None, None, cfg.device, None, {}, cfg)
        >>> spec.ckpt_path.name
        'policy.pt'
    """

    ckpt_path: Path
    checkpoint: Any | None
    checkpoint_hash: str | None
    device: torch.device
    env_specs: Any | None
    policy_kwargs: dict[str, Any]
    config: RenderConfig


@dataclass
class FrameBundle:
    """One rendered step containing one or more named camera frames.

    Args:
        frames: Mapping from camera name to ``uint8`` RGB image arrays.
        step: Step index within the trajectory.
        trajectory_index: Trajectory index.
        timestamp: Optional external timestamp.
        metadata: Backend-specific frame metadata.

    Examples:
        >>> import numpy as np
        >>> from torchrl.render import FrameBundle
        >>> bundle = FrameBundle({"default": np.zeros((2, 2, 3), dtype=np.uint8)}, 0, 0)
        >>> sorted(bundle.frames)
        ['default']
    """

    frames: dict[str, Any]
    step: int
    trajectory_index: int
    timestamp: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RenderResult:
    """Result returned by :func:`torchrl.render.render_policy`.

    Args:
        artifact_path: Main artifact path, if one was written.
        trajectories: Collected rollout TensorDicts.
        frame_paths: Frame or video paths written by artifact writers.
        metadata: JSON-serializable metadata dictionary.
        warnings: Non-fatal warnings collected during rendering.
        frames: In-memory frame bundles used by artifact writers.

    Examples:
        >>> from torchrl.render import RenderResult
        >>> result = RenderResult(None, [], [], {"num_trajs": 0}, [])
        >>> result.metadata["num_trajs"]
        0
    """

    artifact_path: Path | None
    trajectories: list[TensorDictBase]
    frame_paths: list[Path]
    metadata: dict[str, Any]
    warnings: list[str]
    frames: list[list[FrameBundle]] = field(default_factory=list)


def parse_nested_key(value: NestedKey | str | Sequence[str]) -> NestedKey:
    """Parses dotted strings into TensorDict nested keys.

    Args:
        value: Nested key, dotted string, or sequence of key components.

    Returns:
        A TensorDict nested key.
    """
    if isinstance(value, tuple):
        return value
    if isinstance(value, list):
        return tuple(str(item) for item in value)
    if isinstance(value, str):
        value = value.strip()
        if "." in value:
            return tuple(part for part in value.split(".") if part)
        return value
    raise TypeError(f"Expected a nested key value, got {type(value).__name__}.")


def key_to_string(key: NestedKey) -> str:
    """Formats a TensorDict nested key for config and metadata output."""
    if isinstance(key, tuple):
        return ".".join(str(item) for item in key)
    return str(key)


def _split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _callable_name(value: Any) -> str:
    module = getattr(value, "__module__", None)
    qualname = getattr(value, "__qualname__", None)
    if module and qualname:
        return f"{module}:{qualname}"
    return repr(value)


def _validate_choice(name: str, value: str, choices: set[str]) -> None:
    if value not in choices:
        raise ValueError(f"{name} must be one of {sorted(choices)}, got {value!r}.")
