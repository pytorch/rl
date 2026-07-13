# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import hashlib
import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from torchrl.checkpoint import Checkpoint, CheckpointFormat

__all__ = [
    "checkpoint_hash",
    "infer_state_dict",
    "load_checkpoint",
    "save_render_checkpoint",
]

_STATE_DICT_KEYS = ("model_state_dict", "policy", "state_dict")


class _StateDictComponent:
    def __init__(self, state_dict: Mapping[str, Any] | None = None) -> None:
        self.value = state_dict

    def state_dict(self) -> Mapping[str, Any]:
        if self.value is None:
            raise RuntimeError("No state dict has been assigned.")
        return self.value

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        self.value = state_dict


def save_render_checkpoint(
    path: str | Path | None,
    model: Any,
    *,
    env_metadata: Mapping[str, Any] | None = None,
    frames: int | None = None,
    metrics: Mapping[str, Any] | None = None,
    config: Mapping[str, Any] | None = None,
    extra: Mapping[str, Any] | None = None,
    format: CheckpointFormat | None = None,
) -> Path | None:
    """Writes a checkpoint in the layout expected by rlrender factories.

    The model weights are stored under the canonical ``"model_state_dict"`` key
    probed by :func:`infer_state_dict`, and ``env_metadata`` entries are merged
    at the top level of the payload so environment and policy factories can
    rebuild the training setup. Conventional environment metadata keys used by
    the sota-implementations factories are ``"env_name"``, ``"env_backend"``,
    ``"env_config_overrides"``, ``"env_num_envs"``, ``"env_batch_mode"``,
    ``"normalize_observation"``, and ``"vecnorm"`` (frozen observation
    normalization statistics).

    Args:
        path: Destination checkpoint path. ``None`` or ``""`` disables
            checkpointing.
        model: Module exposing ``state_dict()``, or a ready state-dict mapping.
        env_metadata: Environment metadata merged into the payload.
        frames: Number of training frames collected so far.
        metrics: Scalar metrics recorded at checkpoint time.
        config: JSON-serializable training configuration.
        extra: Additional payload entries merged last.
        format: Unified checkpoint container format. When omitted, writes the
            legacy :func:`torch.save` payload during the compatibility window.

    Returns:
        The written checkpoint path, or ``None`` when checkpointing is disabled.

    Examples:
        >>> import tempfile
        >>> import torch
        >>> from torchrl.render import load_checkpoint, save_render_checkpoint
        >>> module = torch.nn.Linear(2, 2)
        >>> with tempfile.TemporaryDirectory() as tmpdir:
        ...     path = save_render_checkpoint(
        ...         f"{tmpdir}/policy.pt",
        ...         module,
        ...         env_metadata={"env_name": "CartPole-v1"},
        ...         format="archive",
        ...     )
        ...     payload = load_checkpoint(path)
        >>> payload["env_name"]
        'CartPole-v1'
    """
    if path in (None, ""):
        return None
    path = Path(path).expanduser()
    if format is None:
        warnings.warn(
            "The default save_render_checkpoint output will change from the "
            "legacy torch.save payload to a unified TorchRL checkpoint in v0.15. "
            "Pass format='archive' or format='directory' to opt in now.",
            FutureWarning,
            stacklevel=2,
        )
        payload: dict[str, Any] = {
            "model_state_dict": model
            if isinstance(model, Mapping)
            else model.state_dict()
        }
        if env_metadata:
            payload.update(dict(env_metadata))
        if frames is not None:
            payload["frames"] = int(frames)
        if metrics is not None:
            payload["metrics"] = dict(metrics)
        if config is not None:
            payload["config"] = dict(config)
        if extra:
            payload.update(dict(extra))
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, path)
        return path

    policy = _StateDictComponent(model) if isinstance(model, Mapping) else model
    components: dict[str, Any] = {"policy": policy}
    if env_metadata is not None:
        metadata = dict(env_metadata)
        components["environment_metadata"] = (
            _StateDictComponent(metadata) if _contains_tensor(metadata) else metadata
        )
    if frames is not None:
        components["trainer_state"] = {"frames": int(frames)}
    if metrics is not None:
        components["metrics"] = dict(metrics)
    if config is not None:
        components["config"] = dict(config)
    if extra is not None:
        extra_payload = dict(extra)
        components["extra"] = (
            _StateDictComponent(extra_payload)
            if _contains_tensor(extra_payload)
            else extra_payload
        )
    return Checkpoint(format=format, **components).save(path)


def load_checkpoint(
    path: str | Path,
    map_location: str | torch.device = "cpu",
    *,
    weights_only: bool | None = None,
) -> Any:
    """Loads a local PyTorch checkpoint.

    Args:
        path: Local checkpoint path.
        map_location: Device mapping passed to :func:`torch.load`.
        weights_only: Whether payloads are restricted to safe weight-only
            types. Unified checkpoints default to ``True``. Legacy payloads
            retain their historical ``False`` default for compatibility.

    Returns:
        The checkpoint payload.
    """
    if "://" in str(path):
        raise ValueError(
            f"Only local checkpoint paths are supported in rlrender MVP, got {path!s}."
        )
    path = Path(path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint path does not exist: {path!s}.")
    try:
        if Checkpoint.is_checkpoint(path):
            return _load_unified_checkpoint(
                path,
                map_location,
                weights_only=True if weights_only is None else weights_only,
            )
        return torch.load(
            path,
            map_location=map_location,
            weights_only=False if weights_only is None else weights_only,
        )
    except Exception as err:
        raise RuntimeError(
            f"Could not load checkpoint {path!s}. If this checkpoint uses a custom "
            "format, provide a policy factory that loads it directly."
        ) from err


def checkpoint_hash(path: str | Path) -> str:
    """Compute a SHA256 digest over all checkpoint bytes."""
    path = Path(path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint path does not exist: {path!s}.")
    digest = hashlib.sha256()
    files = [path] if path.is_file() else sorted(path.rglob("*"))
    for file_path in files:
        if not file_path.is_file():
            continue
        if path.is_dir():
            digest.update(file_path.relative_to(path).as_posix().encode())
            digest.update(b"\0")
        with file_path.open("rb") as file:
            for chunk in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _load_unified_checkpoint(
    path: Path,
    map_location: str | torch.device,
    *,
    weights_only: bool,
) -> dict[str, Any]:
    manifest = Checkpoint.manifest(path)
    policy = _StateDictComponent()
    environment_metadata = _restore_target(manifest, "environment_metadata")
    trainer_state = _restore_target(manifest, "trainer_state")
    metrics = _restore_target(manifest, "metrics")
    config = _restore_target(manifest, "config")
    extra = _restore_target(manifest, "extra")
    checkpoint = Checkpoint(
        strict="ignore",
        policy=policy,
        environment_metadata=environment_metadata,
        trainer_state=trainer_state,
        metrics=metrics,
        config=config,
        extra=extra,
    )
    requested = set(checkpoint.components).intersection(manifest["components"])
    checkpoint.load(
        path,
        components=requested,
        map_location=map_location,
        tensor_load_kwargs={"weights_only": weights_only, "mmap": True},
        strict="ignore",
    )
    environment_metadata = _restored_mapping(environment_metadata)
    trainer_state = _restored_mapping(trainer_state)
    metrics = _restored_mapping(metrics)
    config = _restored_mapping(config)
    extra = _restored_mapping(extra)
    payload: dict[str, Any] = {}
    if policy.value is not None:
        payload["model_state_dict"] = policy.value
    payload.update(environment_metadata)
    frames = trainer_state.get("frames", trainer_state.get("collected_frames"))
    if frames is not None:
        payload["frames"] = frames
    if metrics:
        payload["metrics"] = metrics
    if config:
        payload["config"] = config
    payload.update(extra)
    return payload


def _restore_target(manifest: Mapping[str, Any], name: str) -> Any:
    record = manifest["components"].get(name)
    if record is not None and record["adapter"] == "torchrl.state_dict":
        return _StateDictComponent()
    return {}


def _restored_mapping(component: Any) -> dict[str, Any]:
    if isinstance(component, _StateDictComponent):
        return dict(component.value or {})
    return component


def _contains_tensor(value: Any) -> bool:
    if isinstance(value, Tensor):
        return True
    if isinstance(value, Mapping):
        return any(_contains_tensor(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_tensor(item) for item in value)
    return False


def infer_state_dict(payload: Any, key: str | None = None) -> Mapping[str, Tensor]:
    """Infers a model state dict from common checkpoint payload layouts.

    Args:
        payload: Checkpoint payload.
        key: Explicit state-dict key to read from mapping payloads.

    Returns:
        A mapping from parameter names to tensors.
    """
    if key is not None:
        if not isinstance(payload, Mapping) or key not in payload:
            raise KeyError(
                f"Checkpoint payload does not contain state-dict key {key!r}."
            )
        payload = payload[key]
    elif isinstance(payload, Mapping):
        for candidate in _STATE_DICT_KEYS:
            value = payload.get(candidate)
            if _looks_like_state_dict(value):
                payload = value
                break
    if _looks_like_state_dict(payload):
        return payload
    raise TypeError(
        "Could not infer a tensor state dict from checkpoint payload. Expected one "
        "of 'model_state_dict', 'policy', 'state_dict', or a direct mapping of tensors."
    )


def _looks_like_state_dict(value: Any) -> bool:
    if not isinstance(value, Mapping) or not value:
        return False
    return all(isinstance(key, str) for key in value) and any(
        torch.is_tensor(item) for item in value.values()
    )
