# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

__all__ = ["checkpoint_hash", "infer_state_dict", "load_checkpoint"]

_STATE_DICT_KEYS = ("model_state_dict", "policy", "state_dict")


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> Any:
    """Loads a local PyTorch checkpoint.

    Args:
        path: Local checkpoint path.
        map_location: Device mapping passed to :func:`torch.load`.

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
        return torch.load(path, map_location=map_location, weights_only=False)
    except Exception as err:
        raise RuntimeError(
            f"Could not load checkpoint {path!s}. If this checkpoint uses a custom "
            "format, provide a policy factory that loads it directly."
        ) from err


def checkpoint_hash(path: str | Path) -> str:
    """Computes the SHA256 digest of a local checkpoint file."""
    path = Path(path).expanduser()
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
