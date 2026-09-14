# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Loading frozen MicroDuck controllers without importing training examples."""

from __future__ import annotations

import hashlib
import math
import urllib.request
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from torchrl._utils import logger as torchrl_logger
from torchrl.envs.custom.mujoco.microduck import MicroDuckEnv, MicroDuckTask

if TYPE_CHECKING:
    from torchrl.modules import ProbabilisticActor


def _fetch_checkpoint(
    source: str | Path,
    *,
    root: str | Path | None = None,
    sha256: str | None = None,
) -> Path:
    """Return a local path to ``source``, downloading a URL into the cache once.

    ``root`` is the MicroDuck cache directory (``~/.cache/torchrl/microduck``
    by default); URLs land in its ``checkpoints`` folder under a URL-specific directory
    and their original file name. The SHA-256 digest of the file is checked against ``sha256`` when
    given, for downloads and local files alike.
    """
    source = str(source)
    if source.startswith(("http://", "https://")):
        cache_root = (
            Path("~/.cache/torchrl/microduck").expanduser()
            if root is None
            else Path(root).expanduser()
        )
        path = (
            cache_root
            / "checkpoints"
            / hashlib.sha256(source.encode()).hexdigest()[:16]
            / Path(source.split("?")[0]).name
        )
        if not path.is_file():
            path.parent.mkdir(parents=True, exist_ok=True)
            torchrl_logger.info(
                "Downloading the walker checkpoint %s to %s", source, path
            )
            partial = path.with_name(path.name + ".partial")
            urllib.request.urlretrieve(source, partial)
            partial.replace(path)
    else:
        path = Path(source).expanduser()
        if not path.is_file():
            raise FileNotFoundError(path)
    if sha256 is not None:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != sha256.lower():
            raise ValueError(
                f"The checkpoint {path} has SHA-256 {digest}, expected {sha256}."
            )
    return path


def load_microduck_walker(
    source: str | Path,
    *,
    device: torch.device | str = "cpu",
    root: str | Path | None = None,
    sha256: str | None = None,
    action_scale: float | None = None,
) -> tuple[ProbabilisticActor, list[MicroDuckTask]]:
    """Load a MicroDuck skill checkpoint as a frozen controller.

    Returns the actor, built on ``device`` with the ``policy_kwargs`` the
    checkpoint recorded and set to evaluation mode without gradients, and the
    task library it was trained with (``task_id`` order). ``action_scale``
    must match the one the walker was trained with, since the deployment environment
    applies the walker's actions with its own scale.

    Args:
        source: Local checkpoint path or immutable artifact URL.
        device: Device on which to reconstruct the policy.
        root: Optional checkpoint cache directory.
        sha256: Expected checkpoint SHA-256 digest.
        action_scale: Required deployment scale; checked against the checkpoint.

    Returns:
        The frozen actor and its ordered task library.

    Examples:
        >>> from torchrl.envs import load_microduck_walker
        >>> walker, tasks = load_microduck_walker("walker.ckpt")  # doctest: +SKIP
    """
    # These runtime imports break envs -> modules -> data -> envs and
    # envs -> render -> envs initialization cycles. Types are imported above.
    from torchrl.envs.custom.mujoco._skill_models import make_actor_critic, make_tasks
    from torchrl.render.checkpoint import load_checkpoint

    payload = load_checkpoint(_fetch_checkpoint(source, root=root, sha256=sha256))
    config = payload.get("config") or {}
    tasks = make_tasks((config.get("env") or {})["tasks"])
    trained_scale = float((config.get("env") or {}).get("action_scale", 0.35))
    if action_scale is not None and not math.isclose(trained_scale, action_scale):
        raise ValueError(
            f"The walker was trained with action_scale={trained_scale}; set "
            f"env.action_scale to that value (got {action_scale})."
        )
    walker, _ = make_actor_critic(
        MicroDuckEnv.OBSERVATION_DIM,
        len(tasks),
        device=device,
        **dict(payload["policy_kwargs"]),
    )
    walker.load_state_dict(payload["model_state_dict"])
    walker.requires_grad_(False).eval()
    return walker, tasks
