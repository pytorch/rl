# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import importlib.util
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from torchrl._utils import logger as torchrl_logger
from torchrl.checkpoint._checkpoint import Checkpoint

if TYPE_CHECKING:
    from omegaconf import DictConfig

_has_omegaconf = importlib.util.find_spec("omegaconf") is not None
_has_hydra = importlib.util.find_spec("hydra") is not None

_CONFIG_COMPONENT = "config"


def resume_config(
    cfg: DictConfig,
    checkpoint_path: str | Path,
    *,
    overrides: Sequence[str] | None = None,
) -> DictConfig:
    """Return the configuration of a run resumed from ``checkpoint_path``.

    The configuration saved with the checkpoint under its ``config`` component
    is the base and ``overrides`` are applied on top, so ``resume=<path>`` alone
    rebuilds the original run while ``resume=<path> collector.total_frames=...``
    extends it. Interpolations survive because recipes save the configuration
    unresolved. Config-group overrides such as ``logger@logger=csv``, deletions
    (``~key``) and bare flags cannot be applied to a saved configuration and
    are ignored with a warning. When the checkpoint holds no ``config``
    component, ``cfg`` is returned unchanged with a warning.

    Args:
        cfg (DictConfig): the configuration composed for the current run.
        checkpoint_path (str or Path): the checkpoint being resumed.
        overrides (Sequence[str], optional): ``key=value`` overrides applied over
            the saved configuration. Defaults to the task overrides of the
            current Hydra run, or none outside a Hydra application.

    Returns:
        The configuration to run.

    Examples:
        >>> import tempfile
        >>> from omegaconf import OmegaConf  # doctest: +SKIP
        >>> from torchrl.checkpoint import Checkpoint, resume_config
        >>> saved = {"budget": 100, "trainer": {"total_frames": "${budget}"}}
        >>> with tempfile.TemporaryDirectory() as tmpdir:  # doctest: +SKIP
        ...     path = Checkpoint(config=saved).save(f"{tmpdir}/checkpoint")
        ...     cfg = resume_config(
        ...         OmegaConf.create({"budget": 5}), path, overrides=["budget=200"]
        ...     )
        >>> cfg.trainer.total_frames  # doctest: +SKIP
        200
    """
    if not _has_omegaconf:
        raise ImportError("resume_config requires omegaconf.")
    from omegaconf import OmegaConf

    if overrides is None:
        overrides = _hydra_task_overrides()
    if _CONFIG_COMPONENT not in Checkpoint.manifest(checkpoint_path)["components"]:
        torchrl_logger.warning(
            "Checkpoint %s has no saved configuration; the current configuration "
            "is used as is.",
            checkpoint_path,
        )
        return cfg
    base = OmegaConf.create(
        Checkpoint.read_component(checkpoint_path, _CONFIG_COMPONENT)
    )
    dotlist = []
    for override in overrides:
        key, separator, value = override.partition("=")
        if key.startswith("~") or not separator:
            torchrl_logger.warning(
                "Override %r is ignored on resume: only key=value overrides apply "
                "to a saved configuration.",
                override,
            )
            continue
        key = key.lstrip("+")
        if "@" in key or "/" in key:
            torchrl_logger.warning(
                "Config-group override %r cannot be applied to a saved "
                "configuration and is ignored on resume.",
                override,
            )
            continue
        dotlist.append(f"{key}={value}")
    if not dotlist:
        return base
    return OmegaConf.merge(base, OmegaConf.from_dotlist(dotlist))


def _hydra_task_overrides() -> list[str]:
    if not _has_hydra:
        return []
    from hydra.core.hydra_config import HydraConfig

    try:
        return list(HydraConfig.get().overrides.task)
    except ValueError:
        # Not running under @hydra.main.
        return []
