# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import pathlib
from collections.abc import Mapping, Sequence
from typing import Any

import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf, open_dict

from torchrl._utils import logger as torchrl_logger
from torchrl.checkpoint import Checkpoint, resolve_checkpoint_path, resume_config
from torchrl.checkpoint._hydra import _CONFIG_COMPONENT, _hydra_task_overrides
from torchrl.trainers.trainers import Trainer


def instantiate_trainer(
    cfg: DictConfig, *, overrides: Sequence[str] | None = None
) -> Trainer:
    """Instantiate ``cfg.trainer``, resuming from ``cfg.resume`` when it is set.

    Without ``resume`` this is :func:`hydra.utils.instantiate` on ``cfg.trainer``
    plus registration of the composed configuration on the trainer checkpoint.
    With ``resume`` set to a checkpoint or a
    :class:`~torchrl.checkpoint.CheckpointRotation` directory,
    :func:`~torchrl.checkpoint.resume_config` rebuilds the configuration from
    the saved one with the current command-line overrides on top; the saved
    logger run is reattached before the logger is constructed (W&B resumes the
    saved id with ``resume="must"``, CSV and TensorBoard keep the saved
    directory); checkpoints keep accumulating in the resumed rotation directory
    unless ``checkpoint_rotation.directory`` is overridden; and
    :meth:`~torchrl.trainers.Trainer.load_from_file` restores the trainer state.

    Args:
        cfg (DictConfig): the composed Hydra configuration. It must hold a
            ``trainer`` node and may hold ``resume``.
        overrides (Sequence[str], optional): command-line overrides applied over
            the saved configuration. Defaults to the task overrides of the
            current Hydra run, or none outside a Hydra application.

    Returns:
        The instantiated trainer, restored from the checkpoint when resuming.

    Examples:
        >>> @hydra.main(config_path="config", config_name="config", version_base="1.3")  # doctest: +SKIP
        ... def main(cfg):
        ...     trainer = instantiate_trainer(cfg)
        ...     with trainer.stop_on_signal():
        ...         trainer.train()
    """
    resume = cfg.get("resume", None)
    if resume in (None, ""):
        trainer = hydra.utils.instantiate(cfg.trainer)
        _register_config(trainer, cfg)
        return trainer

    resume_path = pathlib.Path(to_absolute_path(str(resume)))
    checkpoint_path = resolve_checkpoint_path(resume_path)
    if overrides is None:
        overrides = _hydra_task_overrides()
    cfg = resume_config(cfg, checkpoint_path, overrides=overrides)
    logger_cfg = cfg.trainer.get("logger", None)
    saved_logger = Checkpoint.read_component(checkpoint_path, "logger", default=None)
    if logger_cfg is not None and saved_logger is not None:
        _reattach_logger(logger_cfg, saved_logger)
    _continue_rotation(cfg, resume_path, checkpoint_path, overrides)
    trainer = hydra.utils.instantiate(cfg.trainer)
    try:
        trainer.load_from_file(checkpoint_path)
    except BaseException:
        # Release collector workers so a failed restore does not hang at exit.
        trainer.shutdown()
        raise
    # Registered after the restore so the checkpoint does not overwrite the
    # merged configuration with the saved one.
    _register_config(trainer, cfg)
    torchrl_logger.info(
        "Resumed trainer from %s at %s collected frames.",
        checkpoint_path,
        trainer.collected_frames,
    )
    return trainer


def _register_config(trainer: Trainer, cfg: DictConfig) -> None:
    checkpoint = trainer.checkpoint
    if checkpoint is not None and _CONFIG_COMPONENT not in checkpoint:
        checkpoint.register(
            _CONFIG_COMPONENT, OmegaConf.to_container(cfg, resolve=False)
        )


def _reattach_logger(logger_cfg: DictConfig, saved_logger: Mapping[str, Any]) -> None:
    target = str(logger_cfg.get("_target_", "")).lower()
    local = saved_logger.get("local") or {}
    with open_dict(logger_cfg):
        if "wandb" in target:
            run_id = local.get("id")
            if not run_id:
                raise ValueError(
                    "The saved W&B logger state has no run ID; the run cannot be "
                    "resumed."
                )
            logger_cfg.id = run_id
            wandb_kwargs = dict(logger_cfg.get("wandb_kwargs", None) or {})
            wandb_kwargs["resume"] = "must"
            logger_cfg.wandb_kwargs = wandb_kwargs
        elif "csv" in target or "tensorboard" in target:
            logger_cfg.exp_name = saved_logger["exp_name"]
            if saved_logger.get("log_dir"):
                logger_cfg.log_dir = saved_logger["log_dir"]
        else:
            torchrl_logger.warning(
                "The saved logger run is not restored for %r; a new run is created.",
                target,
            )


def _continue_rotation(
    cfg: DictConfig,
    resume_path: pathlib.Path,
    checkpoint_path: pathlib.Path,
    overrides: Sequence[str],
) -> None:
    rotation_cfg = cfg.trainer.get("checkpoint_rotation", None)
    if rotation_cfg is None:
        return
    directory_keys = (
        "checkpoint_rotation.directory=",
        "trainer.checkpoint_rotation.directory=",
    )
    if any(override.lstrip("+").startswith(directory_keys) for override in overrides):
        return
    if resume_path.is_dir() and not Checkpoint.is_checkpoint(resume_path):
        directory = resume_path
    else:
        directory = pathlib.Path(checkpoint_path).parent
    with open_dict(rotation_cfg):
        rotation_cfg.directory = str(directory.resolve())
