# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from torchrl.record.loggers.trackio import TrackioLogger
from torchrl.record.loggers.wandb import WandbLogger
from torchrl.trainers.algorithms.configs.common import ConfigBase


@dataclass
class LoggerConfig(ConfigBase):
    """A class to configure a logger.

    Args:
        logger: The logger to use.
    """

    def __post_init__(self) -> None:
        pass


@dataclass
class WandbLoggerConfig(LoggerConfig):
    """A class to configure a Wandb logger.

    .. seealso::
        :class:`~torchrl.record.loggers.wandb.WandbLogger`
    """

    exp_name: str
    offline: bool = False
    save_dir: str | None = None
    id: str | None = None
    project: str | None = None
    video_fps: int = 32
    log_dir: str | None = None
    wandb_kwargs: dict[str, Any] = field(default_factory=dict)

    _target_: str = "torchrl.trainers.algorithms.configs.logging._make_wandb_logger"

    def __post_init__(self) -> None:
        pass


def _make_wandb_logger(
    exp_name: str,
    offline: bool = False,
    save_dir: str | None = None,
    id: str | None = None,
    project: str | None = None,
    video_fps: int = 32,
    log_dir: str | None = None,
    wandb_kwargs: dict[str, Any] | None = None,
) -> WandbLogger:
    wandb_kwargs = dict(wandb_kwargs or {})
    return WandbLogger(
        exp_name=exp_name,
        offline=offline,
        save_dir=save_dir,
        id=id,
        project=project,
        video_fps=video_fps,
        log_dir=log_dir,
        **wandb_kwargs,
    )


@dataclass
class TensorboardLoggerConfig(LoggerConfig):
    """A class to configure a Tensorboard logger.

    .. seealso::
        :class:`~torchrl.record.loggers.tensorboard.TensorboardLogger`
    """

    exp_name: str
    log_dir: str = "tb_logs"

    _target_: str = "torchrl.record.loggers.tensorboard.TensorboardLogger"

    def __post_init__(self) -> None:
        pass


@dataclass
class TrackioLoggerConfig(LoggerConfig):
    """A class to configure a Trackio logger.

    .. seealso::
        :class:`~torchrl.record.loggers.trackio.TrackioLogger`
    """

    exp_name: str
    project: str
    video_fps: int = 32
    trackio_kwargs: dict[str, Any] = field(default_factory=dict)

    _target_: str = "torchrl.trainers.algorithms.configs.logging._make_trackio_logger"

    def __post_init__(self) -> None:
        pass


def _make_trackio_logger(
    exp_name: str,
    project: str,
    video_fps: int = 32,
    trackio_kwargs: dict[str, Any] | None = None,
) -> TrackioLogger:
    trackio_kwargs = dict(trackio_kwargs or {})
    return TrackioLogger(
        exp_name=exp_name,
        project=project,
        video_fps=video_fps,
        **trackio_kwargs,
    )


@dataclass
class CSVLoggerConfig(LoggerConfig):
    """A class to configure a CSV logger.

    .. seealso::
        :class:`~torchrl.record.loggers.csv.CSVLogger`
    """

    exp_name: str
    log_dir: str | None = None
    video_format: str = "pt"
    video_fps: int = 30

    _target_: str = "torchrl.record.loggers.csv.CSVLogger"

    def __post_init__(self) -> None:
        pass
