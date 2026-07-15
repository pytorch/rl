# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, TYPE_CHECKING

from torchrl.record.loggers.trackio import TrackioLogger
from torchrl.record.loggers.wandb import WandbLogger
from torchrl.trainers.algorithms.configs.common import ConfigBase

if TYPE_CHECKING:
    _LoggerServiceBackend = Literal["direct", "process", "ray"]
else:
    # OmegaConf structured configs resolve this alias at runtime and do not
    # support Literal on all TorchRL-supported versions.
    _LoggerServiceBackend = str


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
    """Hydra configuration for :class:`~torchrl.record.loggers.WandbLogger`.

    .. seealso::
        :class:`~torchrl.record.loggers.wandb.WandbLogger`
    """

    exp_name: str
    offline: bool = False
    save_dir: str | None = None
    id: str | None = None
    project: str | None = None
    video_fps: int = 32
    log_env_packages: bool = True
    log_dir: str | None = None
    wandb_kwargs: dict[str, Any] = field(default_factory=dict)
    service_backend: _LoggerServiceBackend = "direct"
    service_backend_options: dict[str, Any] = field(default_factory=dict)

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
    log_env_packages: bool = True,
    log_dir: str | None = None,
    wandb_kwargs: dict[str, Any] | None = None,
    service_backend: Literal["direct", "process", "ray"] = "direct",
    service_backend_options: dict[str, Any] | None = None,
) -> WandbLogger:
    wandb_kwargs = dict(wandb_kwargs or {})
    return WandbLogger(
        exp_name=exp_name,
        offline=offline,
        save_dir=save_dir,
        id=id,
        project=project,
        video_fps=video_fps,
        log_env_packages=log_env_packages,
        log_dir=log_dir,
        service_backend=service_backend,
        service_backend_options=service_backend_options,
        **wandb_kwargs,
    )


@dataclass
class TensorboardLoggerConfig(LoggerConfig):
    """Hydra configuration for :class:`~torchrl.record.loggers.TensorboardLogger`.

    .. seealso::
        :class:`~torchrl.record.loggers.tensorboard.TensorboardLogger`
    """

    exp_name: str
    log_dir: str = "tb_logs"
    service_backend: _LoggerServiceBackend = "direct"
    service_backend_options: dict[str, Any] = field(default_factory=dict)

    _target_: str = "torchrl.record.loggers.tensorboard.TensorboardLogger"

    def __post_init__(self) -> None:
        pass


@dataclass
class TrackioLoggerConfig(LoggerConfig):
    """Hydra configuration for :class:`~torchrl.record.loggers.TrackioLogger`.

    .. seealso::
        :class:`~torchrl.record.loggers.trackio.TrackioLogger`
    """

    exp_name: str
    project: str
    video_fps: int = 32
    trackio_kwargs: dict[str, Any] = field(default_factory=dict)
    service_backend: _LoggerServiceBackend = "direct"
    service_backend_options: dict[str, Any] = field(default_factory=dict)

    _target_: str = "torchrl.trainers.algorithms.configs.logging._make_trackio_logger"

    def __post_init__(self) -> None:
        pass


def _make_trackio_logger(
    exp_name: str,
    project: str,
    video_fps: int = 32,
    trackio_kwargs: dict[str, Any] | None = None,
    service_backend: Literal["direct", "process", "ray"] = "direct",
    service_backend_options: dict[str, Any] | None = None,
) -> TrackioLogger:
    trackio_kwargs = dict(trackio_kwargs or {})
    return TrackioLogger(
        exp_name=exp_name,
        project=project,
        video_fps=video_fps,
        service_backend=service_backend,
        service_backend_options=service_backend_options,
        **trackio_kwargs,
    )


@dataclass
class CSVLoggerConfig(LoggerConfig):
    """Hydra configuration for :class:`~torchrl.record.loggers.CSVLogger`.

    .. seealso::
        :class:`~torchrl.record.loggers.csv.CSVLogger`
    """

    exp_name: str
    log_dir: str | None = None
    video_format: str = "pt"
    video_fps: int = 30
    service_backend: _LoggerServiceBackend = "direct"
    service_backend_options: dict[str, Any] = field(default_factory=dict)

    _target_: str = "torchrl.record.loggers.csv.CSVLogger"

    def __post_init__(self) -> None:
        pass
