# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass

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

    _target_: str = "torchrl.record.loggers.wandb.WandbLogger"

    def __post_init__(self) -> None:
        pass


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
