# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Any

from torchrl.trainers.algorithms.configs.common import ConfigBase

class LoggerConfig(ConfigBase):
    """A class to configure a logger.

    Args:
        logger: The logger to use.
    """
    pass

class WandbLoggerConfig(LoggerConfig):
    """A class to configure a Wandb logger.

    Args:
        logger: The logger to use.
    """
    _target_: str = "torchrl.trainers.algorithms.configs.logging.WandbLogger"

class TensorboardLoggerConfig(LoggerConfig):
    """A class to configure a Tensorboard logger.

    Args:
        logger: The logger to use.
    """
    _target_: str = "torchrl.trainers.algorithms.configs.logging.TensorboardLogger"

class CSVLoggerConfig(LoggerConfig):
    """A class to configure a CSV logger.

    Args:
        logger: The logger to use.
    """
    _target_: str = "torchrl.trainers.algorithms.configs.logging.CSVLogger"
