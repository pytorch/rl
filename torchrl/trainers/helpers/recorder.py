# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field
from typing import Any


@dataclass
class RecorderConfig:
    record_video: bool = False
    # whether a video of the task should be rendered during logging.
    no_video: bool = True
    # whether a video of the task should be rendered during logging.
    exp_name: str = ""
    # experiment name. Used for logging directory.
    # A date and uuid will be joined to account for multiple experiments with the same name.
    record_interval: int = 1000
    # number of batch collections in between two collections of validation rollouts. Default=1000.
    record_frames: int = 1000
    # number of steps in validation rollouts. " "Default=1000.
    recorder_log_keys: Any = field(default_factory=lambda: ["reward"])
    # Keys to log in the recorder
