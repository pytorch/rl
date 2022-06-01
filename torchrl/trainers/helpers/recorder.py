# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass

__all__ = ["parser_recorder_args"]

@dataclass
class RecorderConfig: 
    record_video: bool = False
    no_video: bool = True
    exp_name: str = ""
    record_interval: int = 1000
    record_frames: int = 1000
