# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Vision-Language-Action (VLA) data primitives.

This subpackage gathers the data-side building blocks shared by the VLA
transforms, policies and losses: the canonical TensorDict schema, robot
dataset metadata, and action tokenizers.
"""
from __future__ import annotations

from torchrl.data.vla.metadata import ActionSpace, GripperMode, RobotDatasetMetadata
from torchrl.data.vla.schema import (
    ACTION_CHUNK_KEY,
    ACTION_IS_PAD_KEY,
    ACTION_KEY,
    ACTION_TOKENS_KEY,
    IMAGE_KEY,
    INSTRUCTION_KEY,
    OBSERVATION_KEY,
    STATE_KEY,
    validate_vla_tensordict,
)
from torchrl.data.vla.tokenizers import (
    ActionTokenizerBase,
    UniformActionTokenizer,
    VocabTailActionTokenizer,
)

__all__ = [
    "ActionSpace",
    "ActionTokenizerBase",
    "GripperMode",
    "RobotDatasetMetadata",
    "UniformActionTokenizer",
    "VocabTailActionTokenizer",
    "validate_vla_tensordict",
    "OBSERVATION_KEY",
    "IMAGE_KEY",
    "STATE_KEY",
    "INSTRUCTION_KEY",
    "ACTION_KEY",
    "ACTION_CHUNK_KEY",
    "ACTION_IS_PAD_KEY",
    "ACTION_TOKENS_KEY",
]
