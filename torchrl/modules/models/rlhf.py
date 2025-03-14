# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import warnings

from .llm import GPT2RewardModel

__all__ = ["GPT2RewardModel"]

warnings.warn(
    "Imports from torchrl.modules.models.rlhf have moved to torchrl.modules.models.llm. "
    "torchrl.modules.models.rlhf will be deprecated in v0.10.",
    category=DeprecationWarning,
)
