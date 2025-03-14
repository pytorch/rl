# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import warnings

from .llm import (
    as_nested_tensor,
    as_padded_tensor,
    DataLoadingPrimer,
    KLRewardTransform,
)

__all__ = [
    "as_padded_tensor",
    "as_nested_tensor",
    "DataLoadingPrimer",
    "KLRewardTransform",
]

warnings.warn(
    "Imports from torchrl.envs.transforms.rlhf have moved to torchrl.envs.transforms.llm. "
    "torchrl.envs.transforms.rlhf will be deprecated in v0.10.",
    category=DeprecationWarning,
)
