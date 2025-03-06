# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import warnings

from torchrl.data.llm import (
    AdaptiveKLController,
    ConstantKLController,
    create_infinite_iterator,
    get_dataloader,
    PairwiseDataset,
    PromptData,
    PromptTensorDictTokenizer,
    RewardData,
    RolloutFromModel,
    TensorDictTokenizer,
    TokenizedDatasetLoader,
)

__all__ = [
    "create_infinite_iterator",
    "get_dataloader",
    "TensorDictTokenizer",
    "TokenizedDatasetLoader",
    "PromptData",
    "PromptTensorDictTokenizer",
    "PairwiseDataset",
    "RewardData",
    "AdaptiveKLController",
    "ConstantKLController",
    "RolloutFromModel",
]

warnings.warn(
    "Imports from torchrl.data.rlhf have moved to torchrl.data.llm. "
    "torchrl.data.rlhf will be deprecated in v0.10.",
    category=DeprecationWarning,
)
