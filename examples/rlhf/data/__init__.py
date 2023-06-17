# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from torchrl.data.rlhf.dataset import get_reward_dataloader
from .openai_summarize_tldr import get_prompt_dataloader

__all__ = ["get_prompt_dataloader", "get_reward_dataloader"]
