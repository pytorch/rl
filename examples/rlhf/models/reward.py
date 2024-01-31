# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import warnings

import torch
from tensordict.nn import TensorDictModule
from torchrl._utils import logger as torchrl_logger

from torchrl.modules.models.rlhf import GPT2RewardModel


def init_reward_model(
    transformer_path=None, reward_model_path=None, device=None, compile_model=False
):
    if transformer_path is None and reward_model_path is None:
        warnings.warn(
            "You did not provide a path to the reward model, a naive reward model will be used instead."
        )
        model = GPT2RewardModel()
    else:
        if not ((transformer_path is None) ^ (reward_model_path is None)):
            raise ValueError(
                "Exactly one of transformer_path or reward_model_path should be specified."
            )
        if transformer_path is not None:
            model = GPT2RewardModel(transformer_path)
        else:
            model = GPT2RewardModel.from_pretrained(reward_model_path)

    model.to(device)
    if compile_model:
        torchrl_logger.info("Compiling the reward model...")
        model = torch.compile(model)

    model = TensorDictModule(
        model,
        in_keys=["input_ids", "attention_mask"],
        out_keys=["rewards", "end_scores"],
    )
    return model
