# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

from itertools import cycle
from pathlib import Path

import torch
from torchrl.data.llm.prompt import PromptData
from torchrl.data.llm.reward import PairwiseDataset, RewardData
from transformers import GPT2Config, GPT2LMHeadModel


def make_tiny_transformer(path: str | Path) -> Path:
    path = Path(path)
    config = GPT2Config(
        n_embd=16,
        n_head=1,
        n_layer=1,
        n_positions=32,
        n_ctx=32,
        vocab_size=50257,
        pad_token_id=50256,
        eos_token_id=50256,
    )
    GPT2LMHeadModel(config).save_pretrained(path)
    return path


def make_prompt_loader(batch_size: int, block_size: int, device: str):
    input_ids = torch.randint(0, 128, (batch_size, block_size), device=device)
    attention_mask = torch.ones_like(input_ids)
    prompt_rindex = torch.full(
        (batch_size,), max(block_size // 2, 1), dtype=torch.long, device=device
    )
    batch = PromptData(
        input_ids=input_ids,
        attention_mask=attention_mask,
        prompt_rindex=prompt_rindex,
        labels=input_ids.clone(),
        batch_size=[batch_size],
    )
    return cycle((batch,))


def make_pairwise_loader(batch_size: int, block_size: int, device: str):
    chosen_ids = torch.randint(0, 128, (batch_size, block_size), device=device)
    rejected_ids = chosen_ids.clone()
    rejected_ids[:, -1] = (rejected_ids[:, -1] + 1) % 128
    attention_mask = torch.ones_like(chosen_ids)
    batch = PairwiseDataset(
        chosen_data=RewardData(
            input_ids=chosen_ids,
            attention_mask=attention_mask,
            batch_size=[batch_size],
        ),
        rejected_data=RewardData(
            input_ids=rejected_ids,
            attention_mask=attention_mask.clone(),
            batch_size=[batch_size],
        ),
        batch_size=[batch_size],
    )
    return cycle((batch,))
