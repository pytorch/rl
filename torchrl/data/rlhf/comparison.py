import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from datasets import Dataset as HFDataset
from tensordict import tensorclass, TensorDict
from torch.utils.data import Dataset
from tqdm import tqdm

from torchrl.data import TensorStorage, TensorDictReplayBuffer
from torchrl.data.replay_buffers import SamplerWithoutReplacement
from torchrl.data.rlhf.dataset import create_or_load_dataset, \
    create_infinite_dataloader

HERE = Path(__file__).parent
DATASET = "CarperAI/openai_summarize_comparisons"


@tensorclass
class RewardData:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    rewards: Optional[torch.Tensor] = None
    end_scores: Optional[torch.Tensor] = None

@tensorclass
class PairwiseDataset:
    """Represents a dataset in a pairwise manner (chosen vs rejected).

    Attributes:
        chosen_data: data to be chosen.
        rejected_data: corresponding data to be rejected.

    Examples:
        >>> data = PairwiseDataset.from_dataset("train", max_length=550)
        >>> print(data)
        PairwiseDataset(
            chosen_data=RewardData(
                attention_mask=MemmapTensor(shape=torch.Size([92534, 550]), device=cpu, dtype=torch.int64, is_shared=False),
                input_ids=MemmapTensor(shape=torch.Size([92534, 550]), device=cpu, dtype=torch.int64, is_shared=False),
                rewards=None,
                end_scores=None,
                batch_size=torch.Size([92534]),
                device=None,
                is_shared=False),
            rejected_data=RewardData(
                attention_mask=MemmapTensor(shape=torch.Size([92534, 550]), device=cpu, dtype=torch.int64, is_shared=False),
                input_ids=MemmapTensor(shape=torch.Size([92534, 550]), device=cpu, dtype=torch.int64, is_shared=False),
                rewards=None,
                end_scores=None,
                batch_size=torch.Size([92534]),
                device=None,
                is_shared=False),
            batch_size=torch.Size([92534]),
            device=None,
            is_shared=False)

    """
    chosen_data: RewardData
    rejected_data: RewardData

    @classmethod
    def from_dataset(cls, split, max_length=550):
        data = create_or_load_dataset(
            split, max_length, DATASET, make_process_fn_comparison, pre_tokenization_hook
        )
        data = data[split, str(max_length)]
        maxidx = data.shape[0] // 2
        batch_size = [maxidx]
        # this is a zero-copy creation, as we index memmap-arrays without
        # creating new storage.
        chosen_data = data[:maxidx]
        rejected_data = data[maxidx:]
        return cls(
            chosen_data=RewardData(
                **chosen_data,
                batch_size=batch_size,
            ),
            rejected_data=RewardData(
                **rejected_data,
                batch_size=batch_size,
            ),
            batch_size=batch_size,
        )


def make_process_fn_comparison(tokenizer, max_length):
    def process(example):
        return tokenizer(
            example["text"],
            max_length=max_length,
            padding="max_length",
            truncation=True,
        )

    return process


def pre_tokenization_hook(dataset):
    chosen = []
    rejected = []
    for sample in tqdm(dataset):
        prompt = sample["prompt"]
        chosen_summary = sample["chosen"]
        rejected_summary = sample["rejected"]
        if chosen_summary == rejected_summary:
            continue
        if len(chosen_summary.split()) < 5 or len(rejected_summary.split()) < 5:
            continue
        chosen.append({"text": prompt + "\n" + chosen_summary})
        rejected.append({"text": prompt + "\n" + rejected_summary})

    return HFDataset.from_list(chosen + rejected)

def get_reward_dataloader(config, device, split="train"):
    data = PairwiseDataset.from_dataset(split, max_length=config["block_size"])
    return TensorDictReplayBuffer(
        storage=TensorStorage(data),
        collate_fn=lambda x: x.as_tensor().to(device, non_blocking=True),
        sampler=SamplerWithoutReplacement(),
        batch_size=config['batch_size'],
        prefetch=config.get('prefetch', 0),
    )
