# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path
from typing import Optional

import torch
from datasets import Dataset as HFDataset
from tensordict import tensorclass, TensorDict

from torchrl.data.rlhf.dataset import create_or_load_dataset
from tqdm import tqdm

DEFAULT_DATASET = "CarperAI/openai_summarize_comparisons"


@tensorclass
class RewardData:
    """A dataclass for reward model training."""

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
    def from_dataset(
        cls, split, dataset_name=None, max_length=550, root_dir=None, from_disk=False
    ):
        if dataset_name is None:
            dataset_name = DEFAULT_DATASET
        data = create_or_load_dataset(
            split,
            max_length,
            dataset_name,
            make_process_fn_comparison,
            pre_tokenization_hook,
            root_dir=root_dir,
            from_disk=from_disk,
        )
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
