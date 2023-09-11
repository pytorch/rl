# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from datasets import Dataset as HFDataset
from tensordict import tensorclass
from torch.utils.data import Dataset
from tqdm import tqdm

from .utils import create_infinite_dataloader, create_memmaps

HERE = Path(__file__).parent
DATASET = "CarperAI/openai_summarize_comparisons"


@tensorclass
class RewardData:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    rewards: Optional[torch.Tensor] = None
    end_scores: Optional[torch.Tensor] = None


class Collate(nn.Module):
    def __init__(self, device="cpu"):
        super().__init__()
        self.device = torch.device(device)

    def __call__(self, batches):
        chosen_batch, rejected_batch = batches
        if self.device.type == "cuda":
            chosen_batch = chosen_batch.pin_memory()
            rejected_batch = rejected_batch.pin_memory()
        return chosen_batch.to(self.device), rejected_batch.to(self.device)


def make_process_fn(tokenizer, max_length):
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


class PairwiseDataset(Dataset):
    def __init__(self, split, max_length=550):
        data_dir = HERE / "comparisons"
        ids_filename = data_dir / f"input_ids-{split}-{max_length}.bin"
        mask_filename = data_dir / f"attention_mask-{split}-{max_length}.bin"

        if not all(
            (data_dir / file).exists() for file in (ids_filename, mask_filename)
        ):
            create_memmaps(
                split, max_length, DATASET, make_process_fn, pre_tokenization_hook
            )

        self.input_ids = np.memmap(ids_filename, dtype=np.int32, mode="r+")
        self.mask = np.memmap(mask_filename, dtype=np.int32, mode="r+")

        self.input_ids = self.input_ids.reshape(
            (self.input_ids.shape[0] // max_length, max_length)
        )
        self.mask = self.mask.reshape((self.mask.shape[0] // max_length, max_length))

    def __len__(self):
        return len(self.input_ids) // 2

    def __getitems__(self, idx):
        ridx = [i + self.__len__() for i in idx]
        chosen_ids = torch.from_numpy(self.input_ids[idx])
        rejected_ids = torch.from_numpy(self.input_ids[ridx])
        chosen_mask = torch.from_numpy(self.mask[idx])
        rejected_mask = torch.from_numpy(self.mask[ridx])
        batch_size = chosen_ids.shape[0]
        chosen_data = RewardData(
            input_ids=chosen_ids,
            attention_mask=chosen_mask,
            batch_size=[batch_size],
        )
        rejected_data = RewardData(
            input_ids=rejected_ids,
            attention_mask=rejected_mask,
            batch_size=[batch_size],
        )
        return chosen_data, rejected_data


def get_reward_dataloader(config, device, split="train"):
    data = PairwiseDataset(split, max_length=config["block_size"])
    return create_infinite_dataloader(data, config, Collate(device))
