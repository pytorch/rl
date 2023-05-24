from typing import Optional

import tiktoken
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from tensordict import MemmapTensor, tensorclass
from tqdm import tqdm

from .utils import create_infinite_dataloader


class Collate(nn.Module):
    def __init__(self, device="cpu"):
        super().__init__()
        self.device = torch.device(device)

    def __call__(self, batch):
        if self.device.type == "cuda":
            batch = batch.pin_memory()
        return batch.to(self.device)


@tensorclass
class PairwiseDataset:
    chosen: torch.Tensor
    rejected: torch.Tensor
    reward: Optional[torch.Tensor] = None

    @staticmethod
    def _encode(sample, max_length):
        enc = tiktoken.get_encoding("gpt2")

        prompt = sample["prompt"]
        chosen = sample["chosen"]
        rejected = sample["rejected"]

        chosen = "\n".join([prompt, chosen])
        rejected = "\n".join([prompt, rejected])

        chosen = enc.encode(
            "<|startoftext|>" + chosen + "<|endoftext|>", allowed_special="all"
        )[-max_length:]
        rejected = enc.encode(
            "<|startoftext|>" + rejected + "<|endoftext|>", allowed_special="all"
        )[-max_length:]
        return chosen, rejected

    @classmethod
    def from_dataset(cls, dataset, max_length):
        # we perform two passes over the dataset. during the first we determine which
        # datapoints to skip. during the second we load the unskipped examples into
        # a pre-allocated memory map. while we do end up paying the cost of iteration
        # and encoding twice, it means we are able to load the full dataset into the
        # memory map without ever having to hold the whole thing in memory
        indices_to_skip = set()
        for idx, sample in tqdm(enumerate(dataset), total=len(dataset)):
            if idx >= 1000:
                break
            if len(sample["chosen"].split()) < 5 or len(sample["rejected"].split()) < 5:
                indices_to_skip.add(idx)
                continue

            chosen, rejected = cls._encode(sample, max_length)

            if chosen == rejected:
                indices_to_skip.add(idx)

        data = cls(
            chosen=MemmapTensor(
                # len(dataset) - len(indices_to_skip), max_length, dtype=torch.int32
                1000, max_length, dtype=torch.int32
            ),
            rejected=MemmapTensor(
                # len(dataset) - len(indices_to_skip), max_length, dtype=torch.int32
                1000, max_length, dtype=torch.int32
            ),
            # batch_size=[len(dataset)],
            batch_size=[1000],
        )
        i = 0

        for idx, sample in tqdm(enumerate(dataset), total=len(dataset)):
            if idx >= 1000:
                break
            if idx in indices_to_skip:
                continue

            chosen, rejected = cls._encode(sample, max_length)

            data[i] = cls(
                chosen=F.pad(
                    torch.Tensor(chosen), (0, max_length - len(chosen)), value=50256
                ),
                rejected=F.pad(
                    torch.Tensor(rejected), (0, max_length - len(rejected)), value=50256
                ),
                batch_size=[],
            )
            i += 1
        return data


def create_datasets(config):
    # Make pairwise datasets for training
    print("Creating pairwise datasets")
    assert config["dataset"] == "openai_summarize_comparisons"
    data_path = "CarperAI/openai_summarize_comparisons"
    train_data = PairwiseDataset.from_dataset(
        load_dataset(data_path, split="train"), max_length=config["block_size"]
    )
    val_data = PairwiseDataset.from_dataset(
        load_dataset(data_path, split="test"), max_length=config["block_size"]
    )

    return train_data, val_data


def get_reward_dataloaders(config):
    train_data, val_data = create_datasets(config)
    train_data.memmap_()
    val_data.memmap_()

    train_loader = create_infinite_dataloader(
        train_data, config, Collate(config["device"])
    )
    val_loader = create_infinite_dataloader(val_data, config, Collate(config["device"]))

    return train_loader, val_loader
