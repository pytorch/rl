from typing import Optional

import tiktoken
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from tensordict import tensorclass
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

    @classmethod
    def from_dataset(cls, dataset, max_length):
        # TODO: check dtypes
        data = cls(
            chosen=torch.zeros(len(dataset), max_length, dtype=torch.int32),
            rejected=torch.zeros(len(dataset), max_length, dtype=torch.int32),
            batch_size=[len(dataset)],
        )
        enc = tiktoken.get_encoding("gpt2")
        i = 0

        for sample in tqdm(dataset, total=len(dataset)):
            prompt = sample["prompt"]
            chosen = sample["chosen"]
            rejected = sample["rejected"]

            if len(chosen.split()) < 5 or len(rejected.split()) < 5:
                continue

            chosen = "\n".join([prompt, chosen])
            rejected = "\n".join([prompt, rejected])

            chosen = enc.encode(
                "<|startoftext|>" + chosen + "<|endoftext|>", allowed_special="all"
            )[-max_length:]
            rejected = enc.encode(
                "<|startoftext|>" + rejected + "<|endoftext|>", allowed_special="all"
            )[-max_length:]

            if chosen == rejected:
                continue

            data[i] = cls(
                chosen=F.pad(
                    torch.Tensor(chosen), (max_length - len(chosen), 0), value=0
                ),
                rejected=F.pad(
                    torch.Tensor(rejected), (max_length - len(rejected), 0), value=0
                ),
                batch_size=[],
            )
            i += 1

        # index because we will have skipped some datapoints
        return data[:i]


def create_datasets(config):
    # Make pairwise datasets for training
    print("Creating pairwise datasets")
    data_path = "CarperAI/openai_summarize_comparisons"
    train_data = PairwiseDataset.from_dataset(
        load_dataset(data_path, split="train"), max_length=config["block_size"]
    )
    val_data = PairwiseDataset.from_dataset(
        load_dataset(data_path, split="test"), max_length=config["block_size"]
    )

    return train_data, val_data


def get_dataloaders(config):
    train_data, val_data = create_datasets(config)
    train_data.memmap_()
    val_data.memmap_()

    train_loader = create_infinite_dataloader(
        train_data, config, Collate(config["device"])
    )
    val_loader = create_infinite_dataloader(val_data, config, Collate(config["device"]))

    return train_loader, val_loader
