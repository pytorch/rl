from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from datasets import load_dataset, Dataset as HFDataset
from tensordict import tensorclass
from tqdm import tqdm
from transformers import AutoTokenizer

from .utils import create_infinite_dataloader

NUM_PROC = 8
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


def create_comparisons_dataset(split="train"):
    dataset = load_dataset(DATASET, split=split)
    if split.startswith("valid"):
        # reduce size of validation dataset
        dataset = dataset.select(range(2_000))

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


def create_comparisons_memmaps(split, max_length):
    dataset = create_comparisons_dataset(split)

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    # tokenize the dataset
    tokenized = dataset.map(
        make_process_fn(tokenizer, max_length=max_length),
        remove_columns=["text"],
        desc=f"Tokenizing {split} data",
        num_proc=NUM_PROC,
        batched=True,
    )
    tokenized.set_format("torch", columns=["input_ids", "attention_mask"])

    # concatenate all the ids in each dataset into one large file we can use for training
    n_examples = len(tokenized)

    data_dir = HERE / "comparisons"
    if not data_dir.exists():
        data_dir.mkdir()
    ids_filename = data_dir / f"ids-{split}-{max_length}.bin"
    mask_filename = data_dir / f"mask-{split}-{max_length}.bin"

    dtype = np.int32  # (can do since enc.max_token_value == 50256 is < 2**16)
    ids_arr = np.memmap(
        ids_filename, dtype=dtype, mode="w+", shape=(n_examples, max_length)
    )
    mask_arr = np.memmap(
        mask_filename, dtype=dtype, mode="w+", shape=(n_examples, max_length)
    )

    print(f"writing {ids_filename} and {mask_filename}...")
    for idx, example in tqdm(enumerate(tokenized), total=len(tokenized)):
        ids_arr[idx] = example["input_ids"]
        mask_arr[idx] = example["attention_mask"]
    ids_arr.flush()
    mask_arr.flush()


class PairwiseDataset(Dataset):
    def __init__(self, split, max_length=550):
        data_dir = HERE / "comparisons"
        ids_filename = data_dir / f"ids-{split}-{max_length}.bin"
        mask_filename = data_dir / f"mask-{split}-{max_length}.bin"
        if not all(
            (data_dir / file).exists() for file in (ids_filename, mask_filename)
        ):
            create_comparisons_memmaps(split, max_length)

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


def create_datasets(config):
    train_data = PairwiseDataset("train", max_length=config["block_size"])
    val_data = PairwiseDataset("valid1", max_length=config["block_size"])

    return train_data, val_data


def get_reward_dataloaders(config):
    train_data, val_data = create_datasets(config)

    train_loader = create_infinite_dataloader(
        train_data, config, Collate(config["device"])
    )
    val_loader = create_infinite_dataloader(val_data, config, Collate(config["device"]))

    return train_loader, val_loader
