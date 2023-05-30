# download and prepare the openai_summarize_tldr dataset for fine tuning transformers
# adapted from
# https://github.com/sanjeevanahilan/nanoChatGPT/blob/3cde2746c7ea8b0bd32edd44c76ead581bbda5d5/data/openai_summarize_tldr/prepare.py
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset  # huggingface datasets
from tensordict import tensorclass
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import AutoTokenizer

from .utils import create_infinite_dataloader

# number of workers in .map() call
# good number to use is ~order number of cpu cores // 2
NUM_PROC = 16
DATASET = "CarperAI/openai_summarize_tldr"
HERE = Path(__file__).parent


@tensorclass
class TransformerData:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: Optional[torch.Tensor] = None
    logits: Optional[torch.Tensor] = None


@tensorclass
class PromptData:
    transformer_data: TransformerData
    loss: Optional[torch.Tensor] = None


class Collate(nn.Module):
    def __init__(self, device="cpu"):
        super().__init__()
        self.device = torch.device(device)

    def __call__(self, batch):
        if self.device.type == "cuda":
            batch = batch.pin_memory()
        return batch.to(self.device)


def make_process_fn(tokenizer, max_length):
    def process(example):
        return (tokenizer(
            [
                prompt + label
                for prompt, label in zip(example["prompt"], example["label"])
            ],
            max_length=max_length,
            padding="max_length",
            truncation=True,
        ), 
        tokenizer(example["prompt"],
            max_length=max_length,
            padding="max_length",
            truncation=True,
        ),
    )        
    return process


def create_tldr_memmaps(split, max_length):
    dataset = load_dataset(DATASET)
    dataset = dataset[split]

    if split == "valid":
        dataset = dataset.select(range(2_000))

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    # tokenize the dataset
    tokenized, tokenized_prompt_only = dataset.map(
        make_process_fn(tokenizer, max_length=max_length),
        remove_columns=["prompt", "label"],
        desc="tokenizing the splits",
        num_proc=NUM_PROC,
        batched=True,
    )
    tokenized.set_format("torch", columns=["input_ids", "attention_mask"])
    tokenized_prompt_only.set_format("torch", columns=["input_ids", "attention_mask"])

    n_examples = len(tokenized)
    assert n_examples == len(tokenized_prompt_only)

    data_dir = HERE / "tldr"
    if not data_dir.exists():
        data_dir.mkdir()
    ids_filename = data_dir / f"ids-{split}-{max_length}.bin"
    mask_filename = data_dir / f"mask-{split}-{max_length}.bin"
    ids_prompt_only_filename = data_dir / f"ids_prompt_only-{split}-{max_length}.bin"
    mask_prompt_only_filename = data_dir / f"mask_prompt_only-{split}-{max_length}.bin"

    dtype = np.int32  # (can do since enc.max_token_value == 50256 is < 2**16)
    ids_arr = np.memmap(
        ids_filename, dtype=dtype, mode="w+", shape=(n_examples, max_length)
    )
    mask_arr = np.memmap(
        mask_filename, dtype=dtype, mode="w+", shape=(n_examples, max_length)
    )
    ids_prompt_only_arr = np.memmap(
        ids_prompt_only_filename, dtype=dtype, mode="w+", shape=(n_examples, max_length)
    )
    mask_prompt_only_arr = np.memmap(
        mask_prompt_only_filename, dtype=dtype, mode="w+", shape=(n_examples, max_length)
    )

    print(f"writing {ids_filename} and {mask_filename}...")
    for idx in tqdm(range(n_examples)):
        ids_arr[idx] = tokenized[idx]["input_ids"]
        mask_arr[idx] = tokenized[idx]["attention_mask"]
        ids_prompt_only_arr[idx] = tokenized_prompt_only[idx]["input_ids"]
        mask_prompt_only_arr[idx] = tokenized_prompt_only[idx]["attention_mask"]
    ids_arr.flush()
    mask_arr.flush()
    ids_prompt_only_arr.flush()
    mask_prompt_only_arr.flush()


class TLDRDataset(Dataset):
    def __init__(self, split, max_length=550):
        data_dir = HERE / "tldr"
        ids_filename = data_dir / f"ids-{split}-{max_length}.bin"
        mask_filename = data_dir / f"mask-{split}-{max_length}.bin"
        ids_prompt_only_filename = data_dir / f"ids_prompt_only-{split}-{max_length}.bin"
        mask_prompt_only_filename = data_dir / f"mask_prompt_only-{split}-{max_length}.bin"
        if not all(
            (data_dir / file).exists() for file in (ids_filename, mask_filename, ids_prompt_only_filename, mask_prompt_only_filename)
        ):
            create_tldr_memmaps(split, max_length)

        self.input_ids = np.memmap(ids_filename, dtype=np.int32, mode="r+")
        self.mask = np.memmap(mask_filename, dtype=np.int32, mode="r+")
        self.input_ids_prompt_only = np.memmap(mask_prompt_only_filename, dtype=np.int32, mode="r+")
        self.mask_prompt_only = np.memmap(mask_prompt_only_filename, dtype=np.int32, mode="r+")

        # FIXME: messing up here

        self.input_ids = self.input_ids.reshape(
            (self.input_ids.shape[0] // max_length, max_length)
        )
        self.mask = self.mask.reshape((self.mask.shape[0] // max_length, max_length))

    def __len__(self):
        return len(self.input_ids)

    def __getitems__(self, idx):
        input_ids = torch.from_numpy(self.input_ids[idx]).to(torch.int64)
        mask = torch.from_numpy(self.mask[idx])
        batch_size = input_ids.shape[0]
        transformer_data = TransformerData(
            input_ids=input_ids,  
            attention_mask=mask,
            labels=input_ids,  # NOTE: we will use Hugging Face model and label are shiftwd within model
            batch_size=[batch_size],
        )
        return PromptData(transformer_data=transformer_data, batch_size=[])


def create_datasets(config):
    train_data = TLDRDataset("train", max_length=config["block_size"])
    val_data = TLDRDataset("valid", max_length=config["block_size"])

    return train_data, val_data


def get_prompt_dataloaders(config):
    train_data, val_data = create_datasets(config)

    train_loader = create_infinite_dataloader(
        train_data, config, Collate(config["device"])
    )
    val_loader = create_infinite_dataloader(val_data, config, Collate(config["device"]))

    return train_loader, val_loader
