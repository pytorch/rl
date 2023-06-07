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
    prompt_rindex: torch.Tensor
    labels: Optional[torch.Tensor] = None
    logits: Optional[torch.Tensor] = None

    def mask_label(self, pad_token_id=50256):
        batch_size, block_size = self.input_ids.shape
        attention_mask = (
            torch.arange(block_size, device=self.prompt_rindex.device)
            < self.prompt_rindex[:, None]
        ).to(torch.int64)
        input_ids = torch.where(attention_mask == 1, self.input_ids, pad_token_id)
        return self.__class__(
            input_ids=input_ids,
            attention_mask=attention_mask,
            prompt_rindex=self.prompt_rindex,
            batch_size=[batch_size],
        )


@tensorclass
class PromptData:
    transformer_data: TransformerData
    loss: Optional[torch.Tensor] = None

    def mask_label(self, pad_token_id=50256):
        return self.__class__(
            transformer_data=self.transformer_data.mask_label(pad_token_id),
            loss=self.loss,
            batch_size=[],
        )


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
        tokenized_prompts = tokenizer(
            example["prompt"], max_length=max_length, truncation=True
        )
        prompt_rindex = [len(prompt) - 1 for prompt in tokenized_prompts["input_ids"]]
        tokenized_example = tokenizer(
            [
                prompt + label
                for prompt, label in zip(example["prompt"], example["label"])
            ],
            max_length=max_length,
            padding="max_length",
            truncation=True,
        )
        tokenized_example["prompt_rindex"] = prompt_rindex
        return tokenized_example

    return process


def create_tldr_memmaps(split, max_length):
    dataset = load_dataset(DATASET)
    dataset = dataset[split]

    if split == "valid":
        dataset = dataset.select(range(2_000))

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    # tokenize the dataset
    tokenized = dataset.map(
        make_process_fn(tokenizer, max_length=max_length),
        remove_columns=["prompt", "label"],
        desc="tokenizing the splits",
        num_proc=NUM_PROC,
        batched=True,
    )

    tokenized.set_format(
        "torch", columns=["input_ids", "attention_mask", "prompt_rindex"]
    )
    n_examples = len(tokenized)

    data_dir = HERE / "tldr"
    if not data_dir.exists():
        data_dir.mkdir()
    ids_filename = data_dir / f"ids-{split}-{max_length}.bin"
    mask_filename = data_dir / f"mask-{split}-{max_length}.bin"
    rindex_filename = data_dir / f"rindex-{split}-{max_length}.bin"

    dtype = np.int32  # (can do since enc.max_token_value == 50256 is < 2**16)
    ids_arr = np.memmap(
        ids_filename, dtype=dtype, mode="w+", shape=(n_examples, max_length)
    )
    mask_arr = np.memmap(
        mask_filename, dtype=dtype, mode="w+", shape=(n_examples, max_length)
    )
    rindex_arr = np.memmap(rindex_filename, dtype=dtype, mode="w+", shape=(n_examples,))

    print(f"writing {ids_filename} and {mask_filename}...")
    for idx in tqdm(range(n_examples)):
        ids_arr[idx] = tokenized[idx]["input_ids"]
        mask_arr[idx] = tokenized[idx]["attention_mask"]
        rindex_arr[idx] = tokenized[idx]["prompt_rindex"]
    ids_arr.flush()
    mask_arr.flush()
    rindex_arr.flush()


class TLDRDataset(Dataset):
    def __init__(self, split, max_length=550):
        data_dir = HERE / "tldr"
        ids_filename = data_dir / f"ids-{split}-{max_length}.bin"
        mask_filename = data_dir / f"mask-{split}-{max_length}.bin"
        rindex_filename = data_dir / f"rindex-{split}-{max_length}.bin"

        if not all(
            (data_dir / file).exists()
            for file in (ids_filename, mask_filename, rindex_filename)
        ):
            create_tldr_memmaps(split, max_length)

        self.input_ids = np.memmap(ids_filename, dtype=np.int32, mode="r+")
        self.mask = np.memmap(mask_filename, dtype=np.int32, mode="r+")
        self.rindex = np.memmap(rindex_filename, dtype=np.int32, mode="r+")

        self.input_ids = self.input_ids.reshape(
            (self.input_ids.shape[0] // max_length, max_length)
        )
        self.mask = self.mask.reshape((self.mask.shape[0] // max_length, max_length))

    def __len__(self):
        return len(self.input_ids)

    def __getitems__(self, idx):
        input_ids = torch.from_numpy(self.input_ids[idx]).to(torch.int64)
        mask = torch.from_numpy(self.mask[idx])
        rindex = torch.from_numpy(self.rindex[idx])
        batch_size = input_ids.shape[0]
        transformer_data = TransformerData(
            input_ids=input_ids,
            attention_mask=mask,
            prompt_rindex=rindex,
            labels=input_ids,  # NOTE: we will use Hugging Face model and label are shiftwd within model
            batch_size=[batch_size],
        )
        return PromptData(transformer_data=transformer_data, batch_size=[])


def create_datasets(config):
    train_data = TLDRDataset("train", max_length=config["block_size"])
    val_data = TLDRDataset("valid", max_length=config["block_size"])

    return train_data, val_data


def strip_labels(it):
    for batch in it:
        batch.transformer_data.labels = None
        yield batch


def get_prompt_dataloaders(config, inference=False):
    train_data, val_data = create_datasets(config)

    train_loader = create_infinite_dataloader(
        train_data, config, Collate(config["device"])
    )
    val_loader = create_infinite_dataloader(val_data, config, Collate(config["device"]))

    if inference:
        return strip_labels(train_loader), strip_labels(val_loader)
    return train_loader, val_loader
