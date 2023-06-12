from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from tensordict import tensorclass
from torch.utils.data import Dataset

from .utils import create_infinite_dataloader, create_memmaps

HERE = Path(__file__).parent
DATASET = "CarperAI/openai_summarize_tldr"


@tensorclass
class PromptData:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    prompt_rindex: torch.Tensor
    labels: Optional[torch.Tensor] = None
    logits: Optional[torch.Tensor] = None
    loss: Optional[torch.Tensor] = None

    def mask_label(self, pad_token_id=50256):
        _, block_size = self.input_ids.shape
        attention_mask = (
            torch.arange(block_size, device=self.prompt_rindex.device)
            < self.prompt_rindex[:, None]
        ).to(torch.int64)
        input_ids = torch.where(attention_mask == 1, self.input_ids, pad_token_id)
        return self.__class__(
            input_ids=input_ids,
            attention_mask=attention_mask,
            prompt_rindex=self.prompt_rindex,
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
        # drop any examples whose total length when tokenized exceeds block size
        # with recommended block size of 550, this is only ~0.1% of available examples.
        indices_to_drop = {
            i
            for i, input_ids in enumerate(tokenized_example["input_ids"])
            if input_ids[-1] != tokenizer.eos_token_id
        }
        for key in tokenized_example:
            tokenized_example[key] = [
                item
                for i, item in enumerate(tokenized_example[key])
                if i not in indices_to_drop
            ]

        return tokenized_example

    return process


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
            create_memmaps(split, max_length, DATASET, make_process_fn)

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
        return PromptData(
            input_ids=input_ids,
            attention_mask=mask,
            prompt_rindex=rindex,
            labels=input_ids,  # NOTE: we will use Hugging Face model and label are shifted within model
            batch_size=[],
        )


def get_prompt_dataloader(config, split="train"):
    data = TLDRDataset(split, max_length=config["block_size"])
    return create_infinite_dataloader(data, config, Collate(config["device"]))
