# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from tensordict import tensorclass
from torch.utils.data import Dataset

from torchrl.data import TensorDictReplayBuffer, TensorStorage
from torchrl.data.replay_buffers import SamplerWithoutReplacement
from torchrl.data.rlhf.dataset import create_or_load_dataset, \
    create_infinite_dataloader

HERE = Path(__file__).parent
DATASET = "CarperAI/openai_summarize_tldr"


@tensorclass
class PromptDataTLDR:
    """A prompt dataset."""
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

    @classmethod
    def from_dataset(cls, split, max_length=550):
        """

        Args:
            split:
            max_length:

        Returns:

        Examples:
            >>> data = PromptDataTLDR.from_dataset("train")
            >>> print(data)
            PromptDataTLDR(
                attention_mask=MemmapTensor(shape=torch.Size([116722, 550]), device=cpu, dtype=torch.int64, is_shared=False),
                input_ids=MemmapTensor(shape=torch.Size([116722, 550]), device=cpu, dtype=torch.int64, is_shared=False),
                prompt_rindex=MemmapTensor(shape=torch.Size([116722]), device=cpu, dtype=torch.int64, is_shared=False),
                labels=MemmapTensor(shape=torch.Size([116722, 550]), device=cpu, dtype=torch.int64, is_shared=False),
                logits=None,
                loss=None,
                batch_size=torch.Size([116722]),
                device=None,
                is_shared=False)
        """
        data = create_or_load_dataset(
            split, max_length, DATASET, make_process_fn_tldr,
        )
        data = data[split, str(max_length)]
        return cls(**data, labels=data["input_ids"], batch_size=data.shape)

class CollateTLDR(nn.Module):
    def __init__(self, device="cpu"):
        super().__init__()
        self.device = torch.device(device)

    def __call__(self, batch):
        if self.device.type == "cuda":
            batch = batch.pin_memory()
        return batch.to(self.device)


def make_process_fn_tldr(tokenizer, max_length):
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
        # NOTE: to mark as discarded we just save the mask as we cannot change the shape here
        tokenized_example["valid_sample"] = [True] * len(tokenized_example["input_ids"])
        for i, input_ids in enumerate(tokenized_example["input_ids"]):
            if input_ids[-1] != tokenizer.eos_token_id:
                tokenized_example["valid_sample"][i] = False

        return tokenized_example

    return process

#
# @tensorclass
# class TLDRDataset:
#     def __init__(self, split, max_length=550):
#         data_dir = HERE / "tldr"
#         ids_filename = data_dir / f"input_ids-{split}-{max_length}.bin"
#         mask_filename = data_dir / f"attention_mask-{split}-{max_length}.bin"
#         rindex_filename = data_dir / f"prompt_rindex-{split}-{max_length}.bin"
#
#         if not all(
#             (data_dir / file).exists()
#             for file in (ids_filename, mask_filename, rindex_filename)
#         ):
#             create_or_load_dataset(split, max_length, DATASET, make_process_fn_tldr)
#
#         self.input_ids = np.memmap(ids_filename, dtype=np.int32, mode="r+")
#         self.mask = np.memmap(mask_filename, dtype=np.int32, mode="r+")
#         self.rindex = np.memmap(rindex_filename, dtype=np.int32, mode="r+")
#
#         self.input_ids = self.input_ids.reshape(
#             (self.input_ids.shape[0] // max_length, max_length)
#         )
#         self.mask = self.mask.reshape((self.mask.shape[0] // max_length, max_length))
#
#     def __len__(self):
#         return len(self.input_ids)
#
#     def __getitems__(self, idx):
#         input_ids = torch.from_numpy(self.input_ids[idx]).to(torch.int64)
#         mask = torch.from_numpy(self.mask[idx])
#         rindex = torch.from_numpy(self.rindex[idx])
#         return PromptDataTLDR(
#             input_ids=input_ids,
#             attention_mask=mask,
#             prompt_rindex=rindex,
#             labels=input_ids,  # NOTE: we will use Hugging Face model and label are shifted within model
#             batch_size=[],
#         )


def get_prompt_dataloader(config, device, split="train"):
    """Creates a dataset for prompt generation and returns a dataloader from it.

    Args:
        config (dict or equivalent): a configuration dict. Should contain the
            entries ``"block_size"`` indicating the maximum length of a sequence,
            ``"batch_size"`` indicating the batch size of the dataloader samples) and
            optionally ``"prefetch"`` which sets the queue length for
            multithreaded sampling. If none is provided, no prefetching
            is assumed.
        device (torch.device or equivalent): the device where the samples should
            be cast.
        split (str, optional): the data split. Either ``"train"`` or ``"valid"``.
            Defaults to ``"train"``.

    """
    data = PromptDataTLDR.from_dataset(split, max_length=config["block_size"])
    return TensorDictReplayBuffer(
        storage=TensorStorage(data),
        collate_fn=lambda x: x.as_tensor().to(device, non_blocking=True),
        sampler=SamplerWithoutReplacement(drop_last=True),
        batch_size=config['batch_size'],
        prefetch=config.get('prefetch', 0),
    )
