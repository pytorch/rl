# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path
from typing import Optional

import torch
from tensordict import tensorclass

from torchrl.data.rlhf.dataset import create_or_load_dataset

DEFAULT_DATASET = "CarperAI/openai_summarize_tldr"


@tensorclass
class PromptData:
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
    def from_dataset(
        cls, split, dataset_name=None, max_length=550, root_dir=None, from_disk=False
    ):
        """

        Args:
            split:
            max_length:

        Returns:

        Examples:
            >>> data = PromptData.from_dataset("train")
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
        dataset_name = dataset_name if dataset_name is not None else DEFAULT_DATASET
        data = create_or_load_dataset(
            split,
            max_length,
            dataset_name,
            make_process_fn_tldr,
            root_dir=root_dir,
            from_disk=from_disk,
        )
        return cls(**data, labels=data["input_ids"], batch_size=data.shape)


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
