# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional

import torch
from datasets import Dataset as HFDataset
from tensordict import tensorclass

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
        """TODO
        Returns a ``PairwiseDataset`` from a dataset name.

        Args:
            split:
            dataset_name:
            max_length:
            root_dir:
            from_disk:

        Returns:

        Examples:
            >>> data = PairwiseDataset.from_dataset("train")
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


def make_process_fn_comparison(
    tokenizer, max_length, key="text", padding="max_length", truncation=True
):
    """Factory for a process function that applies a tokenizer over a text example.

    Args:
        tokenizer (tokenizer from transformers library): the tokenizer to use.
        max_length (int): maximum length of the sequence.
        key (str, optional): the key where to find the text. Defaults to ``"text"``.
        padding (str, optional): type of padding. Defaults to ``"max_length"``.
        truncation (bool, optional): whether the sequences should be truncated to max_length.

    See transformers library for more information about tokenizers:
        Padding and truncation: `<https://huggingface.co/docs/transformers/pad_truncation>`_

    Examples:
        >>> from transformers import AutoTokenizer
        >>> tokenizer = AutoTokenizer.from_pretrained("gpt2")
        >>> tokenizer.pad_token = 100
        >>> process = make_process_fn_comparison(tokenizer, max_length=10)
        >>> example = {"text": "I am a little worried"}
        >>> process(example)
        {'input_ids': [40, 716, 257, 1310, 7960, 3064, 3064, 3064, 3064, 3064], 'attention_mask': [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]}

    """

    def process(example):
        return tokenizer(
            example[key],
            max_length=max_length,
            padding=padding,
            truncation=truncation,
        )

    return process


def pre_tokenization_hook(dataset, min_length=5):
    """Pre-tokenizer for the eward model (comparison) dataset.

    This function selects all samples where the length of the prompt is
    sufficient and where the ``"chosen"`` and ``"rejected"`` entries differ.

    Args:
        dataset (datasets.Dataset): the dataset to process. Should have entries
            ``"prompt"``, ``"chosen"`` and ``"rejected"``.
        min_length (int, optional): minimum length of a prompt (in word count).

    Returns: a new ``datasets.Dataset`` with selected prompts under ``"text"``.
        The first half are the chosen strings and the second the rejected ones,
        always preceded by the original prompt.

    Examples:
        >>> from datasets import Dataset
        >>> data = Dataset.from_dict({
        ...     "prompt": ["I'm the king"],
        ...     "chosen": ["It is true, you are the king"],
        ...     "rejected": ["No, I am the king, you are not"]})
        >>> print(pre_tokenization_hook(data))
        Dataset({
            features: ['text'],
            num_rows: 2
        })
        >>> data = Dataset.from_dict({
        ...     "prompt": ["I'm the king"],
        ...     "chosen": ["It is true, you are the king"],
        ...     "rejected": ["It is true, you are the king"]}) # chosen and rejected match
        >>> print(pre_tokenization_hook(data))
        Dataset({
            features: [],
            num_rows: 0
        })
        >>> data = Dataset.from_dict({
        ...     "prompt": ["I'm the king"],
        ...     "chosen": ["Yes"],
        ...     "rejected": ["No"]}) # chosen and rejected are too short
        >>> print(pre_tokenization_hook(data))
        Dataset({
            features: [],
            num_rows: 0
        })

    """
    chosen = []
    rejected = []
    for sample in tqdm(dataset):
        prompt = sample["prompt"]
        chosen_summary = sample["chosen"]
        rejected_summary = sample["rejected"]
        if chosen_summary == rejected_summary:
            continue
        if (
            len(chosen_summary.split()) < min_length
            or len(rejected_summary.split()) < min_length
        ):
            continue
        chosen.append({"text": prompt + "\n" + chosen_summary})
        rejected.append({"text": prompt + "\n" + rejected_summary})

    return HFDataset.from_list(chosen + rejected)
