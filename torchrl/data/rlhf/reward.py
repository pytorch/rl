# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import importlib
from typing import Optional

import torch

from tensordict import tensorclass

from torchrl.data.rlhf.dataset import TensorDictTokenizer, TokenizedDatasetLoader

DEFAULT_DATASET = "CarperAI/openai_summarize_comparisons"
_has_datasets = importlib.util.find_spec("datasets") is not None
_has_tqdm = importlib.util.find_spec("tqdm") is not None


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
                attention_mask=MemoryMappedTensor(shape=torch.Size([92534, 550]), device=cpu, dtype=torch.int64, is_shared=False),
                input_ids=MemoryMappedTensor(shape=torch.Size([92534, 550]), device=cpu, dtype=torch.int64, is_shared=False),
                rewards=None,
                end_scores=None,
                batch_size=torch.Size([92534]),
                device=None,
                is_shared=False),
            rejected_data=RewardData(
                attention_mask=MemoryMappedTensor(shape=torch.Size([92534, 550]), device=cpu, dtype=torch.int64, is_shared=False),
                input_ids=MemoryMappedTensor(shape=torch.Size([92534, 550]), device=cpu, dtype=torch.int64, is_shared=False),
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
        cls,
        split,
        dataset_name: str | None = None,
        max_length: int = 550,
        root_dir: str | None = None,
        from_disk: bool = False,
        num_workers: int | None = None,
    ):
        """Returns a :class:`PairwiseDataset` from a dataset name.

        Args:
            split (str): ``"train"`` or ``"valid"`` depending on the data split needed.
            dataset_name (str, optional): name of the dataset to be processed. Defaults to
                ``"CarperAI/openai_summarize_comparisons"``.
            max_length (int, optional): maximum length of the dataset sequenes.
                Defaults to 550.
            root_dir (path, optional): the path where the datasets are stored.
                Defaults to ``"$HOME/.cache/torchrl/data"``
            from_disk (bool, optional): if ``True``, :func:`datasets.load_from_disk`
                will be used. Otherwise, :func:`datasets.load_dataset` will be used.
                Defaults to ``False``.

        Returns: a :class:`PairwiseDataset` instance containing a memory-mapped
            version of the required dataset.

        Examples:
            >>> data = PairwiseDataset.from_dataset("train")
            >>> print(data)
            PairwiseDataset(
                chosen_data=RewardData(
                    attention_mask=MemoryMappedTensor(shape=torch.Size([92534, 550]), device=cpu, dtype=torch.int64, is_shared=False),
                    input_ids=MemoryMappedTensor(shape=torch.Size([92534, 550]), device=cpu, dtype=torch.int64, is_shared=False),
                    rewards=None,
                    end_scores=None,
                    batch_size=torch.Size([92534]),
                    device=None,
                    is_shared=False),
                rejected_data=RewardData(
                    attention_mask=MemoryMappedTensor(shape=torch.Size([92534, 550]), device=cpu, dtype=torch.int64, is_shared=False),
                    input_ids=MemoryMappedTensor(shape=torch.Size([92534, 550]), device=cpu, dtype=torch.int64, is_shared=False),
                    rewards=None,
                    end_scores=None,
                    batch_size=torch.Size([92534]),
                    device=None,
                    is_shared=False),
                batch_size=torch.Size([92534]),
                device=None,
                is_shared=False)
            >>> # data can be sampled from using regular indexing
            >>> sub_data = data[:3]

        """
        if dataset_name is None:
            dataset_name = DEFAULT_DATASET
        loader = TokenizedDatasetLoader(
            split,
            max_length,
            dataset_name,
            TensorDictTokenizer,
            pre_tokenization_hook,
            root_dir=root_dir,
            from_disk=from_disk,
            num_workers=num_workers,
        )
        data = loader.load()
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


def pre_tokenization_hook(dataset, min_length=5):
    """Pre-tokenizer for the reward model (comparison) dataset.

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
    if not _has_datasets:
        raise ImportError(
            "datasets module couldn't be found. Make sure it is installed in your environment."
        )
    from datasets import Dataset as HFDataset

    chosen = []
    rejected = []
    if _has_tqdm:
        from tqdm import tqdm

        pbar = tqdm(dataset)
    else:
        pbar = dataset
    for sample in pbar:
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
