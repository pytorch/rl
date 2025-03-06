# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import torch
from tensordict import tensorclass, TensorDict

from torchrl.data.llm.dataset import TensorDictTokenizer, TokenizedDatasetLoader

DEFAULT_DATASET = "CarperAI/openai_summarize_tldr"


@tensorclass
class PromptData:
    """A prompt dataset."""

    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    prompt_rindex: torch.Tensor
    labels: torch.Tensor | None = None
    logits: torch.Tensor | None = None
    loss: torch.Tensor | None = None

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
        cls,
        split,
        dataset_name=None,
        max_length=550,
        root_dir=None,
        from_disk=False,
        num_workers: int | None = None,
    ):
        """Returns a :class:`PromptData` from a dataset name.

        Args:
            split (str): ``"train"`` or ``"valid"`` depending on the data split needed.
            dataset_name (str, optional): name of the dataset to be processed. Defaults to
                ``"CarperAI/openai_summarize_comparisons"``.
            max_length (int, optional): maximum length of the dataset sequences.
                Defaults to 550.
            root_dir (path, optional): the path where the datasets are stored.
                Defaults to ``"$HOME/.cache/torchrl/data"``
            from_disk (bool, optional): if ``True``, :func:`datasets.load_from_disk`
                will be used. Otherwise, :func:`datasets.load_dataset` will be used.
                Defaults to ``False``.
            num_workers (int, optional): number of workers for :meth:`datasets.dataset.map`
                which is called during tokenization.
                Defaults to ``max(os.cpu_count() // 2, 1)``.

        Returns: a :class:`PromptData` instance containing a memory-mapped
            version of the required dataset.

        Examples:
            >>> data = PromptData.from_dataset("train")
            >>> print(data)
            PromptDataTLDR(
                attention_mask=MemoryMappedTensor(shape=torch.Size([116722, 550]), device=cpu, dtype=torch.int64, is_shared=False),
                input_ids=MemoryMappedTensor(shape=torch.Size([116722, 550]), device=cpu, dtype=torch.int64, is_shared=False),
                prompt_rindex=MemoryMappedTensor(shape=torch.Size([116722]), device=cpu, dtype=torch.int64, is_shared=False),
                labels=MemoryMappedTensor(shape=torch.Size([116722, 550]), device=cpu, dtype=torch.int64, is_shared=False),
                logits=None,
                loss=None,
                batch_size=torch.Size([116722]),
                device=None,
                is_shared=False)
            >>> # data can be sampled from using regular indexing
            >>> sub_data = data[:3]

        """
        dataset_name = dataset_name if dataset_name is not None else DEFAULT_DATASET
        loader = TokenizedDatasetLoader(
            split,
            max_length,
            dataset_name,
            PromptTensorDictTokenizer,
            root_dir=root_dir,
            from_disk=from_disk,
            num_workers=num_workers,
        )
        data = loader.load()
        return cls(**data, labels=data["input_ids"], batch_size=data.shape)


class PromptTensorDictTokenizer(TensorDictTokenizer):
    """Tokenization recipe for prompt datasets.

    Returns a tokenizer function, which reads an example containing a prompt
    and a label and tokenizes them.

    Args:
        tokenizer (tokenizer from transformers library): the tokenizer to use.
        max_length (int): maximum length of the sequence.
        key (str, optional): the key where to find the text. Defaults to ``"prompt"``.
        padding (str, optional): type of padding. Defaults to ``"max_length"``.
        truncation (bool, optional): whether the sequences should be truncated to max_length.
        return_tensordict (bool, optional): if ``True``, a TensoDict is returned.
            Otherwise, a the original data will be returned.
        device (torch.device, optional): the device where to store the data.
            This option is ignored if ``return_tensordict=False``.

    The :meth:`__call__` method of this class will execute the following operations:

        - Read the ``prompt`` string contacted with the ``label`` string and tokenize
          them. The results will be stored in the ``"input_ids"`` TensorDict entry.
        - Write a ``"prompt_rindex"`` entry with the index of the last valid
          token from the prompt.
        - Write a ``"valid_sample"`` which identifies which entry in the
          tensordict has eough toknens to meet the ``max_length`` criterion.
        - Return a :class:`tensordict.TensorDict` instance with tokenized inputs.

    The tensordict batch-size will match the batch-size of the input.

    Examples:
        >>> from transformers import AutoTokenizer
        >>> tokenizer = AutoTokenizer.from_pretrained("gpt2")
        >>> tokenizer.pad_token = tokenizer.eos_token
        >>> example = {
        ...     "prompt": ["This prompt is long enough to be tokenized.", "this one too!"],
        ...     "label": ["Indeed it is.", 'It might as well be.'],
        ... }
        >>> fn = PromptTensorDictTokenizer(tokenizer, 50)
        >>> print(fn(example))
        TensorDict(
            fields={
                attention_mask: Tensor(shape=torch.Size([2, 50]), device=cpu, dtype=torch.int64, is_shared=False),
                input_ids: Tensor(shape=torch.Size([2, 50]), device=cpu, dtype=torch.int64, is_shared=False),
                prompt_rindex: Tensor(shape=torch.Size([2]), device=cpu, dtype=torch.int64, is_shared=False),
                valid_sample: Tensor(shape=torch.Size([2]), device=cpu, dtype=torch.bool, is_shared=False)},
            batch_size=torch.Size([2]),
            device=None,
            is_shared=False)

    """

    def __init__(
        self,
        tokenizer,
        max_length,
        key="prompt",
        padding="max_length",
        truncation=True,
        return_tensordict=True,
        device=None,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.key = key
        self.padding = padding
        self.truncation = truncation
        self.return_tensordict = return_tensordict
        self.device = device

    def __call__(self, sample):
        tokenizer = self.tokenizer
        max_length = self.max_length

        tokenized_prompts = tokenizer(
            sample[self.key], max_length=max_length, truncation=True
        )
        prompt_rindex = [len(prompt) - 1 for prompt in tokenized_prompts["input_ids"]]
        tokenized_example = tokenizer(
            [
                prompt + label
                for prompt, label in zip(sample[self.key], sample["label"])
            ],
            max_length=max_length,
            padding=self.padding,
            truncation=self.truncation,
        )
        tokenized_example["prompt_rindex"] = prompt_rindex
        # drop any examples whose total length when tokenized exceeds block size
        # with recommended block size of 550, this is only ~0.1% of available examples.
        # NOTE: to mark as discarded we just save the mask as we cannot change the shape here
        tokenized_example["valid_sample"] = [True] * len(tokenized_example["input_ids"])
        for i, input_ids in enumerate(tokenized_example["input_ids"]):
            if input_ids[-1] != tokenizer.eos_token_id:
                tokenized_example["valid_sample"][i] = False
        if self.return_tensordict:
            return TensorDict.from_dict(dict(tokenized_example), device=self.device)
        return tokenized_example
