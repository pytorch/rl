# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from typing import Sequence

import torch
from tensordict import TensorDict, TensorDictBase
from tensordict.utils import NestedKey
from torchrl._utils import logger as torchrl_logger
from torchrl.data.replay_buffers import (
    SamplerWithoutReplacement,
    TensorDictReplayBuffer,
    TensorStorage,
)

_has_transformers = importlib.util.find_spec("transformers") is not None
_has_datasets = importlib.util.find_spec("datasets") is not None


class TokenizedDatasetLoader:
    """Loads a tokenizes dataset, and caches a memory-mapped copy of it.

    Args:
        split (str): One of ``"train"`` or ``"valid"``.
        max_length (int): the maximum sequence length.
        dataset_name (str): the name of the dataset.
        tokenizer_fn (callable): the tokeinizing method constructor, such as
            :class:`torchrl.data.llm.TensorDictTokenizer`. When called,
            it should return a :class:`tensordict.TensorDict` instance
            or a dictionary-like structure with the tokenized data.
        pre_tokenization_hook (callable, optional): called on
            the Dataset before tokenization. It should return a modified
            Dataset object.
            The intended use is for carrying out tasks that
            require modifying the dataset as a whole as opposed to modifying
            individual datapoints, for example discarding certain datapoints
            based on a particular condition. Tokenization and other
            "elementwise" operations on the data are performed by the process
            function which is mapped over the dataset.
        root_dir (path, optional): the path where the datasets are stored.
            Defaults to ``"$HOME/.cache/torchrl/data"``
        from_disk (bool, optional): if ``True``, :func:`datasets.load_from_disk`
            will be used. Otherwise, :func:`datasets.load_dataset` will be used.
            Defaults to ``False``.
        valid_size (int, optional): the size of the validation dataset (if split
            starts with ``"valid"``) will be truncated to this value.
            Defaults to 2000 items.
        num_workers (int, optional): number of workers for :meth:`datasets.dataset.map`
            which is called during tokenization.
            Defaults to ``max(os.cpu_count() // 2, 1)``.
        tokenizer_class (Type, optional): A tokenizer class, such as
            :class:`~transformers.AutoTokenizer` (default).
        tokenizer_model_name (str, optional): The model from which the vocabulary
            should be gathered. Defaults to ``"gpt2"``.

    The dataset will be stored in ``<root_dir>/<split>/<max_length>/``.

    Examples:
        >>> from torchrl.data.llm import TensorDictTokenizer
        >>> from torchrl.data.llm.reward import  pre_tokenization_hook
        >>> split = "train"
        >>> max_length = 550
        >>> dataset_name = "CarperAI/openai_summarize_comparisons"
        >>> loader = TokenizedDatasetLoader(
        ...     split,
        ...     max_length,
        ...     dataset_name,
        ...     TensorDictTokenizer,
        ...     pre_tokenization_hook=pre_tokenization_hook,
        ... )
        >>> dataset = loader.load()
        >>> print(dataset)
        TensorDict(
            fields={
                attention_mask: MemoryMappedTensor(shape=torch.Size([185068, 550]), device=cpu, dtype=torch.int64, is_shared=False),
                input_ids: MemoryMappedTensor(shape=torch.Size([185068, 550]), device=cpu, dtype=torch.int64, is_shared=False)},
            batch_size=torch.Size([185068]),
            device=None,
            is_shared=False)

    """

    def __init__(
        self,
        split,
        max_length,
        dataset_name,
        tokenizer_fn: type[TensorDictTokenizer],
        pre_tokenization_hook=None,
        root_dir=None,
        from_disk=False,
        valid_size: int = 2000,
        num_workers: int = None,
        tokenizer_class=None,
        tokenizer_model_name=None,
    ):
        self.split = split
        self.max_length = max_length
        self.dataset_name = dataset_name
        self.tokenizer_fn = tokenizer_fn
        self.pre_tokenization_hook = pre_tokenization_hook
        self.root_dir = root_dir
        self.from_disk = from_disk
        self.valid_size = valid_size
        if num_workers is None:
            num_workers = max(os.cpu_count() // 2, 1)
        self.num_workers = num_workers
        if tokenizer_class is None:
            from transformers import AutoTokenizer

            tokenizer_class = AutoTokenizer
        if tokenizer_model_name is None:
            tokenizer_model_name = "gpt2"
        self.make_tokenizer(
            tokenizer_class=AutoTokenizer, tokenizer_model_name=tokenizer_model_name
        )

    def make_tokenizer(self, *, tokenizer_class, tokenizer_model_name):
        tokenizer = tokenizer_class.from_pretrained(tokenizer_model_name)
        tokenizer.pad_token = tokenizer.eos_token
        self.tokenizer = tokenizer

    def load(self):
        """Loads a pre-processed, memory-mapped dataset if it exists, and creates it otherwise."""
        root_dir = self.root_dir
        max_length = self.max_length
        split = self.split
        if root_dir is None:
            root_dir = Path(os.environ.get("HOME")) / ".cache/torchrl/data/"
            os.makedirs(root_dir, exist_ok=True)
        root_dir = Path(root_dir)
        data_dir = root_dir / str(Path(self.dataset_name).name).split("-")[0]
        data_dir_total = data_dir / split / str(max_length)
        # search for data
        torchrl_logger.info(f"Looking for data in {data_dir_total}")
        if os.path.exists(data_dir_total):
            dataset = TensorDict.load_memmap(data_dir_total)
            return dataset
        dataset = self._load_dataset()
        dataset = self._tokenize(dataset)
        prefix = (split, str(max_length))
        result = self.dataset_to_tensordict(
            dataset, data_dir=data_dir, prefix=prefix, valid_mask_key="valid_sample"
        )
        return result[prefix]

    def _load_dataset(self):
        """Loads a text dataset from ``datasets``.

        Returns: a dataset of type ``datasets.Dataset``.
        """
        if not _has_datasets:
            raise ImportError(
                "preproc_data requires the datasets package to be installed."
            )
        from datasets import load_dataset, load_from_disk

        if self.from_disk:
            dataset = load_from_disk(str(self.dataset_name))[self.split]
        else:
            dataset = load_dataset(self.dataset_name, split=self.split)
        if self.split.startswith("valid"):
            # reduce size of validation dataset
            dataset = dataset.select(range(self.valid_size))
        if self.pre_tokenization_hook is not None:
            dataset = self.pre_tokenization_hook(dataset)
        return dataset

    def _tokenize(
        self,
        dataset,
        excluded_features: Sequence[str] | None = None,
    ):
        """Preprocesses a text dataset from ``datasets``.

        Args:
            dataset (datasets.Dataset): a dataset loaded using :meth:`load_dataset`.
            excluded_features (sequence of str, optional): the features to exclude
                once tokenization is complete. Defaults to ``{"text", "prompt", "label", "valid_sample"}``.

        Returns: a dataset of type ``datasets.Dataset``.
        """
        if not _has_transformers:
            raise ImportError("The transformers library is missing.")

        num_workers = self.num_workers
        if excluded_features is None:
            excluded_features = {"text", "prompt", "label", "valid_sample"}
        tokenizer = self.tokenizer
        # tokenize the dataset
        # TODO: replace this by TensorDict.map
        dataset = dataset.map(
            self.tokenizer_fn(
                tokenizer, max_length=self.max_length, return_tensordict=False
            ),
            desc="Tokenizing...",
            num_proc=num_workers,
            batched=True,
        )
        if not isinstance(dataset, TensorDictBase):
            dataset_dict = dataset.to_dict()
            if excluded_features:
                dataset_dict = {
                    key: value
                    for key, value in dataset_dict.items()
                    if key not in excluded_features
                }
            dataset = TensorDict.from_dict(
                dataset_dict, auto_batch_size=True, batch_dims=1
            )
        elif excluded_features:
            dataset = dataset.exclude(*excluded_features)
        # keep non empty rows (i.e. where at least one token is not eos)
        if "valid_sample" in dataset.keys():
            mask = dataset.get("valid_sample")
            dataset = dataset[mask]
        return dataset

    @staticmethod
    def dataset_to_tensordict(
        dataset: datasets.Dataset | TensorDict,  # noqa: F821
        data_dir: Path,
        prefix: NestedKey = None,
        features: Sequence[str] = None,
        batch_dims=1,
        valid_mask_key=None,
    ):
        """Converts a dataset to a memory-mapped TensorDict.

        If the dataset is already a :class:`TensorDict` instance, it is simply converted
        to a memory-mapped TensorDict.
        Otherwise, the dataset is expected to have a ``features`` attribute
        which is a sequence of strings indicating the features that can be found
        in the dataset. If it does not, the ``features`` must be passed explicitly
        to this function.

        Args:
            dataset (datasets.Dataset, TensorDict or equivalent): a dataset to convert
                to a memory-mapped TensorDict.
                If ``features`` is ``None``, it must have a ``features`` attribute
                with the list of keys to write in the tensordict.
            data_dir (Path or equivalent): directory where the data should be written.
            prefix (NestedKey, optional): the prefix of the dataset location. This can
                be used to differentiate several copies of a same dataset that have
                undergone different preprocessings.
            features (sequence of str, optional): a sequence of str indicating the
                features that can be found in the dataset.
            batch_dims (int, optional): the number of batch_dimensions of the data
                (ie number of dimensions along which the tensordict can be indexed).
                Defaults to 1.
            valid_mask_key (NestedKey, optional): if provided, this entry will be
                tentatively gathered and used to filder the data. Defaults to
                ``None`` (ie, no filter key).

        Returns: a TensorDict containing memory-mapped tensors with the dataset.

        Examples:
            >>> from datasets import Dataset
            >>> import tempfile
            >>> data = Dataset.from_dict({"tokens": torch.randint(20, (10, 11)), "labels": torch.zeros(10, 11)})
            >>> with tempfile.TemporaryDirectory() as tmpdir:
            ...     data_memmap = TokenizedDatasetLoader.dataset_to_tensordict(
            ...         data, data_dir=tmpdir, prefix=("some", "prefix"), features=["tokens", "labels"]
            ...     )
            ...     print(data_memmap)
            TensorDict(
                fields={
                    some: TensorDict(
                        fields={
                            prefix: TensorDict(
                                fields={
                                    labels: MemoryMappedTensor(shape=torch.Size([10, 11]), device=cpu, dtype=torch.float32, is_shared=False),
                                    tokens: MemoryMappedTensor(shape=torch.Size([10, 11]), device=cpu, dtype=torch.int64, is_shared=False)},
                                batch_size=torch.Size([10]),
                                device=None,
                                is_shared=False)},
                        batch_size=torch.Size([]),
                        device=None,
                        is_shared=False)},
                batch_size=torch.Size([]),
                device=None,
                is_shared=False)

        """
        if not isinstance(dataset, TensorDict):
            if features is None:
                features = dataset.features
            if prefix is None:
                prefix = ()
            data_dict = {key: torch.as_tensor(dataset[key]) for key in features}
            out = TensorDict.from_dict(
                data_dict, batch_dims=batch_dims, auto_batch_size=True
            )
        else:
            out = dataset
        if valid_mask_key is not None and valid_mask_key in out.keys(
            include_nested=True
        ):
            out = out[out.get(valid_mask_key)]
        out = TensorDict({prefix: out})
        out.memmap_(prefix=data_dir)
        return out


def create_infinite_iterator(iterator):
    """Iterates indefinitely over an iterator."""
    while True:
        yield from iterator


def get_dataloader(
    batch_size: int,
    block_size: int,
    tensorclass_type: type,
    device: torch.device,
    dataset_name: str | None = None,
    infinite: bool = True,
    prefetch: int = 0,
    split: str = "train",
    root_dir: str | None = None,
    from_disk: bool = False,
    num_workers: int | None = None,
):
    """Creates a dataset and returns a dataloader from it.

    Args:
        batch_size (int): the batch size of the dataloader samples.
        block_size (int): the maximum length of a sequence in the dataloader.
        tensorclass_type (tensorclass class): a tensorclass with a :meth:`from_dataset`
            method that must accept three keyword arguments: ``split`` (see below),
            ``max_length`` which is the block size to be used for training and
            ``dataset_name``, a string indicating the dataset. The ``root_dir``
            and ``from_disk`` arguments should also be supported.
        device (torch.device or equivalent): the device where the samples should
            be cast.
        dataset_name (str, optional): the dataset name. If not provided and if
            the tensorclass supports it, a default dataset name will be gathered
            for the tensorclass being used.
        infinite (bool, optional): if ``True``, the iteration will be infinite
            such that ``next(iterator)`` will always return a value.
            Defaults to ``True``.
        prefetch (int, optional): the number of items to be prefetched if
            multithreaded dataloading is being used.
        split (str, optional): the data split. Either ``"train"`` or ``"valid"``.
            Defaults to ``"train"``.
        root_dir (path, optional): the path where the datasets are stored.
            Defaults to ``"$HOME/.cache/torchrl/data"``
        from_disk (bool, optional): if ``True``, :func:`datasets.load_from_disk`
            will be used. Otherwise, :func:`datasets.load_dataset` will be used.
            Defaults to ``False``.
        num_workers (int, optional): number of workers for :meth:`datasets.dataset.map`
            which is called during tokenization.
            Defaults to ``max(os.cpu_count() // 2, 1)``.

    Examples:
        >>> from torchrl.data.llm.reward import PairwiseDataset
        >>> dataloader = get_dataloader(
        ...     batch_size=256, block_size=550, tensorclass_type=PairwiseDataset, device="cpu")
        >>> for d in dataloader:
        ...     print(d)
        ...     break
        PairwiseDataset(
            chosen_data=RewardData(
                attention_mask=Tensor(shape=torch.Size([256, 550]), device=cpu, dtype=torch.int64, is_shared=False),
                input_ids=Tensor(shape=torch.Size([256, 550]), device=cpu, dtype=torch.int64, is_shared=False),
                rewards=None,
                end_scores=None,
                batch_size=torch.Size([256]),
                device=cpu,
                is_shared=False),
            rejected_data=RewardData(
                attention_mask=Tensor(shape=torch.Size([256, 550]), device=cpu, dtype=torch.int64, is_shared=False),
                input_ids=Tensor(shape=torch.Size([256, 550]), device=cpu, dtype=torch.int64, is_shared=False),
                rewards=None,
                end_scores=None,
                batch_size=torch.Size([256]),
                device=cpu,
                is_shared=False),
            batch_size=torch.Size([256]),
            device=cpu,
            is_shared=False)
    """
    data = tensorclass_type.from_dataset(
        split=split,
        dataset_name=dataset_name,
        max_length=block_size,
        root_dir=root_dir,
        from_disk=from_disk,
        num_workers=num_workers,
    )
    out = TensorDictReplayBuffer(
        storage=TensorStorage(data),
        collate_fn=lambda x: x.as_tensor().to(device, non_blocking=True),
        sampler=SamplerWithoutReplacement(drop_last=True),
        batch_size=batch_size,
        prefetch=prefetch,
    )
    if infinite:
        return create_infinite_iterator(out)
    return out


class TensorDictTokenizer:
    """Factory for a process function that applies a tokenizer over a text example.

    Args:
        tokenizer (tokenizer from transformers library): the tokenizer to use.
        max_length (int): maximum length of the sequence.
        key (str, optional): the key where to find the text. Defaults to ``"text"``.
        padding (str, optional): type of padding. Defaults to ``"max_length"``.
        truncation (bool, optional): whether the sequences should be truncated to max_length.
        return_tensordict (bool, optional): if ``True``, a TensoDict is returned.
            Otherwise, a the original data will be returned.
        device (torch.device, optional): the device where to store the data.
            This option is ignored if ``return_tensordict=False``.

    See transformers library for more information about tokenizers:
        Padding and truncation: `<https://huggingface.co/docs/transformers/pad_truncation>`_

    Returns: a :class:`tensordict.TensorDict` instance with the same batch-size
    as the input data.

    Examples:
        >>> from transformers import AutoTokenizer
        >>> tokenizer = AutoTokenizer.from_pretrained("gpt2")
        >>> tokenizer.pad_token = 100
        >>> process = TensorDictTokenizer(tokenizer, max_length=10)
        >>> # example with a single input
        >>> example = {"text": "I am a little worried"}
        >>> process(example)
        TensorDict(
            fields={
                attention_mask: Tensor(shape=torch.Size([10]), device=cpu, dtype=torch.int64, is_shared=False),
                input_ids: Tensor(shape=torch.Size([10]), device=cpu, dtype=torch.int64, is_shared=False)},
            batch_size=torch.Size([]),
            device=None,
            is_shared=False)
        >>> # example with a multiple inputs
        >>> example = {"text": ["Let me reassure you", "It will be ok"]}
        >>> process(example)
        TensorDict(
            fields={
                attention_mask: Tensor(shape=torch.Size([2, 10]), device=cpu, dtype=torch.int64, is_shared=False),
                input_ids: Tensor(shape=torch.Size([2, 10]), device=cpu, dtype=torch.int64, is_shared=False)},
            batch_size=torch.Size([2]),
            device=None,
            is_shared=False)

    """

    def __init__(
        self,
        tokenizer,
        max_length,
        key="text",
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
        input = sample[self.key]
        tokenized_sample = self.tokenizer(
            input,
            max_length=self.max_length,
            padding=self.padding,
            truncation=self.truncation,
        )
        batch_size = [] if isinstance(input, str) else [len(input)]
        if self.return_tensordict:
            return TensorDict.from_dict(
                dict(tokenized_sample),
                batch_size=batch_size,
                device=self.device,
                auto_batch_size=True,
            )
        return tokenized_sample
