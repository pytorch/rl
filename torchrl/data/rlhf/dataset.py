# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import numpy as np
import torch

_has_datasets = importlib.util.find_spec("datasets") is not None
from typing import Sequence

from tensordict.tensordict import NestedKey, TensorDict
from torch.utils.data import DataLoader
from torchrl.data import TensorDictReplayBuffer, TensorStorage
from torchrl.data.replay_buffers import SamplerWithoutReplacement
from tqdm import trange
from transformers import AutoTokenizer


def create_or_load_dataset(
    split,
    max_length,
    dataset_name,
    make_process_fn,
    pre_tokenization_hook=None,
    root_dir=None,
    from_disk=False,
):
    """Loads a pre-processed, memory-mapped dataset if it exists, and creates it otherwise.

    Args:
        split (str): One of ``"train"`` or ``"valid"``.
        max_length (int): the maximum sequence length.
        dataset_name (str): the name of the dataset.
        make_process_fn (callable): a preprocess function.
        pre_tokenization_hook (callable): TODO
        root_dir (path, optional): the path where the datasets are stored.
            Defaults to ``"$HOME/.cache/torchrl/data"``
        from_disk (bool, optional): if ``True``, :func:`datasets.load_from_disk`
            will be used. Otherwise, :func:`datasets.load_dataset` will be used.
            Defaults to ``False``.

    The dataset will be stored in ``root/<split>/<max_length>/``
    Examples:
        >>> from torchrl.data.rlhf.comparison import make_process_fn_comparison, pre_tokenization_hook
        >>> split = "train"
        >>> max_length = 550
        >>> dataset_name = "CarperAI--openai_summarize_comparisons"
        >>> dataset = create_or_load_dataset(
        ...     split,
        ...     max_length,
        ...     dataset_name,
        ...     make_process_fn_comparison,
        ...     pre_tokenization_hook=pre_tokenization_hook,
        ... )
        >>> print(dataset)
        TensorDict(
            fields={
                train: TensorDict(
                    fields={
                        550: TensorDict(
                            fields={
                                attention_mask: MemmapTensor(shape=torch.Size([185068, 550]), device=cpu, dtype=torch.int64, is_shared=False),
                                input_ids: MemmapTensor(shape=torch.Size([185068, 550]), device=cpu, dtype=torch.int64, is_shared=False)},
                            batch_size=torch.Size([185068, 550]),
                            device=None,
                            is_shared=False)},
                    batch_size=torch.Size([185068, 550]),
                    device=None,
                    is_shared=False)},
            batch_size=torch.Size([185068, 550]),
            device=None,
            is_shared=False)

    """
    if root_dir is None:
        root_dir = Path(os.environ.get("HOME")) / ".cache/torchrl/data/"
        os.makedirs(root_dir, exist_ok=True)
    root_dir = Path(root_dir)
    data_dir = root_dir / str(Path(dataset_name).name).split("-")[0]
    data_dir_total = data_dir / split / str(max_length)
    # search for data
    if os.path.exists(data_dir_total):
        dataset = TensorDict.load_memmap(data_dir_total)
        return dataset
    dataset = load_dataset(
        split=split,
        dataset_name=dataset_name,
        pre_tokenization_hook=pre_tokenization_hook,
        from_disk=from_disk,
    )
    dataset = tokenize(
        dataset,
        max_length=max_length,
        make_process_fn=make_process_fn,
    )
    prefix = (split, str(max_length))
    return dataset_to_tensordict(
        dataset, data_dir=data_dir, prefix=prefix, valid_mask_key="valid_sample"
    )[prefix]


def load_dataset(
    split,
    dataset_name,
    pre_tokenization_hook=None,
    from_disk: bool = False,
):
    """Loads a text dataset from ``datasets``.

    Args:
        split (str): One of ``"train"`` or ``"valid"``.
        dataset_name (str): the name or path of the dataset.
        pre_tokenization_hook (callable): TODO
        from_disk (bool, optional): if ``True``, :func:`datasets.load_from_disk`
            will be used. Otherwise, :func:`datasets.load_dataset` will be used.
            Defaults to ``False``.

    Returns: a dataset of type ``datasets.Dataset``.
    """
    if not _has_datasets:
        raise ImportError("preproc_data requires the datasets package to be installed.")
    from datasets import load_dataset, load_from_disk

    if from_disk:
        dataset = load_from_disk(dataset_name)[split]
    else:
        dataset = load_dataset(dataset_name, split=split)
    if split.startswith("valid"):
        # reduce size of validation dataset
        dataset = dataset.select(range(2_000))
    if pre_tokenization_hook is not None:
        dataset = pre_tokenization_hook(dataset)
    return dataset


def tokenize(
    dataset,
    max_length,
    make_process_fn,
    num_workers: int = None,
    excluded_features: Sequence[str] | None = None,
):
    """Preprocesses a text dataset from ``datasets``.

    Args:
        dataset (datasets.Dataset): a dataset loaded using :func:`~.load_dataset`.
        max_length (int): the maximum sequence length.
        make_process_fn (callable): a preprocess function.
        num_workers (int, optional): number of workers for :meth:`datasets.dataset.map`.
            Defaults to ``max(os.cpu_count() // 2, 1)``.
        excluded_features (sequence of str, optional): the features to exclude
            once tokenization is complete. Defaults to ``{"text", "prompt", "label", "valid_sample"}``.

    Returns: a dataset of type ``datasets.Dataset``.
    """
    if num_workers is None:
        num_workers = max(os.cpu_count() // 2, 1)
    if excluded_features is None:
        excluded_features = {"text", "prompt", "label", "valid_sample"}
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    # tokenize the dataset
    dataset = dataset.map(
        make_process_fn(tokenizer, max_length=max_length),
        desc="Tokenizing...",
        num_proc=num_workers,
        batched=True,
    )
    if excluded_features:
        dataset = dataset.select_columns(
            list({*dataset.column_names} - excluded_features)
        )
    # keep non empty rows (i.e. where at least one token is not eos)
    if "valid_sample" in dataset.features:
        raise RuntimeError("move to tensordict")
        dataset.set_format("numpy")
        mask = dataset["valid_sample"]
        filtered_ = dataset.data.filter(mask)
        dataset = dataset.__class__(filtered_, dataset.info, dataset.split)
    dataset.set_format("torch")
    return dataset


def dataset_to_tensordict(
    dataset: "datasets.Dataset",
    data_dir: Path,
    prefix: NestedKey = None,
    features: Sequence[str] = None,
    batch_dims=1,
    valid_mask_key=None,
):
    """Convers a dataset to a TensorDict.

    The dataset is expected to have a ``features`` attribute which is a sequence of
    strings indicating the features that can be found in the dataset.

    Args:
        dataset (datasets.Dataset or equivalent): a dataset to map to a TensorDict.
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
        ...     data_memmap = dataset_to_tensordict(data, data_dir=tmpdir, prefix=("some", "prefix"), features=["tokens", "labels"])
        ...     print(data_memmap)
        TensorDict(
            fields={
                some: TensorDict(
                    fields={
                        prefix: TensorDict(
                            fields={
                                labels: MemmapTensor(shape=torch.Size([10, 11]), device=cpu, dtype=torch.float32, is_shared=False),
                                tokens: MemmapTensor(shape=torch.Size([10, 11]), device=cpu, dtype=torch.int64, is_shared=False)},
                            batch_size=torch.Size([10]),
                            device=None,
                            is_shared=False)},
                    batch_size=torch.Size([10]),
                    device=None,
                    is_shared=False)},
            batch_size=torch.Size([10]),
            device=None,
            is_shared=False)

    """
    if features is None:
        features = dataset.features
    if prefix is None:
        prefix = ()
    data_dict = {key: torch.as_tensor(dataset[key]) for key in features}
    out = TensorDict.from_dict(data_dict, batch_dims=batch_dims)
    if valid_mask_key is not None and valid_mask_key in out.keys(include_nested=True):
        out = out[out.get(valid_mask_key)]
    out = TensorDict({prefix: out}, [])
    out.memmap_(prefix=data_dir)
    return out


def create_infinite_iterator(iterator):
    """Iterates indefinitely over an iterator."""
    while True:
        yield from iterator


def get_dataloader(
    batch_size,
    block_size,
    tensorclass_type,
    device,
    dataset_name=None,
    infinite=True,
    prefetch=0,
    split="train",
    root_dir=None,
    from_disk=False,
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

    Examples:
        >>> from torchrl.data.rlhf.comparison import PairwiseDataset
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
