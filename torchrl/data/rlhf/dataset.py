import numpy as np
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import trange
from transformers import AutoTokenizer
import os
from pathlib import Path

from tensordict import TensorDict

NUM_PROC = max(os.cpu_count() // 2, 1)
HERE = Path(__file__).parent


def create_or_load_dataset(
    split,
    max_length,
    dataset_name,
    make_process_fn,
    pre_tokenization_hook=None,
    root_dir=None,
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
    print("root_dir:", root_dir)
    data_dir = root_dir / dataset_name.split("-")[0]
    print("data_dir", data_dir)
    data_dir_total = data_dir / split / str(max_length)
    print("data_dir_total", data_dir_total)
    # search for data
    if os.path.exists(data_dir_total):
        print("found existing dataset")
        dataset = TensorDict.load_memmap(data_dir)
        # exclude other datasets, if needed
        dataset = dataset.select((split, str(max_length)))
        return dataset
    print("preproc")
    dataset = preproc_data(split, max_length,    dataset_name,    make_process_fn,    pre_tokenization_hook)
    data_spec = (split, str(max_length))
    return dataset_to_tensordict(dataset, data_dir, data_spec)

def preproc_data(    split,
    max_length,
    dataset_name,
    make_process_fn,
    pre_tokenization_hook=None,
                     ):
    dataset = load_dataset(dataset_name, split=split)
    if split.startswith("valid"):
        # reduce size of validation dataset
        dataset = dataset.select(range(2_000))

    if pre_tokenization_hook is not None:
        dataset = pre_tokenization_hook(dataset)

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    # tokenize the dataset
    dataset = dataset.map(
        make_process_fn(tokenizer, max_length=max_length),
        desc="Tokenizing...",
        num_proc=NUM_PROC,
        batched=True,
    )
    dataset = dataset.select_columns(
        list({*dataset.column_names} - {"text", "prompt", "label", "valid_sample"})
    )
    # keep non empty rows (i.e. where at least one token is not eos)
    if "valid_sample" in dataset.features:
        dataset.set_format("numpy")
        mask = dataset["valid_sample"]
        filtered_ = dataset.data.filter(mask)
        dataset = dataset.__class__(filtered_, dataset.info, dataset.split)
    dataset.set_format("torch")
    return dataset

def dataset_to_tensordict(dataset, data_dir, suffix):
    data_dict = {(suffix, key): torch.as_tensor(dataset[key]) for key in dataset.features}
    out = TensorDict.from_dict(data_dict).memmap_(prefix=data_dir)
    out.batch_size = out.batch_size[:1]
    print("dataset:", out)
    return out

def create_infinite_dataloader(data, config, collate_fn):
    """
    Creates a dataloader and yields batches from it indefinitely, so that we can request
    batches whenever we like with next.
    """
    dl = DataLoader(
        data,
        shuffle=True,
        batch_size=config["batch_size"],
        collate_fn=collate_fn,
        drop_last=True,
    )
    while True:
        yield from dl
