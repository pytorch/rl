# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import os
from pathlib import Path

import numpy as np
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import trange
from transformers import AutoTokenizer

NUM_PROC = max(os.cpu_count() // 2, 1)
HERE = Path(__file__).parent


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


def create_memmaps(
    split,
    max_length,
    dataset_name,
    make_process_fn,
    pre_tokenization_hook=None,
    batch_size=256,
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

    suffix = f"{split}-{max_length}"
    data_dir = HERE / dataset_name.rsplit("_", 1)[1]

    if not data_dir.exists():
        data_dir.mkdir()

    dtype = np.int32  # (can do since enc.max_token_value == 50256 is < 2**16)
    for feature in dataset.features:
        filename = data_dir / f"{feature}-{suffix}.bin"
        feature_data = dataset[feature]
        arr = np.memmap(filename, dtype=dtype, mode="w+", shape=feature_data.shape)

        print(f"writing {filename}...")
        for idx in trange(0, feature_data.shape[0], batch_size):
            arr[idx : idx + batch_size] = feature_data[idx : idx + batch_size]

        arr.flush()
