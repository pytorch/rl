# download and prepare the openai_summarize_tldr dataset for fine tuning transformers
# adapted from
# https://github.com/sanjeevanahilan/nanoChatGPT/blob/3cde2746c7ea8b0bd32edd44c76ead581bbda5d5/data/openai_summarize_tldr/prepare.py
from pathlib import Path

import numpy as np
import tiktoken
from datasets import load_dataset  # huggingface datasets
from tqdm import tqdm

# number of workers in .map() call
# good number to use is ~order number of cpu cores // 2
NUM_PROC = 16
DATASET = "CarperAI/openai_summarize_tldr"
HERE = Path(__file__).parent

# takes 54GB in huggingface .cache dir, about 8M documents (8,013,769)


def _process(example):
    enc = tiktoken.get_encoding("gpt2")
    ids = enc.encode_ordinary(
        example["text"]
    )  # encode_ordinary ignores any special tokens
    ids.append(enc.eot_token)  # add the end of text token, e.g. 50256 for gpt2 bpe
    # note: I think eot should be prepended not appended... hmm. it's called "eot" though...
    return {"ids": ids, "len": len(ids)}


def create_tldr_memmaps():
    dataset = load_dataset(DATASET)

    train_text_list = []
    for sample in dataset["train"]:
        train_text_list.append(sample["prompt"] + sample["label"])

    # add the text column to the train dataset
    dataset["train"] = dataset["train"].add_column("text", train_text_list)
    dataset["val"] = dataset.pop("valid")  # rename the valid dataset to val

    val_text_list = []
    for sample in dataset["val"]:
        val_text_list.append(sample["prompt"] + sample["label"])
    # add the text column to the val dataset
    dataset["val"] = dataset["val"].add_column("text", val_text_list)
    dataset.pop("test")  # remove the test dataset

    split_dataset = dataset
    # this results in:
    # >>> split_dataset
    # DatasetDict({
    #     train: Dataset({
    #         features: ['prompt', 'label', 'text'],
    #         num_rows: 116722
    #     })
    #     val: Dataset({
    #         features: ['prompt', 'label', 'text'],
    #         num_rows: 6447
    #     })
    # })

    # tokenize the dataset
    tokenized = split_dataset.map(
        _process,
        remove_columns=["text", "prompt", "label"],
        desc="tokenizing the splits",
        num_proc=NUM_PROC,
    )

    # concatenate all the ids in each dataset into one large file we can use for training
    for split, dset in tokenized.items():
        arr_len = np.sum(dset["len"])
        data_dir = HERE / "tldr"
        if not data_dir.exists():
            data_dir.mkdir()
        filename = data_dir / f"{split}.bin"
        dtype = np.uint16  # (can do since enc.max_token_value == 50256 is < 2**16)
        arr = np.memmap(filename, dtype=dtype, mode="w+", shape=(arr_len,))

        print(f"writing {filename}...")
        idx = 0
        for example in tqdm(dset):
            arr[idx : idx + example["len"]] = example["ids"]
            idx += example["len"]
        arr.flush()
