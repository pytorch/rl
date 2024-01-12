# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Script used to generate the mini datasets."""
import multiprocessing as mp

try:
    mp.set_start_method("spawn")
except Exception:
    pass
from tempfile import TemporaryDirectory

from datasets import Dataset, DatasetDict, load_dataset

from torchrl.data.rlhf.dataset import get_dataloader
from torchrl.data.rlhf.prompt import PromptData


def generate_small_dataset(comparison=True):
    if comparison:
        d = load_dataset("CarperAI/openai_summarize_comparisons")
    else:
        d = load_dataset("CarperAI/openai_summarize_tldr")

    smalld = {}
    for key in list(d.keys()):
        if any(key.startswith(sub) for sub in ("train", "valid", "test")):
            smalld[key] = Dataset.from_dict(d[key][:50])
    smalld = DatasetDict(smalld)

    if comparison:
        smalld.save_to_disk("test/datasets_mini/openai_summarize_comparisons")
    else:
        smalld.save_to_disk("test/datasets_mini/openai_summarize_tldr")


def get_minibatch():
    with TemporaryDirectory() as tmpdir:
        dl = get_dataloader(
            batch_size=16,
            block_size=33,
            tensorclass_type=PromptData,
            dataset_name="../datasets_mini/openai_summarize_tldr",
            device="cpu",
            num_workers=2,
            infinite=False,
            prefetch=0,
            split="train",
            from_disk=True,
            root_dir=tmpdir,
        )
        for data in dl:
            data = data.clone().memmap_("test/datasets_mini/tldr_batch/")
            break


if __name__ == "__main__":
    get_minibatch()
