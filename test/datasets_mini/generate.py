# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Script used to generate the mini datasets."""

from datasets import Dataset, DatasetDict, load_dataset

d = load_dataset("CarperAI/openai_summarize_comparisons")
# d = load_dataset("CarperAI/openai_summarize_tldr")

smalld = {}
for key in list(d.keys()):
    if any(key.startswith(sub) for sub in ("train", "valid", "test")):
        smalld[key] = Dataset.from_dict(d[key][:1000])
smalld = DatasetDict(smalld)

smalld.save_to_disk("test/datasets_mini/openai_summarize_comparisons")
# smalld.save_to_disk("test/datasets_mini/openai_summarize_tldr")
