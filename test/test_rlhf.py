# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
from pathlib import Path

import pytest

from tensordict import TensorDict
from torchrl.data.rlhf.comparison import (
    make_process_fn_comparison,
    pre_tokenization_hook,
)
from torchrl.data.rlhf.dataset import create_or_load_dataset
from torchrl.data.rlhf.tldr import make_process_fn_tldr

HERE = Path(__file__).parent


@pytest.mark.parametrize("max_length", [12, 550])
@pytest.mark.parametrize(
    "dataset,make_process_fn,pre_tokenization_hook",
    [
        (
            f"{HERE}/datasets_mini/openai_summarize_comparisons",
            make_process_fn_comparison,
            pre_tokenization_hook,
        ),
        (f"{HERE}/datasets_mini/openai_summarize_tldr", make_process_fn_tldr, None),
    ],
)
def test_create_or_load_dataset(
    tmpdir, max_length, dataset, make_process_fn, pre_tokenization_hook
):
    data = create_or_load_dataset(
        split="train",
        max_length=max_length,
        dataset_name=dataset,
        make_process_fn=make_process_fn,
        pre_tokenization_hook=pre_tokenization_hook,
        from_disk=True,
        root_dir=tmpdir,
    )
    assert isinstance(data, TensorDict)
    assert "train" in data.keys()
    assert ("train", str(max_length)) in data.keys(True)
    for key, val in data.items(True, True):
        if val.ndim > 1:
            assert val.shape[1] == max_length


def test_preproc_data():
    pass


def dataset_to_tensordict():
    pass


def test_get_dataloader():
    pass


def test_promptdata():
    pass


def test_pairwise_dataset():
    pass


if __name__ == "__main__":
    args, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst"] + unknown)
