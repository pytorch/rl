# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
from copy import deepcopy
from pathlib import Path

import datasets
import numpy as np
import pytest
import torch

from _utils_internal import get_default_devices
from tensordict import is_tensor_collection, MemmapTensor, TensorDict
from torchrl.data.rlhf.dataset import (
    create_or_load_dataset,
    dataset_to_tensordict,
    get_dataloader,
    load_dataset,
    tokenize,
)
from torchrl.data.rlhf.prompt import make_process_fn_tldr, PromptData
from torchrl.data.rlhf.reward import (
    make_process_fn_comparison,
    PairwiseDataset,
    pre_tokenization_hook,
)
from torchrl.modules.models.rlhf import GPT2RewardModel

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
    tmpdir, max_length, dataset, make_process_fn, pre_tokenization_hook, mocker
):
    # test caching of the values
    lmemmap_save = deepcopy(TensorDict.load_memmap)
    mocked_hello = mocker.patch("tensordict.TensorDict.load_memmap")
    mocked_hello.side_effect = lmemmap_save

    for i in range(2):
        data = create_or_load_dataset(
            split="train",
            max_length=max_length,
            dataset_name=dataset,
            make_process_fn=make_process_fn,
            pre_tokenization_hook=pre_tokenization_hook,
            from_disk=True,
            root_dir=tmpdir,
        )
        if i == 0:
            mocked_hello.assert_not_called()
        else:
            mocked_hello.assert_called()

        assert isinstance(data, TensorDict)
        # assert "train" in data.keys(), data
        # assert ("train", str(max_length)) in data.keys(True), data
        for val in data.values(True, True):
            if val.ndim > 1:
                assert val.shape[1] == max_length


@pytest.mark.parametrize("max_length", [12, 550])
@pytest.mark.parametrize(
    "dataset_name,make_process_fn,pre_tokenization_hook",
    [
        (
            f"{HERE}/datasets_mini/openai_summarize_comparisons",
            make_process_fn_comparison,
            pre_tokenization_hook,
        ),
        (f"{HERE}/datasets_mini/openai_summarize_tldr", make_process_fn_tldr, None),
    ],
)
def test_preproc_data(
    max_length, dataset_name, make_process_fn, pre_tokenization_hook, split="train"
):
    dataset = load_dataset(
        split=split,
        dataset_name=dataset_name,
        pre_tokenization_hook=pre_tokenization_hook,
        from_disk=True,
    )
    assert isinstance(dataset, datasets.Dataset)
    dataset = tokenize(
        dataset,
        max_length=max_length,
        make_process_fn=make_process_fn,
    )
    assert isinstance(dataset, datasets.Dataset)


@pytest.mark.parametrize("suffix", ["c", ("c", "d")])
def test_dataset_to_tensordict(tmpdir, suffix):
    dataset = datasets.Dataset.from_dict({"a": np.zeros((10,)), "b": np.ones((10,))})
    td = dataset_to_tensordict(dataset, tmpdir, prefix=suffix)
    if suffix == "c":
        assert ("c", "a") in td.keys(True)
        assert ("c", "b") in td.keys(True)
    else:
        assert ("c", "d", "a") in td.keys(True)
        assert ("c", "d", "b") in td.keys(True)
    assert isinstance(td.get((suffix, "a")), MemmapTensor)
    assert isinstance(td.get((suffix, "b")), MemmapTensor)


@pytest.mark.parametrize("batch_size", [5, 6])
@pytest.mark.parametrize("block_size", [15, 50])
@pytest.mark.parametrize(
    "tensorclass_type,dataset_name",
    [
        (PromptData, f"{HERE}/datasets_mini/openai_summarize_tldr"),
        (PairwiseDataset, f"{HERE}/datasets_mini/openai_summarize_comparisons"),
    ],
)
@pytest.mark.parametrize("device", get_default_devices())
@pytest.mark.parametrize("split", ["train"])
@pytest.mark.parametrize("infinite", [True, False])
def test_get_dataloader(
    tmpdir,
    tensorclass_type,
    batch_size,
    block_size,
    device,
    dataset_name,
    split,
    infinite,
):
    dl = get_dataloader(
        batch_size,
        block_size,
        tensorclass_type,
        device,
        dataset_name=dataset_name,
        infinite=infinite,
        prefetch=0,
        split=split,
        root_dir=tmpdir,
        from_disk=True,
    )
    for data in dl:  # noqa: B007
        break
    assert data.shape[0] == batch_size
    for value in data.values():
        if value.ndim > 1:
            assert value.shape[1] == block_size
    assert data.device == device
    if infinite:
        assert not is_tensor_collection(dl)
    else:
        assert not is_tensor_collection(dl)


@pytest.mark.parametrize("batch_size", [5, 6])
@pytest.mark.parametrize("block_size", [15, 50])
@pytest.mark.parametrize("device", get_default_devices())
def test_reward_model(tmpdir, batch_size, block_size, device):
    dataset_name = f"{HERE}/datasets_mini/openai_summarize_tldr"
    tensorclass_type = PromptData
    dl = get_dataloader(
        batch_size,
        block_size,
        tensorclass_type,
        device,
        dataset_name=dataset_name,
        infinite=True,
        prefetch=0,
        split="train",
        root_dir=tmpdir,
        from_disk=True,
    )

    reward_model = GPT2RewardModel().to(device)

    batch = next(dl)

    rewards, end_scores = reward_model(
        input_ids=batch.input_ids, attention_mask=batch.attention_mask
    )

    assert rewards.shape == torch.Size([batch_size, block_size])
    assert end_scores.shape == torch.Size([batch_size])


if __name__ == "__main__":
    args, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst"] + unknown)
