# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import zipfile
from copy import deepcopy
from pathlib import Path

import datasets
import numpy as np
import pytest
import torch

from _utils_internal import get_default_devices
from tensordict import is_tensor_collection, MemmapTensor, TensorDict, TensorDictBase
from torchrl.data.rlhf import TensorDictTokenizer
from torchrl.data.rlhf.dataset import (
    create_or_load_dataset,
    dataset_to_tensordict,
    get_dataloader,
    load_dataset,
    tokenize,
)
from torchrl.data.rlhf.prompt import PromptData, PromptTensorDictTokenizer
from torchrl.data.rlhf.reward import PairwiseDataset, pre_tokenization_hook
from torchrl.modules.models.rlhf import GPT2RewardModel
from transformers import AutoTokenizer

HERE = Path(__file__).parent


@pytest.fixture
def tmpdir1(tmp_path_factory):
    yield tmp_path_factory.mktemp("tmpdir1")


@pytest.fixture
def tmpdir2(tmp_path_factory):
    yield tmp_path_factory.mktemp("tmpdir2")


@pytest.mark.parametrize("max_length", [12, 550])
@pytest.mark.parametrize(
    "dataset_path,make_process_fn,pre_tokenization_hook",
    [
        (
            f"{HERE}/assets/openai_summarize_comparisons.zip",
            TensorDictTokenizer,
            pre_tokenization_hook,
        ),
        (
            f"{HERE}/assets/openai_summarize_tldr.zip",
            PromptTensorDictTokenizer,
            None,
        ),
    ],
)
def test_create_or_load_dataset(
    tmpdir1,
    tmpdir2,
    max_length,
    dataset_path,
    make_process_fn,
    pre_tokenization_hook,
    mocker,
):
    # test caching of the values
    lmemmap_save = deepcopy(TensorDict.load_memmap)
    mocked_hello = mocker.patch("tensordict.TensorDict.load_memmap")
    mocked_hello.side_effect = lmemmap_save
    with zipfile.ZipFile(dataset_path, "r") as zip_ref:
        zip_ref.extractall(tmpdir2)

        for i in range(2):

            # shutil.copyfileobj(gzip.open(dataset_path), tmpdir2)
            data = create_or_load_dataset(
                split="train",
                max_length=max_length,
                dataset_name=tmpdir2 / Path(dataset_path).name[:-4],
                tokenizer_fn=make_process_fn,
                pre_tokenization_hook=pre_tokenization_hook,
                from_disk=True,
                root_dir=tmpdir1,
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
    "dataset_path,make_process_fn,pre_tokenization_hook",
    [
        (
            f"{HERE}/assets/openai_summarize_comparisons.zip",
            TensorDictTokenizer,
            pre_tokenization_hook,
        ),
        (
            f"{HERE}/assets/openai_summarize_tldr.zip",
            PromptTensorDictTokenizer,
            None,
        ),
    ],
)
def test_preproc_data(
    tmpdir1,
    max_length,
    dataset_path,
    make_process_fn,
    pre_tokenization_hook,
    split="train",
):
    with zipfile.ZipFile(dataset_path, "r") as zip_ref:
        zip_ref.extractall(tmpdir1)
        dataset = load_dataset(
            split=split,
            dataset_name=tmpdir1 / Path(dataset_path).name[:-4],
            pre_tokenization_hook=pre_tokenization_hook,
            from_disk=True,
        )
        assert isinstance(dataset, datasets.Dataset)
        dataset = tokenize(
            dataset,
            max_length=max_length,
            tokenizer_fn=make_process_fn,
        )
        assert isinstance(dataset, TensorDictBase)


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
    "tensorclass_type,dataset_path",
    [
        (PromptData, f"{HERE}/assets/openai_summarize_tldr.zip"),
        (PairwiseDataset, f"{HERE}/assets/openai_summarize_comparisons.zip"),
    ],
)
@pytest.mark.parametrize("device", get_default_devices())
@pytest.mark.parametrize("split", ["train"])
@pytest.mark.parametrize("infinite", [True, False])
def test_get_dataloader(
    tmpdir1,
    tmpdir2,
    tensorclass_type,
    batch_size,
    block_size,
    device,
    dataset_path,
    split,
    infinite,
):
    with zipfile.ZipFile(dataset_path, "r") as zip_ref:
        zip_ref.extractall(tmpdir1)
        dl = get_dataloader(
            batch_size,
            block_size,
            tensorclass_type,
            device,
            dataset_name=tmpdir1 / Path(dataset_path).name[:-4],
            infinite=infinite,
            prefetch=0,
            split=split,
            root_dir=tmpdir2,
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


class TestTokenizers:
    @pytest.mark.parametrize("max_length", [10, 15])
    @pytest.mark.parametrize("key", ["text", "other"])
    @pytest.mark.parametrize("padding", ["max_length"])
    @pytest.mark.parametrize("truncation", [True, False])
    @pytest.mark.parametrize("return_tensordict", [True, False])
    @pytest.mark.parametrize("device", get_default_devices())
    def test_tensordict_tokenizer(
        self, max_length, key, padding, truncation, return_tensordict, device
    ):
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = 100
        process = TensorDictTokenizer(
            tokenizer,
            max_length=max_length,
            key=key,
            padding=padding,
            truncation=truncation,
            return_tensordict=return_tensordict,
            device=device,
        )
        example = {
            key: [
                "Knock, knock.",
                "Who's there?",
                "Lettuce.",
                "Lettuce who?",
                "Lettuce in, it's cold out here!",
            ]
        }
        if not truncation and return_tensordict and max_length == 10:
            with pytest.raises(ValueError, match="TensorDict conversion only supports"):
                out = process(example)
            return
        out = process(example)
        if return_tensordict:
            assert out.get("input_ids").shape[-1] == max_length
        else:
            obj = out.get("input_ids")
            while not isinstance(obj[-1], int):
                obj = obj[-1]
            if not truncation:
                assert len(obj) >= max_length
            else:
                assert len(obj) == max_length

    @pytest.mark.parametrize("max_length", [10, 15])
    @pytest.mark.parametrize("key", ["text", "other"])
    @pytest.mark.parametrize("padding", ["max_length"])
    @pytest.mark.parametrize("truncation", [True, False])
    @pytest.mark.parametrize("return_tensordict", [True, False])
    @pytest.mark.parametrize("device", get_default_devices())
    def test_prompt_tensordict_tokenizer(
        self, max_length, key, padding, truncation, return_tensordict, device
    ):
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = 100
        process = PromptTensorDictTokenizer(
            tokenizer,
            max_length=max_length,
            key=key,
            padding=padding,
            truncation=truncation,
            return_tensordict=return_tensordict,
            device=device,
        )
        example = {
            key: [
                "Knock, knock.",
                "Who's there?",
                "Lettuce.",
                "Lettuce who?",
                "Lettuce in, it's cold out here!",
            ],
            "label": ["right", "wrong", "right", "wrong", "right"],
        }
        if not truncation and return_tensordict and max_length == 10:
            with pytest.raises(ValueError, match="TensorDict conversion only supports"):
                out = process(example)
            return
        out = process(example)
        if return_tensordict:
            assert out.get("input_ids").shape[-1] == max_length
        else:
            obj = out.get("input_ids")
            while not isinstance(obj[-1], int):
                obj = obj[-1]
            if not truncation:
                assert len(obj) >= max_length
            else:
                assert len(obj) == max_length


@pytest.mark.parametrize("batch_size", [5, 6])
@pytest.mark.parametrize("block_size", [15, 50])
@pytest.mark.parametrize("device", get_default_devices())
def test_reward_model(tmpdir1, tmpdir2, batch_size, block_size, device):
    dataset_path = f"{HERE}/assets/openai_summarize_tldr.zip"
    with zipfile.ZipFile(dataset_path, "r") as zip_ref:
        zip_ref.extractall(tmpdir2)
        tensorclass_type = PromptData
        dl = get_dataloader(
            batch_size,
            block_size,
            tensorclass_type,
            device,
            dataset_name=tmpdir2 / "openai_summarize_tldr",
            infinite=True,
            prefetch=0,
            split="train",
            root_dir=tmpdir1,
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
