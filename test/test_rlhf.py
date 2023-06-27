# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import zipfile
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import torch

from _utils_internal import get_default_devices
from tensordict import is_tensor_collection, MemmapTensor, TensorDict, TensorDictBase
from torchrl.data.rlhf import TensorDictTokenizer
from torchrl.data.rlhf.dataset import (
    _has_datasets,
    _has_transformers,
    get_dataloader,
    TokenizedDatasetLoader,
)
from torchrl.data.rlhf.prompt import PromptData, PromptTensorDictTokenizer
from torchrl.data.rlhf.reward import PairwiseDataset, pre_tokenization_hook
from torchrl.modules.models.rlhf import GPT2RewardModel
from transformers import AutoTokenizer

HERE = Path(__file__).parent


@pytest.fixture
def tmpdir1(tmp_path_factory):
    yield tmp_path_factory.mktemp("tmpdir1")


@pytest.fixture(scope="session")
def minidata_dir_comparison(tmp_path_factory):
    dest = tmp_path_factory.mktemp("tldr")
    dataset_path = f"{HERE}/assets/openai_summarize_comparisons.zip"
    with zipfile.ZipFile(dataset_path, "r") as zip_ref:
        zip_ref.extractall(dest)
        yield dest / Path(dataset_path).name[:-4]


@pytest.fixture(scope="session")
def minidata_dir_tldr(tmp_path_factory):
    dest = tmp_path_factory.mktemp("tldr")
    dataset_path = f"{HERE}/assets/openai_summarize_tldr.zip"
    with zipfile.ZipFile(dataset_path, "r") as zip_ref:
        zip_ref.extractall(dest)
        yield dest / Path(dataset_path).name[:-4]


@pytest.mark.skipif(
    not (_has_transformers and _has_datasets), reason="missing dependencies"
)
@pytest.mark.parametrize("max_length", [12, 550])
@pytest.mark.parametrize(
    "dataset,make_process_fn,pre_tokenization_hook",
    [
        (
            "comp",
            TensorDictTokenizer,
            pre_tokenization_hook,
        ),
        (
            "tldr",
            PromptTensorDictTokenizer,
            None,
        ),
    ],
)
def test_create_or_load_dataset(
    tmpdir1,
    minidata_dir_tldr,
    minidata_dir_comparison,
    max_length,
    dataset,
    make_process_fn,
    pre_tokenization_hook,
    mocker,
):
    # test caching of the values
    lmemmap_save = deepcopy(TensorDict.load_memmap)
    mocked_hello = mocker.patch("tensordict.TensorDict.load_memmap")
    mocked_hello.side_effect = lmemmap_save
    if dataset == "tldr":
        dataset = minidata_dir_tldr
    elif dataset == "comp":
        dataset = minidata_dir_comparison
    else:
        raise NotImplementedError

    for i in range(2):
        data = TokenizedDatasetLoader(
            split="train",
            max_length=max_length,
            dataset_name=dataset,
            tokenizer_fn=make_process_fn,
            pre_tokenization_hook=pre_tokenization_hook,
            from_disk=True,
            root_dir=tmpdir1,
        ).load()
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


@pytest.mark.skipif(
    not (_has_transformers and _has_datasets), reason="missing dependencies"
)
@pytest.mark.parametrize("max_length", [12, 550])
@pytest.mark.parametrize(
    "dataset,make_process_fn,pre_tokenization_hook",
    [
        (
            "comp",
            TensorDictTokenizer,
            pre_tokenization_hook,
        ),
        (
            "tldr",
            PromptTensorDictTokenizer,
            None,
        ),
    ],
)
def test_preproc_data(
    tmpdir1,
    max_length,
    dataset,
    make_process_fn,
    pre_tokenization_hook,
    minidata_dir_tldr,
    minidata_dir_comparison,
    split="train",
):
    import datasets

    if dataset == "tldr":
        dataset = minidata_dir_tldr
    elif dataset == "comp":
        dataset = minidata_dir_comparison
    else:
        raise NotImplementedError
    loader = TokenizedDatasetLoader(
        split=split,
        max_length=max_length,
        dataset_name=dataset,
        tokenizer_fn=make_process_fn,
        pre_tokenization_hook=pre_tokenization_hook,
        from_disk=True,
        root_dir=tmpdir1,
    )
    dataset = loader._load_dataset()
    assert isinstance(dataset, datasets.Dataset)
    dataset = loader._tokenize(dataset)
    assert isinstance(dataset, TensorDictBase)


@pytest.mark.skipif(
    not (_has_transformers and _has_datasets), reason="missing dependencies"
)
@pytest.mark.parametrize("suffix", ["c", ("c", "d")])
def test_dataset_to_tensordict(tmpdir, suffix):
    import datasets

    dataset = datasets.Dataset.from_dict({"a": np.zeros((10,)), "b": np.ones((10,))})
    td = TokenizedDatasetLoader.dataset_to_tensordict(dataset, tmpdir, prefix=suffix)
    if suffix == "c":
        assert ("c", "a") in td.keys(True)
        assert ("c", "b") in td.keys(True)
    else:
        assert ("c", "d", "a") in td.keys(True)
        assert ("c", "d", "b") in td.keys(True)
    assert isinstance(td.get((suffix, "a")), MemmapTensor)
    assert isinstance(td.get((suffix, "b")), MemmapTensor)


@pytest.mark.skipif(
    not (_has_transformers and _has_datasets), reason="missing dependencies"
)
@pytest.mark.parametrize("batch_size", [5, 6])
@pytest.mark.parametrize("block_size", [15, 50])
@pytest.mark.parametrize(
    "tensorclass_type,dataset",
    [
        (PromptData, "tldr"),
        (PairwiseDataset, "comp"),
    ],
)
@pytest.mark.parametrize("device", get_default_devices())
@pytest.mark.parametrize("split", ["train"])
@pytest.mark.parametrize("infinite", [True, False])
def test_get_dataloader(
    tmpdir1,
    tensorclass_type,
    batch_size,
    block_size,
    device,
    dataset,
    split,
    infinite,
    minidata_dir_tldr,
    minidata_dir_comparison,
):
    if dataset == "tldr":
        dataset = minidata_dir_tldr
    elif dataset == "comp":
        dataset = minidata_dir_comparison
    else:
        raise NotImplementedError
    dl = get_dataloader(
        batch_size,
        block_size,
        tensorclass_type,
        device,
        dataset_name=dataset,
        infinite=infinite,
        prefetch=0,
        split=split,
        root_dir=tmpdir1,
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


@pytest.mark.skipif(
    not (_has_transformers and _has_datasets), reason="missing dependencies"
)
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
        from transformers import AutoTokenizer

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
        from transformers import AutoTokenizer

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
