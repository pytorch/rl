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
import torch.nn.functional as F

from _utils_internal import get_default_devices
from tensordict import (
    is_tensor_collection,
    MemoryMappedTensor,
    TensorDict,
    TensorDictBase,
)
from tensordict.nn import TensorDictModule
from torchrl.data.rlhf import TensorDictTokenizer
from torchrl.data.rlhf.dataset import (
    _has_datasets,
    _has_transformers,
    get_dataloader,
    TokenizedDatasetLoader,
)
from torchrl.data.rlhf.prompt import PromptData, PromptTensorDictTokenizer
from torchrl.data.rlhf.reward import PairwiseDataset, pre_tokenization_hook
from torchrl.data.rlhf.utils import RolloutFromModel
from torchrl.modules.models.rlhf import GPT2RewardModel

HERE = Path(__file__).parent


@pytest.fixture
def tmpdir1(tmp_path_factory):
    yield tmp_path_factory.mktemp("tmpdir1")


@pytest.fixture(scope="session")
def minidata_dir_comparison(tmp_path_factory):
    dest = tmp_path_factory.mktemp("comparisons")
    dataset_path = HERE / "assets" / "openai_summarize_comparisons.zip"
    with zipfile.ZipFile(dataset_path, "r") as zip_ref:
        zip_ref.extractall(dest)
        yield dest / Path(dataset_path).stem


@pytest.fixture(scope="session")
def minidata_dir_tldr(tmp_path_factory):
    dest = tmp_path_factory.mktemp("tldr")
    dataset_path = HERE / "assets" / "openai_summarize_tldr.zip"
    with zipfile.ZipFile(dataset_path, "r") as zip_ref:
        zip_ref.extractall(dest)
        yield dest / Path(dataset_path).stem


@pytest.fixture(scope="session")
def tldr_batch_dir(tmp_path_factory):
    dest = tmp_path_factory.mktemp("tldr_batch")
    dataset_path = HERE / "assets" / "tldr_batch.zip"
    with zipfile.ZipFile(dataset_path, "r") as zip_ref:
        zip_ref.extractall(dest)
        yield dest / Path(dataset_path).stem
    from torchrl._utils import print_directory_tree

    print_directory_tree(dest)


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
    "dataset,make_process_fn,pre_tokenization_hook,split",
    [
        ("comp", TensorDictTokenizer, pre_tokenization_hook, "train"),
        ("comp", TensorDictTokenizer, pre_tokenization_hook, "valid1"),
        ("tldr", PromptTensorDictTokenizer, None, "train"),
        ("tldr", PromptTensorDictTokenizer, None, "valid"),
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
    split,
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
        valid_size=20,
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
    assert isinstance(td.get((suffix, "a")), MemoryMappedTensor)
    assert isinstance(td.get((suffix, "b")), MemoryMappedTensor)


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
        tokenizer.pad_token = "-pad-"
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
        tokenizer.pad_token = "-pad-"
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


@pytest.mark.skipif(
    not (_has_transformers and _has_datasets), reason="missing dependencies"
)
@pytest.mark.parametrize("batch_size", [5, 6])
@pytest.mark.parametrize("block_size", [550, 560])
@pytest.mark.parametrize("device", get_default_devices())
def test_reward_model(tmpdir1, minidata_dir_comparison, batch_size, block_size, device):
    dl = get_dataloader(
        batch_size,
        block_size,
        PairwiseDataset,
        device,
        dataset_name=minidata_dir_comparison,
        infinite=True,
        prefetch=0,
        split="train",
        root_dir=tmpdir1,
        from_disk=True,
    )

    reward_model = GPT2RewardModel().to(device)

    batch = next(dl)
    chosen_rewards, chosen_end_scores = reward_model(
        input_ids=batch.chosen_data.input_ids,
        attention_mask=batch.chosen_data.attention_mask,
    )
    rejected_rewards, _ = reward_model(
        input_ids=batch.rejected_data.input_ids,
        attention_mask=batch.rejected_data.attention_mask,
    )

    assert chosen_rewards.shape == torch.Size([batch_size, block_size])
    assert chosen_end_scores.shape == torch.Size([batch_size])

    batch.chosen_data.rewards = chosen_rewards
    batch.rejected_data.rewards = rejected_rewards

    loss = reward_model.compute_reward_loss(batch.chosen_data, batch.rejected_data)
    assert loss.shape == torch.Size([])


@pytest.mark.skipif(
    not (_has_transformers and _has_datasets), reason="missing dependencies"
)
class TestRollout:
    kl_coef = 0.1

    @staticmethod
    def init_transformer(device="cpu", as_tensordictmodule=True, inference=False):
        from transformers import GPT2Config, GPT2LMHeadModel

        model = GPT2LMHeadModel(GPT2Config())
        model.to(device)

        if as_tensordictmodule:
            model = TensorDictModule(
                model,
                in_keys={
                    "input_ids": "input_ids",
                    "attention_mask": "attention_mask",
                    "labels": "labels",
                },
                out_keys=["logits"] if inference else ["loss", "logits"],
            )
        return model

    @staticmethod
    def init_reward_model(device=None):
        model = GPT2RewardModel()
        model.to(device)

        model = TensorDictModule(
            model,
            in_keys=["input_ids", "attention_mask"],
            out_keys=["rewards", "end_scores"],
        )
        return model

    @staticmethod
    def _get_dummy_batch(batch_dir):
        return TensorDict.load_memmap(batch_dir)

    @property
    def _model(self):
        return self.init_transformer(
            as_tensordictmodule=False,
            inference=True,
        )

    @property
    def _ref_model(self):
        return self.init_transformer(
            as_tensordictmodule=False,
            inference=True,
        )

    @property
    def _reward_model(self):
        return self.init_reward_model()

    def _get_rollout_model(self, max_new_tokens=10):
        return RolloutFromModel(
            model=self._model,
            ref_model=self._ref_model,
            reward_model=self._reward_model,
            max_new_tokens=max_new_tokens,
        )

    def test_padded_right_to_left(self):
        x = torch.arange(12).view(3, 4)
        x[0, -2:] = 100
        x[1, -1:] = 100
        x[2, -3:] = 100
        y = RolloutFromModel._padded_right_to_left(x, eos_token_id=100)
        y_test = torch.tensor([[100, 100, 0, 1], [100, 4, 5, 6], [100, 100, 100, 8]])
        assert (y == y_test).all()

    @pytest.mark.parametrize("right_padded", [False, True])
    @pytest.mark.parametrize("sequence_length", [None, 5])
    def test_padded_left_to_right(self, right_padded, sequence_length):
        x = torch.arange(12).view(3, 4)
        x[0, :2] = 100
        x[1, :1] = 100
        x[2, :3] = 100
        if right_padded:
            x[..., -1] = 100
        y = RolloutFromModel._padded_left_to_right(
            x, eos_token_id=100, sequence_length=sequence_length
        )
        if not right_padded:
            y_test = torch.tensor(
                [[2, 3, 100, 100], [5, 6, 7, 100], [11, 100, 100, 100]]
            )
        else:
            y_test = torch.tensor(
                [[2, 100, 100, 100], [5, 6, 100, 100], [100, 100, 100, 100]]
            )
        if sequence_length:
            y_test = F.pad(y_test, (0, 1), value=100)

        assert (y == y_test).all()

    @pytest.mark.parametrize("batch_size", [2])
    @pytest.mark.parametrize("max_new_tokens", [10])
    @pytest.mark.parametrize("use_max", [True, False])
    def test_get_scores(self, batch_size, max_new_tokens, use_max):
        scores = torch.arange(batch_size * max_new_tokens**2, dtype=torch.float).view(
            batch_size, max_new_tokens, max_new_tokens
        )
        gen_tokens = torch.arange(max_new_tokens).expand(1, max_new_tokens)
        scores_comp = self._get_rollout_model(
            max_new_tokens=max_new_tokens
        )._get_scores(scores.unbind(1), generated_tokens=gen_tokens, use_max=use_max)
        if not use_max:
            assert (
                scores_comp.squeeze()
                == torch.diagonal(scores.log_softmax(-1), 0, -2, -1).squeeze()
            ).all()
        else:
            assert (
                scores_comp.squeeze() == scores.log_softmax(-1)[..., -1].squeeze()
            ).all()

    def test_generate(self, tldr_batch_dir, max_new_tokens=10):
        model = self._get_rollout_model(max_new_tokens)
        batch = self._get_dummy_batch(tldr_batch_dir)
        generated, log_probs, log_ratio = model.generate(batch)
        batch_size = batch.shape[0]

        assert generated.shape == torch.Size(
            [batch_size, batch.input_ids.shape[1] + max_new_tokens]
        )
        assert log_probs.shape == torch.Size([batch_size, max_new_tokens, 1])
        assert (log_probs <= 0).all().item()
        assert log_ratio.shape == torch.Size([batch_size, max_new_tokens])

    def test_rollout_from_data(self, tldr_batch_dir, max_new_tokens=10):
        model = self._get_rollout_model(max_new_tokens)
        batch = self._get_dummy_batch(tldr_batch_dir)
        td = model.rollout_from_data(batch)
        batch_size = batch.shape[0]

        expected_keys = {
            ("next", "attention_mask"),
            ("next", "done"),
            ("next", "terminated"),
            ("next", "input_ids"),
            ("next", "reward"),
            "action",
            "attention_mask",
            "input_ids",
            "sample_log_prob",
        }
        keys = set(td.keys(True, True))
        assert all(key in keys for key in expected_keys)
        assert td.batch_size == torch.Size([batch_size, max_new_tokens])


if __name__ == "__main__":
    args, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst"] + unknown)
