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
from tensordict.nn import TensorDictModule
from torch.nn import functional as F
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
from torchrl.data.rlhf.utils import RolloutFromModel
from torchrl.modules.models.rlhf import GPT2RewardModel
from transformers import GPT2Config

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
    for data in dl:
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


class TestRollout:
    kl_coef = 0.1

    @staticmethod
    def init_transformer(
        dropout=0.1,
        device="cpu",
        as_tensordictmodule=True,
        inference=False,
    ):
        from transformers import GPT2LMHeadModel

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

    def init_reward_model(self, device=None):
        model = GPT2RewardModel()
        model.to(device)

        model = TensorDictModule(
            model,
            in_keys=["input_ids", "attention_mask"],
            out_keys=["rewards", "end_scores"],
        )
        return model

    @property
    def _dummy_batch(self):
        return PromptData.from_tensordict(
            TensorDict.load_memmap(f"{HERE}/datasets_mini/tldr_batch/")
        )

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

    @property
    def _rollout_model(self):
        return RolloutFromModel(
            self._model,
            self._ref_model,
            self._reward_model,
            self.max_new_tokens,
            self.kl_coef,
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

    @pytest.mark.parametrize("use_max", [True, False])
    def test_get_scores(self, use_max):
        scores = torch.arange(32, dtype=torch.float).view(2, 4, 4)
        gen_tokens = torch.arange(4).view(4).expand(1, 4)
        scores_comp = RolloutFromModel._get_scores(
            scores.unbind(1), generated_tokens=gen_tokens, use_max=use_max
        )
        if not use_max:
            assert (
                scores_comp.squeeze()
                == torch.diagonal(scores.log_softmax(-1), 0, -2, -1).squeeze()
            ).all()
        else:
            assert (
                scores_comp.squeeze() == scores.log_softmax(-1)[..., -1].squeeze()
            ).all()

    def test_generate(self, max_new_tokens=10):
        self.max_new_tokens = max_new_tokens
        model = self._rollout_model
        batch = self._dummy_batch
        generated, log_probs, log_ratio = model.generate(batch)

    def test_rollout_from_data(self, max_new_tokens=10):
        self.max_new_tokens = max_new_tokens
        model = self._rollout_model
        batch = self._dummy_batch
        model.rollout_from_data(batch)


if __name__ == "__main__":
    args, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst"] + unknown)
