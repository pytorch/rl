# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import argparse
import importlib.util

import pytest
import torch

from tensordict import set_list_to_stack
from torchrl.data import History

_has_transformers = importlib.util.find_spec("transformers")
_has_vllm = importlib.util.find_spec("vllm")


class TestHistory:
    @pytest.fixture(scope="class", autouse=True)
    def set_context(self):
        with set_list_to_stack(True):
            yield

    def test_history_construct(self):
        hst0 = History(role="user", content="a message")
        assert not hst0.shape
        hst1 = History(role="user", content="another message")
        with pytest.raises(RuntimeError, match="unsqueeze"):
            hst0.append(hst1)
        hst0 = hst0.unsqueeze(0)

        # In an env.step, we typically have one more piece of history to add to the stack
        assert not hst1.shape
        assert not hst1.batch_size
        assert not hst1.batch_dims
        # test out-place
        hst0_copy = hst0.copy()
        hst0b = hst0.append(hst1, inplace=False)
        assert hst0b is not hst0
        assert (hst0 == hst0_copy).all()
        assert (hst0b[:-1] == hst0).all()

        # test in-place
        hst0b = hst0.append(hst1)
        assert hst0b is hst0
        assert hst0b.shape == (2,)

        assert hst0b.content == ["a message", "another message"]
        hst2 = History(
            role=["assistant", "user"],
            content=["i'm the assistant", "i'm the user"],
            batch_size=2,
        )
        assert hst2[0].role == "assistant"
        assert hst2[0].content == "i'm the assistant"
        assert hst2[1].role == "user"
        assert hst2[1].content == "i'm the user"
        with pytest.raises(RuntimeError, match="The new history to extend"):
            hst0.extend(hst1)

        # test out-place
        hst0_copy = hst0.copy()
        hst0b = hst0.extend(hst2, inplace=False)
        assert hst0b is not hst0
        assert (hst0 == hst0_copy).all()
        assert (hst0b[:-2] == hst0).all()

        # test in-place
        hst0b = hst0.extend(hst2)

        assert hst0b is hst0
        assert hst0.__dict__["_tensordict"].shape == (4,)
        assert hst0.shape == (4,)
        assert hst0.role == ["user", "user", "assistant", "user"]
        assert hst0.content == [
            "a message",
            "another message",
            "i'm the assistant",
            "i'm the user",
        ]

    def test_history_construct_ndim(self):
        hst0 = History(role="user", content="a message").unsqueeze(0).unsqueeze(0)
        hst1 = History(role="user", content="another message").unsqueeze(0)

        # test out-place
        hst0_copy = hst0.copy()
        assert isinstance(hst0_copy, History)
        assert hst0.shape == (1, 1)
        hst0b = hst0.append(hst1, inplace=False, dim=1)
        assert hst0b is not hst0
        assert hst0.shape == (1, 1)
        assert (hst0 == hst0_copy).all()
        assert (hst0b[:, :-1] == hst0).all()

        # test in-place
        assert hst0b.shape == (1, 2)
        assert hst0.shape == (1, 1)
        hst0b = hst0.append(hst1, dim=1)
        assert hst0b is hst0
        assert hst0b._tensordict.shape == (1, 2)
        assert hst0b.batch_size == (1, 2)
        assert hst0b.shape == (1, 2)

        assert hst0b.content == [["a message", "another message"]]
        hst2 = History(
            role=["assistant", "user"],
            content=["i'm the assistant", "i'm the user"],
            batch_size=2,
        ).unsqueeze(0)

        # test out-place
        hst0_copy = hst0.copy()
        hst0b = hst0.extend(hst2, inplace=False, dim=1)
        assert hst0b is not hst0
        assert (hst0 == hst0_copy).all()
        assert (hst0b[:, :-2] == hst0).all()

        # test in-place
        hst0b = hst0.extend(hst2, dim=1)

        assert hst0b is hst0
        assert hst0.__dict__["_tensordict"].shape == (
            1,
            4,
        )
        assert hst0.shape == (
            1,
            4,
        )
        assert hst0.role == [["user", "user", "assistant", "user"]]
        assert hst0.content == [
            [
                "a message",
                "another message",
                "i'm the assistant",
                "i'm the user",
            ]
        ]

    @pytest.fixture(scope="class")
    def mock_history(self):
        history0 = History(
            role="system",
            content="""CONTENT
        This is the setup""",
        )
        history1 = History(
            role="user",
            content="""CONTENT
        This is the first user prompt""",
        )
        history2 = History(
            role="assistant",
            content="""CONTENT
        This is the second prompt, the first for the assistant.""",
        )
        history = torch.stack([history0, history1, history2])
        return history

    @pytest.fixture(scope="class")
    def tokenizer(self):
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained("GPT2")
        yield tokenizer

    @pytest.mark.skipif(not _has_transformers, reason="requires transformers library")
    def test_history_template(self, mock_history, tokenizer):
        history = mock_history
        data_str = history.apply_chat_template(
            tokenizer=tokenizer, add_generation_prompt=False
        )
        assert isinstance(data_str, str)
        data_token = history.apply_chat_template(
            tokenizer=tokenizer, tokenize=True, add_generation_prompt=False
        )
        assert isinstance(data_token, torch.Tensor)

        # test add_generation_prompt
        data_str = history.apply_chat_template(
            tokenizer=tokenizer, add_generation_prompt=True
        )
        assert isinstance(data_str, str)
        assert data_str.endswith("<|im_start|>assistant\n"), data_str

    @pytest.mark.skipif(not _has_transformers, reason="requires transformers library")
    def test_history_template_recover(self, mock_history, tokenizer):
        history = mock_history
        data_str = history.apply_chat_template(tokenizer=tokenizer)
        # Test inverse
        recovered = history._inv_chatml(data_str)
        assert recovered.role == history.role
        assert recovered.content == history.content
        data_token = history.apply_chat_template(
            tokenizer=tokenizer, tokenize=True, add_generation_prompt=False
        )
        recovered = history._inv_chatml(tokenizer.batch_decode(data_token)[0])

    def test_history_spec(self):
        history = History(
            role=["system", "user", "assistant", "user"],
            content=[
                "i'm the system",
                "i'm the user",
                "I'm the assistant",
                "I'm the user again",
            ],
        )
        spec = history.default_spec()
        r = spec.zero()
        assert isinstance(r, History)
        assert spec.is_in(r)
        assert spec.is_in(history)


if __name__ == "__main__":
    args, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst"] + unknown)
