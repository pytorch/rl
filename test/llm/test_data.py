# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import argparse
import importlib.util
from typing import Mapping

import pytest
import torch
from tensordict import lazy_stack, set_list_to_stack

from torchrl import torchrl_logger

from torchrl.data import History
from torchrl.data.llm.chat import ContentBase

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
        data_str = history.apply_chat_template(
            tokenizer=tokenizer, add_generation_prompt=False
        )
        # Test inverse
        recovered = history._inv_chatml(data_str)
        assert recovered.role == history.role, (recovered.role, history.role)
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

    def test_content_base(self):
        from transformers import AutoProcessor

        processor = AutoProcessor.from_pretrained(
            "llava-hf/llava-onevision-qwen2-0.5b-ov-hf"
        )

        content_text = ContentBase(type="text", text="Hello, world!")
        content_img = ContentBase(
            type="image",
            url="https://github.com/pytorch/rl/blob/main/docs/source/_static/img/icon.png?raw=true",
        )
        content = lazy_stack([content_text, content_img])
        history0 = History(
            role="assistant",
            content=ContentBase(
                type="text",
                text="You are going to see an image and a hello world message. Ignore both.",
                batch_size=1,
            ),
        )
        history1 = History(role="user", content=content)
        history = lazy_stack([history0, history1])
        proc = history.apply_chat_template(
            tokenizer=processor,
            add_generation_prompt=False,
            return_dict=True,
            tokenize=False,
        )
        assert (
            proc
            == "<|im_start|>assistant \nYou are going to see an image and a hello world message. Ignore both.<|im_end|><|im_start|>user <image>\nHello, world!<|im_end|>"
        )
        proc = history.apply_chat_template(
            tokenizer=processor,
            add_generation_prompt=False,
            return_dict=True,
            tokenize=True,
        )
        assert isinstance(proc, Mapping)
        assert proc["input_ids"].shape == (1, 7294)
        assert proc["attention_mask"].shape == (1, 7294)
        assert proc["pixel_values"].shape == (1, 37, 3, 384, 384), proc[
            "pixel_values"
        ].shape
        assert (proc["image_sizes"] == torch.tensor([[2096, 2324]])).all()

    TEST_CASES = [
        # Case 1: All messages complete
        """<|im_start|>system
I am a helpful assistant.<|im_end|>
<|im_start|>user
What is the capital of France?<|im_end|>
<|im_start|>assistant
The capital of France is Paris.<|im_end|>""",
        # Case 2: Last message incomplete
        """<|im_start|>system
I am a helpful assistant.<|im_end|>
<|im_start|>user
What is the capital of France?<|im_end|>
<|im_start|>assistant
The capital of France is""",
        # Case 3: Multiple messages with mix of endings
        """<|im_start|>system
I am a helpful assistant.<|im_end|>
<|im_start|>user
Tell me about Python.<|im_end|>
<|im_start|>assistant
Python is a programming language.<|endoftext|>
<|im_start|>user
Can you elaborate?<|im_end|>
<|im_start|>assistant
Python is known for its simplicity""",
        # Case 4: Single incomplete message
        """<|im_start|>assistant
Let me help you with that""",
        # Case 5: Empty content but complete
        """<|im_start|>system
<|im_end|>
<|im_start|>user
<|im_end|>""",
        # Case 6: Message with tool calls
        """<|im_start|>system
I am an assistant that can use tools.<|im_end|>
<|im_start|>assistant
Let me help you with that.
<tool_call>
{"name": "calculator", "arguments": {"expression": "2+2"}}
</tool_call>
<|im_end|>
<|im_start|>tool
4<|im_end|>
<|im_start|>assistant
The result is""",
    ]

    def test_history_completion(self):
        """Test the History class's handling of complete and incomplete messages."""

        for i, test_case in enumerate(self.TEST_CASES):
            history = History.from_text(test_case, chat_template_name="qwen")

            # Print details about each message
            for j, (role, content, is_complete) in enumerate(
                zip(history.role, history.content, history.is_complete)
            ):
                torchrl_logger.info(f"Message {j}:")
                torchrl_logger.info(f"  Role: {role}")
                torchrl_logger.info(f"  Content: {content[:50]}...")
                torchrl_logger.info(f"  Complete: {is_complete}")

            # Basic assertions
            assert len(history.role) > 0, f"Case {i} should have at least one message"
            assert (
                len(history.role) == len(history.content) == len(history.is_complete)
            ), f"Case {i} should have matching lengths for role, content, and is_complete"

            # Case-specific assertions
            if i == 0:  # All messages complete
                assert all(
                    history.is_complete
                ), "Case 0 should have all complete messages"

            elif i == 1:  # Last message incomplete
                assert all(
                    history.is_complete[:-1]
                ), "Case 1 should have all but last message complete"
                assert not history.is_complete[
                    -1
                ], "Case 1 should have last message incomplete"

            elif i == 2:  # Mix of endings
                assert not history.is_complete[
                    -1
                ], "Case 2 should have last message incomplete"
                assert history.is_complete[
                    -2
                ], "Case 2 should have second-to-last message complete"

            elif i == 3:  # Single incomplete message
                assert len(history.role) == 1, "Case 3 should have exactly one message"
                assert not history.is_complete[
                    0
                ], "Case 3 should have an incomplete message"

            elif i == 4:  # Empty but complete messages
                assert all(
                    history.is_complete
                ), "Case 4 should have all complete messages"
                assert all(
                    not content.strip() for content in history.content
                ), "Case 4 should have empty content"

            elif i == 5:  # Tool calls
                assert not history.is_complete[
                    -1
                ], "Case 5 should have last message incomplete"
                assert history[2].role == "tool"


if __name__ == "__main__":
    args, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst"] + unknown)
