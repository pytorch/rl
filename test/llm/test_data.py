# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import argparse
import importlib.util
from collections.abc import Mapping

import pytest
import torch
from tensordict import lazy_stack, set_list_to_stack, TensorDict

from torchrl import torchrl_logger

from torchrl.data import (
    History,
    LazyStackStorage,
    ReplayBuffer,
    SamplerWithoutReplacement,
)
from torchrl.data.llm.history import ContentBase
from torchrl.data.llm.topk import TopKRewardSelector

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
        assert proc["input_ids"].shape == (7294,)
        assert proc["attention_mask"].shape == (7294,)
        assert proc["pixel_values"].shape == (37, 3, 384, 384), proc[
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

    @pytest.mark.parametrize(
        "test_case",
        TEST_CASES,
        ids=["case_1", "case_2", "case_3", "case_4", "case_5", "case_6"],
    )
    def test_history_assistant_mask_qwen(self, test_case):
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B")
        history = History.from_text(test_case, chat_template_name="qwen")
        proc = history.apply_chat_template(
            tokenizer=tokenizer,
            chat_template_name="qwen",
            add_generation_prompt=False,
            return_dict=True,
            return_assistant_tokens_mask=True,
        )
        role_assistant = torch.tensor([r == "assistant" for r in history.role])
        last_item: str = history[role_assistant].apply_chat_template(
            tokenizer=tokenizer,
            chat_template_name="qwen",
            add_generation_prompt=False,
        )

        if "assistant" in history.role:
            assert proc["assistant_masks"].any()
        else:
            assert not proc["assistant_masks"].any()
        if last_item:
            decoded = tokenizer.decode(
                proc["input_ids"][proc["assistant_masks"].bool()]
            )
            assert type(decoded) is str
            assert last_item.endswith(decoded), (decoded, last_item)

    LLAMA_TEST_CASES = [
        # Case 1: All messages complete
        """<|begin_of_text|><|header_start|>system<|header_end|>

I am a helpful assistant.<|eot|><|header_start|>user<|header_end|>

What is the capital of France?<|eot|><|header_start|>assistant<|header_end|>

The capital of France is Paris.<|eot|>""",
        # Case 2: Last message incomplete
        """<|begin_of_text|><|header_start|>system<|header_end|>

I am a helpful assistant.<|eot|><|header_start|>user<|header_end|>

What is the capital of France?<|eot|><|header_start|>assistant<|header_end|>

The capital of France is""",
        # Case 3: Multiple messages with mix of endings
        """<|begin_of_text|><|header_start|>system<|header_end|>

I am a helpful assistant.<|eot|><|header_start|>user<|header_end|>

Tell me about Python.<|eot|><|header_start|>assistant<|header_end|>

Python is a programming language.<|eot|><|header_start|>user<|header_end|>

Can you elaborate?<|eot|><|header_start|>assistant<|header_end|>

Python is known for its simplicity""",
        # Case 4: Single incomplete message
        """<|header_start|>assistant<|header_end|>

Let me help you with that""",
        #         # Case 5: Empty content but complete -- not supported by LLAMA 4
        #         """<|begin_of_text|><|header_start|>system<|header_end|>
        # <|eot|><|header_start|>user<|header_end|>
        # <|eot|>""",
        # Case 6: Message with tool calls
        """<|begin_of_text|><|header_start|>system<|header_end|>

I am an assistant that can use tools.<|eot|><|header_start|>user<|header_end|>

<|eot|><|header_start|>assistant<|header_end|>

Let me help you with that.
<tool_call>
{"name": "calculator", "arguments": {"expression": "2+2"}}
</tool_call><|eot|><|header_start|>user<|header_end|>

4<|eot|><|header_start|>assistant<|header_end|>

The result is""",
    ]

    @pytest.mark.parametrize(
        "test_case",
        LLAMA_TEST_CASES,
        ids=["case_1", "case_2", "case_3", "case_4", "case_6"],
    )
    def test_history_assistant_mask_llama(self, test_case):
        from transformers import AutoTokenizer

        try:
            tokenizer = AutoTokenizer.from_pretrained(
                "meta-llama/Llama-4-Scout-17B-16E-Instruct"
            )
        except Exception:
            pytest.skip("Could not load Llama tokenizer")

        history = History.from_text(test_case, chat_template_name="llama")
        proc = history.apply_chat_template(
            tokenizer=tokenizer,
            chat_template_name="llama",
            add_generation_prompt=False,
            return_dict=True,
            return_assistant_tokens_mask=True,
        )
        role_assistant = torch.tensor([r == "assistant" for r in history.role])
        last_item: str = history[role_assistant].apply_chat_template(
            tokenizer=tokenizer,
            chat_template_name="llama",
            add_generation_prompt=False,
        )

        if "assistant" in history.role:
            assert proc["assistant_masks"].any()
        else:
            assert not proc["assistant_masks"].any()
        if last_item:
            decoded = tokenizer.decode(
                proc["input_ids"][proc["assistant_masks"].bool()]
            )
            assert type(decoded) is str
            assert last_item.endswith(decoded), (decoded, last_item)

    def test_history_completion(self):
        """Test the History class's handling of complete and incomplete messages."""

        for i, test_case in enumerate(self.TEST_CASES):
            history = History.from_text(test_case, chat_template_name="qwen")

            # torchrl_logger.info details about each message
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

    @pytest.mark.parametrize(
        "model_name, expected_template",
        [
            ("Qwen/Qwen2.5-0.5B", "qwen"),
            ("microsoft/phi-2", "chatml_format"),
            ("mosaicml/mpt-7b-instruct", "chatml_format"),
            ("facebook/opt-125m", "chatml_format"),
            ("gpt2", "chatml_format"),
            ("EleutherAI/pythia-70m", "chatml_format"),
            ("bigscience/bloom-560m", "chatml_format"),
            ("deepseek-ai/deepseek-coder-6.7b-base", "deepseek"),
        ],
    )
    def test_assistant_mask_model_families(self, model_name, expected_template):
        """Test assistant token masking support across different model families."""
        from transformers import AutoTokenizer

        torchrl_logger.info(f"\nTesting {model_name} with {expected_template} template")
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

        # Create a simple history
        history = History.from_chats(
            [
                [
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi there!"},
                ]
            ]
        )

        # Test with expected template
        result = history.apply_chat_template(
            tokenizer=tokenizer,
            chat_template_name=expected_template,
            add_generation_prompt=False,
            return_dict=True,
            return_assistant_tokens_mask=True,
        )

        # Verify assistant mask is present
        assert (
            "assistant_masks" in result
        ), f"Model {model_name} should support assistant masking"
        assert (
            result["assistant_masks"].shape[0] == 1
        ), "Should have batch dimension of 1"
        assert result["assistant_masks"].shape[1] > 0, "Should have sequence length > 0"

        # Verify some assistant tokens are masked
        assistant_token_count = result["assistant_masks"].sum().item()
        assert (
            assistant_token_count > 0
        ), f"Model {model_name} should have assistant tokens masked"
        torchrl_logger.info(
            f"  ✓ {model_name}: {assistant_token_count} assistant tokens masked"
        )

    @pytest.mark.parametrize(
        "template_name", ["qwen", "dialogpt", "falcon", "deepseek"]
    )
    def test_assistant_mask_with_custom_templates(self, template_name):
        """Test that models with custom templates can still use assistant masking."""
        from transformers import AutoTokenizer

        # Test Qwen with its custom template
        tokenizer = AutoTokenizer.from_pretrained(
            "Qwen/Qwen2.5-0.5B", trust_remote_code=True
        )

        history = History.from_chats(
            [
                [
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi there!"},
                ]
            ]
        )

        # Test with Qwen's custom template
        result = history.apply_chat_template(
            tokenizer=tokenizer,
            chat_template_name=template_name,
            add_generation_prompt=False,
            return_dict=True,
            return_assistant_tokens_mask=True,
        )

        assert "assistant_masks" in result
        assert result["assistant_masks"].sum().item() > 0

    @pytest.mark.parametrize(
        "model_name, template_name",
        [
            ("Qwen/Qwen2.5-0.5B", "qwen"),
            ("microsoft/DialoGPT-medium", "dialogpt"),
            ("tiiuae/falcon-7b-instruct", "falcon"),
            ("deepseek-ai/deepseek-coder-6.7b-base", "deepseek"),
        ],
    )
    def test_custom_template_equivalence(self, model_name, template_name):
        """Test that our custom templates produce the same output as the model's default template (except for masking)."""
        import re

        import transformers

        # Simple multi-turn chat for each model
        def norm(s):
            if isinstance(s, list):
                return [re.sub(r"\s+", " ", x.strip()) for x in s]
            elif isinstance(s, str):
                return re.sub(r"\s+", " ", s.strip())
            else:
                return s

        chat = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "How are you?"},
            {"role": "assistant", "content": "I'm good, thanks!"},
        ]

        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )
        history = History.from_chats([chat])

        # Output with model's default template
        try:
            default_out = history.apply_chat_template(
                tokenizer=tokenizer,
                add_generation_prompt=False,
                chat_template=tokenizer.chat_template,  # Use model's default
                chat_template_name=None,
                tokenize=False,
            )
        except Exception as e:
            default_out = None
            torchrl_logger.info(
                f"[WARN] Could not get default template for {model_name}: {e}"
            )

        # Output with our custom template
        custom_out = history.apply_chat_template(
            tokenizer=tokenizer,
            add_generation_prompt=False,
            chat_template_name=template_name,
            chat_template=None,
            tokenize=False,
        )

        if default_out is not None:
            assert norm(default_out) == norm(custom_out), (
                f"Custom template for {model_name} does not match default!\n"
                f"Default: {default_out}\nCustom: {custom_out}"
            )
        else:
            torchrl_logger.info(
                f"[INFO] Skipped equivalence check for {model_name} (no default template available)"
            )

    def test_add_chat_template_parameters_used(self):
        """Test that add_chat_template actually uses inverse_parser and model_family_keywords parameters with a real tokenizer."""
        import re
        import uuid

        from torchrl.data.llm.history import add_chat_template, History
        from transformers import AutoTokenizer

        try:
            # Track if the inverse parser is called
            inverse_parser_called = {"called": False}

            template_name = f"qwen_custom_{uuid.uuid4()}"

            # Create a custom template (trivially different from Qwen)
            custom_template = """
            {% for message in messages %}
            {%- if message['role'] == 'user' %}
            [USER] {{ message['content'] }}
            {%- elif message['role'] == 'assistant' %}
            {% generation %}[ASSISTANT] {{ message['content'] }}{% endgeneration %}
            {%- endif %}
            {% endfor %}
            """

            # Custom inverse parser
            def custom_inverse_parser(text: str) -> History:
                inverse_parser_called["called"] = True
                user_msgs = re.findall(
                    r"\[USER\] (.*?)(?=\[ASSISTANT\]|$)", text, re.DOTALL
                )
                assistant_msgs = re.findall(
                    r"\[ASSISTANT\] (.*?)(?=\[USER\]|$)", text, re.DOTALL
                )
                messages = []
                for i, user_content in enumerate(user_msgs):
                    messages.append(History(role="user", content=user_content.strip()))
                    if i < len(assistant_msgs):
                        messages.append(
                            History(role="assistant", content=assistant_msgs[i].strip())
                        )
                return lazy_stack(messages)

            # Register the custom template and parser for Qwen
            add_chat_template(
                template_name=template_name,
                template=custom_template,
                inverse_parser=custom_inverse_parser,
                model_family_keywords=["qwen"],
            )

            # Use a real Qwen tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                "Qwen/Qwen2.5-3B", trust_remote_code=True
            )
            history = History.from_chats(
                [
                    [
                        {"role": "user", "content": "Hello"},
                        {"role": "assistant", "content": "Hi there!"},
                    ]
                ]
            )

            # This should trigger auto-detection using our custom template
            result = history.apply_chat_template(
                tokenizer=tokenizer,
                add_generation_prompt=False,
                tokenize=False,
            )
            # The result should use our custom format
            if isinstance(result, list):
                result_str = result[0]
            else:
                result_str = result
            assert "[USER]" in result_str
            assert "[ASSISTANT]" in result_str

            # Test that inverse parser works
            parsed = History.from_text(result, chat_template_name=template_name)
            assert inverse_parser_called["called"], "Inverse parser was not called"
            assert parsed.role == history.role
            assert parsed.content == history.content
        finally:
            from torchrl.data.llm.history import (
                _CHAT_TEMPLATES,
                _CUSTOM_INVERSE_PARSERS,
                _CUSTOM_MODEL_FAMILY_KEYWORDS,
            )

            if template_name in _CHAT_TEMPLATES:
                del _CHAT_TEMPLATES[template_name]
            if template_name in _CUSTOM_INVERSE_PARSERS:
                del _CUSTOM_INVERSE_PARSERS[template_name]
            if template_name in _CUSTOM_MODEL_FAMILY_KEYWORDS:
                del _CUSTOM_MODEL_FAMILY_KEYWORDS[template_name]

    chats_round_trip = [
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is the capital of France?"},
            {"role": "assistant", "content": "The capital of France is Paris."},
        ],
        [
            {"role": "user", "content": "Tell me a joke."},
            {
                "role": "assistant",
                "content": "Why did the chicken cross the road? To get to the other side!",
            },
        ],
        [
            {"role": "system", "content": "You are a coding assistant."},
            {"role": "user", "content": "Write a Python function to add two numbers."},
            {"role": "assistant", "content": "def add(a, b):\n    return a + b"},
            {"role": "user", "content": "What about subtraction?"},
            {"role": "assistant", "content": "def subtract(a, b):\n    return a - b"},
        ],
    ]

    @pytest.mark.skipif(not _has_transformers, reason="requires transformers library")
    @pytest.mark.parametrize(
        "tokenizer_name",
        [
            "meta-llama/Llama-4-Scout-17B-16E-Instruct",
            "Qwen/Qwen2.5-0.5B",
            "microsoft/phi-2",
        ],
    )
    @pytest.mark.parametrize(
        "use_tokenizer_chat_template",
        [False, True],
        ids=["no_use_tokenizer_chat_template", "use_tokenizer_chat_template"],
    )
    @pytest.mark.parametrize("chat", chats_round_trip)
    def test_history_round_trip(
        self, tokenizer_name, use_tokenizer_chat_template, chat
    ):
        """Test round-trip conversion: History -> string -> History for various templates and tokenizers."""
        import re

        from transformers import AutoTokenizer

        # Example chats

        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name, trust_remote_code=True
        )

        history = History.from_chats(chat)
        if use_tokenizer_chat_template:
            if (
                not hasattr(tokenizer, "chat_template")
                or tokenizer.chat_template is None
            ):
                pytest.skip(f"Tokenizer {tokenizer_name} does not have a chat template")
            chat_template = tokenizer.chat_template
            chat_template_name = None
        else:
            chat_template = None
            chat_template_name = None  # Let History auto-detect

        # Serialize
        chat_str = history.apply_chat_template(
            tokenizer=tokenizer,
            add_generation_prompt=False,
            chat_template=chat_template,
            chat_template_name=chat_template_name,
            return_dict=False,
        )
        # Parse back
        parsed = History.from_text(
            text=chat_str,
            tokenizer=tokenizer,
            chat_template=chat_template,
            chat_template_name=chat_template_name,
        )

        # Normalize whitespace for comparison
        def norm(x):
            if isinstance(x, list):
                return [re.sub(r"\\s+", " ", str(xx).strip()) for xx in x]
            return re.sub(r"\\s+", " ", str(x).strip())
            # Compare roles and content
            assert norm(parsed.role) == norm(
                history.role
            ), f"Roles do not match!\nOriginal: {history.role}\nParsed: {parsed.role}"
            assert norm(parsed.content) == norm(
                history.content
            ), f"Content does not match!\nOriginal: {history.content}\nParsed: {parsed.content}"

        # All messages should be complete
        assert all(
            parsed.is_complete
        ), f"All messages should be complete after round-trip. is_complete: {parsed.is_complete}"

    @pytest.mark.skipif(not _has_transformers, reason="requires transformers library")
    @pytest.mark.parametrize(
        "tokenizer_name",
        [
            "Qwen/Qwen2.5-0.5B",
            "microsoft/phi-2",
            "meta-llama/Llama-4-Scout-17B-16E-Instruct",
        ],
    )
    @pytest.mark.parametrize(
        "use_tokenizer_chat_template",
        [False, True],
        ids=["no_use_tokenizer_chat_template", "use_tokenizer_chat_template"],
    )
    @pytest.mark.parametrize("chat", chats_round_trip)
    def test_history_round_trip_incomplete(
        self, tokenizer_name, use_tokenizer_chat_template, chat
    ):
        """Test that truncated strings are properly parsed with the last message marked as incomplete."""
        if chat[0]["role"] != "system":
            pytest.skip("Skipping test for non-system message")
        import re

        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name, trust_remote_code=True
        )

        history = History.from_chats(chat)

        if use_tokenizer_chat_template:
            if (
                not hasattr(tokenizer, "chat_template")
                or tokenizer.chat_template is None
            ):
                pytest.skip(f"Tokenizer {tokenizer_name} does not have a chat template")
            chat_template = tokenizer.chat_template
            chat_template_name = None
        else:
            chat_template = None
            chat_template_name = None  # Let History auto-detect

        # Serialize
        chat_str = history.apply_chat_template(
            tokenizer=tokenizer,
            add_generation_prompt=False,
            chat_template=chat_template,
            chat_template_name=chat_template_name,
            return_dict=False,
        )

        # Truncate the last 10 characters to simulate incomplete response
        truncated_chat_str = chat_str[:-10]

        # Parse back the truncated string
        parsed = History.from_text(
            text=truncated_chat_str,
            tokenizer=tokenizer,
            chat_template=chat_template,
            chat_template_name=chat_template_name,
        )

        # Normalize whitespace for comparison
        def norm(x):
            if isinstance(x, list):
                return [re.sub(r"\\s+", " ", str(xx).strip()) for xx in x]
            return re.sub(r"\\s+", " ", str(x).strip())

        # Check that we have the same number of messages as the original
        assert len(parsed.role) == len(
            history.role
        ), f"Number of messages should match original. Original: {len(history.role)}, Parsed: {len(parsed.role)}"
        assert len(parsed.content) == len(
            history.content
        ), f"Number of content items should match original. Original: {len(history.content)}, Parsed: {len(parsed.content)}"
        assert len(parsed.is_complete) == len(
            history.is_complete
        ), f"Number of completion flags should match original. Original: {len(history.is_complete)}, Parsed: {len(parsed.is_complete)}"

        # Check that all messages except the last one are complete
        if len(parsed.is_complete) > 0:
            assert all(
                parsed.is_complete[:-1]
            ), f"All messages except the last should be complete. is_complete: {parsed.is_complete}"
            assert not parsed.is_complete[
                -1
            ], f"Last message should be incomplete. is_complete: {parsed.is_complete}"

        # Check that roles match the original (except potentially the last one if it was truncated mid-message)
        assert norm(parsed.role[:-1]) == norm(
            history.role[:-1]
        ), f"All roles except the last should match original. Original: {history.role[:-1]}, Parsed: {parsed.role[:-1]}"

    @pytest.mark.skipif(not _has_transformers, reason="requires transformers library")
    def test_extract_responses_from_full_histories_batch_issue(self):
        """Test the isolated function for handling different response shapes in batch processing."""
        from torchrl.modules.llm.policies.common import (
            _extract_responses_from_full_histories,
        )
        from transformers import AutoTokenizer

        # Create a batch of 2 prompt histories
        prompt_histories = History.from_chats(
            [
                [
                    {"role": "user", "content": "Hello, how are you?"},
                ],
                [
                    {"role": "user", "content": "Tell me a joke."},
                ],
            ]
        )

        # Simulate generated text with different response counts
        text_full = [
            # First element: 1 assistant response
            """<|im_start|>user
Hello, how are you?<|im_end|>
<|im_start|>assistant
I'm doing well, thank you for asking!<|im_end|>""",
            # Second element: 3 messages (1 assistant + 1 user + 1 assistant)
            """<|im_start|>user
Tell me a joke.<|im_end|>
<|im_start|>assistant
Why did the chicken cross the road?<|im_end|>
<|im_start|>user
I don't know, why?<|im_end|>
<|im_start|>assistant
To get to the other side!<|im_end|>""",
        ]

        # Test the isolated function
        h_responses = _extract_responses_from_full_histories(
            text_full, prompt_histories, chat_template_name="qwen"
        )

        # Verify the responses have the expected shapes and content
        assert len(h_responses) == 2, f"Expected 2 responses, got {len(h_responses)}"

        # Check first response (should be padded to match second response length)
        response_0 = h_responses[0]
        assert response_0.shape == (3,), f"Expected shape (3,), got {response_0.shape}"
        assert response_0.role == [
            "assistant",
            "<none>",
            "<none>",
        ], f"Expected roles ['assistant', '<none>', '<none>'], got {response_0.role}"
        assert response_0.content == [
            "I'm doing well, thank you for asking!",
            "",
            "",
        ], f"Expected content ['I\\'m doing well, thank you for asking!', '', ''], got {response_0.content}"

        # Check second response (should have 3 messages)
        response_1 = h_responses[1]
        assert response_1.shape == (3,), f"Expected shape (3,), got {response_1.shape}"
        assert response_1.role == [
            "assistant",
            "user",
            "assistant",
        ], f"Expected roles ['assistant', 'user', 'assistant'], got {response_1.role}"
        assert response_1.content == [
            "Why did the chicken cross the road?",
            "I don't know, why?",
            "To get to the other side!",
        ], f"Expected content ['Why did the chicken cross the road?', 'I don\\'t know, why?', 'To get to the other side!'], got {response_1.content}"

        assert isinstance(h_responses, History)
        h_responses.shape == (
            2,
            3,
        ), f"Expected stacked shape (2, 3), got {h_responses.shape}"

        # Extract individual responses for testing
        response_0 = h_responses[0]
        response_1 = h_responses[1]

        # Test chat template application
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")

        # Test first response (should only show the assistant message, ignore padding)
        template_0 = response_0.apply_chat_template(
            tokenizer=tokenizer, add_generation_prompt=False, chat_template_name="qwen"
        )
        expected_0 = """<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>assistant
I'm doing well, thank you for asking!<|im_end|>
    """
        assert template_0 == expected_0

        # Test second response (should show all 3 messages)
        template_1 = response_1.apply_chat_template(
            tokenizer=tokenizer, add_generation_prompt=False, chat_template_name="qwen"
        )
        expected_1 = """<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>assistant
Why did the chicken cross the road?<|im_end|>
    <|im_start|>user
I don't know, why?<|im_end|>
<|im_start|>assistant
To get to the other side!<|im_end|>
    """
        assert template_1 == expected_1


class TestTopK:
    @pytest.mark.parametrize("per_token_reward", [True, False])
    def test_topk(self, per_token_reward):
        rb = ReplayBuffer(
            storage=LazyStackStorage(50),
            sampler=SamplerWithoutReplacement,
            batch_size=5,
        )

        def _per_token_reward(i):
            if per_token_reward:
                return torch.full((i + 5, 1), i)
            else:
                return torch.full((1, 1), i)

        td = lazy_stack(
            [
                TensorDict(
                    {
                        ("next", "done"): torch.full((1, 1), True),
                        ("next", "reward"): _per_token_reward(i),
                        # total of 10 dialogs per prompt
                        ("text", "prompt"): f"Prompt {i // 5}",
                    }
                )
                for i in range(50)
            ]
        )
        topk = TopKRewardSelector(total_dialog_turns=5, topk_size=3)
        rb.append_transform(topk)
        for _td in td.chunk(25):
            rb.extend(_td)
        # Only wrote top3 of 50 items in 10 groups of 5
        #  Because we only write items that are strictly greater than the median,
        #  only 20 items are written.
        assert rb.write_count == 20
        assert len(rb) == 20
        r3 = rb[:2].get(("next", "reward"), as_padded_tensor=True).squeeze()
        # 0 and 1 are missing because they're not part of the top-k
        if per_token_reward:
            assert (
                r3
                == torch.tensor(
                    [
                        [4, 4, 4, 4, 4, 4, 4, 4, 4],
                        [3, 3, 3, 3, 3, 3, 3, 3, 0],
                    ]
                )
            ).all()
        else:
            assert (r3 == torch.tensor([[4, 3]])).all()


if __name__ == "__main__":
    args, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst"] + unknown)
