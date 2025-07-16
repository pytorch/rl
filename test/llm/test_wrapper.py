# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import argparse
import importlib.util

import os
from functools import partial

import pytest
import torch
from tensordict import assert_close, lazy_stack, set_list_to_stack, TensorDict

from tensordict.utils import _zip_strict
from torchrl.data.llm import History
from torchrl.envs.llm.transforms.kl import KLComputation, RetrieveKL, RetrieveLogProb
from torchrl.modules.llm.policies.common import (
    ChatHistory,
    LogProbs,
    Masks,
    Text,
    Tokens,
)
from torchrl.modules.llm.policies.transformers_wrapper import TransformersWrapper
from torchrl.modules.llm.policies.vllm_wrapper import vLLMWrapper
from transformers import AutoTokenizer


# Set environment variable for vLLM V0 engine
os.environ["VLLM_USE_V1"] = "0"

_has_transformers = importlib.util.find_spec("transformers") is not None
_has_vllm = importlib.util.find_spec("vllm") is not None
_has_datasets = importlib.util.find_spec("datasets") is not None

TransformersWrapperMaxTokens = partial(
    TransformersWrapper, generate_kwargs={"max_new_tokens": 10, "do_sample": True}
)


@pytest.fixture(scope="function", autouse=True)
def set_seed():
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    yield


@pytest.fixture(scope="module", autouse=True)
def set_list_to_stack_fixture():
    with set_list_to_stack(True):
        yield


@pytest.fixture(scope="module")
def vllm_instance():
    """Create vLLM model and tokenizer for testing."""
    if not _has_vllm:
        pytest.skip("vllm not available")

    import vllm.envs as envs
    from vllm import LLM

    envs.VLLM_HOST_IP = "0.0.0.0" or "127.0.0.1"

    assert os.environ.get("VLLM_USE_V1") == "0"

    try:
        model = LLM("Qwen/Qwen2.5-0.5B")
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
        tokenizer.pad_token = tokenizer.eos_token
        return model, tokenizer
    except Exception as e:
        pytest.skip(f"Failed to load vLLM model: {e}")


@pytest.fixture(scope="module")
def transformers_instance():
    """Create transformers model and tokenizer for testing."""
    if not _has_transformers:
        pytest.skip("transformers not available")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


@pytest.fixture
def sample_history():
    """Create sample conversation history for testing."""
    chats = [
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Are you happy? Say yes or no."},
        ],
        [
            {
                "role": "system",
                "content": "You are a very helpful assistant, but more handsome.",
            },
            {
                "role": "user",
                "content": "Explain the difference between a cat and a dog. Be very detailed.",
            },
        ],
    ]
    return History.from_chats(chats)


@pytest.fixture
def sample_history_assistant():
    """Create sample conversation history for testing."""
    chats = [
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Are you happy? Say yes or no."},
            {"role": "assistant", "content": "Yes."},
        ],
        [
            {
                "role": "system",
                "content": "You are a very helpful assistant, but more handsome.",
            },
            {
                "role": "user",
                "content": "Explain the difference between a cat and a dog. Be very detailed.",
            },
            {
                "role": "assistant",
                "content": "A cat is a small animal that meows, while a dog is a larger animal that barks.",
            },
        ],
    ]
    return History.from_chats(chats)


@pytest.fixture
def sample_text():
    """Create sample text for testing."""
    return [
        "Are you happy? Say yes or no.",
        "Explain the difference between a cat and a dog. Be very detailed.",
    ]


@pytest.fixture
def sample_tokens(vllm_instance):
    """Create sample tokens for testing."""
    model, tokenizer = vllm_instance
    text = [
        "Are you happy? Say yes or no.",
        "Explain the difference between a cat and a dog. Be very detailed.",
    ]
    tokenized = tokenizer(text, return_tensors="pt", padding=True, padding_side="left")
    return tokenized["input_ids"], tokenized["attention_mask"]


@pytest.fixture
def sample_tokens_unpadded(vllm_instance):
    """Create sample tokens for testing."""
    model, tokenizer = vllm_instance
    text = [
        "Are you happy? Say yes or no.",
        "Explain the difference between a cat and a dog. Be very detailed.",
    ]
    tokenized = tokenizer(text, padding=False)
    return torch.nested.nested_tensor(
        [torch.tensor(t) for t in tokenized["input_ids"]], layout=torch.jagged
    ), torch.nested.nested_tensor(
        [torch.tensor(t) for t in tokenized["attention_mask"]], layout=torch.jagged
    )


def check_output_shapes(out, pad_output, requested_log_probs=False):
    if pad_output:
        # We can get all tensors or they are none
        log_probs = out.get("log_probs")
        masks = out.get("masks")
        tokens = out.get("tokens")
        text = out.get("text")
        history = out.get("history")

        # Test the all_ tensors
        if log_probs is not None:
            assert isinstance(log_probs, LogProbs)
            all_logprobs = log_probs.full
        else:
            all_logprobs = None
        if masks is not None:
            assert isinstance(masks, Masks)
            all_attention_masks = masks.all_attention_mask
            all_assistant_masks = masks.all_assistant_mask
        else:
            all_attention_masks = None
            all_assistant_masks = None
        if tokens is not None:
            assert isinstance(tokens, Tokens)
            all_tokens = tokens.full
        else:
            all_tokens = None
        if text is not None:
            assert isinstance(text, Text)
            text.full
        else:
            pass
        if history is not None:
            assert isinstance(history, ChatHistory)
            history.full
        else:
            pass

        shapes = set()
        if all_logprobs is not None:
            shapes.add(all_logprobs.shape)
        if all_attention_masks is not None:
            shapes.add(all_attention_masks.shape)
        if all_assistant_masks is not None:
            shapes.add(all_assistant_masks.shape)
        if all_tokens is not None:
            shapes.add(all_tokens.shape)
        assert len(shapes) <= 1, ("all_tensors shapes differ", out)

        # Check the response tensors
        shapes = set()
        if log_probs is not None and log_probs.response is not None:
            shapes.add(log_probs.response.shape)
        if tokens is not None and tokens.response is not None:
            shapes.add(tokens.response.shape)
        assert len(shapes) <= 1, (shapes, out)

        # Check the prompt tensors
        shapes = set()
        if log_probs is not None and log_probs.prompt is not None:
            shapes.add(log_probs.prompt.shape)
        if tokens is not None and tokens.prompt is not None:
            shapes.add(tokens.prompt.shape)

        if (
            log_probs is not None
            and log_probs.response is not None
            and log_probs.prompt is not None
        ):
            assert (
                log_probs.response.shape[-1] + log_probs.prompt.shape[-1]
                == log_probs.full.shape[-1]
            )
        if (
            tokens is not None
            and tokens.response is not None
            and tokens.prompt is not None
        ):
            assert (
                tokens.response.shape[-1] + tokens.prompt.shape[-1]
                == tokens.full.shape[-1]
            )

        assert len(shapes) <= 1, shapes

        # Check that if 'full' is defined, either both 'prompt' and 'response' must be set or neither of them
        if requested_log_probs:
            for obj_name, obj in [
                ("log_probs", log_probs),
                ("tokens", tokens),
                ("text", text),
            ]:
                if obj is not None and obj.get("full", as_list=True) is not None:
                    has_prompt = obj.get("prompt", as_list=True) is not None
                    has_response = obj.get("response", as_list=True) is not None
                    assert (has_prompt and has_response) or (
                        not has_prompt and not has_response
                    ), (
                        f"{obj_name}: if 'full' is defined, either both 'prompt' and 'response' must be set or neither of them. "
                        f"prompt={has_prompt}, response={has_response}, full={obj.full is not None}"
                    )
    else:
        # we can simply iterate over out
        for _out in out.unbind(0):
            check_output_shapes(
                _out, pad_output=not _out.ndim, requested_log_probs=requested_log_probs
            )


@pytest.mark.skipif(not _has_vllm, reason="vllm not available")
class TestWrappers:
    """Comprehensive tests for vLLMWrapper and TransformersWrapper covering all modalities and configurations."""

    # ================================================
    # History Input Mode Tests
    # ================================================

    @pytest.mark.parametrize(
        "wrapper_class",
        [vLLMWrapper, TransformersWrapperMaxTokens],
        ids=["vllm", "transformers"],
    )
    @pytest.mark.parametrize("generate", [True, False], ids=["generate", "no_generate"])
    @pytest.mark.parametrize("pad_output", [True, False], ids=["padded", "unpadded"])
    def test_history_input_mode(
        self,
        wrapper_class,
        vllm_instance,
        transformers_instance,
        sample_history,
        sample_history_assistant,
        generate,
        pad_output,
    ):
        """Test history input mode with various configurations."""

        if wrapper_class == vLLMWrapper:
            model, tokenizer = vllm_instance
        else:
            model, tokenizer = transformers_instance
        wrapper = wrapper_class(
            model,
            tokenizer=tokenizer,
            input_mode="history",
            generate=generate,
            pad_output=pad_output,
        )

        # Check input keys
        assert (
            wrapper.in_keys == [("history", "prompt")]
            if generate
            else [("history", "full")]
        )

        # Check output keys - always return everything
        expected_out_keys = ["text", "masks", "tokens", "log_probs", "history"]
        assert wrapper.out_keys == expected_out_keys

        # Create input data
        if generate:
            data = TensorDict(
                history=ChatHistory(prompt=sample_history),
                batch_size=(2,),
            )
        else:
            data = TensorDict(
                history=ChatHistory(full=sample_history_assistant),
                batch_size=(2,),
            )

        # Run wrapper
        result = wrapper(data)
        check_output_shapes(result, pad_output, requested_log_probs=not generate)

        # Check output structure
        for key in expected_out_keys:
            assert key in result
            assert hasattr(result[key], "__class__")

        # Check specific outputs - always check everything
        text_obj = result["text"]
        assert hasattr(text_obj, "prompt")
        assert hasattr(text_obj, "response")
        assert hasattr(text_obj, "full")

        if generate:
            assert text_obj.response is not None
            assert isinstance(text_obj.response, list)
            assert isinstance(text_obj.response[0], str)

        tokens_obj = result["tokens"]
        if pad_output:
            assert hasattr(tokens_obj, "prompt")
            assert hasattr(tokens_obj, "response")
            assert hasattr(tokens_obj, "full")
            assert hasattr(tokens_obj, "padded")
        assert all(tokens_obj.padded) == pad_output

        if generate:
            if pad_output:
                assert tokens_obj.response is not None
            else:
                assert tokens_obj.get("response", as_list=True) is not None
            if not pad_output:
                response_tokens = result["tokens"].get("response", as_list=True)
                assert isinstance(response_tokens, list)
            else:
                assert isinstance(tokens_obj.response, torch.Tensor)

        masks_obj = result["masks"]
        if pad_output:
            assert hasattr(masks_obj, "all_attention_mask")
            assert hasattr(masks_obj, "all_assistant_mask")
            assert hasattr(masks_obj, "padded")
        assert all(masks_obj.padded) == pad_output

        log_probs_obj = result["log_probs"]
        if pad_output:
            assert hasattr(log_probs_obj, "prompt")
            assert hasattr(log_probs_obj, "response")
            assert hasattr(log_probs_obj, "full")
            assert hasattr(log_probs_obj, "padded")
        assert all(log_probs_obj.padded) == pad_output

    # ================================================
    # Text Input Mode Tests
    # ================================================

    @pytest.mark.parametrize(
        "wrapper_class",
        [vLLMWrapper, TransformersWrapperMaxTokens],
        ids=["vllm", "transformers"],
    )
    @pytest.mark.parametrize("generate", [True, False], ids=["generate", "no_generate"])
    @pytest.mark.parametrize("pad_output", [True, False], ids=["padded", "unpadded"])
    def test_text_input_mode(
        self,
        wrapper_class,
        vllm_instance,
        transformers_instance,
        sample_text,
        generate,
        pad_output,
    ):
        """Test text input mode with various configurations."""
        model, tokenizer = vllm_instance

        if wrapper_class == vLLMWrapper:
            model, tokenizer = vllm_instance
        else:
            model, tokenizer = transformers_instance
        wrapper = wrapper_class(
            model,
            tokenizer=tokenizer,
            input_mode="text",
            generate=generate,
            pad_output=pad_output,
        )

        # Check input keys
        if generate:
            assert wrapper.in_keys == [("text", "prompt")]
        else:
            assert wrapper.in_keys == [("text", "full")]

        # Create input data
        if generate:
            data = TensorDict(text=Text(prompt=sample_text), batch_size=(2,))
        else:
            data = TensorDict(text=Text(full=sample_text), batch_size=(2,))

        # Run wrapper
        result = wrapper(data)
        check_output_shapes(result, pad_output, requested_log_probs=not generate)

        # Check output structure - always return everything
        expected_keys = ["text", "masks", "tokens", "log_probs"]
        for key in expected_keys:
            assert key in result

        # Check text output
        text_obj = result["text"]
        if generate:
            assert text_obj.prompt == sample_text
        else:
            assert text_obj.full == sample_text
        if generate:
            assert text_obj.response is not None

        # Check tokens output
        tokens_obj = result["tokens"]
        if generate:
            if not pad_output:
                response_tokens = tokens_obj.get("response", as_list=True)
                assert isinstance(tokens_obj.get("response", as_list=True), list)
            else:
                assert isinstance(tokens_obj.response, torch.Tensor)

    # ================================================
    # Tokens Input Mode Tests
    # ================================================

    @pytest.mark.parametrize(
        "wrapper_class",
        [vLLMWrapper, TransformersWrapperMaxTokens],
        ids=["vllm", "transformers"],
    )
    @pytest.mark.parametrize("generate", [True, False], ids=["generate", "no_generate"])
    @pytest.mark.parametrize("pad_output", [True, False], ids=["padded", "unpadded"])
    def test_tokens_input_mode(
        self,
        wrapper_class,
        vllm_instance,
        transformers_instance,
        sample_tokens,
        generate,
        pad_output,
    ):
        """Test tokens input mode with various configurations."""
        if wrapper_class == vLLMWrapper:
            model, tokenizer = vllm_instance
        else:
            model, tokenizer = transformers_instance

        input_ids, attention_mask = sample_tokens

        wrapper = wrapper_class(
            model,
            tokenizer=tokenizer,
            input_mode="tokens",
            attention_mask_key="attention_mask",
            generate=generate,
            pad_output=pad_output,
        )

        # Check input keys
        assert (
            wrapper.in_keys == [("tokens", "prompt")]
            if generate
            else [("tokens", "full")]
        )

        # Create input data
        data = TensorDict(
            tokens=Tokens(prompt=input_ids) if generate else Tokens(full=input_ids),
            attention_mask=attention_mask,
            batch_size=(2,),
        )

        # Run wrapper
        result = wrapper(data)
        check_output_shapes(result, pad_output, requested_log_probs=not generate)

        # Check output structure
        expected_keys = ["masks", "tokens", "log_probs"]
        for key in expected_keys:
            assert key in result

        # Check tokens output
        tokens_obj = result["tokens"]
        if generate:
            if not pad_output:
                response_tokens = result["tokens"].get("response", as_list=True)
                assert isinstance(response_tokens, list)
            else:
                assert isinstance(tokens_obj.response, torch.Tensor)

    # ================================================
    # Error Handling Tests
    # ================================================

    @pytest.mark.parametrize(
        "wrapper_class",
        [vLLMWrapper, TransformersWrapperMaxTokens],
        ids=["vllm", "transformers"],
    )
    def test_invalid_input_mode(
        self, wrapper_class, vllm_instance, transformers_instance
    ):
        """Test that invalid input_mode raises an error."""
        if wrapper_class == vLLMWrapper:
            model, tokenizer = vllm_instance
        else:
            model, tokenizer = transformers_instance

        with pytest.raises(ValueError, match="input_mode must be one of"):
            wrapper_class(
                model,
                tokenizer=tokenizer,
                input_mode="invalid_mode",
            )

    @pytest.mark.parametrize(
        "wrapper_class",
        [vLLMWrapper, TransformersWrapperMaxTokens],
        ids=["vllm", "transformers"],
    )
    def test_missing_input_key(
        self, wrapper_class, vllm_instance, transformers_instance, sample_history
    ):
        """Test that missing input key raises an error."""
        if wrapper_class == vLLMWrapper:
            model, tokenizer = vllm_instance
        else:
            model, tokenizer = transformers_instance

        wrapper = wrapper_class(
            model,
            tokenizer=tokenizer,
            input_mode="history",
            input_key="history",
        )

        # Create data without the required key
        data = TensorDict(batch_size=(2,))

        with pytest.raises(ValueError, match="Expected 'history' key"):
            wrapper(data)

    @pytest.mark.parametrize(
        "wrapper_class",
        [vLLMWrapper, TransformersWrapperMaxTokens],
        ids=["vllm", "transformers"],
    )
    def test_invalid_history_type(
        self, wrapper_class, vllm_instance, transformers_instance
    ):
        """Test that invalid history type raises an error."""
        if wrapper_class == vLLMWrapper:
            model, tokenizer = vllm_instance
        else:
            model, tokenizer = transformers_instance

        wrapper = wrapper_class(
            model,
            tokenizer=tokenizer,
            input_mode="history",
        )

        # Create data with wrong type
        data = TensorDict(
            history=ChatHistory(prompt="not a history object"), batch_size=(2,)
        )

        with pytest.raises(TypeError, match="Expected History object"):
            wrapper(data)

    @pytest.mark.parametrize(
        "wrapper_class",
        [vLLMWrapper, TransformersWrapperMaxTokens],
        ids=["vllm", "transformers"],
    )
    def test_generate_false_without_log_probs(
        self, wrapper_class, vllm_instance, transformers_instance
    ):
        """Test that generate=False without return_log_probs=True raises an error."""
        if wrapper_class == vLLMWrapper:
            model, tokenizer = vllm_instance
        else:
            model, tokenizer = transformers_instance

        with pytest.raises(ValueError, match="return_log_probs must be True"):
            wrapper_class(
                model,
                tokenizer=tokenizer,
                generate=False,
                return_log_probs=False,
            )

    # ================================================
    # Batch Size Tests
    # ================================================

    @pytest.mark.parametrize(
        "batch_size", [1, 2, 3], ids=["batch_size_1", "batch_size_2", "batch_size_3"]
    )
    @pytest.mark.parametrize("pad_output", [True, False], ids=["padded", "unpadded"])
    @pytest.mark.parametrize(
        "wrapper_class",
        [vLLMWrapper, TransformersWrapperMaxTokens],
        ids=["vllm", "transformers"],
    )
    def test_batch_sizes(
        self,
        wrapper_class,
        vllm_instance,
        transformers_instance,
        batch_size,
        pad_output,
    ):
        """Test wrapper with different batch sizes."""
        if wrapper_class == vLLMWrapper:
            model, tokenizer = vllm_instance
        else:
            model, tokenizer = transformers_instance

        # Create history with specified batch size
        chats = [
            [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"Question {i}?"},
            ]
            for i in range(batch_size)
        ]
        history = History.from_chats(chats)

        wrapper = wrapper_class(
            model,
            tokenizer=tokenizer,
            input_mode="history",
            generate=True,
            return_log_probs=True,
            pad_output=pad_output,
        )

        data = TensorDict(history=ChatHistory(prompt=history), batch_size=(batch_size,))
        result = wrapper(data)
        check_output_shapes(
            result, pad_output=wrapper.pad_output, requested_log_probs=False
        )

        # Check that all expected keys are present
        expected_keys = ["text", "masks", "tokens", "log_probs"]
        for key in expected_keys:
            assert key in result

        # Check batch size consistency
        if pad_output:
            # For padded output, tensors should have the correct batch dimension
            assert len(result["text"].response) == batch_size
            assert len(result["tokens"].response) == batch_size
        else:
            # For unpadded output, use as_list=True to get lists
            response_text = result["text"].get("response", as_list=True)
            response_tokens = result["tokens"].get("response", as_list=True)
            assert len(response_text) == batch_size
            assert len(response_tokens) == batch_size

    # ================================================
    # Custom Input Key Tests
    # ================================================

    @pytest.mark.parametrize(
        "wrapper_class",
        [vLLMWrapper, TransformersWrapperMaxTokens],
        ids=["vllm", "transformers"],
    )
    def test_custom_input_key(
        self, wrapper_class, vllm_instance, transformers_instance, sample_history
    ):
        """Test wrapper with custom input key."""
        if wrapper_class == vLLMWrapper:
            model, tokenizer = vllm_instance
        else:
            model, tokenizer = transformers_instance

        wrapper = wrapper_class(
            model,
            tokenizer=tokenizer,
            input_mode="history",
            input_key=("custom_history_key", "prompt"),
            generate=True,
            return_log_probs=True,
        )

        # Check input keys
        assert wrapper.in_keys == [("custom_history_key", "prompt")]

        # Create data with custom key
        data = TensorDict(
            custom_history_key=ChatHistory(prompt=sample_history), batch_size=(2,)
        )
        result = wrapper(data)
        check_output_shapes(
            result, pad_output=wrapper.pad_output, requested_log_probs=False
        )

        # Check that wrapper works correctly
        expected_keys = ["text", "masks", "tokens", "log_probs"]
        for key in expected_keys:
            assert key in result

    # ================================================
    # Selective Output Tests
    # ================================================

    @pytest.mark.parametrize(
        "return_log_probs", [True, False], ids=["log_probs", "no_log_probs"]
    )
    @pytest.mark.parametrize(
        "wrapper_class",
        [vLLMWrapper, TransformersWrapperMaxTokens],
        ids=["vllm", "transformers"],
    )
    def test_selective_outputs(
        self,
        wrapper_class,
        vllm_instance,
        transformers_instance,
        sample_history,
        return_log_probs,
    ):
        """Test wrapper with selective output configurations."""
        if wrapper_class == vLLMWrapper:
            model, tokenizer = vllm_instance
        else:
            model, tokenizer = transformers_instance

        wrapper = wrapper_class(
            model,
            tokenizer=tokenizer,
            input_mode="history",
            generate=True,
            return_log_probs=return_log_probs,
        )

        # Check output keys
        expected_out_keys = []
        if wrapper.return_text:
            expected_out_keys.append("text")
        if wrapper.return_masks:
            expected_out_keys.append("masks")
        if wrapper.return_tokens:
            expected_out_keys.append("tokens")
        if return_log_probs:
            expected_out_keys.append("log_probs")
        if wrapper.return_history:
            expected_out_keys.append("history")

        assert wrapper.out_keys == expected_out_keys

        # Run wrapper
        data = TensorDict(history=ChatHistory(prompt=sample_history), batch_size=(2,))
        result = wrapper(data)
        check_output_shapes(
            result, pad_output=wrapper.pad_output, requested_log_probs=False
        )

        # Check that only expected keys are present
        for key in expected_out_keys:
            assert key in result

        # Check that unexpected keys are not present
        all_possible_keys = ["text", "masks", "tokens", "log_probs"]
        for key in all_possible_keys:
            if key not in expected_out_keys:
                assert key not in result

    # ================================================
    # Log-probs Only Mode Tests
    # ================================================

    @pytest.mark.parametrize(
        "wrapper_class",
        [vLLMWrapper, TransformersWrapperMaxTokens],
        ids=["vllm", "transformers"],
    )
    def test_log_probs_only_mode(
        self,
        wrapper_class,
        vllm_instance,
        transformers_instance,
        sample_history_assistant,
    ):
        """Test wrapper in log-probs only mode (generate=False)."""
        if wrapper_class == vLLMWrapper:
            model, tokenizer = vllm_instance
        else:
            model, tokenizer = transformers_instance

        wrapper = wrapper_class(
            model,
            tokenizer=tokenizer,
            input_mode="history",
            generate=False,  # Only compute log-probs
            return_log_probs=True,  # Must be True when generate=False
        )

        data = TensorDict(
            history=ChatHistory(full=sample_history_assistant), batch_size=(2,)
        )
        result = wrapper(data)
        check_output_shapes(
            result, pad_output=wrapper.pad_output, requested_log_probs=True
        )

        # Check that log_probs are present
        assert "log_probs" in result

        # Check that response_text is None (no generation)
        assert result["text"].response is None

        # Check that prompt_logprobs are present
        log_probs_obj = result["log_probs"]
        assert log_probs_obj.get("full", as_list=True) is not None

    # ================================================
    # TensorClass Structure Tests
    # ================================================

    @pytest.mark.parametrize(
        "wrapper_class",
        [vLLMWrapper, TransformersWrapperMaxTokens],
        ids=["vllm", "transformers"],
    )
    def test_tensorclass_structure(
        self, wrapper_class, vllm_instance, transformers_instance, sample_history
    ):
        """Test that TensorClass objects have the correct structure."""
        if wrapper_class == vLLMWrapper:
            model, tokenizer = vllm_instance
        else:
            model, tokenizer = transformers_instance
        pad_output = False

        wrapper = wrapper_class(
            model,
            tokenizer=tokenizer,
            input_mode="history",
            generate=True,
            return_log_probs=True,
        )

        data = TensorDict(history=ChatHistory(prompt=sample_history), batch_size=(2,))
        result = wrapper(data)

        # Test Text TensorClass
        text_obj = result["text"]
        assert hasattr(text_obj, "prompt")
        assert hasattr(text_obj, "response")
        assert hasattr(text_obj, "full")

        # Test Tokens TensorClass
        tokens_obj = result["tokens"]
        if pad_output:
            # if not padded, we will fail to stack
            assert hasattr(tokens_obj, "prompt")
            assert hasattr(tokens_obj, "response")
            assert hasattr(tokens_obj, "full")
            assert hasattr(tokens_obj, "padded")
        else:
            assert (
                tokens_obj.get("response", as_list=True) is not None
            )  # if not padded, we will fail to stack

        # Test LogProbs TensorClass
        log_probs_obj = result["log_probs"]
        if pad_output:
            # if not padded, we will fail to stack
            assert hasattr(log_probs_obj, "prompt")
            assert hasattr(log_probs_obj, "response")
            assert hasattr(log_probs_obj, "full")
            assert hasattr(log_probs_obj, "padded")
        else:
            assert (
                log_probs_obj.get("response", as_list=True) is not None
            )  # if not padded, we will fail to stack

        # Test Masks TensorClass
        masks_obj = result["masks"]
        if pad_output:
            # if not padded, we will fail to stack
            assert hasattr(masks_obj, "all_attention_mask")
            assert hasattr(masks_obj, "all_assistant_mask")
            assert hasattr(masks_obj, "padded")

    # ================================================
    # Unpadded Output Tests (with as_list=True)
    # ================================================

    @pytest.mark.parametrize(
        "wrapper_class",
        [vLLMWrapper, TransformersWrapperMaxTokens],
        ids=["vllm", "transformers"],
    )
    def test_unpadded_output_with_as_list(
        self, wrapper_class, vllm_instance, transformers_instance, sample_history
    ):
        """Test unpadded output using as_list=True to avoid stacking issues."""
        if wrapper_class == vLLMWrapper:
            model, tokenizer = vllm_instance
        else:
            model, tokenizer = transformers_instance

        wrapper = wrapper_class(
            model,
            tokenizer=tokenizer,
            input_mode="history",
            generate=True,
            return_log_probs=True,
            pad_output=False,  # Unpadded output
        )

        data = TensorDict(history=ChatHistory(prompt=sample_history), batch_size=(2,))
        result = wrapper(data)
        check_output_shapes(
            result, pad_output=wrapper.pad_output, requested_log_probs=False
        )

        # Use as_list=True to get lists instead of trying to stack
        text_list = result.get("text", as_list=True)
        tokens_list = result.get("tokens", as_list=True)
        masks_list = result.get("masks", as_list=True)
        log_probs_list = result.get("log_probs", as_list=True)

        # Check that we get lists
        assert isinstance(text_list.response, list)
        assert isinstance(tokens_list.get("response", as_list=True), list)
        assert isinstance(log_probs_list.get("response", as_list=True), list)

        # Check list lengths
        assert len(text_list.response) == 2
        assert len(tokens_list.get("response", as_list=True)) == 2
        assert len(log_probs_list.get("response", as_list=True)) == 2

        # Check that individual elements are tensors
        assert isinstance(text_list.response[0], str)
        assert isinstance(tokens_list.get("response", as_list=True)[0], torch.Tensor)
        assert isinstance(log_probs_list.get("response", as_list=True)[0], torch.Tensor)

    @pytest.mark.parametrize("num_samples", [2], ids=["num_samples_2"])
    @pytest.mark.parametrize("pad_output", [True, False], ids=["padded", "unpadded"])
    @pytest.mark.parametrize(
        "return_log_probs", [True, False], ids=["log_probs", "no_log_probs"]
    )
    @pytest.mark.parametrize(
        "input_mode", ["history", "text", "tokens"], ids=["history", "text", "tokens"]
    )
    @pytest.mark.parametrize(
        "wrapper_class",
        [vLLMWrapper, TransformersWrapperMaxTokens],
        ids=["vllm", "transformers"],
    )
    def test_num_samples(
        self,
        wrapper_class,
        vllm_instance,
        transformers_instance,
        sample_history,
        sample_text,
        sample_tokens,
        num_samples,
        pad_output,
        return_log_probs,
        input_mode,
    ):
        """Test wrapper with num_samples."""
        if wrapper_class == vLLMWrapper:
            model, tokenizer = vllm_instance
        else:
            model, tokenizer = transformers_instance

        wrapper = wrapper_class(
            model,
            tokenizer=tokenizer,
            input_mode=input_mode,
            generate=True,
            return_log_probs=return_log_probs,
            pad_output=pad_output,
            num_samples=num_samples,
        )
        if input_mode == "history":
            data = TensorDict(
                history=ChatHistory(prompt=sample_history), batch_size=(2,)
            )
        elif input_mode == "text":
            data = TensorDict(text=Text(prompt=sample_text), batch_size=(2,))
        elif input_mode == "tokens":
            data = TensorDict(tokens=Tokens(prompt=sample_tokens[0]), batch_size=(2,))
        else:
            raise ValueError(f"Invalid input mode: {input_mode}")
        result = wrapper(data)
        assert result.batch_size == (2, num_samples)
        check_output_shapes(
            result, pad_output=wrapper.pad_output, requested_log_probs=False
        )


class TestKLTransforms:
    """Comprehensive tests for KL-related transforms with different input modes and configurations."""

    @pytest.mark.skipif(not _has_transformers, reason="transformers not available")
    @pytest.mark.parametrize("pad_output", [True, False], ids=["padded", "unpadded"])
    @pytest.mark.parametrize(
        "assistant_only", [True, False], ids=["assistant_only", "all_tokens"]
    )
    @pytest.mark.parametrize(
        "input_mode", ["history", "text", "tokens"], ids=["history", "text", "tokens"]
    )
    def test_retrieve_log_prob_input_modes(
        self,
        transformers_instance,
        sample_history_assistant,
        sample_text,
        sample_tokens,
        pad_output,
        assistant_only,
        input_mode,
    ):
        """Test RetrieveLogProb with different input modes and assistant_only settings."""
        model, tokenizer = transformers_instance

        # Skip invalid combinations
        if assistant_only and input_mode != "history":
            pytest.skip("assistant_only=True requires input_mode='history'")

        # Create test data based on input mode
        if input_mode == "history":
            history = sample_history_assistant
            data = TensorDict(history=ChatHistory(full=history), batch_size=(2,))
        elif input_mode == "text":
            history = None  # Not used in text mode
            prompts = sample_text
            data = TensorDict(text=Text(full=prompts), batch_size=(2,))
        elif input_mode == "tokens":
            history = None  # Not used in tokens mode
            prompts = sample_tokens
            data = TensorDict(
                tokens=Tokens(full=prompts[0]),
                masks=Masks(all_attention_mask=prompts[1]),
                batch_size=(2,),
            )
        else:
            raise ValueError(f"Invalid input_mode: {input_mode}")

        # Create reference model with appropriate input mode
        ref_model = TransformersWrapper(
            model,
            tokenizer=tokenizer,
            input_mode=input_mode,
            generate=False,
            pad_output=pad_output,
        )

        # Create RetrieveLogProb transform
        transform = RetrieveLogProb(
            ref_model,
            assistant_only=assistant_only,
            tokenizer=tokenizer,
        )

        # Apply transform
        result = transform(data)

        # The log-probs key should be based on the model's log_probs_key
        log_probs_key = (ref_model.log_probs_key, "full")
        assert log_probs_key in result

        # Check log-probs structure
        if pad_output:
            log_probs = result.get(log_probs_key)
            assert isinstance(log_probs, torch.Tensor)
            assert log_probs.shape[0] == 2  # batch size
        else:
            # For unpadded output, we get a list of tensors
            log_probs = result.get(log_probs_key, as_list=True)
            assert isinstance(log_probs, list)
            assert len(log_probs) == 2  # batch size

    @pytest.mark.skipif(not _has_transformers, reason="transformers not available")
    @pytest.mark.parametrize("pad_output", [True, False], ids=["padded", "unpadded"])
    @pytest.mark.parametrize(
        "assistant_only", [True, False], ids=["assistant_only", "all_tokens"]
    )
    @pytest.mark.parametrize(
        "input_mode", ["history", "text", "tokens"], ids=["history", "text", "tokens"]
    )
    def test_retrieve_kl_input_modes(
        self,
        transformers_instance,
        sample_history_assistant,
        sample_text,
        sample_tokens,
        pad_output,
        assistant_only,
        input_mode,
    ):
        """Test RetrieveKL with different input modes and assistant_only settings."""
        model, tokenizer = transformers_instance

        # Skip invalid combinations
        if assistant_only and input_mode != "history":
            pytest.skip("assistant_only=True requires input_mode='history'")

        # Create test data based on input mode
        if input_mode == "history":
            history = sample_history_assistant
            data = TensorDict(history=ChatHistory(full=history), batch_size=(2,))
        elif input_mode == "text":
            history = None  # Not used in text mode
            prompts = sample_text
            data = TensorDict(text=Text(full=prompts), batch_size=(2,))
        elif input_mode == "tokens":
            history = None  # Not used in tokens mode
            prompts = sample_tokens
            data = TensorDict(
                tokens=Tokens(full=prompts[0]),
                masks=Masks(all_attention_mask=prompts[1]),
                batch_size=(2,),
            )
        else:
            raise ValueError(f"Invalid input_mode: {input_mode}")

        # Create generation and reference models with appropriate input mode
        gen_model = TransformersWrapper(
            model,
            tokenizer=tokenizer,
            input_mode=input_mode,
            generate=False,
            pad_output=pad_output,
            log_probs_key="gen_log_probs",
        )

        ref_model = TransformersWrapper(
            model,
            tokenizer=tokenizer,
            input_mode=input_mode,
            generate=False,
            pad_output=pad_output,
            log_probs_key="ref_log_probs",
        )

        # Create RetrieveKL transform
        transform = RetrieveKL(
            gen_model=gen_model,
            ref_model=ref_model,
            assistant_only=assistant_only,
            tokenizer=tokenizer,
            gen_log_probs_full_key=("gen_log_probs", "full"),
            ref_log_probs_full_key=("ref_log_probs", "full"),
        )

        # Apply transform
        data = data.to_lazystack(0)
        result = transform(data)

        # Check that KL is present
        # Check that both log-probs and KL are present
        assert ("gen_log_probs", "full") in result
        assert ("ref_log_probs", "full") in result
        assert "kl_penalty" in result

        # Check KL structure
        if pad_output:
            kl = result.get("kl_penalty")
            assert isinstance(kl, torch.Tensor)
            assert kl.shape[0] == 2  # batch size
        else:
            kl = result.get("kl_penalty", as_list=True)
            # For unpadded output, we get a list of tensors
            assert isinstance(kl, list)
            assert len(kl) == 2  # batch size

    @pytest.mark.skipif(not _has_transformers, reason="transformers not available")
    def test_retrieve_log_prob_assistant_only_validation(
        self, transformers_instance, sample_text
    ):
        """Test that assistant_only=True with non-history input_mode raises an error."""
        model, tokenizer = transformers_instance

        # Create reference model with text input mode
        ref_model = TransformersWrapper(
            model,
            tokenizer=tokenizer,
            input_mode="text",
            generate=False,
            return_log_probs=True,
            pad_output=True,
        )

        # This should raise an error
        with pytest.raises(
            ValueError, match="The model must have `input_mode='history'` when"
        ):
            RetrieveLogProb(
                ref_model,
                assistant_only=True,  # This should fail with text input_mode
                tokenizer=tokenizer,
            )

    @pytest.mark.skipif(not _has_transformers, reason="transformers not available")
    def test_retrieve_kl_assistant_only_validation(
        self, transformers_instance, sample_text
    ):
        """Test that assistant_only=True with non-history input_mode raises an error."""
        model, tokenizer = transformers_instance

        # Create models with text input mode
        gen_model = TransformersWrapper(
            model,
            tokenizer=tokenizer,
            input_mode="text",
            generate=False,
            return_log_probs=True,
            pad_output=True,
            log_probs_key="gen_log_probs",
        )

        ref_model = TransformersWrapper(
            model,
            tokenizer=tokenizer,
            input_mode="text",
            generate=False,
            return_log_probs=True,
            pad_output=True,
            log_probs_key="ref_log_probs",
        )

        # This should raise an error
        with pytest.raises(
            ValueError, match="The model must have `input_mode='history'` when"
        ):
            RetrieveKL(
                gen_model=gen_model,
                ref_model=ref_model,
                assistant_only=True,  # This should fail with text input_mode
                tokenizer=tokenizer,
            )

    @pytest.mark.skipif(not _has_transformers, reason="transformers not available")
    @pytest.mark.parametrize("pad_output", [True, False], ids=["padded", "unpadded"])
    def test_retrieve_kl_pad_output_consistency(
        self, transformers_instance, sample_history_assistant, pad_output
    ):
        """Test that RetrieveKL enforces pad_output consistency between models."""
        model, tokenizer = transformers_instance

        # Create models with different pad_output settings
        gen_model = TransformersWrapper(
            model,
            tokenizer=tokenizer,
            input_mode="history",
            generate=False,
            return_log_probs=True,
            pad_output=pad_output,
            log_probs_key="gen_log_probs",
        )

        ref_model = TransformersWrapper(
            model,
            tokenizer=tokenizer,
            input_mode="history",
            generate=False,
            return_log_probs=True,
            pad_output=not pad_output,  # Different pad_output setting
            log_probs_key="ref_log_probs",
        )

        # This should raise an error
        with pytest.raises(ValueError, match="pad_output mismatch"):
            RetrieveKL(
                gen_model=gen_model,
                ref_model=ref_model,
                assistant_only=False,
                tokenizer=tokenizer,
            )

    @pytest.mark.skipif(not _has_transformers, reason="transformers not available")
    @pytest.mark.parametrize("pad_output", [True, False], ids=["padded", "unpadded"])
    def test_kl_computation_transform(
        self, transformers_instance, sample_history_assistant, pad_output
    ):
        """Test the KLComputation transform directly."""
        model, tokenizer = transformers_instance

        # Create models
        gen_model = TransformersWrapper(
            model,
            tokenizer=tokenizer,
            input_mode="history",
            generate=False,
            return_log_probs=True,
            pad_output=pad_output,
            log_probs_key="gen_log_probs",
        )

        ref_model = TransformersWrapper(
            model,
            tokenizer=tokenizer,
            input_mode="history",
            generate=False,
            return_log_probs=True,
            pad_output=pad_output,
            log_probs_key="ref_log_probs",
        )

        # Create data
        data = TensorDict(
            history=ChatHistory(full=sample_history_assistant), batch_size=(2,)
        )

        # Get log-probs from both models
        data = data.to_lazystack(0)
        gen_result = gen_model(data)
        ref_result = ref_model(data)

        # Create next tensordict with log-probs and reward
        next_td = TensorDict(batch_size=(2,)).to_lazystack(0)
        next_td.update(gen_result, keys_to_update=[("gen_log_probs", "full")])
        next_td.update(ref_result, keys_to_update=[("ref_log_probs", "full")])
        next_td.update({"reward": torch.randn(2, 1, 1)})

        # Create KLComputation transform
        kl_transform = KLComputation(
            gen_log_probs_full_key=("gen_log_probs", "full"),
            ref_log_probs_full_key=("ref_log_probs", "full"),
            kl_key="kl",
            add_to_reward=True,
            coeff=1.0,
        )

        # Apply transform
        result = kl_transform(data.set("next", next_td))

        # Check that KL is computed
        result = result["next"]
        assert "kl" in result

        if pad_output:
            kl = result.get("kl")
            assert isinstance(kl, torch.Tensor)
            assert kl.shape[0] == 2  # batch size
        else:
            kl = result.get("kl", as_list=True)
            assert isinstance(kl, list)
            assert len(kl) == 2  # batch size

        # Check that reward is modified
        assert "reward" in result
        reward = result.get("reward", as_list=True)
        assert reward is not None


class TestLogProbsComparison:
    """Test log-probability consistency between vLLM and Transformers wrappers."""

    @pytest.mark.skipif(not _has_vllm, reason="vllm not available")
    @pytest.mark.skipif(not _has_transformers, reason="transformers not available")
    @pytest.mark.parametrize(
        "input_mode", ["history", "text", "tokens"], ids=["history", "text", "tokens"]
    )
    @pytest.mark.parametrize("pad_output", [True, False], ids=["padded", "unpadded"])
    def test_log_probs_consistency(
        self,
        vllm_instance,
        transformers_instance,
        input_mode,
        pad_output,
        sample_history,
        sample_text,
        sample_tokens,
    ):
        """Test that log-probabilities are consistent between vLLM and Transformers wrappers."""
        vllm_model, vllm_tokenizer = vllm_instance
        tf_model, tf_tokenizer = transformers_instance

        # Create test data based on input mode
        if input_mode == "history":
            history = sample_history
            data = TensorDict(history=history, batch_size=(2,))
            input_key = "history"
        elif input_mode == "text":
            history = None  # Not used in text mode
            prompts = sample_text
            data = TensorDict(text=prompts, batch_size=(2,))
            input_key = "text"
        elif input_mode == "tokens":
            history = None  # Not used in tokens mode
            prompts = sample_tokens
            data = TensorDict(
                input_ids=prompts[0],
                attention_mask=prompts[1],
                batch_size=(2,),
            )
            input_key = "input_ids"
        else:
            raise ValueError(f"Invalid input_mode: {input_mode}")

        # Create vLLM wrapper for generation
        vllm_gen_wrapper = vLLMWrapper(
            vllm_model,
            tokenizer=vllm_tokenizer,
            input_mode=input_mode,
            input_key=input_key,
            generate=True,
            pad_output=pad_output,
            generate_kwargs={"max_tokens": 5, "temperature": 0.0},  # Deterministic
        )

        # Create Transformers wrapper for generation
        tf_gen_wrapper = TransformersWrapper(
            tf_model,
            tokenizer=tf_tokenizer,
            input_mode=input_mode,
            input_key=input_key,
            generate=True,
            pad_output=pad_output,
            generate_kwargs={
                "max_new_tokens": 5,
                "do_sample": False,
                "temperature": 0.0,
            },  # Deterministic
        )

        # Step 1: Generate tokens with both wrappers
        vllm_gen_result = vllm_gen_wrapper(data.copy())
        tf_gen_wrapper(data.copy())

        # Step 2: Extract generated tokens and create new input for log-probs computation
        if input_mode == "history":
            # For history mode, we need to create new history with generated responses
            generated_texts = vllm_gen_result["text"].response
            new_chats = []
            assert history is not None  # Type assertion for linter
            for chat, gen_text in _zip_strict(history.unbind(0), generated_texts):
                new_chat = chat.copy().append(
                    History(role="assistant", content=gen_text)
                )
                new_chats.append(new_chat)
            new_history = lazy_stack(new_chats)
            new_data = TensorDict(history=new_history, batch_size=(2,))
        elif input_mode == "text":
            # For text mode, concatenate original text with generated text
            original_texts = data["text"]
            generated_texts = vllm_gen_result["text"].response
            new_texts = [
                orig + gen for orig, gen in zip(original_texts, generated_texts)
            ]
            new_data = TensorDict(text=new_texts, batch_size=(2,))
        elif input_mode == "tokens":
            # For tokens mode, concatenate original tokens with generated tokens
            original_tokens = data["input_ids"]
            generated_tokens = vllm_gen_result["tokens"].response
            if pad_output:
                # Remove padding from generated tokens
                mask = generated_tokens != vllm_tokenizer.pad_token_id
                new_tokens = []
                for i in range(len(original_tokens)):
                    valid_tokens = generated_tokens[i][mask[i]]
                    combined = torch.cat([original_tokens[i], valid_tokens])
                    new_tokens.append(combined)
                new_tokens = torch.stack(new_tokens)
            else:
                new_tokens = []
                for i in range(len(original_tokens)):
                    combined = torch.cat([original_tokens[i], generated_tokens[i]])
                    new_tokens.append(combined)
            new_data = TensorDict(input_ids=new_tokens, batch_size=(2,))
        else:
            raise ValueError(f"Invalid input_mode: {input_mode}")

        # Step 3: Create log-probs only wrappers
        vllm_lp_wrapper = vLLMWrapper(
            vllm_model,
            tokenizer=vllm_tokenizer,
            input_mode=input_mode,
            input_key=input_key,
            generate=False,
            pad_output=pad_output,
        )

        tf_lp_wrapper = TransformersWrapper(
            tf_model,
            tokenizer=tf_tokenizer,
            input_mode=input_mode,
            input_key=input_key,
            generate=False,
            pad_output=pad_output,
        )

        # Step 4: Compute log-probs for the full sequence (original + generated)
        vllm_lp_result = vllm_lp_wrapper(new_data.copy())
        tf_lp_result = tf_lp_wrapper(new_data.copy())

        assert_close(
            vllm_lp_result, tf_lp_result, atol=1e-1, rtol=1e-1, intersection=True
        )


class TestDistributionMethods:
    """Test the new distribution methods and masking strategies."""

    @pytest.mark.skipif(not _has_vllm, reason="vllm not available")
    @pytest.mark.parametrize("masking_strategy", ["sft", "rlhf", "generic"])
    def test_vllm_distribution_methods(
        self, vllm_instance, sample_history_assistant, sample_tokens, masking_strategy
    ):
        """Test that vLLM wrapper distribution methods work correctly."""
        model, tokenizer = vllm_instance

        # vLLM doesn't support get_dist methods
        wrapper = vLLMWrapper(
            model,
            tokenizer=tokenizer,
            input_mode="history",
            generate=False,
            return_log_probs=True,
        )

        # Create test data
        td = TensorDict({"history": sample_history_assistant}, batch_size=(2,))

        # Test that all distribution methods raise NotImplementedError
        with pytest.raises(NotImplementedError, match="vLLM does not return logits"):
            wrapper.get_dist(td)

        with pytest.raises(NotImplementedError, match="vLLM does not return logits"):
            wrapper._get_sft_dist(td)

        with pytest.raises(NotImplementedError, match="vLLM does not return logits"):
            wrapper._get_rlhf_dist(td)

        with pytest.raises(NotImplementedError, match="vLLM does not return logits"):
            wrapper._get_generic_dist(td)

    @pytest.mark.skipif(not _has_transformers, reason="transformers not available")
    @pytest.mark.parametrize("masking_strategy", ["sft", "rlhf", "generic"])
    @pytest.mark.parametrize("pad_output", [True, False], ids=["padded", "unpadded"])
    def test_transformers_distribution_methods(
        self,
        transformers_instance,
        sample_history_assistant,
        sample_tokens,
        masking_strategy,
        pad_output,
    ):
        """Test that Transformers wrapper distribution methods work correctly."""
        model, tokenizer = transformers_instance

        # Use tokens input mode for SFT, history for RLHF/generic
        if masking_strategy == "sft":
            input_mode = "tokens"
            input_ids, attention_mask = sample_tokens
            assistant_mask = attention_mask.bool().clone()
            assistant_mask[:, : attention_mask.shape[-1] // 2] = False
            input_data = {
                "tokens": Tokens(full=input_ids),
                "masks": Masks(
                    all_attention_mask=attention_mask.bool(),
                    all_assistant_mask=assistant_mask,
                ),
            }

            # Create test data with correct batch size
            td = TensorDict(input_data, batch_size=(2,)).to_lazystack(0)
            if not pad_output:
                for _td in td.unbind(0):
                    _td["tokens"].full = _td["tokens"].full[
                        _td["masks"].all_attention_mask
                    ]
                    _td["masks"].all_assistant_mask = _td["masks"].all_assistant_mask[
                        _td["masks"].all_attention_mask
                    ]
                    _td["masks"].all_attention_mask = _td["masks"].all_attention_mask[
                        _td["masks"].all_attention_mask
                    ]
        else:
            input_mode = "history"
            input_data = {"history": ChatHistory(full=sample_history_assistant)}

            # Create test data with correct batch size
            td = TensorDict(input_data, batch_size=(2,)).to_lazystack(0)

        wrapper = TransformersWrapper(
            model,
            tokenizer=tokenizer,
            input_mode=input_mode,
            generate=False,
            pad_output=pad_output,
        )

        # Test the appropriate distribution method
        if masking_strategy == "sft":
            dist = wrapper._get_sft_dist(td)
        elif masking_strategy == "rlhf":
            dist = wrapper._get_rlhf_dist(td)
        elif masking_strategy == "generic":
            dist = wrapper._get_generic_dist(td)

        # Verify that we get a distribution
        assert dist is not None
        assert hasattr(dist, "log_prob")
        assert hasattr(dist, "sample")

        # Test that logits are available in the output
        td_out = wrapper(td.copy())

        # Test log_prob computation
        if masking_strategy == "sft":
            # For SFT, we need tokens to compute log_prob
            tokens = td_out.get(
                ("tokens", "full"),
                as_padded_tensor=True,
                padding_side="left",
                padding_value=tokenizer.pad_token_id,
            )
            if tokens is not None:
                log_probs = dist.log_prob(tokens.long())
                assert log_probs.shape == tokens.shape
        else:
            # For RLHF/generic, we can test with dummy tokens
            logits = td_out.get("logits")
            if logits is not None:
                dummy_tokens = torch.randint(0, logits.shape[-1], logits.shape[:-1])
                log_probs = dist.log_prob(dummy_tokens)
                assert log_probs.shape == dummy_tokens.shape

    @pytest.mark.skipif(not _has_transformers, reason="transformers not available")
    def test_transformers_custom_masking(
        self, transformers_instance, sample_history_assistant
    ):
        """Test custom masking functionality."""
        model, tokenizer = transformers_instance

        wrapper = TransformersWrapper(
            model,
            tokenizer=tokenizer,
            input_mode="history",
            generate=False,
            return_log_probs=True,
            pad_output=True,
        )

        td = TensorDict(
            {"history": ChatHistory(full=sample_history_assistant)}, batch_size=(2,)
        )

        # Get the actual logits shape from the wrapper
        result = wrapper(td)
        lp = result["log_probs"].get("full")

        # Create a custom mask matching the logits shape
        custom_mask = torch.zeros_like(lp, dtype=torch.bool)
        custom_mask[:, :5] = True  # Only first 5 tokens

        dist = wrapper._get_dist_with_custom_mask(td, custom_mask)

        assert dist is not None
        assert hasattr(dist, "log_prob")


@pytest.mark.skipif(not _has_transformers, reason="transformers not available")
@pytest.mark.parametrize("pad_output", [False, True])
class TestPacking:
    def test_packing_history(
        self, transformers_instance, sample_history_assistant, pad_output
    ):
        model, tokenizer = transformers_instance

        wrapper_packed = TransformersWrapper(
            model,
            tokenizer=tokenizer,
            input_mode="history",
            generate=False,
            return_log_probs=True,
            pad_output=pad_output,
            pad_model_input=False,
        )
        wrapped_padded = TransformersWrapper(
            model,
            tokenizer=tokenizer,
            input_mode="history",
            generate=False,
            return_log_probs=True,
            pad_output=pad_output,
            pad_model_input=True,
        )

        td = TensorDict(
            {"history": ChatHistory(full=sample_history_assistant)}, batch_size=(2,)
        ).to_lazystack(0)

        result_padded = wrapped_padded(td)
        result_packed = wrapper_packed(td)
        assert_close(result_packed["log_probs"], result_padded["log_probs"])

    def test_packing_text(self, transformers_instance, sample_text, pad_output):
        model, tokenizer = transformers_instance
        wrapper_packed = TransformersWrapper(
            model,
            tokenizer=tokenizer,
            input_mode="text",
            generate=False,
            return_log_probs=True,
            pad_output=pad_output,
            pad_model_input=False,
        )
        wrapped_padded = TransformersWrapper(
            model,
            tokenizer=tokenizer,
            input_mode="text",
            generate=False,
            return_log_probs=True,
            pad_output=pad_output,
            pad_model_input=True,
        )
        td = TensorDict({"text": Text(full=sample_text)}, batch_size=(2,))
        result_packed = wrapper_packed(td)
        result_padded = wrapped_padded(td)
        assert_close(result_packed["log_probs"], result_padded["log_probs"])

    def test_packing_tokens(
        self, transformers_instance, sample_tokens_unpadded, pad_output
    ):
        model, tokenizer = transformers_instance
        wrapper_packed = TransformersWrapper(
            model,
            tokenizer=tokenizer,
            input_mode="tokens",
            generate=False,
            return_log_probs=True,
            pad_output=pad_output,
            pad_model_input=False,
        )
        wrapped_padded = TransformersWrapper(
            model,
            tokenizer=tokenizer,
            input_mode="tokens",
            generate=False,
            return_log_probs=True,
            pad_output=pad_output,
            pad_model_input=True,
        )
        td = TensorDict(
            {
                "tokens": Tokens(full=sample_tokens_unpadded[0]),
                "masks": Masks(all_attention_mask=sample_tokens_unpadded[1]),
            },
            batch_size=(2,),
        ).to_lazystack(0)
        result_padded = wrapped_padded(td)
        result_packed = wrapper_packed(td)
        assert_close(result_packed["log_probs"], result_padded["log_probs"])


if __name__ == "__main__":
    args, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst"] + unknown)
