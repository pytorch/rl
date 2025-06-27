# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import argparse
import importlib.util

import os

import pytest
import torch

from tensordict import lazy_stack, set_list_to_stack, TensorDict
from torchrl.data.llm import History
from torchrl.modules.llm import TransformersWrapper, vLLMWrapper
from transformers import AutoTokenizer


# Set environment variable for vLLM V0 engine
os.environ["VLLM_USE_V1"] = "0"

_has_transformers = importlib.util.find_spec("transformers") is not None
_has_vllm = importlib.util.find_spec("vllm") is not None
_has_datasets = importlib.util.find_spec("datasets") is not None


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


def check_output_shapes(out, pad_output):
    if pad_output:
        # We can get all tensors or they are none
        log_probs = out.get("log_probs")
        masks = out.get("masks")
        tokens = out.get("tokens")
        # Test the all_ tensors
        if log_probs is not None:
            all_logprobs = log_probs.full
        else:
            all_logprobs = None
        if masks is not None:
            all_attention_masks = masks.all_attention_mask
            all_assistant_masks = masks.all_assistant_mask
        else:
            all_attention_masks = None
            all_assistant_masks = None
        if tokens is not None:
            all_tokens = tokens.full
        else:
            all_tokens = None
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
    else:
        # we can simply iterate over out
        for _out in out.unbind(0):
            check_output_shapes(_out, pad_output=not _out.ndim)


@pytest.mark.skipif(not _has_vllm, reason="vllm not available")
class TestVLLMWrapper:
    """Comprehensive tests for vLLMWrapper covering all modalities and configurations."""

    # ================================================
    # History Input Mode Tests
    # ================================================

    @pytest.mark.parametrize("generate", [True, False], ids=["generate", "no_generate"])
    @pytest.mark.parametrize(
        "return_log_probs", [True, False], ids=["log_probs", "no_log_probs"]
    )
    @pytest.mark.parametrize("return_text", [True, False], ids=["text", "no_text"])
    @pytest.mark.parametrize(
        "return_tokens", [True, False], ids=["tokens", "no_tokens"]
    )
    @pytest.mark.parametrize("return_masks", [True, False], ids=["masks", "no_masks"])
    @pytest.mark.parametrize("pad_output", [True, False], ids=["padded", "unpadded"])
    def test_history_input_mode(
        self,
        vllm_instance,
        sample_history,
        sample_history_assistant,
        generate,
        return_log_probs,
        return_text,
        return_tokens,
        return_masks,
        pad_output,
    ):
        if return_masks and not return_tokens:
            pytest.skip("return_masks cannot be True if return_tokens is False")
        """Test history input mode with various configurations."""
        model, tokenizer = vllm_instance

        # Skip invalid combinations
        if not generate and not return_log_probs:
            pytest.skip("generate=False requires return_log_probs=True")

        wrapper = vLLMWrapper(
            model,
            tokenizer=tokenizer,
            input_mode="history",
            input_key="history",
            generate=generate,
            return_log_probs=return_log_probs,
            return_text=return_text,
            return_tokens=return_tokens,
            return_masks=return_masks,
            pad_output=pad_output,
        )

        # Check input keys
        assert wrapper.in_keys == ["history"]

        # Check output keys
        expected_out_keys = []
        if return_text:
            expected_out_keys.append("text")
        if return_masks:
            expected_out_keys.append("masks")
        if return_tokens:
            expected_out_keys.append("tokens")
        if return_log_probs:
            expected_out_keys.append("log_probs")
        assert wrapper.out_keys == expected_out_keys

        # Create input data
        data = TensorDict(
            history=sample_history if generate else sample_history_assistant,
            batch_size=(2,),
        )

        # Run wrapper
        result = wrapper(data)
        check_output_shapes(result, pad_output)

        # Check output structure
        for key in expected_out_keys:
            assert key in result
            assert hasattr(result[key], "__class__")

        # Check specific outputs
        if return_text:
            text_obj = result["text"]
            assert hasattr(text_obj, "prompt")
            assert hasattr(text_obj, "response")
            assert hasattr(text_obj, "full")
            assert hasattr(text_obj, "padded")
            assert all(text_obj.padded) == pad_output

            if generate:
                assert text_obj.response is not None
                assert isinstance(text_obj.response, list)
                assert isinstance(text_obj.response[0], str)

        if return_tokens:
            tokens_obj = result["tokens"]
            if pad_output:
                # if not padded, we will fail to stack
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
                    # For unpadded output, use as_list=True to avoid stacking issues
                    response_tokens = result["tokens"].get("response", as_list=True)
                    assert isinstance(response_tokens, list)
                else:
                    assert isinstance(tokens_obj.response, torch.Tensor)

        if return_masks:
            masks_obj = result["masks"]
            if pad_output:
                # if not padded, we will fail to stack
                assert hasattr(masks_obj, "all_attention_mask")
                assert hasattr(masks_obj, "all_assistant_mask")
                assert hasattr(masks_obj, "padded")
            assert all(masks_obj.padded) == pad_output

        if return_log_probs:
            log_probs_obj = result["log_probs"]
            if pad_output:
                # if not padded, we will fail to stack
                assert hasattr(log_probs_obj, "prompt")
                assert hasattr(log_probs_obj, "response")
                assert hasattr(log_probs_obj, "full")
                assert hasattr(log_probs_obj, "padded")
            assert all(log_probs_obj.padded) == pad_output

    # ================================================
    # Text Input Mode Tests
    # ================================================

    @pytest.mark.parametrize("generate", [True, False], ids=["generate", "no_generate"])
    @pytest.mark.parametrize(
        "return_log_probs", [True, False], ids=["log_probs", "no_log_probs"]
    )
    @pytest.mark.parametrize("pad_output", [True, False], ids=["padded", "unpadded"])
    def test_text_input_mode(
        self,
        vllm_instance,
        sample_text,
        generate,
        return_log_probs,
        pad_output,
    ):
        """Test text input mode with various configurations."""
        model, tokenizer = vllm_instance

        # Skip invalid combinations
        if not generate and not return_log_probs:
            pytest.skip("generate=False requires return_log_probs=True")

        wrapper = vLLMWrapper(
            model,
            tokenizer=tokenizer,
            input_mode="text",
            input_key="prompt",
            generate=generate,
            return_log_probs=return_log_probs,
            return_text=True,
            return_tokens=True,
            return_masks=True,
            pad_output=pad_output,
        )

        # Check input keys
        assert wrapper.in_keys == ["prompt"]

        # Create input data
        data = TensorDict(prompt=sample_text, batch_size=(2,))

        # Run wrapper
        result = wrapper(data)
        check_output_shapes(result, pad_output)

        # Check output structure
        expected_keys = ["text", "masks", "tokens"]
        if return_log_probs:
            expected_keys.append("log_probs")

        for key in expected_keys:
            assert key in result

        # Check text output
        text_obj = result["text"]
        assert text_obj.prompt == sample_text
        if generate:
            assert text_obj.response is not None

        # Check tokens output
        tokens_obj = result["tokens"]
        if generate:
            if not pad_output:
                # For unpadded output, use as_list=True
                response_tokens = tokens_obj.get("response", as_list=True)
                assert isinstance(tokens_obj.get("response", as_list=True), list)
            else:
                assert isinstance(tokens_obj.response, torch.Tensor)

    # ================================================
    # Tokens Input Mode Tests
    # ================================================

    @pytest.mark.parametrize("generate", [True, False], ids=["generate", "no_generate"])
    @pytest.mark.parametrize(
        "return_log_probs", [True, False], ids=["log_probs", "no_log_probs"]
    )
    @pytest.mark.parametrize("pad_output", [True, False], ids=["padded", "unpadded"])
    def test_tokens_input_mode(
        self,
        vllm_instance,
        sample_tokens,
        generate,
        return_log_probs,
        pad_output,
    ):
        """Test tokens input mode with various configurations."""
        model, tokenizer = vllm_instance

        # Skip invalid combinations
        if not generate and not return_log_probs:
            pytest.skip("generate=False requires return_log_probs=True")

        input_ids, attention_mask = sample_tokens

        wrapper = vLLMWrapper(
            model,
            tokenizer=tokenizer,
            input_mode="tokens",
            input_key="input_ids",
            attention_mask_key="attention_mask",
            generate=generate,
            return_log_probs=return_log_probs,
            return_text=True,
            return_tokens=True,
            return_masks=True,
            pad_output=pad_output,
        )

        # Check input keys
        assert wrapper.in_keys == ["input_ids"]

        # Create input data
        data = TensorDict(
            input_ids=input_ids, attention_mask=attention_mask, batch_size=(2,)
        )

        # Run wrapper
        result = wrapper(data)
        check_output_shapes(result, pad_output)

        # Check output structure
        expected_keys = ["text", "masks", "tokens"]
        if return_log_probs:
            expected_keys.append("log_probs")

        for key in expected_keys:
            assert key in result

        # Check tokens output
        tokens_obj = result["tokens"]
        if generate:
            if not pad_output:
                # For unpadded output, use as_list=True
                response_tokens = result["tokens"].get("response", as_list=True)
                assert isinstance(response_tokens, list)
            else:
                assert isinstance(tokens_obj.response, torch.Tensor)

    # ================================================
    # Error Handling Tests
    # ================================================

    def test_invalid_input_mode(self, vllm_instance):
        """Test that invalid input_mode raises an error."""
        model, tokenizer = vllm_instance

        with pytest.raises(ValueError, match="input_mode must be one of"):
            vLLMWrapper(
                model,
                tokenizer=tokenizer,
                input_mode="invalid_mode",
            )

    def test_missing_input_key(self, vllm_instance, sample_history):
        """Test that missing input key raises an error."""
        model, tokenizer = vllm_instance

        wrapper = vLLMWrapper(
            model,
            tokenizer=tokenizer,
            input_mode="history",
            input_key="history",
        )

        # Create data without the required key
        data = TensorDict(batch_size=(2,))

        with pytest.raises(ValueError, match="Expected 'history' key"):
            wrapper(data)

    def test_invalid_history_type(self, vllm_instance):
        """Test that invalid history type raises an error."""
        model, tokenizer = vllm_instance

        wrapper = vLLMWrapper(
            model,
            tokenizer=tokenizer,
            input_mode="history",
            input_key="history",
        )

        # Create data with wrong type
        data = TensorDict(history="not a history object", batch_size=(2,))

        with pytest.raises(TypeError, match="Expected History object"):
            wrapper(data)

    def test_generate_false_without_log_probs(self, vllm_instance):
        """Test that generate=False without return_log_probs=True raises an error."""
        model, tokenizer = vllm_instance

        with pytest.raises(ValueError, match="return_log_probs must be True"):
            vLLMWrapper(
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
    def test_batch_sizes(self, vllm_instance, batch_size, pad_output):
        """Test wrapper with different batch sizes."""
        model, tokenizer = vllm_instance

        # Create history with specified batch size
        chats = [
            [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"Question {i}?"},
            ]
            for i in range(batch_size)
        ]
        history = History.from_chats(chats)

        wrapper = vLLMWrapper(
            model,
            tokenizer=tokenizer,
            input_mode="history",
            generate=True,
            return_text=True,
            return_tokens=True,
            return_masks=True,
            return_log_probs=True,
            pad_output=pad_output,
        )

        data = TensorDict(history=history, batch_size=(batch_size,))
        result = wrapper(data)
        check_output_shapes(result, pad_output=wrapper.pad_output)

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

    def test_custom_input_key(self, vllm_instance, sample_history):
        """Test wrapper with custom input key."""
        model, tokenizer = vllm_instance

        wrapper = vLLMWrapper(
            model,
            tokenizer=tokenizer,
            input_mode="history",
            input_key="custom_history_key",
            generate=True,
            return_text=True,
            return_tokens=True,
            return_masks=True,
            return_log_probs=True,
        )

        # Check input keys
        assert wrapper.in_keys == ["custom_history_key"]

        # Create data with custom key
        data = TensorDict(custom_history_key=sample_history, batch_size=(2,))
        result = wrapper(data)
        check_output_shapes(result, pad_output=wrapper.pad_output)

        # Check that wrapper works correctly
        expected_keys = ["text", "masks", "tokens", "log_probs"]
        for key in expected_keys:
            assert key in result

    # ================================================
    # Selective Output Tests
    # ================================================

    @pytest.mark.parametrize("return_text", [True, False], ids=["text", "no_text"])
    @pytest.mark.parametrize(
        "return_tokens", [True, False], ids=["tokens", "no_tokens"]
    )
    @pytest.mark.parametrize("return_masks", [True, False], ids=["masks", "no_masks"])
    @pytest.mark.parametrize(
        "return_log_probs", [True, False], ids=["log_probs", "no_log_probs"]
    )
    def test_selective_outputs(
        self,
        vllm_instance,
        sample_history,
        return_text,
        return_tokens,
        return_masks,
        return_log_probs,
    ):
        """Test wrapper with selective output configurations."""
        if return_masks and not return_tokens:
            pytest.skip("return_masks cannot be True if return_tokens is False")
        model, tokenizer = vllm_instance

        wrapper = vLLMWrapper(
            model,
            tokenizer=tokenizer,
            input_mode="history",
            generate=True,
            return_text=return_text,
            return_tokens=return_tokens,
            return_masks=return_masks,
            return_log_probs=return_log_probs,
        )

        # Check output keys
        expected_out_keys = []
        if return_text:
            expected_out_keys.append("text")
        if return_masks:
            expected_out_keys.append("masks")
        if return_tokens:
            expected_out_keys.append("tokens")
        if return_log_probs:
            expected_out_keys.append("log_probs")

        assert wrapper.out_keys == expected_out_keys

        # Run wrapper
        data = TensorDict(history=sample_history, batch_size=(2,))
        result = wrapper(data)
        check_output_shapes(result, pad_output=wrapper.pad_output)

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

    def test_log_probs_only_mode(self, vllm_instance, sample_history):
        """Test wrapper in log-probs only mode (generate=False)."""
        model, tokenizer = vllm_instance

        wrapper = vLLMWrapper(
            model,
            tokenizer=tokenizer,
            input_mode="history",
            generate=False,  # Only compute log-probs
            return_log_probs=True,  # Must be True when generate=False
            return_text=True,
            return_tokens=True,
            return_masks=True,
        )

        data = TensorDict(history=sample_history, batch_size=(2,))
        result = wrapper(data)
        check_output_shapes(result, pad_output=wrapper.pad_output)

        # Check that log_probs are present
        assert "log_probs" in result

        # Check that response_text is None (no generation)
        assert result["text"].response is None

        # Check that prompt_logprobs are present
        log_probs_obj = result["log_probs"]
        assert log_probs_obj.get("prompt", as_list=True) is not None

    # ================================================
    # TensorClass Structure Tests
    # ================================================

    def test_tensorclass_structure(self, vllm_instance, sample_history):
        """Test that TensorClass objects have the correct structure."""
        model, tokenizer = vllm_instance
        pad_output = False

        wrapper = vLLMWrapper(
            model,
            tokenizer=tokenizer,
            input_mode="history",
            generate=True,
            return_text=True,
            return_tokens=True,
            return_masks=True,
            return_log_probs=True,
        )

        data = TensorDict(history=sample_history, batch_size=(2,))
        result = wrapper(data)

        # Test Text TensorClass
        text_obj = result["text"]
        assert hasattr(text_obj, "prompt")
        assert hasattr(text_obj, "response")
        assert hasattr(text_obj, "full")
        assert hasattr(text_obj, "padded")

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

    def test_unpadded_output_with_as_list(self, vllm_instance, sample_history):
        """Test unpadded output using as_list=True to avoid stacking issues."""
        model, tokenizer = vllm_instance

        wrapper = vLLMWrapper(
            model,
            tokenizer=tokenizer,
            input_mode="history",
            generate=True,
            return_text=True,
            return_tokens=True,
            return_masks=True,
            return_log_probs=True,
            pad_output=False,  # Unpadded output
        )

        data = TensorDict(history=sample_history, batch_size=(2,))
        result = wrapper(data)
        check_output_shapes(result, pad_output=wrapper.pad_output)

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
    @pytest.mark.parametrize("return_text", [True, False], ids=["text", "no_text"])
    @pytest.mark.parametrize(
        "return_tokens", [True, False], ids=["tokens", "no_tokens"]
    )
    @pytest.mark.parametrize("return_masks", [True, False], ids=["masks", "no_masks"])
    @pytest.mark.parametrize(
        "return_log_probs", [True, False], ids=["log_probs", "no_log_probs"]
    )
    @pytest.mark.parametrize(
        "input_mode", ["history", "text", "tokens"], ids=["history", "text", "tokens"]
    )
    def test_num_samples(
        self,
        vllm_instance,
        sample_history,
        sample_text,
        sample_tokens,
        num_samples,
        pad_output,
        return_text,
        return_tokens,
        return_masks,
        return_log_probs,
        input_mode,
    ):
        """Test wrapper with num_samples."""
        model, tokenizer = vllm_instance
        if return_masks and not return_tokens:
            pytest.skip("return_masks cannot be True if return_tokens is False")

        wrapper = vLLMWrapper(
            model,
            tokenizer=tokenizer,
            input_mode=input_mode,
            generate=True,
            return_text=return_text,
            return_tokens=return_tokens,
            return_masks=return_masks,
            return_log_probs=return_log_probs,
            pad_output=pad_output,
            num_samples=num_samples,
            input_key="prompt" if input_mode == "text" else None,
        )
        if input_mode == "history":
            data = TensorDict(history=sample_history, batch_size=(2,))
        elif input_mode == "text":
            data = TensorDict(prompt=sample_text, batch_size=(2,))
        elif input_mode == "tokens":
            data = TensorDict(tokens=sample_tokens[0], batch_size=(2,))
        else:
            raise ValueError(f"Invalid input mode: {input_mode}")
        result = wrapper(data)
        assert result.batch_size == (2, num_samples)
        check_output_shapes(result, pad_output=wrapper.pad_output)


@pytest.mark.skipif(not _has_vllm, reason="vllm not available")
class TestTransformersWrapper:
    """Comprehensive tests for TransformersWrapper covering all modalities and configurations."""

    # ================================================
    # History Input Mode Tests
    # ================================================

    @pytest.mark.parametrize("generate", [True, False], ids=["generate", "no_generate"])
    @pytest.mark.parametrize(
        "return_log_probs", [True, False], ids=["log_probs", "no_log_probs"]
    )
    @pytest.mark.parametrize("return_text", [True, False], ids=["text", "no_text"])
    @pytest.mark.parametrize(
        "return_tokens", [True, False], ids=["tokens", "no_tokens"]
    )
    @pytest.mark.parametrize("return_masks", [True, False], ids=["masks", "no_masks"])
    @pytest.mark.parametrize("pad_output", [True, False], ids=["padded", "unpadded"])
    def test_history_input_mode(
        self,
        transformers_instance,
        sample_history,
        sample_history_assistant,
        generate,
        return_log_probs,
        return_text,
        return_tokens,
        return_masks,
        pad_output,
    ):
        if return_masks and not return_tokens:
            pytest.skip("return_masks cannot be True if return_tokens is False")
        """Test history input mode with various configurations."""
        model, tokenizer = transformers_instance

        # Skip invalid combinations
        if not generate and not return_log_probs:
            pytest.skip("generate=False requires return_log_probs=True")

        wrapper = TransformersWrapper(
            model,
            tokenizer=tokenizer,
            input_mode="history",
            input_key="history",
            generate=generate,
            return_log_probs=return_log_probs,
            return_text=return_text,
            return_tokens=return_tokens,
            return_masks=return_masks,
            pad_output=pad_output,
            generate_kwargs={"max_new_tokens": 10},
        )

        # Check input keys
        assert wrapper.in_keys == ["history"]

        # Check output keys
        expected_out_keys = []
        if return_text:
            expected_out_keys.append("text")
        if return_masks:
            expected_out_keys.append("masks")
        if return_tokens:
            expected_out_keys.append("tokens")
        if return_log_probs:
            expected_out_keys.append("log_probs")
        assert wrapper.out_keys == expected_out_keys

        # Create input data
        data = TensorDict(
            history=sample_history if generate else sample_history_assistant,
            batch_size=(2,),
        )

        # Run wrapper
        result = wrapper(data)
        check_output_shapes(result, pad_output)

        # Check output structure
        for key in expected_out_keys:
            assert key in result
            assert hasattr(result[key], "__class__")

        # Check specific outputs
        if return_text:
            text_obj = result["text"]
            assert hasattr(text_obj, "prompt")
            assert hasattr(text_obj, "response")
            assert hasattr(text_obj, "full")
            assert hasattr(text_obj, "padded")
            assert all(text_obj.padded) == pad_output

            if generate:
                assert text_obj.response is not None
                assert isinstance(text_obj.response, list)
                assert isinstance(text_obj.response[0], str)

        if return_tokens:
            tokens_obj = result["tokens"]
            if pad_output:
                # if not padded, we will fail to stack
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
                    # For unpadded output, use as_list=True to avoid stacking issues
                    response_tokens = result["tokens"].get("response", as_list=True)
                    assert isinstance(response_tokens, list)
                else:
                    assert isinstance(tokens_obj.response, torch.Tensor)

        if return_masks:
            masks_obj = result["masks"]
            if pad_output:
                # if not padded, we will fail to stack
                assert hasattr(masks_obj, "all_attention_mask")
                assert hasattr(masks_obj, "all_assistant_mask")
                assert hasattr(masks_obj, "padded")
            assert all(masks_obj.padded) == pad_output

        if return_log_probs:
            log_probs_obj = result["log_probs"]
            if pad_output:
                # if not padded, we will fail to stack
                assert hasattr(log_probs_obj, "prompt")
                assert hasattr(log_probs_obj, "response")
                assert hasattr(log_probs_obj, "full")
                assert hasattr(log_probs_obj, "padded")
            assert all(log_probs_obj.padded) == pad_output

    # ================================================
    # Text Input Mode Tests
    # ================================================

    @pytest.mark.parametrize("generate", [True, False], ids=["generate", "no_generate"])
    @pytest.mark.parametrize(
        "return_log_probs", [True, False], ids=["log_probs", "no_log_probs"]
    )
    @pytest.mark.parametrize("pad_output", [True, False], ids=["padded", "unpadded"])
    def test_text_input_mode(
        self,
        transformers_instance,
        sample_text,
        generate,
        return_log_probs,
        pad_output,
    ):
        """Test text input mode with various configurations."""
        model, tokenizer = transformers_instance

        # Skip invalid combinations
        if not generate and not return_log_probs:
            pytest.skip("generate=False requires return_log_probs=True")

        wrapper = TransformersWrapper(
            model,
            tokenizer=tokenizer,
            input_mode="text",
            input_key="prompt",
            generate=generate,
            return_log_probs=return_log_probs,
            return_text=True,
            return_tokens=True,
            return_masks=True,
            pad_output=pad_output,
            generate_kwargs={"max_new_tokens": 10},
        )

        # Check input keys
        assert wrapper.in_keys == ["prompt"]

        # Create input data
        data = TensorDict(prompt=sample_text, batch_size=(2,))

        # Run wrapper
        result = wrapper(data)
        check_output_shapes(result, pad_output)

        # Check output structure
        expected_keys = ["text", "masks", "tokens"]
        if return_log_probs:
            expected_keys.append("log_probs")

        for key in expected_keys:
            assert key in result

        # Check text output
        text_obj = result["text"]
        assert text_obj.prompt == sample_text
        if generate:
            assert text_obj.response is not None

        # Check tokens output
        tokens_obj = result["tokens"]
        if generate:
            if not pad_output:
                # For unpadded output, use as_list=True
                response_tokens = tokens_obj.get("response", as_list=True)
                assert isinstance(tokens_obj.get("response", as_list=True), list)
            else:
                assert isinstance(tokens_obj.response, torch.Tensor)

    # ================================================
    # Tokens Input Mode Tests
    # ================================================

    @pytest.mark.parametrize("generate", [True, False], ids=["generate", "no_generate"])
    @pytest.mark.parametrize(
        "return_log_probs", [True, False], ids=["log_probs", "no_log_probs"]
    )
    @pytest.mark.parametrize("pad_output", [True, False], ids=["padded", "unpadded"])
    def test_tokens_input_mode(
        self,
        transformers_instance,
        sample_tokens,
        generate,
        return_log_probs,
        pad_output,
    ):
        """Test tokens input mode with various configurations."""
        model, tokenizer = transformers_instance

        # Skip invalid combinations
        if not generate and not return_log_probs:
            pytest.skip("generate=False requires return_log_probs=True")

        input_ids, attention_mask = sample_tokens

        wrapper = TransformersWrapper(
            model,
            tokenizer=tokenizer,
            input_mode="tokens",
            input_key="input_ids",
            attention_mask_key="attention_mask",
            generate=generate,
            return_log_probs=return_log_probs,
            return_text=True,
            return_tokens=True,
            return_masks=True,
            pad_output=pad_output,
            generate_kwargs={"max_new_tokens": 10},
        )

        # Check input keys
        assert wrapper.in_keys == ["input_ids"]

        # Create input data
        data = TensorDict(
            input_ids=input_ids, attention_mask=attention_mask, batch_size=(2,)
        )

        # Run wrapper
        result = wrapper(data)
        check_output_shapes(result, pad_output)

        # Check output structure
        expected_keys = ["text", "masks", "tokens"]
        if return_log_probs:
            expected_keys.append("log_probs")

        for key in expected_keys:
            assert key in result

        # Check tokens output
        tokens_obj = result["tokens"]
        if generate:
            if not pad_output:
                # For unpadded output, use as_list=True
                response_tokens = result["tokens"].get("response", as_list=True)
                assert isinstance(response_tokens, list)
            else:
                assert isinstance(tokens_obj.response, torch.Tensor)

    # ================================================
    # Error Handling Tests
    # ================================================

    def test_invalid_input_mode(self, transformers_instance):
        """Test that invalid input_mode raises an error."""
        model, tokenizer = transformers_instance

        with pytest.raises(ValueError, match="input_mode must be one of"):
            TransformersWrapper(
                model,
                tokenizer=tokenizer,
                input_mode="invalid_mode",
            )

    def test_missing_input_key(self, transformers_instance, sample_history):
        """Test that missing input key raises an error."""
        model, tokenizer = transformers_instance

        wrapper = TransformersWrapper(
            model,
            tokenizer=tokenizer,
            input_mode="history",
            input_key="history",
        )

        # Create data without the required key
        data = TensorDict(batch_size=(2,))

        with pytest.raises(ValueError, match="Expected 'history' key"):
            wrapper(data)

    def test_invalid_history_type(self, transformers_instance):
        """Test that invalid history type raises an error."""
        model, tokenizer = transformers_instance

        wrapper = TransformersWrapper(
            model,
            tokenizer=tokenizer,
            input_mode="history",
            input_key="history",
        )

        # Create data with wrong type
        data = TensorDict(history="not a history object", batch_size=(2,))

        with pytest.raises(TypeError, match="Expected History object"):
            wrapper(data)

    def test_generate_false_without_log_probs(self, transformers_instance):
        """Test that generate=False without return_log_probs=True raises an error."""
        model, tokenizer = transformers_instance

        with pytest.raises(ValueError, match="return_log_probs must be True"):
            TransformersWrapper(
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
    def test_batch_sizes(self, transformers_instance, batch_size, pad_output):
        """Test wrapper with different batch sizes."""
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

        wrapper = TransformersWrapper(
            model,
            tokenizer=tokenizer,
            input_mode="history",
            generate=True,
            return_text=True,
            return_tokens=True,
            return_masks=True,
            return_log_probs=True,
            pad_output=pad_output,
            generate_kwargs={"max_new_tokens": 10},
        )

        data = TensorDict(history=history, batch_size=(batch_size,))
        result = wrapper(data)
        check_output_shapes(result, pad_output=wrapper.pad_output)

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

    def test_custom_input_key(self, transformers_instance, sample_history):
        """Test wrapper with custom input key."""
        model, tokenizer = transformers_instance

        wrapper = TransformersWrapper(
            model,
            tokenizer=tokenizer,
            input_mode="history",
            input_key="custom_history_key",
            generate=True,
            return_text=True,
            return_tokens=True,
            return_masks=True,
            return_log_probs=True,
            generate_kwargs={"max_new_tokens": 10},
        )

        # Check input keys
        assert wrapper.in_keys == ["custom_history_key"]

        # Create data with custom key
        data = TensorDict(custom_history_key=sample_history, batch_size=(2,))
        result = wrapper(data)
        check_output_shapes(result, pad_output=wrapper.pad_output)

        # Check that wrapper works correctly
        expected_keys = ["text", "masks", "tokens", "log_probs"]
        for key in expected_keys:
            assert key in result

    # ================================================
    # Selective Output Tests
    # ================================================

    @pytest.mark.parametrize("return_text", [True, False], ids=["text", "no_text"])
    @pytest.mark.parametrize(
        "return_tokens", [True, False], ids=["tokens", "no_tokens"]
    )
    @pytest.mark.parametrize("return_masks", [True, False], ids=["masks", "no_masks"])
    @pytest.mark.parametrize(
        "return_log_probs", [True, False], ids=["log_probs", "no_log_probs"]
    )
    def test_selective_outputs(
        self,
        transformers_instance,
        sample_history,
        return_text,
        return_tokens,
        return_masks,
        return_log_probs,
    ):
        """Test wrapper with selective output configurations."""
        if return_masks and not return_tokens:
            pytest.skip("return_masks cannot be True if return_tokens is False")
        model, tokenizer = transformers_instance

        wrapper = TransformersWrapper(
            model,
            tokenizer=tokenizer,
            input_mode="history",
            generate=True,
            return_text=return_text,
            return_tokens=return_tokens,
            return_masks=return_masks,
            return_log_probs=return_log_probs,
            generate_kwargs={"max_new_tokens": 10},
        )

        # Check output keys
        expected_out_keys = []
        if return_text:
            expected_out_keys.append("text")
        if return_masks:
            expected_out_keys.append("masks")
        if return_tokens:
            expected_out_keys.append("tokens")
        if return_log_probs:
            expected_out_keys.append("log_probs")

        assert wrapper.out_keys == expected_out_keys

        # Run wrapper
        data = TensorDict(history=sample_history, batch_size=(2,))
        result = wrapper(data)
        check_output_shapes(result, pad_output=wrapper.pad_output)

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

    def test_log_probs_only_mode(self, transformers_instance, sample_history):
        """Test wrapper in log-probs only mode (generate=False)."""
        model, tokenizer = transformers_instance

        wrapper = TransformersWrapper(
            model,
            tokenizer=tokenizer,
            input_mode="history",
            generate=False,  # Only compute log-probs
            return_log_probs=True,  # Must be True when generate=False
            return_text=True,
            return_tokens=True,
            return_masks=True,
            generate_kwargs={"max_new_tokens": 10},
        )

        data = TensorDict(history=sample_history, batch_size=(2,))
        result = wrapper(data)
        check_output_shapes(result, pad_output=wrapper.pad_output)

        # Check that log_probs are present
        assert "log_probs" in result

        # Check that response_text is None (no generation)
        assert result["text"].response is None

        # Check that prompt_logprobs are present
        log_probs_obj = result["log_probs"]
        assert log_probs_obj.prompt is not None

    # ================================================
    # TensorClass Structure Tests
    # ================================================

    def test_tensorclass_structure(self, transformers_instance, sample_history):
        """Test that TensorClass objects have the correct structure."""
        model, tokenizer = transformers_instance
        pad_output = False

        wrapper = TransformersWrapper(
            model,
            tokenizer=tokenizer,
            input_mode="history",
            generate=True,
            return_text=True,
            return_tokens=True,
            return_masks=True,
            return_log_probs=True,
            generate_kwargs={"max_new_tokens": 10},
        )

        data = TensorDict(history=sample_history, batch_size=(2,))
        result = wrapper(data)

        # Test Text TensorClass
        text_obj = result["text"]
        assert hasattr(text_obj, "prompt")
        assert hasattr(text_obj, "response")
        assert hasattr(text_obj, "full")
        assert hasattr(text_obj, "padded")

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

    def test_unpadded_output_with_as_list(self, transformers_instance, sample_history):
        """Test unpadded output using as_list=True to avoid stacking issues."""
        model, tokenizer = transformers_instance

        wrapper = TransformersWrapper(
            model,
            tokenizer=tokenizer,
            input_mode="history",
            generate=True,
            return_text=True,
            return_tokens=True,
            return_masks=True,
            return_log_probs=True,
            pad_output=False,  # Unpadded output
            generate_kwargs={"max_new_tokens": 10},
        )

        data = TensorDict(history=sample_history, batch_size=(2,))
        result = wrapper(data)
        check_output_shapes(result, pad_output=wrapper.pad_output)

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
    @pytest.mark.parametrize("return_text", [True, False], ids=["text", "no_text"])
    @pytest.mark.parametrize(
        "return_tokens", [True, False], ids=["tokens", "no_tokens"]
    )
    @pytest.mark.parametrize("return_masks", [True, False], ids=["masks", "no_masks"])
    @pytest.mark.parametrize(
        "return_log_probs", [True, False], ids=["log_probs", "no_log_probs"]
    )
    @pytest.mark.parametrize(
        "input_mode", ["history", "text", "tokens"], ids=["history", "text", "tokens"]
    )
    def test_num_samples(
        self,
        transformers_instance,
        sample_history,
        sample_text,
        sample_tokens,
        num_samples,
        pad_output,
        return_text,
        return_tokens,
        return_masks,
        return_log_probs,
        input_mode,
    ):
        """Test wrapper with num_samples."""
        model, tokenizer = transformers_instance
        if return_masks and not return_tokens:
            pytest.skip("return_masks cannot be True if return_tokens is False")

        wrapper = TransformersWrapper(
            model,
            tokenizer=tokenizer,
            input_mode=input_mode,
            generate=True,
            return_text=return_text,
            return_tokens=return_tokens,
            return_masks=return_masks,
            return_log_probs=return_log_probs,
            pad_output=pad_output,
            num_samples=num_samples,
            input_key="prompt" if input_mode == "text" else None,
            generate_kwargs={"max_new_tokens": 10, "do_sample": True},
        )
        if input_mode == "history":
            data = TensorDict(history=sample_history, batch_size=(2,))
        elif input_mode == "text":
            data = TensorDict(prompt=sample_text, batch_size=(2,))
        elif input_mode == "tokens":
            data = TensorDict(tokens=sample_tokens[0], batch_size=(2,))
        else:
            raise ValueError(f"Invalid input mode: {input_mode}")
        result = wrapper(data)
        assert result.batch_size == (2, num_samples)
        check_output_shapes(result, pad_output=wrapper.pad_output)


class TestChatEnvIntegration:
    @pytest.mark.skipif(not _has_vllm, reason="vllm not available")
    @pytest.mark.skipif(not _has_datasets, reason="datasets not available")
    def test_chat_env_integration_gsm8k(self):
        """Test that the wrapper works correctly with the ChatEnv."""
        import vllm.envs as envs
        from torchrl.envs.llm import GSM8KEnv

        envs.VLLM_HOST_IP = "0.0.0.0" or "127.0.0.1"

        policy = vLLMWrapper(
            model="Qwen/Qwen2.5-0.5B",
            tokenizer="Qwen/Qwen2.5-0.5B",
            input_mode="history",
            generate=True,
            return_text=True,
        )
        env = GSM8KEnv(max_steps=10)
        r = env.reset()
        r = policy(r)
        r, r_ = env.step_and_maybe_reset(r)
        r = policy(r_)
        r, r_ = env.step_and_maybe_reset(r)


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
            prompts = sample_text
            data = TensorDict(text=prompts, batch_size=(2,))
            input_key = "text"
        elif input_mode == "tokens":
            prompts = sample_tokens
            data = TensorDict(
                input_ids=prompts[0],
                attention_mask=prompts[1],
                batch_size=(2,),
            )
            input_key = "input_ids"

        # Create vLLM wrapper for generation
        vllm_gen_wrapper = vLLMWrapper(
            vllm_model,
            tokenizer=vllm_tokenizer,
            input_mode=input_mode,
            input_key=input_key,
            generate=True,
            return_log_probs=True,
            return_text=True,
            return_tokens=True,
            return_masks=True,
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
            return_log_probs=True,
            return_text=True,
            return_tokens=True,
            return_masks=True,
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
            for i, (chat, gen_text) in enumerate(
                zip(history.unbind(0), generated_texts)
            ):
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

        # Step 3: Create log-probs only wrappers
        vllm_lp_wrapper = vLLMWrapper(
            vllm_model,
            tokenizer=vllm_tokenizer,
            input_mode=input_mode,
            input_key=input_key,
            generate=False,
            return_log_probs=True,
            return_text=True,
            return_tokens=True,
            return_masks=True,
            pad_output=pad_output,
        )

        tf_lp_wrapper = TransformersWrapper(
            tf_model,
            tokenizer=tf_tokenizer,
            input_mode=input_mode,
            input_key=input_key,
            generate=False,
            return_log_probs=True,
            return_text=True,
            return_tokens=True,
            return_masks=True,
            pad_output=pad_output,
        )

        # Step 4: Compute log-probs for the full sequence (original + generated)
        vllm_lp_result = vllm_lp_wrapper(new_data.copy())
        tf_lp_result = tf_lp_wrapper(new_data.copy())

        from tensordict import assert_close

        assert_close(
            vllm_lp_result, tf_lp_result, atol=1e-1, rtol=1e-1, intersection=True
        )


if __name__ == "__main__":
    args, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst"] + unknown)
