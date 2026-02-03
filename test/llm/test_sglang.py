# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Tests for SGLang backend and wrapper."""
from __future__ import annotations

import argparse
import importlib.util

import pytest
import torch
from tensordict import set_list_to_stack, TensorDict
from torchrl.data.llm import History

_has_sglang = importlib.util.find_spec("sglang") is not None
_has_transformers = importlib.util.find_spec("transformers") is not None

# Skip entire module if SGLang is not available
pytestmark = pytest.mark.skipif(not _has_sglang, reason="sglang not available")

MODEL_NAME = "Qwen/Qwen2.5-0.5B"


@pytest.fixture(scope="module", autouse=True)
def set_list_to_stack_fixture():
    with set_list_to_stack(True):
        yield


@pytest.fixture(scope="module")
def tokenizer():
    """Create tokenizer for testing."""
    if not _has_transformers:
        pytest.skip("transformers not available")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


# Maximum allowed time for SGLang server startup (seconds)
# With --disable-cuda-graph, startup should be fast (< 2 minutes).
# We use a generous timeout to accommodate model loading and initialization.
SGLANG_STARTUP_TIMEOUT = 300


@pytest.fixture(scope="module")
def sglang_service():
    """Create AsyncSGLang service for testing.

    This fixture launches a managed SGLang server for the test module.
    The server should start within SGLANG_STARTUP_TIMEOUT seconds if the
    environment is properly warmed (model downloaded, kernels compiled).
    """
    if not _has_sglang:
        pytest.skip("sglang not available")
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    import time

    from torchrl.modules.llm.backends import AsyncSGLang

    start_time = time.time()
    service = AsyncSGLang.from_pretrained(
        MODEL_NAME,
        tp_size=1,
        dp_size=1,
        mem_fraction_static=0.3,  # Low memory for testing
        disable_cuda_graph=True,  # Skip CUDA graph compilation for faster startup
        timeout=SGLANG_STARTUP_TIMEOUT,
    )
    startup_time = time.time() - start_time

    # Log startup time for debugging
    from torchrl._utils import logger as torchrl_logger

    torchrl_logger.info(f"SGLang server started in {startup_time:.1f}s")

    yield service

    # Cleanup
    service.shutdown()


@pytest.fixture
def sample_history():
    """Create sample conversation history for testing."""
    chats = [
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Are you happy? Say yes or no."},
        ],
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is 2+2?"},
        ],
    ]
    return History.from_chats(chats)


@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.timeout(300)  # 5 minutes should be enough with --disable-cuda-graph
@pytest.mark.skipif(not _has_sglang, reason="sglang not available")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestAsyncSGLangIntegration:
    """Integration tests for AsyncSGLang with real models."""

    def test_server_starts_fast(self, sglang_service):
        """Test that SGLang server starts within acceptable time.

        The server may take up to 10 minutes on first run due to CUDA kernel
        compilation. Subsequent runs with cached kernels should be much faster.
        """
        # If we got here, the fixture succeeded within SGLANG_STARTUP_TIMEOUT
        # The server is running and ready
        assert sglang_service is not None
        assert sglang_service.server_url is not None

    def test_connect_to_server(self, sglang_service):
        """Test that AsyncSGLang service is connected and functional."""
        from torchrl.modules.llm.backends import AsyncSGLang, RLSGLangEngine

        assert isinstance(sglang_service, AsyncSGLang)
        assert isinstance(sglang_service, RLSGLangEngine)

        # Check server is running
        assert sglang_service.server_url is not None

    def test_get_tp_size(self, sglang_service):
        """Test tensor parallel size retrieval."""
        tp_size = sglang_service.get_tp_size()
        assert isinstance(tp_size, int)
        assert tp_size >= 1

    def test_get_dp_size(self, sglang_service):
        """Test data parallel size retrieval."""
        dp_size = sglang_service.get_dp_size()
        assert isinstance(dp_size, int)
        assert dp_size >= 1

    def test_get_model_metadata(self, sglang_service):
        """Test model metadata extraction."""
        metadata = sglang_service.get_model_metadata()
        assert isinstance(metadata, dict)
        # Metadata should contain parameter names with (dtype, shape) tuples
        for name, (dtype, shape) in metadata.items():
            assert isinstance(name, str)
            assert isinstance(dtype, torch.dtype)
            assert isinstance(shape, (tuple, torch.Size))

    def test_generate_text(self, sglang_service):
        """Test text generation with single prompt."""
        result = sglang_service.generate(
            prompts=["Hello, how are you?"],
            max_new_tokens=10,
            temperature=0.0,
        )

        assert isinstance(result, list)
        assert len(result) == 1
        # Check result structure
        assert hasattr(result[0], "text") or isinstance(result[0], dict)

    def test_generate_batch(self, sglang_service):
        """Test batch text generation."""
        prompts = [
            "What is 1+1?",
            "What is 2+2?",
            "What is 3+3?",
        ]
        result = sglang_service.generate(
            prompts=prompts,
            max_new_tokens=10,
            temperature=0.0,
        )

        assert isinstance(result, list)
        assert len(result) == 3

    def test_generate_from_tokens(self, sglang_service, tokenizer):
        """Test generation from token IDs instead of text."""
        # Tokenize a prompt
        prompt_text = "Hello, how are you?"
        input_ids = tokenizer.encode(prompt_text)

        # Generate using input_ids
        result = sglang_service.generate(
            input_ids=input_ids,
            sampling_params={"max_new_tokens": 10, "temperature": 0.0},
        )

        assert isinstance(result, dict)
        # Should have output_ids in the result
        assert "output_ids" in result or "text" in result

    def test_generate_from_tokens_batch(self, sglang_service, tokenizer):
        """Test batch generation from token IDs."""
        prompts = ["What is 1+1?", "What is 2+2?"]
        input_ids_batch = [tokenizer.encode(p) for p in prompts]

        result = sglang_service.generate(
            input_ids=input_ids_batch,
            sampling_params={"max_new_tokens": 10, "temperature": 0.0},
        )

        assert isinstance(result, list)
        assert len(result) == 2

    def test_generate_mutually_exclusive_inputs(self, sglang_service, tokenizer):
        """Test that prompts and input_ids are mutually exclusive."""
        input_ids = tokenizer.encode("Hello")

        # Should raise when both are provided
        with pytest.raises(ValueError, match="Cannot provide both"):
            sglang_service.generate(
                prompts="Hello",
                input_ids=input_ids,
            )

        # Should raise when neither is provided
        with pytest.raises(ValueError, match="Must provide either"):
            sglang_service.generate()

    def test_flush_cache(self, sglang_service):
        """Test cache flushing."""
        success = sglang_service.flush_cache()
        assert success is True


@pytest.mark.gpu
@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.timeout(300)  # 5 minutes should be enough with --disable-cuda-graph
@pytest.mark.skipif(not _has_sglang, reason="sglang not available")
@pytest.mark.skipif(not _has_transformers, reason="transformers not available")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestSGLangWrapper:
    """Tests for SGLangWrapper policy module."""

    def test_wrapper_creation_from_service(self, sglang_service, tokenizer):
        """Test SGLangWrapper creation from AsyncSGLang service."""
        from torchrl.modules.llm.policies import SGLangWrapper

        wrapper = SGLangWrapper(
            model=sglang_service,
            tokenizer=tokenizer,
            input_mode="history",
            generate=True,
            return_log_probs=True,
        )

        assert wrapper is not None
        assert wrapper.tokenizer is tokenizer

    def test_history_mode(self, sglang_service, tokenizer, sample_history):
        """Test SGLangWrapper with history input mode."""
        from torchrl.modules.llm.policies import SGLangWrapper
        from torchrl.modules.llm.policies.common import ChatHistory

        wrapper = SGLangWrapper(
            model=sglang_service,
            tokenizer=tokenizer,
            input_mode="history",
            generate=True,
            generate_kwargs={"max_new_tokens": 10, "temperature": 0.0},
        )

        # Create input tensordict
        td = TensorDict(
            {
                "history": ChatHistory(
                    prompt=sample_history,
                )
            },
            batch_size=[2],
        )

        # Run generation
        result = wrapper(td)

        # Check outputs
        assert "history" in result.keys()
        assert hasattr(result["history"], "response")
        assert hasattr(result["history"], "full")

    def test_text_mode(self, sglang_service, tokenizer):
        """Test SGLangWrapper with text input mode."""
        from torchrl.modules.llm.policies import SGLangWrapper
        from torchrl.modules.llm.policies.common import Text

        wrapper = SGLangWrapper(
            model=sglang_service,
            tokenizer=tokenizer,
            input_mode="text",
            generate=True,
            generate_kwargs={"max_new_tokens": 10, "temperature": 0.0},
        )

        # Create input tensordict
        td = TensorDict(
            {
                "text": Text(
                    prompt=["Hello, how are you?", "What is 2+2?"],
                )
            },
            batch_size=[2],
        )

        # Run generation
        result = wrapper(td)

        # Check outputs
        assert "text" in result.keys()
        assert hasattr(result["text"], "response")
        assert hasattr(result["text"], "full")

    def test_tokens_mode(self, sglang_service, tokenizer):
        """Test SGLangWrapper with tokens input mode."""
        from torchrl.modules.llm.policies import SGLangWrapper
        from torchrl.modules.llm.policies.common import Tokens

        wrapper = SGLangWrapper(
            model=sglang_service,
            tokenizer=tokenizer,
            input_mode="tokens",
            generate=True,
            generate_kwargs={"max_new_tokens": 10, "temperature": 0.0},
        )

        # Tokenize prompts
        prompts = ["Hello, how are you?", "What is 2+2?"]
        encoded = tokenizer(prompts, return_tensors="pt", padding=True)

        # Create input tensordict
        td = TensorDict(
            {
                "tokens": Tokens(
                    prompt=encoded["input_ids"],
                )
            },
            batch_size=[2],
        )

        # Run generation
        result = wrapper(td)

        # Check outputs
        assert "tokens" in result.keys()
        assert hasattr(result["tokens"], "response")
        assert hasattr(result["tokens"], "full")

    def test_log_probs(self, sglang_service, tokenizer):
        """Test log probability extraction."""
        from torchrl.modules.llm.policies import SGLangWrapper
        from torchrl.modules.llm.policies.common import Text

        wrapper = SGLangWrapper(
            model=sglang_service,
            tokenizer=tokenizer,
            input_mode="text",
            generate=True,
            return_log_probs=True,
            generate_kwargs={"max_new_tokens": 10, "temperature": 0.0},
        )

        # Create input tensordict
        td = TensorDict(
            {
                "text": Text(
                    prompt=["Hello, how are you?"],
                )
            },
            batch_size=[1],
        )

        # Run generation
        result = wrapper(td)

        # Check log probs are returned
        assert "log_probs" in result.keys() or hasattr(result.get("text"), "log_probs")

    def test_get_new_version(self, sglang_service, tokenizer):
        """Test get_new_version for policy version tracking."""
        from torchrl.modules.llm.policies import SGLangWrapper

        wrapper = SGLangWrapper(
            model=sglang_service,
            tokenizer=tokenizer,
            input_mode="history",
        )

        # Should return new wrapper
        new_wrapper = wrapper.get_new_version()
        assert new_wrapper is not wrapper
        # Both should be valid SGLangWrapper instances
        assert isinstance(new_wrapper, SGLangWrapper)


if __name__ == "__main__":
    args, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst", "-v", "-s"] + unknown)
