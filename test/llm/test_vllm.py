# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import asyncio
import importlib.util
import os
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import torch

# Set environment variable for vLLM V0 engine
os.environ["VLLM_USE_V1"] = "0"

_has_vllm = importlib.util.find_spec("vllm") is not None
_has_ray = importlib.util.find_spec("ray") is not None

# Skip entire module if vLLM is not available
pytestmark = pytest.mark.skipif(not _has_vllm, reason="vllm not available")


# Fixtures for async vLLM testing
@pytest.fixture
def mock_async_engine_args():
    """Mock AsyncEngineArgs for testing."""
    if not _has_vllm:
        pytest.skip("vllm not available")

    from vllm import AsyncEngineArgs

    return AsyncEngineArgs(
        model="Qwen/Qwen2.5-0.5B",
        tensor_parallel_size=1,
        max_model_len=1024,
        max_num_batched_tokens=1024,
        enable_prefix_caching=True,
    )


@pytest.fixture
def mock_sampling_params():
    """Mock SamplingParams for testing."""
    if not _has_vllm:
        pytest.skip("vllm not available")

    from vllm import SamplingParams

    return SamplingParams(
        temperature=0.7,
        top_p=0.9,
        max_tokens=50,
    )


@pytest.fixture
def mock_request_output():
    """Mock RequestOutput for testing."""
    if not _has_vllm:
        pytest.skip("vllm not available")

    from vllm import RequestOutput
    from vllm.outputs import CompletionOutput

    # Create a mock completion output
    completion = MagicMock(spec=CompletionOutput)
    completion.text = "Hello, world!"
    completion.token_ids = [1, 2, 3, 4, 5]
    completion.logprobs = None
    completion.finish_reason = "stop"

    # Create a mock request output
    output = MagicMock(spec=RequestOutput)
    output.request_id = str(uuid.uuid4())
    output.prompt = "Test prompt"
    output.prompt_token_ids = [1, 2, 3]
    output.prompt_logprobs = None
    output.outputs = [completion]
    output.finished = True
    output.metrics = None
    output.lora_request = None
    output.encoder_prompt = None
    output.encoder_prompt_token_ids = None
    output.num_cached_tokens = 0

    return output


class TestAsyncvLLMWorker:
    """Test cases for AsyncvLLMWorker."""

    @pytest.mark.skipif(not _has_vllm, reason="vllm not available")
    def test_worker_initialization(self):
        """Test AsyncvLLMWorker initialization."""
        from torchrl.modules.llm.backends.vllm_async import AsyncvLLMWorker

        with patch(
            "torchrl.modules.llm.backends.vllm_async.Worker.__init__"
        ) as mock_init:
            mock_init.return_value = None

            worker = AsyncvLLMWorker()

            # Check that parent __init__ was called
            mock_init.assert_called_once()

            # Check that model_update_group is initialized to None
            assert worker.model_update_group is None

    @pytest.mark.skipif(not _has_vllm, reason="vllm not available")
    def test_init_weight_update_group(self):
        """Test weight update group initialization."""
        from torchrl.modules.llm.backends.vllm_async import AsyncvLLMWorker

        # Mock dependencies
        with patch("torchrl.modules.llm.backends.vllm_async.Worker.__init__"):
            with patch(
                "vllm.distributed.parallel_state.get_world_group"
            ) as mock_get_group:
                with patch(
                    "torchrl.modules.llm.backends.vllm_async.stateless_init_process_group_async"
                ) as mock_init_group:

                    # Setup mocks
                    mock_tp_group = MagicMock()
                    mock_tp_group.rank = 0
                    mock_get_group.return_value = mock_tp_group

                    mock_comm_group = MagicMock()
                    mock_init_group.return_value = mock_comm_group

                    worker = AsyncvLLMWorker()
                    worker.device = torch.device("cuda:0")

                    # Test the method
                    worker.init_weight_update_group("localhost", "12345", 1, 4)

                    # Verify calls
                    mock_get_group.assert_called_once()
                    mock_init_group.assert_called_once_with(
                        "localhost", "12345", 1, 4, torch.device("cuda:0")
                    )

                    # Check that model_update_group is set
                    assert worker.model_update_group == mock_comm_group

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.skipif(not _has_vllm, reason="vllm not available")
    def test_update_weight_broadcast(self):
        """Test weight update via broadcast."""
        from torchrl.modules.llm.backends.vllm_async import AsyncvLLMWorker

        with patch("torchrl.modules.llm.backends.vllm_async.Worker.__init__"):
            worker = AsyncvLLMWorker()

            # Mock model_update_group
            mock_group = MagicMock()
            worker.model_update_group = mock_group

            # Mock model_runner
            mock_model_runner = MagicMock()
            worker.model_runner = mock_model_runner

            # Test the method
            with patch("torch.empty") as mock_empty:
                mock_weight = MagicMock()
                mock_empty.return_value = mock_weight

                worker.update_weight_broadcast(
                    "test_param", torch.float32, torch.Size([2, 3])
                )

                # Verify calls
                mock_empty.assert_called_once_with(
                    torch.Size([2, 3]), dtype=torch.float32, device="cuda"
                )
                mock_group.broadcast.assert_called_once()
                mock_model_runner.model.load_weights.assert_called_once_with(
                    weights=[("test_param", mock_weight)]
                )

    @pytest.mark.skipif(not _has_vllm, reason="vllm not available")
    def test_update_weight_broadcast_no_group(self):
        """Test weight update broadcast fails when group is not initialized."""
        from torchrl.modules.llm.backends.vllm_async import AsyncvLLMWorker

        with patch("torchrl.modules.llm.backends.vllm_async.Worker.__init__"):
            worker = AsyncvLLMWorker()
            # model_update_group is None by default

            # Test that it raises RuntimeError
            with pytest.raises(
                RuntimeError, match="Weight update group not initialized"
            ):
                worker.update_weight_broadcast(
                    "test_param", torch.float32, torch.Size([2, 3])
                )

    @pytest.mark.skipif(not _has_vllm, reason="vllm not available")
    def test_update_weight(self):
        """Test direct weight update."""
        from torchrl.modules.llm.backends.vllm_async import AsyncvLLMWorker

        with patch("torchrl.modules.llm.backends.vllm_async.Worker.__init__"):
            worker = AsyncvLLMWorker()

            # Mock model_runner
            mock_model_runner = MagicMock()
            worker.model_runner = mock_model_runner

            # Test the method
            mock_weight = torch.randn(2, 3)
            worker.update_weight("test_param", mock_weight)

            # Verify call
            mock_model_runner.model.load_weights.assert_called_once_with(
                weights=[("test_param", mock_weight)]
            )


class TestAsyncLLMEngineExtended:
    """Test cases for AsyncLLMEngineExtended."""

    @pytest.mark.skipif(not _has_vllm, reason="vllm not available")
    def test_initialization(self, mock_async_engine_args):
        """Test AsyncLLMEngineExtended initialization."""
        from torchrl.modules.llm.backends.vllm_async import AsyncLLMEngineExtended

        with patch("vllm.AsyncLLMEngine.from_engine_args") as mock_from_args:
            mock_engine = MagicMock()
            mock_from_args.return_value = mock_engine

            engine = AsyncLLMEngineExtended(
                mock_async_engine_args, bundle_indices=[0, 1]
            )

            # Check that engine args were modified correctly
            assert (
                mock_async_engine_args.worker_cls
                == "torchrl.modules.llm.backends.vllm_async.AsyncvLLMWorker"
            )
            assert mock_async_engine_args.distributed_executor_backend == "ray"
            assert mock_async_engine_args.enable_prefix_caching is True

            # Check engine was created
            mock_from_args.assert_called_once_with(mock_async_engine_args)
            assert engine.engine == mock_engine
            assert engine.bundle_indices == [0, 1]

    @pytest.mark.skipif(not _has_vllm, reason="vllm not available")
    def test_ready(self, mock_async_engine_args):
        """Test ready method."""
        from torchrl.modules.llm.backends.vllm_async import AsyncLLMEngineExtended

        with patch("vllm.AsyncLLMEngine.from_engine_args"):
            engine = AsyncLLMEngineExtended(mock_async_engine_args)
            assert engine.ready() is True

    @pytest.mark.skipif(not _has_vllm, reason="vllm not available")
    @pytest.mark.asyncio
    async def test_generate_with_tokens(
        self, mock_async_engine_args, mock_sampling_params, mock_request_output
    ):
        """Test async generate method with prompt_token_ids."""
        from torchrl.modules.llm.backends.vllm_async import AsyncLLMEngineExtended

        with patch("vllm.AsyncLLMEngine.from_engine_args") as mock_from_args:
            # Mock the async engine
            mock_engine = AsyncMock()
            mock_from_args.return_value = mock_engine

            # Mock the async generator
            async def mock_generate(*args, **kwargs):
                yield mock_request_output

            mock_engine.generate = mock_generate

            engine = AsyncLLMEngineExtended(mock_async_engine_args)

            # Test generation with prompt_token_ids
            tokens = [1, 2, 3, 4, 5]
            result = await engine.generate(
                prompt_token_ids=tokens, sampling_params=mock_sampling_params
            )

            assert result == mock_request_output

    @pytest.mark.skipif(not _has_vllm, reason="vllm not available")
    @pytest.mark.asyncio
    async def test_generate_with_text(
        self, mock_async_engine_args, mock_sampling_params, mock_request_output
    ):
        """Test async generate method with text prompts."""
        from torchrl.modules.llm.backends.vllm_async import AsyncLLMEngineExtended

        with patch("vllm.AsyncLLMEngine.from_engine_args") as mock_from_args:
            # Mock the async engine
            mock_engine = AsyncMock()
            mock_from_args.return_value = mock_engine

            # Mock the async generator
            async def mock_generate(*args, **kwargs):
                yield mock_request_output

            mock_engine.generate = mock_generate

            engine = AsyncLLMEngineExtended(mock_async_engine_args)

            # Test generation with text
            text = "Hello, world!"
            result = await engine.generate(
                prompts=text, sampling_params=mock_sampling_params
            )

            assert result == mock_request_output

    @pytest.mark.skipif(not _has_vllm, reason="vllm not available")
    @pytest.mark.asyncio
    async def test_generate_timeout(self, mock_async_engine_args, mock_sampling_params):
        """Test generate with timeout."""
        from torchrl.modules.llm.backends.vllm_async import AsyncLLMEngineExtended

        with patch("vllm.AsyncLLMEngine.from_engine_args") as mock_from_args:
            # Mock the async engine
            mock_engine = AsyncMock()
            mock_from_args.return_value = mock_engine

            # Mock a slow async generator
            async def mock_slow_generate(*args, **kwargs):
                await asyncio.sleep(2.0)  # Sleep longer than timeout
                yield MagicMock()

            mock_engine.generate = mock_slow_generate
            mock_engine.abort = AsyncMock()  # Mock abort method

            engine = AsyncLLMEngineExtended(mock_async_engine_args)

            # Test timeout
            tokens = [1, 2, 3, 4, 5]
            with pytest.raises(TimeoutError, match="vLLM generation timed out"):
                await engine.generate(
                    prompt_token_ids=tokens,
                    sampling_params=mock_sampling_params,
                    timeout_seconds=0.1,
                )

    @pytest.mark.skipif(not _has_vllm, reason="vllm not available")
    @pytest.mark.asyncio
    async def test_generate_batch_prompts(
        self, mock_async_engine_args, mock_sampling_params, mock_request_output
    ):
        """Test async generate method with batch text prompts."""
        from torchrl.modules.llm.backends.vllm_async import AsyncLLMEngineExtended

        with patch("vllm.AsyncLLMEngine.from_engine_args") as mock_from_args:
            # Mock the async engine
            mock_engine = AsyncMock()
            mock_from_args.return_value = mock_engine

            # Mock returning multiple outputs for batch
            async def mock_generate(*args, **kwargs):
                yield mock_request_output
                yield mock_request_output

            mock_engine.generate = mock_generate

            engine = AsyncLLMEngineExtended(mock_async_engine_args)

            # Test generation with batch of text prompts
            prompts = ["Hello, world!", "How are you?"]
            result = await engine.generate(
                prompts=prompts, sampling_params=mock_sampling_params
            )

            # Should return a list of outputs
            assert isinstance(result, list)
            assert len(result) == 2
            assert result[0] == mock_request_output
            assert result[1] == mock_request_output

    @pytest.mark.skipif(not _has_vllm, reason="vllm not available")
    @pytest.mark.asyncio
    async def test_generate_batch_token_ids(
        self, mock_async_engine_args, mock_sampling_params, mock_request_output
    ):
        """Test async generate method with batch token IDs."""
        from torchrl.modules.llm.backends.vllm_async import AsyncLLMEngineExtended

        with patch("vllm.AsyncLLMEngine.from_engine_args") as mock_from_args:
            # Mock the async engine
            mock_engine = AsyncMock()
            mock_from_args.return_value = mock_engine

            # Mock returning multiple outputs for batch
            async def mock_generate(*args, **kwargs):
                yield mock_request_output
                yield mock_request_output

            mock_engine.generate = mock_generate

            engine = AsyncLLMEngineExtended(mock_async_engine_args)

            # Test generation with batch of token IDs
            token_ids = [[1, 2, 3], [4, 5, 6]]
            result = await engine.generate(
                prompt_token_ids=token_ids, sampling_params=mock_sampling_params
            )

            # Should return a list of outputs
            assert isinstance(result, list)
            assert len(result) == 2
            assert result[0] == mock_request_output
            assert result[1] == mock_request_output

    @pytest.mark.skipif(not _has_vllm, reason="vllm not available")
    @pytest.mark.asyncio
    async def test_generate_tokens_prompt_objects(
        self, mock_async_engine_args, mock_sampling_params, mock_request_output
    ):
        """Test async generate method with TokensPrompt objects."""
        from torchrl.modules.llm.backends.vllm_async import AsyncLLMEngineExtended

        with patch("vllm.AsyncLLMEngine.from_engine_args") as mock_from_args:
            with patch("vllm.TokensPrompt") as MockTokensPrompt:
                # Mock the async engine
                mock_engine = AsyncMock()
                mock_from_args.return_value = mock_engine

                # Mock the async generator
                async def mock_generate(*args, **kwargs):
                    yield mock_request_output

                mock_engine.generate = mock_generate

                # Mock TokensPrompt
                mock_tokens_prompt = MagicMock()
                MockTokensPrompt.return_value = mock_tokens_prompt

                engine = AsyncLLMEngineExtended(mock_async_engine_args)

                # Test generation with single token list (should create TokensPrompt)
                tokens = [1, 2, 3, 4, 5]
                result = await engine.generate(
                    prompt_token_ids=tokens, sampling_params=mock_sampling_params
                )

                # Verify TokensPrompt was created
                MockTokensPrompt.assert_called_with(prompt_token_ids=tokens)
                assert result == mock_request_output

    @pytest.mark.skipif(not _has_vllm, reason="vllm not available")
    @pytest.mark.asyncio
    async def test_generate_missing_inputs(
        self, mock_async_engine_args, mock_sampling_params
    ):
        """Test generate method with missing inputs."""
        from torchrl.modules.llm.backends.vllm_async import AsyncLLMEngineExtended

        with patch("vllm.AsyncLLMEngine.from_engine_args") as mock_from_args:
            mock_engine = AsyncMock()
            mock_from_args.return_value = mock_engine

            engine = AsyncLLMEngineExtended(mock_async_engine_args)

            # Test with neither prompts nor prompt_token_ids
            with pytest.raises(
                ValueError, match="Either prompts or prompt_token_ids must be provided"
            ):
                await engine.generate(sampling_params=mock_sampling_params)

    @pytest.mark.skipif(not _has_vllm, reason="vllm not available")
    @pytest.mark.asyncio
    async def test_generate_with_lora_request(
        self, mock_async_engine_args, mock_sampling_params, mock_request_output
    ):
        """Test async generate method with LoRA request."""
        from torchrl.modules.llm.backends.vllm_async import AsyncLLMEngineExtended

        with patch("vllm.AsyncLLMEngine.from_engine_args") as mock_from_args:
            # Mock the async engine
            mock_engine = AsyncMock()
            mock_from_args.return_value = mock_engine

            # Mock the async generator
            async def mock_generate(*args, **kwargs):
                yield mock_request_output

            mock_engine.generate = mock_generate

            engine = AsyncLLMEngineExtended(mock_async_engine_args)

            # Test generation with LoRA request
            mock_lora_request = MagicMock()
            result = await engine.generate(
                prompts="Test prompt",
                sampling_params=mock_sampling_params,
                lora_request=mock_lora_request,
            )

            assert result == mock_request_output

    @pytest.mark.skipif(not _has_vllm, reason="vllm not available")
    @pytest.mark.asyncio
    async def test_get_tokenizer(self, mock_async_engine_args):
        """Test get_tokenizer method."""
        from torchrl.modules.llm.backends.vllm_async import AsyncLLMEngineExtended

        with patch("vllm.AsyncLLMEngine.from_engine_args") as mock_from_args:
            mock_engine = AsyncMock()
            mock_tokenizer = MagicMock()
            mock_engine.get_tokenizer.return_value = mock_tokenizer
            mock_from_args.return_value = mock_engine

            engine = AsyncLLMEngineExtended(mock_async_engine_args)

            result = await engine.get_tokenizer()
            assert result == mock_tokenizer

    @pytest.mark.skipif(not _has_vllm, reason="vllm not available")
    def test_collective_rpc_v0(self, mock_async_engine_args):
        """Test collective_rpc_v0 method."""
        from torchrl.modules.llm.backends.vllm_async import AsyncLLMEngineExtended

        with patch("vllm.AsyncLLMEngine.from_engine_args") as mock_from_args:
            mock_engine = MagicMock()
            mock_inner_engine = MagicMock()
            mock_engine.engine = mock_inner_engine
            mock_from_args.return_value = mock_engine

            engine = AsyncLLMEngineExtended(mock_async_engine_args)

            # Test method call
            engine.collective_rpc_v0(
                "test_method", args=(1, 2), kwargs={"key": "value"}
            )

            # Verify call was forwarded correctly
            mock_inner_engine.collective_rpc.assert_called_once_with(
                "test_method", None, (1, 2), {"key": "value"}
            )


class TestAsyncVLLMEngineService:
    """Test cases for AsyncVLLMEngineService."""

    @pytest.mark.skipif(not _has_vllm, reason="vllm not available")
    @pytest.mark.skipif(not _has_ray, reason="ray not available")
    def test_initialization(self, mock_async_engine_args):
        """Test AsyncVLLMEngineService initialization."""
        from torchrl.modules.llm.backends.vllm_async import AsyncVLLMEngineService

        service = AsyncVLLMEngineService(mock_async_engine_args, num_replicas=2)

        assert service.engine_args == mock_async_engine_args
        assert service.num_replicas == 2
        assert service.actors == []
        assert service._launched is False
        assert service._placement_group is None
        assert len(service._service_id) == 8  # UUID hex[:8]
        assert mock_async_engine_args.enable_prefix_caching is True

    @pytest.mark.skipif(not _has_vllm, reason="vllm not available")
    @pytest.mark.skipif(not _has_ray, reason="ray not available")
    def test_get_random_actor_index(self, mock_async_engine_args):
        """Test get_random_actor_index method."""
        from torchrl.modules.llm.backends.vllm_async import AsyncVLLMEngineService

        service = AsyncVLLMEngineService(mock_async_engine_args, num_replicas=3)

        # Mock actors
        service.actors = [MagicMock(), MagicMock(), MagicMock()]

        # Test multiple calls to ensure randomness works
        indices = [service.get_random_actor_index() for _ in range(10)]

        # All indices should be valid
        for idx in indices:
            assert 0 <= idx < 3

    @pytest.mark.skipif(not _has_vllm, reason="vllm not available")
    def test_gpus_per_replica(self, mock_async_engine_args):
        """Test gpus_per_replica function."""
        from torchrl.modules.llm.backends.vllm_async import gpus_per_replica

        # Test default values
        result = gpus_per_replica(mock_async_engine_args)
        expected = (
            mock_async_engine_args.tensor_parallel_size * 1 * 1
        )  # Default data_parallel_size and pipeline_parallel_size
        assert result == expected

        # Test with custom values
        mock_async_engine_args.tensor_parallel_size = 2
        # Note: data_parallel_size and pipeline_parallel_size might not exist in all vLLM versions
        # So we test with the defaults
        result = gpus_per_replica(mock_async_engine_args)
        assert result == 2

    @pytest.mark.skipif(not _has_vllm, reason="vllm not available")
    @pytest.mark.skipif(not _has_ray, reason="ray not available")
    def test_shutdown_no_actors(self, mock_async_engine_args):
        """Test shutdown when no actors are present."""
        from torchrl.modules.llm.backends.vllm_async import AsyncVLLMEngineService

        service = AsyncVLLMEngineService(mock_async_engine_args)

        # This should not raise any errors
        service.shutdown()

        assert service.actors == []
        assert service._launched is False
        assert service._placement_group is None

    @pytest.mark.skipif(not _has_vllm, reason="vllm not available")
    @pytest.mark.skipif(not _has_ray, reason="ray not available")
    def test_generate_with_text(
        self, mock_async_engine_args, mock_sampling_params, mock_request_output
    ):
        """Test AsyncVLLMEngineService generate method with text."""
        from torchrl.modules.llm.backends.vllm_async import AsyncVLLMEngineService

        service = AsyncVLLMEngineService(mock_async_engine_args, num_replicas=2)

        # Mock actors
        mock_actor1 = MagicMock()
        mock_actor2 = MagicMock()
        service.actors = [mock_actor1, mock_actor2]

        # Mock ray.get to return our mock output
        with patch("ray.get", return_value=mock_request_output):
            result = service.generate(
                prompts="Hello, world!",
                sampling_params=mock_sampling_params,
                timeout_seconds=10.0,
            )

            assert result == mock_request_output

    @pytest.mark.skipif(not _has_vllm, reason="vllm not available")
    @pytest.mark.skipif(not _has_ray, reason="ray not available")
    def test_generate_with_tokens(
        self, mock_async_engine_args, mock_sampling_params, mock_request_output
    ):
        """Test AsyncVLLMEngineService generate method with token IDs."""
        from torchrl.modules.llm.backends.vllm_async import AsyncVLLMEngineService

        service = AsyncVLLMEngineService(mock_async_engine_args, num_replicas=2)

        # Mock actors
        mock_actor1 = MagicMock()
        mock_actor2 = MagicMock()
        service.actors = [mock_actor1, mock_actor2]

        # Mock ray.get to return our mock output
        with patch("ray.get", return_value=mock_request_output):
            result = service.generate(
                prompt_token_ids=[1, 2, 3, 4, 5],
                sampling_params=mock_sampling_params,
                actor_index=0,  # Use specific actor
            )

            assert result == mock_request_output
            # Verify the correct actor was used
            mock_actor1.generate.remote.assert_called_once()

    @pytest.mark.skipif(not _has_vllm, reason="vllm not available")
    @pytest.mark.skipif(not _has_ray, reason="ray not available")
    def test_generate_batch_prompts(
        self, mock_async_engine_args, mock_sampling_params, mock_request_output
    ):
        """Test AsyncVLLMEngineService generate method with batch prompts."""
        from torchrl.modules.llm.backends.vllm_async import AsyncVLLMEngineService

        service = AsyncVLLMEngineService(mock_async_engine_args, num_replicas=1)

        # Mock actor
        mock_actor = MagicMock()
        service.actors = [mock_actor]

        # Mock ray.get to return list of outputs
        mock_outputs = [mock_request_output, mock_request_output]
        with patch("ray.get", return_value=mock_outputs):
            result = service.generate(
                prompts=["Hello, world!", "How are you?"],
                sampling_params=mock_sampling_params,
            )

            assert result == mock_outputs

    @pytest.mark.skipif(not _has_vllm, reason="vllm not available")
    @pytest.mark.skipif(not _has_ray, reason="ray not available")
    def test_generate_with_lora_request(
        self, mock_async_engine_args, mock_sampling_params, mock_request_output
    ):
        """Test AsyncVLLMEngineService generate method with LoRA request."""
        from torchrl.modules.llm.backends.vllm_async import AsyncVLLMEngineService

        service = AsyncVLLMEngineService(mock_async_engine_args, num_replicas=1)

        # Mock actor
        mock_actor = MagicMock()
        service.actors = [mock_actor]

        # Mock ray.get to return our mock output
        with patch("ray.get", return_value=mock_request_output):
            mock_lora_request = MagicMock()
            result = service.generate(
                prompts="Test prompt",
                sampling_params=mock_sampling_params,
                lora_request=mock_lora_request,
            )

            assert result == mock_request_output
            # Verify the LoRA request was passed through
            mock_actor.generate.remote.assert_called_once()
            call_args = mock_actor.generate.remote.call_args
            assert call_args.kwargs["lora_request"] == mock_lora_request


class TestMakeAsyncVLLMEngine:
    """Test cases for make_async_vllm_engine function."""

    @pytest.mark.skipif(not _has_vllm, reason="vllm not available")
    @pytest.mark.skipif(not _has_ray, reason="ray not available")
    def test_make_async_vllm_engine_basic(self):
        """Test basic make_async_vllm_engine functionality."""
        from torchrl.modules.llm.backends.vllm_async import make_async_vllm_engine

        with patch(
            "torchrl.modules.llm.backends.vllm_async.AsyncVLLMEngineService"
        ) as MockService:
            mock_service_instance = MagicMock()
            MockService.launch.return_value = mock_service_instance

            # Test with basic parameters
            result = make_async_vllm_engine("test-model", num_devices=2, num_replicas=3)

            # Verify AsyncVLLMEngineService.launch was called
            MockService.launch.assert_called_once()
            args, kwargs = MockService.launch.call_args

            # Check engine_args
            engine_args = args[0]
            assert engine_args.model == "test-model"
            assert engine_args.tensor_parallel_size == 2
            assert engine_args.distributed_executor_backend == "ray"
            assert (
                engine_args.worker_cls
                == "torchrl.modules.llm.backends.vllm_async.AsyncvLLMWorker"
            )
            assert engine_args.enable_prefix_caching is True

            # Check num_replicas
            assert args[1] == 3

            assert result == mock_service_instance

    @pytest.mark.skipif(not _has_vllm, reason="vllm not available")
    def test_make_async_vllm_engine_device_validation(self):
        """Test device validation in make_async_vllm_engine."""
        from torchrl.modules.llm.backends.vllm_async import make_async_vllm_engine

        # Test conflicting parameters
        with pytest.raises(
            ValueError, match="Cannot specify both num_devices and devices"
        ):
            make_async_vllm_engine("test-model", devices=[0, 1], num_devices=2)

        # Test invalid device index
        with patch("torch.cuda.device_count", return_value=2):
            with pytest.raises(ValueError, match="Invalid device index: 3"):
                make_async_vllm_engine("test-model", devices=[0, 1, 3])

    @pytest.mark.skipif(not _has_vllm, reason="vllm not available")
    @pytest.mark.skipif(not _has_ray, reason="ray not available")
    def test_make_async_vllm_engine_defaults(self):
        """Test make_async_vllm_engine with default parameters."""
        from torchrl.modules.llm.backends.vllm_async import make_async_vllm_engine

        with patch(
            "torchrl.modules.llm.backends.vllm_async.AsyncVLLMEngineService"
        ) as MockService:
            with patch("torch.cuda.device_count", return_value=4):
                mock_service_instance = MagicMock()
                MockService.launch.return_value = mock_service_instance

                # Test with minimal parameters
                make_async_vllm_engine("test-model")

                # Verify defaults
                MockService.launch.assert_called_once()
                args, kwargs = MockService.launch.call_args

                engine_args = args[0]
                assert (
                    engine_args.tensor_parallel_size == 1
                )  # Default when no devices specified
                assert args[1] == 1  # Default num_replicas


class TestStatelessInitProcessGroupAsync:
    """Test cases for stateless_init_process_group_async function."""

    @pytest.mark.skipif(not _has_vllm, reason="vllm not available")
    def test_stateless_init_process_group_async(self):
        """Test stateless_init_process_group_async function."""
        from torchrl.modules.llm.backends.vllm_async import (
            stateless_init_process_group_async,
        )

        with patch(
            "torchrl.modules.llm.backends.vllm_async.StatelessProcessGroup"
        ) as MockPG:
            with patch(
                "torchrl.modules.llm.backends.vllm_async.PyNcclCommunicator"
            ) as MockComm:
                with patch(
                    "torchrl.modules.llm.backends.vllm_async.get_open_port",
                    return_value="12345",
                ):

                    mock_pg = MagicMock()
                    MockPG.create.return_value = mock_pg

                    mock_comm = MagicMock()
                    MockComm.return_value = mock_comm

                    device = torch.device("cuda:0")
                    result = stateless_init_process_group_async(
                        "localhost", "12345", 0, 4, device
                    )

                    # Verify calls
                    MockPG.create.assert_called_once_with(
                        host="localhost", port=12345, rank=0, world_size=4
                    )
                    MockComm.assert_called_once_with(mock_pg, device=device)

                    assert result == mock_comm

    @pytest.mark.skipif(not _has_vllm, reason="vllm not available")
    def test_stateless_init_process_group_async_defaults(self):
        """Test stateless_init_process_group_async with default parameters."""
        from torchrl.modules.llm.backends.vllm_async import (
            stateless_init_process_group_async,
        )

        with patch(
            "torchrl.modules.llm.backends.vllm_async.StatelessProcessGroup"
        ) as MockPG:
            with patch(
                "torchrl.modules.llm.backends.vllm_async.PyNcclCommunicator"
            ) as MockComm:
                with patch(
                    "torchrl.modules.llm.backends.vllm_async.get_open_port",
                    return_value="54321",
                ):

                    mock_pg = MagicMock()
                    MockPG.create.return_value = mock_pg

                    mock_comm = MagicMock()
                    MockComm.return_value = mock_comm

                    device = torch.device("cuda:0")
                    result = stateless_init_process_group_async(
                        None, None, 1, 2, device
                    )

                    # Verify defaults were applied
                    MockPG.create.assert_called_once_with(
                        host="localhost", port=54321, rank=1, world_size=2
                    )
                    MockComm.assert_called_once_with(mock_pg, device=device)

                    assert result == mock_comm


# Integration test with actual vLLM components (if available)
class TestIntegration:
    """Integration tests that require actual vLLM components."""

    @pytest.mark.skipif(not _has_vllm, reason="vllm not available")
    def test_async_engine_args_compatibility(self):
        """Test that our AsyncEngineArgs usage is compatible with vLLM."""
        from torchrl.modules.llm.backends.vllm_async import AsyncLLMEngineExtended
        from vllm import AsyncEngineArgs

        # Create real AsyncEngineArgs
        engine_args = AsyncEngineArgs(
            model="Qwen/Qwen2.5-0.5B",
            tensor_parallel_size=1,
            max_model_len=512,
            max_num_batched_tokens=512,
        )

        # This should not raise any errors during initialization
        # (though it won't actually create the engine without proper setup)
        with patch("vllm.AsyncLLMEngine.from_engine_args") as mock_from_args:
            mock_from_args.return_value = MagicMock()

            AsyncLLMEngineExtended(engine_args)

            # Verify the args were modified correctly
            assert (
                engine_args.worker_cls
                == "torchrl.modules.llm.backends.vllm_async.AsyncvLLMWorker"
            )
            assert engine_args.distributed_executor_backend == "ray"
            assert engine_args.enable_prefix_caching is True

    @pytest.mark.skipif(not _has_vllm, reason="vllm not available")
    def test_sampling_params_compatibility(self):
        """Test that our SamplingParams usage is compatible with vLLM."""
        from vllm import SamplingParams

        # Test various parameter combinations
        params1 = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=50)
        assert params1.temperature == 0.7
        assert params1.top_p == 0.9
        assert params1.max_tokens == 50

        params2 = SamplingParams(temperature=0.0)  # Greedy decoding
        assert params2.temperature == 0.0

        params3 = SamplingParams(n=3, temperature=1.0)  # Multiple samples
        assert params3.n == 3
        assert params3.temperature == 1.0


if __name__ == "__main__":
    pytest.main([__file__, "--capture", "no", "--exitfirst", "-v"])
