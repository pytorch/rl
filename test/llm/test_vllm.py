# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import importlib.util
import os
from unittest.mock import MagicMock, patch

import pytest
import torch

# Set environment variable for vLLM V0 engine
os.environ["VLLM_USE_V1"] = "0"

_has_vllm = importlib.util.find_spec("vllm") is not None
_has_ray = importlib.util.find_spec("ray") is not None

# Skip entire module if vLLM is not available
pytestmark = pytest.mark.skipif(not _has_vllm, reason="vllm not available")


@pytest.fixture
def mock_sampling_params():
    """Mock SamplingParams for testing."""
    if not _has_vllm:
        pytest.skip("vllm not available")

    from vllm import SamplingParams

    return SamplingParams(temperature=0.7, top_p=0.9, max_tokens=50)


@pytest.fixture
def mock_request_output():
    """Mock RequestOutput for testing."""
    if not _has_vllm:
        pytest.skip("vllm not available")

    from vllm.outputs import CompletionOutput

    # Create a mock completion output
    completion = MagicMock(spec=CompletionOutput)
    completion.text = "Hello, world!"
    completion.token_ids = [1, 2, 3, 4, 5]
    completion.logprobs = None
    completion.finish_reason = "stop"

    # Create a mock request output
    output = MagicMock()
    output.prompt = "Test prompt"
    output.outputs = [completion]
    output.finished = True

    return output


class TestAsyncVLLM:
    """Test the public AsyncVLLM API."""

    @pytest.mark.skipif(not _has_vllm, reason="vllm not available")
    @pytest.mark.skipif(not _has_ray, reason="ray not available")
    def test_from_pretrained_single_replica(self, mock_sampling_params):
        """Test AsyncVLLM.from_pretrained with single replica."""
        from torchrl.modules.llm.backends.vllm_async import AsyncVLLM

        with patch.object(AsyncVLLM, "_launch") as mock_launch:
            with patch("torch.cuda.device_count", return_value=2):
                service = AsyncVLLM.from_pretrained(
                    "Qwen/Qwen2.5-0.5B", num_replicas=1, max_model_len=512
                )

                # Verify service was configured correctly
                assert service.num_replicas == 1
                assert service.engine_args.model == "Qwen/Qwen2.5-0.5B"
                assert service.engine_args.max_model_len == 512
                assert service.engine_args.enable_prefix_caching is True
                mock_launch.assert_called_once()

    @pytest.mark.skipif(not _has_vllm, reason="vllm not available")
    @pytest.mark.skipif(not _has_ray, reason="ray not available")
    def test_from_pretrained_multi_replica(self, mock_sampling_params):
        """Test AsyncVLLM.from_pretrained with multiple replicas."""
        from torchrl.modules.llm.backends.vllm_async import AsyncVLLM

        with patch.object(AsyncVLLM, "_launch") as mock_launch:
            service = AsyncVLLM.from_pretrained(
                "Qwen/Qwen2.5-0.5B", num_replicas=2, num_devices=1
            )

            # Verify service was configured correctly
            assert service.num_replicas == 2
            assert service.engine_args.tensor_parallel_size == 1
            mock_launch.assert_called_once()

    @pytest.mark.skipif(not _has_vllm, reason="vllm not available")
    @pytest.mark.skipif(not _has_ray, reason="ray not available")
    def test_generate_text_input(self, mock_sampling_params, mock_request_output):
        """Test generation with text input (same as vLLM.LLM API)."""
        from torchrl.modules.llm.backends.vllm_async import AsyncVLLM

        with patch.object(AsyncVLLM, "_launch"):
            service = AsyncVLLM.from_pretrained("Qwen/Qwen2.5-0.5B", num_replicas=1)

            # Mock actors
            mock_actor = MagicMock()
            service.actors = [mock_actor]

            # Mock ray.get to return our mock output
            with patch("ray.get", return_value=mock_request_output):
                result = service.generate(
                    prompts="Hello, world!", sampling_params=mock_sampling_params
                )

                assert result == mock_request_output
                mock_actor.generate.remote.assert_called_once()

    @pytest.mark.skipif(not _has_vllm, reason="vllm not available")
    @pytest.mark.skipif(not _has_ray, reason="ray not available")
    def test_generate_token_input(self, mock_sampling_params, mock_request_output):
        """Test generation with token IDs input (same as vLLM.LLM API)."""
        from torchrl.modules.llm.backends.vllm_async import AsyncVLLM

        with patch.object(AsyncVLLM, "_launch"):
            service = AsyncVLLM.from_pretrained("Qwen/Qwen2.5-0.5B", num_replicas=1)

            # Mock actors
            mock_actor = MagicMock()
            service.actors = [mock_actor]

            # Mock ray.get to return our mock output
            with patch("ray.get", return_value=mock_request_output):
                result = service.generate(
                    prompt_token_ids=[1, 2, 3, 4, 5],
                    sampling_params=mock_sampling_params,
                )

                assert result == mock_request_output
                mock_actor.generate.remote.assert_called_once()

    @pytest.mark.skipif(not _has_vllm, reason="vllm not available")
    @pytest.mark.skipif(not _has_ray, reason="ray not available")
    def test_generate_batch_input(self, mock_sampling_params, mock_request_output):
        """Test generation with batch input (same as vLLM.LLM API)."""
        from torchrl.modules.llm.backends.vllm_async import AsyncVLLM

        with patch.object(AsyncVLLM, "_launch"):
            service = AsyncVLLM.from_pretrained("Qwen/Qwen2.5-0.5B", num_replicas=1)

            # Mock actors
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
                mock_actor.generate.remote.assert_called_once()

    @pytest.mark.skipif(not _has_vllm, reason="vllm not available")
    @pytest.mark.skipif(not _has_ray, reason="ray not available")
    def test_weight_updates(self):
        """Test weight update functionality across replicas."""
        from torchrl.modules.llm.backends.vllm_async import AsyncVLLM

        with patch.object(AsyncVLLM, "_launch"):
            service = AsyncVLLM.from_pretrained("Qwen/Qwen2.5-0.5B", num_replicas=2)

            # Mock actors
            mock_actor1 = MagicMock()
            mock_actor2 = MagicMock()
            service.actors = [mock_actor1, mock_actor2]

            # Test weight update broadcast
            with patch("ray.get"):
                service.update_weight_broadcast(
                    "test_param", torch.float32, torch.Size([2, 3])
                )

                # Verify both actors received the update
                mock_actor1.update_weight_broadcast.remote.assert_called_once_with(
                    "test_param", torch.float32, torch.Size([2, 3])
                )
                mock_actor2.update_weight_broadcast.remote.assert_called_once_with(
                    "test_param", torch.float32, torch.Size([2, 3])
                )

            # Test direct weight update
            test_weight = torch.randn(2, 3)
            with patch("ray.get"):
                service.update_weight("test_param", test_weight)

                # Verify both actors received the update
                mock_actor1.update_weight.remote.assert_called_once_with(
                    "test_param", test_weight
                )
                mock_actor2.update_weight.remote.assert_called_once_with(
                    "test_param", test_weight
                )

    @pytest.mark.skipif(not _has_vllm, reason="vllm not available")
    @pytest.mark.skipif(not _has_ray, reason="ray not available")
    def test_shutdown(self):
        """Test proper cleanup of resources."""
        from torchrl.modules.llm.backends.vllm_async import AsyncVLLM

        with patch.object(AsyncVLLM, "_launch"):
            service = AsyncVLLM.from_pretrained("Qwen/Qwen2.5-0.5B", num_replicas=1)

            # Mock placement group and actors
            mock_placement_group = MagicMock()
            service._placement_group = mock_placement_group
            mock_actor = MagicMock()
            service.actors = [mock_actor]

            with patch("ray.kill") as mock_kill:
                with patch(
                    "ray.util.placement_group.remove_placement_group"
                ) as mock_remove_pg:
                    service.shutdown()

                    # Verify cleanup
                    mock_kill.assert_called_once_with(mock_actor, no_restart=True)
                    mock_remove_pg.assert_called_once_with(mock_placement_group)
                    assert service.actors == []
                    assert service._placement_group is None
                    assert service._launched is False


if __name__ == "__main__":
    pytest.main([__file__, "--capture", "no", "--exitfirst", "-v"])
