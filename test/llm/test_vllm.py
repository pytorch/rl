# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import importlib.util

import pytest
import torch

_has_vllm = importlib.util.find_spec("vllm") is not None
_has_ray = importlib.util.find_spec("ray") is not None

# Skip entire module if vLLM is not available
pytestmark = pytest.mark.skipif(not _has_vllm, reason="vllm not available")

MODEL_NAME = "Qwen/Qwen2.5-0.5B"


@pytest.fixture(scope="module")
def sampling_params():
    """Real SamplingParams for testing."""
    if not _has_vllm:
        pytest.skip("vllm not available")

    from vllm import SamplingParams

    return SamplingParams(
        temperature=0.0, max_tokens=10
    )  # Use greedy decoding for reproducibility


@pytest.mark.xfail(
    reason="AsyncVLLM tests fail due to Ray placement group timeout. "
    "ray.get(pg.ready(), timeout=180) times out. See LLM_TEST_ISSUES.md for details.",
    strict=False,
)
class TestAsyncVLLMIntegration:
    """Integration tests for AsyncVLLM with real models."""

    @pytest.mark.gpu
    @pytest.mark.skipif(not _has_vllm, reason="vllm not available")
    @pytest.mark.skipif(not _has_ray, reason="ray not available")
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.slow
    def test_vllm_api_compatibility(self, sampling_params):
        """Test that AsyncVLLM supports the same inputs as vLLM.LLM.generate()."""
        from torchrl.modules.llm.backends import AsyncVLLM

        # Create AsyncVLLM service
        service = AsyncVLLM.from_pretrained(
            MODEL_NAME,
            num_replicas=1,
            max_model_len=512,
            gpu_memory_utilization=0.3,  # Use less GPU memory for CI
        )

        try:
            # Get tokenizer first for token tests
            import ray

            tokenizer = ray.get(service.actors[0].get_tokenizer.remote())

            # Test 1: Single text prompt (string)
            result1 = service.generate("Hello, world!", sampling_params)
            # Handle potential list return for single input
            if isinstance(result1, list):
                output1 = result1[0]
            else:
                output1 = result1
            assert hasattr(output1, "outputs") and output1.outputs
            assert hasattr(output1.outputs[0], "text")

            # Test 2: List of text prompts
            prompts = ["Hello, world!", "How are you?"]
            result2 = service.generate(prompts, sampling_params)
            assert isinstance(result2, list)
            assert len(result2) == 2
            for output in result2:
                assert hasattr(output, "outputs") and output.outputs

            # Test 3: Token IDs (single sequence)
            token_ids = tokenizer.encode("Hello, world!")
            result3 = service.generate(
                prompt_token_ids=token_ids, sampling_params=sampling_params
            )
            # Handle potential list return for single input
            if isinstance(result3, list):
                output3 = result3[0]
            else:
                output3 = result3
            assert hasattr(output3, "outputs") and output3.outputs

            # Test 4: List of token ID sequences
            token_ids_batch = [
                tokenizer.encode("Hello, world!"),
                tokenizer.encode("How are you?"),
            ]
            result4 = service.generate(
                prompt_token_ids=token_ids_batch, sampling_params=sampling_params
            )
            assert isinstance(result4, list)
            assert len(result4) == 2

            # Verify outputs contain text
            text_output = output1.outputs[0].text.strip()
            token_output = output3.outputs[0].text.strip()
            assert len(text_output) > 0
            assert len(token_output) > 0

        finally:
            service.shutdown()

    @pytest.mark.gpu
    @pytest.mark.skipif(not _has_vllm, reason="vllm not available")
    @pytest.mark.skipif(not _has_ray, reason="ray not available")
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.slow
    def test_weight_updates_with_transformer(self, sampling_params):
        """Test weight updates using vLLMUpdater with a real transformer model."""
        from torchrl.collectors.llm.weight_update.vllm import vLLMUpdater
        from torchrl.modules.llm.backends import AsyncVLLM
        from torchrl.modules.llm.policies.transformers_wrapper import (
            TransformersWrapper,
        )

        # Create a transformer policy with the same model
        policy = TransformersWrapper(
            model=MODEL_NAME,
            device=torch.device("cuda"),
            generate_kwargs={
                "max_new_tokens": 256,
                "do_sample": False,  # Use greedy decoding for consistency
            },
        )

        # Create AsyncVLLM service
        service = AsyncVLLM.from_pretrained(
            MODEL_NAME,
            num_replicas=1,
            max_model_len=256,
            gpu_memory_utilization=0.3,
            dtype="float16",
        )

        try:
            # Get initial output before weight update
            initial_result = service.generate("Hello, world!", sampling_params)
            # Handle potential list return for single input
            if isinstance(initial_result, list):
                initial_output = initial_result[0]
            else:
                initial_output = initial_result
            initial_text = initial_output.outputs[0].text  # type: ignore

            # Create weight updater
            updater = vLLMUpdater(vllm_tp_size=1)

            # Get model metadata
            model_metadata = updater.get_model_metadata(policy)

            # Create a proper collector mock that provides access to the AsyncVLLM service
            class MockCollector:
                def __init__(self, policy_ref, vllm_service):
                    self.policy = policy_ref
                    # The vLLMUpdater expects the collector to have a _collector attribute
                    # for Ray-based collectors, or a policy.model for local collectors
                    # We'll use the local collector pattern and patch policy.model to be the Ray actor
                    self.policy.model = vllm_service.actors[0]

                def increment_version(self):
                    pass

            mock_collector = MockCollector(policy, service)
            updater.register_collector(mock_collector)

            # Initialize the updater - this will now have access to the collector
            updater.init(model_metadata)

            # Modify some weights slightly in the transformer
            with torch.no_grad():
                for name, param in policy.model.named_parameters():
                    if "embed_tokens" in name and param.requires_grad:
                        # Add small noise to embedding weights
                        param.data += torch.randn_like(param.data) * 0.001
                        break

            # Update weights in vLLM
            policy_weights = updater._maybe_map_weights(policy)
            updater._sync_weights_with_worker(server_weights=policy_weights)

            # Get output after weight update
            updated_result = service.generate("Hello, world!", sampling_params)
            # Handle potential list return for single input
            if isinstance(updated_result, list):
                updated_output = updated_result[0]
            else:
                updated_output = updated_result
            updated_text = updated_output.outputs[0].text  # type: ignore

            # Verify the weight update process completed without errors
            assert isinstance(updated_text, str)
            assert len(updated_text) > 0
            # Verify initial generation also worked
            assert len(initial_text) > 0

        finally:
            service.shutdown()


if __name__ == "__main__":
    import argparse

    args, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst", "-v", "-s"] + unknown)
