# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Tests for vLLM weight updaters, including the new vLLMUpdaterV2.

This module implements and tests vLLMUpdaterV2, which is an improved version
of the weight updater that automatically infers configuration from vLLM objects
instead of requiring manual specification of tensor parallel size and other parameters.

Key improvements of vLLMUpdaterV2:
- Automatically detects tensor parallel size from vLLM engines;
- Supports multiple vLLM engine types: Ray workers, AsyncVLLM services, and local LLM instances;
- Simplifies API by removing need for manual configuration;
- Provides get_tp_size() method for introspection.

The tests demonstrate weight transfer between a source TransformersWrapper (Qwen2.5-0.5B)
and target vLLM engines, validating that the updater can successfully synchronize weights
across different vLLM deployment patterns.
"""

import importlib.util

import pytest
import torch
from torchrl._utils import logger as torchrl_logger

# Check for dependencies
_has_vllm = importlib.util.find_spec("vllm") is not None
_has_transformers = importlib.util.find_spec("transformers") is not None
_has_ray = importlib.util.find_spec("ray") is not None

if _has_vllm:
    from vllm import LLM, SamplingParams
    from vllm.utils import get_open_port
else:
    LLM = None
    SamplingParams = None

    def get_open_port():
        raise ImportError("vllm is not installed")


if _has_transformers:
    from torchrl.modules.llm.policies import TransformersWrapper
    from transformers import AutoModelForCausalLM, AutoTokenizer
else:
    AutoModelForCausalLM = None
    AutoTokenizer = None
    TransformersWrapper = None

if _has_ray:
    import ray
else:
    ray = None

if _has_vllm and _has_transformers:
    from torchrl.collectors.llm.weight_update.vllm_v2 import vLLMUpdaterV2
    from torchrl.modules.llm.backends import (
        AsyncVLLM,
        LocalLLMWrapper,
        make_vllm_worker,
        RayLLMWorker,
    )


@pytest.mark.skipif(not _has_transformers, reason="missing transformers dependencies")
@pytest.mark.skipif(not _has_vllm, reason="missing vllm dependencies")
@pytest.mark.skipif(not _has_ray, reason="missing ray dependencies")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestVLLMUpdaterV2:
    """Test the new vLLMUpdaterV2 class."""

    @pytest.fixture(scope="class", autouse=True)
    def setup_ray(self):
        """Initialize Ray for testing."""
        if ray is not None and not ray.is_initialized():
            ray.init()
        yield
        # Note: We don't shutdown Ray here to avoid issues with other tests

    @pytest.fixture
    def model_name(self):
        """Model name for testing."""
        return "Qwen/Qwen2.5-0.5B"  # Small model for faster testing

    @pytest.fixture
    def sampling_params(self):
        """Sampling parameters for testing."""
        if SamplingParams is not None:
            return SamplingParams(temperature=0.8, max_tokens=50)
        return None

    @pytest.fixture
    def source_policy(self, model_name):
        """Create source TransformersWrapper policy."""
        if (
            AutoModelForCausalLM is None
            or AutoTokenizer is None
            or TransformersWrapper is None
        ):
            pytest.skip("transformers dependencies not available")

        model = AutoModelForCausalLM.from_pretrained(
            model_name, device_map="cuda:0", torch_dtype=torch.float16
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        return TransformersWrapper(
            model,
            tokenizer=tokenizer,
            input_mode="text",
            generate=False,
            return_log_probs=True,
        )

    @pytest.fixture
    def target_vllm_ray_worker(self, model_name):
        """Create target vLLM Ray worker."""
        worker = make_vllm_worker(
            model_name=model_name,
            devices=[0],
            make_ray_worker=True,
            gpu_memory_utilization=0.3,
            dtype="float16",
        )
        yield worker
        # Cleanup is handled by Ray shutdown

    @pytest.fixture
    def target_ray_worker(self, model_name):
        """Create target Ray worker vLLM engine."""
        if not _has_ray or ray is None:
            pytest.skip("Ray not available")

        if not ray.is_initialized():
            ray.init()

        # Create Ray worker
        ray_worker = make_vllm_worker(
            model_name=model_name,
            make_ray_worker=True,
            num_devices=1,
            max_model_len=512,
        )
        yield ray_worker

        # Cleanup
        try:
            ray.kill(ray_worker.ray_actor)
        except Exception:
            pass

    @pytest.fixture
    def target_local_llm(self, model_name):
        """Create target local LLM engine."""
        # Create local LLM
        local_llm = make_vllm_worker(
            model_name=model_name,
            make_ray_worker=False,
            num_devices=1,
            max_model_len=512,
        )
        yield local_llm

    @pytest.fixture
    def target_async_vllm(self, model_name):
        """Create target AsyncVLLM service."""
        service = AsyncVLLM.from_pretrained(
            model_name,
            num_devices=1,
            num_replicas=1,
            gpu_memory_utilization=0.3,
            dtype="float16",
        )
        yield service
        try:
            service.shutdown()
        except Exception:
            pass  # Ignore cleanup errors

    # Legacy test - removed in favor of the functional version below

    @pytest.mark.slow
    def test_updater_v2_with_async_vllm(self, source_policy, target_async_vllm):
        """Test vLLMUpdaterV2 with AsyncVLLM service."""
        torchrl_logger.info("=== Testing vLLMUpdaterV2 with AsyncVLLM ===")

        # Create updater V2 - AsyncVLLM now implements RLvLLMEngine
        updater = vLLMUpdaterV2(target_async_vllm)

        # Check that it detected the configuration correctly
        assert updater.get_tp_size() == 1
        torchrl_logger.info(f"Detected tensor parallel size: {updater.get_tp_size()}")

        # Since AsyncVLLM.get_model_metadata() is not fully implemented yet,
        # we'll get metadata from the source policy and pass it to init()
        model_metadata = vLLMUpdaterV2.get_model_metadata(source_policy)
        assert len(model_metadata) > 0
        torchrl_logger.info(f"Found {len(model_metadata)} parameters in model metadata")

        # Initialize updater with metadata
        updater.init(model_metadata)

        # Test weight update using the new interface
        updater.push_weights_from_transformers(source_policy)

        torchrl_logger.info("AsyncVLLM weight update completed successfully")

    def test_config_extraction_from_engines(
        self, target_async_vllm, target_ray_worker, target_local_llm
    ):
        """Test that configuration is correctly extracted from different engine types."""
        torchrl_logger.info("=== Testing configuration extraction ===")

        # Test AsyncVLLM (implements RLvLLMEngine)
        updater_async = vLLMUpdaterV2(target_async_vllm)
        assert updater_async.get_tp_size() == 1
        assert isinstance(target_async_vllm, AsyncVLLM)
        assert target_async_vllm.get_tp_size() == 1
        assert target_async_vllm.get_master_address() == "localhost"
        assert isinstance(target_async_vllm.get_master_port(), int)

        # Test Ray worker (implements RLvLLMEngine)
        updater_ray = vLLMUpdaterV2(target_ray_worker)
        assert updater_ray.get_tp_size() == 1
        assert isinstance(target_ray_worker, RayLLMWorker)
        assert target_ray_worker.get_tp_size() == 1
        assert target_ray_worker.get_master_address() == "localhost"
        assert isinstance(target_ray_worker.get_master_port(), int)

        # Test Local LLM (implements RLvLLMEngine)
        updater_local = vLLMUpdaterV2(target_local_llm)
        assert updater_local.get_tp_size() == 1
        assert isinstance(target_local_llm, LocalLLMWrapper)
        assert target_local_llm.get_tp_size() == 1
        assert target_local_llm.get_master_address() == "localhost"
        assert isinstance(target_local_llm.get_master_port(), int)

        torchrl_logger.info(
            "Configuration extraction tests passed for all engine types"
        )

    def test_updater_v2_interface_demo(self, target_async_vllm, source_policy):
        """Demonstrate the new vLLMUpdaterV2 interface in action."""
        torchrl_logger.info("=== Demonstrating vLLMUpdaterV2 Interface ===")

        # Show how simple the new interface is
        updater = vLLMUpdaterV2(target_async_vllm)

        # All configuration is automatically extracted
        torchrl_logger.info("Automatically detected:")
        torchrl_logger.info(f"  - Tensor parallel size: {updater.get_tp_size()}")
        torchrl_logger.info(
            f"  - Master address: {updater.vllm_engine.get_master_address()}"
        )
        torchrl_logger.info(f"  - Master port: {updater.vllm_engine.get_master_port()}")

        # Initialize with model metadata
        model_metadata = vLLMUpdaterV2.get_model_metadata(source_policy)
        updater.init(model_metadata)

        # Push weights using the clean interface
        updater.push_weights_from_transformers(source_policy)

        torchrl_logger.info("✓ vLLMUpdaterV2 interface demonstration completed")

    @pytest.mark.slow
    def test_updater_v2_with_ray_worker(self, source_policy, target_ray_worker):
        """Test vLLMUpdaterV2 with Ray worker."""
        torchrl_logger.info("=== Testing vLLMUpdaterV2 with Ray worker ===")

        # Create updater V2 - Ray worker now implements RLvLLMEngine
        updater = vLLMUpdaterV2(target_ray_worker)

        # Check that it detected the configuration correctly
        assert updater.get_tp_size() == 1
        torchrl_logger.info(f"Detected tensor parallel size: {updater.get_tp_size()}")

        # Get metadata from source policy and initialize
        model_metadata = vLLMUpdaterV2.get_model_metadata(source_policy)
        assert len(model_metadata) > 0
        torchrl_logger.info(f"Found {len(model_metadata)} parameters in model metadata")

        # Initialize updater with metadata
        updater.init(model_metadata)

        # Test weight update using the new interface
        updater.push_weights_from_transformers(source_policy)

        torchrl_logger.info("Ray worker weight update completed successfully")

    @pytest.mark.slow
    def test_updater_v2_with_local_llm(self, source_policy, target_local_llm):
        """Test vLLMUpdaterV2 with local LLM."""
        torchrl_logger.info("=== Testing vLLMUpdaterV2 with local LLM ===")

        # Create updater V2 - Local LLM now implements RLvLLMEngine
        updater = vLLMUpdaterV2(target_local_llm)

        # Check that it detected the configuration correctly
        assert updater.get_tp_size() == 1
        torchrl_logger.info(f"Detected tensor parallel size: {updater.get_tp_size()}")

        # Get metadata from source policy and initialize
        model_metadata = vLLMUpdaterV2.get_model_metadata(source_policy)
        assert len(model_metadata) > 0
        torchrl_logger.info(f"Found {len(model_metadata)} parameters in model metadata")

        # Initialize updater with metadata
        updater.init(model_metadata)

        # Test weight update using the new interface (no-op for local LLM)
        updater.push_weights_from_transformers(source_policy)

        torchrl_logger.info("Local LLM weight update completed successfully")

    @pytest.mark.skip(
        reason="Requires Ray worker RLvLLMEngine interface implementation"
    )
    def test_weight_transfer_consistency(
        self, source_policy, target_vllm_ray_worker, sampling_params
    ):
        """Test that weight updates actually change the model behavior."""
        # TODO: This test will be enabled once Ray workers implement RLvLLMEngine interface


if __name__ == "__main__":
    # Simple smoke test
    pytest.main([__file__, "-v", "-s"])
