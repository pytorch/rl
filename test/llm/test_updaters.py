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

The tests are organized by engine type to optimize GPU memory usage:
- Each test class manages its own vLLM instance and cleans up when done
- Abstract base class provides common test patterns
- Low KV cache settings are used to minimize GPU memory utilization
"""

import gc
import importlib.util
from abc import ABC, abstractmethod

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
        RLvLLMEngine,
    )


@pytest.mark.skipif(not _has_transformers, reason="missing transformers dependencies")
@pytest.mark.skipif(not _has_vllm, reason="missing vllm dependencies")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class BaseVLLMUpdaterTest(ABC):
    """Abstract base class for vLLM updater tests.

    This class provides common test patterns and ensures proper memory management
    by having each concrete test class handle its own vLLM instance lifecycle.
    """

    @pytest.fixture
    def model_name(self):
        """Model name for testing - small model for faster testing."""
        return "Qwen/Qwen2.5-0.5B"

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

        # Use low memory settings
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="cuda:0",
            torch_dtype=torch.float16,
            attn_implementation="flash_attention_2"
            if hasattr(torch.nn, "scaled_dot_product_attention")
            else "eager",
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        return TransformersWrapper(
            model,
            tokenizer=tokenizer,
            input_mode="text",
            generate=False,
            return_log_probs=True,
        )

    @abstractmethod
    @pytest.fixture
    def target_vllm_engine(self, model_name):
        """Create target vLLM engine - must be implemented by subclasses."""

    def test_config_extraction(self, target_vllm_engine):
        """Test that configuration is correctly extracted from engine."""
        torchrl_logger.info(
            f"=== Testing config extraction for {type(target_vllm_engine).__name__} ==="
        )

        # Verify it implements RLvLLMEngine interface
        assert isinstance(target_vllm_engine, RLvLLMEngine)

        # Test interface methods
        tp_size = target_vllm_engine.get_tp_size()
        assert isinstance(tp_size, int) and tp_size >= 1

        master_address = target_vllm_engine.get_master_address()
        assert isinstance(master_address, str) and len(master_address) > 0

        master_port = target_vllm_engine.get_master_port()
        assert isinstance(master_port, int) and master_port > 0

        model_metadata = target_vllm_engine.get_model_metadata()
        assert isinstance(model_metadata, dict)

        torchrl_logger.info(
            f"✓ Config extraction test passed for {type(target_vllm_engine).__name__}"
        )

    def test_updater_v2_creation(self, target_vllm_engine):
        """Test vLLMUpdaterV2 creation and configuration detection."""
        torchrl_logger.info(
            f"=== Testing vLLMUpdaterV2 creation for {type(target_vllm_engine).__name__} ==="
        )

        # Create updater
        updater = vLLMUpdaterV2(target_vllm_engine)

        # Verify configuration is extracted
        assert updater.get_tp_size() >= 1
        assert isinstance(updater.master_address, str)
        assert isinstance(updater.master_port, int)
        assert isinstance(updater.model_metadata, dict)

        torchrl_logger.info(
            f"✓ vLLMUpdaterV2 creation test passed for {type(target_vllm_engine).__name__}"
        )

    @pytest.mark.slow
    def test_weight_update_interface(self, source_policy, target_vllm_engine):
        """Test weight update using the vLLMUpdaterV2 interface."""
        torchrl_logger.info(
            f"=== Testing weight update for {type(target_vllm_engine).__name__} ==="
        )

        # Create updater
        updater = vLLMUpdaterV2(target_vllm_engine)

        # Get model metadata from source policy
        model_metadata = vLLMUpdaterV2.get_model_metadata(source_policy)
        assert len(model_metadata) > 0
        torchrl_logger.info(f"Found {len(model_metadata)} parameters in model metadata")

        # Initialize updater
        updater.init(model_metadata)

        # Test weight update
        updater.push_weights_from_transformers(source_policy)

        torchrl_logger.info(
            f"✓ Weight update test passed for {type(target_vllm_engine).__name__}"
        )

    def test_error_handling(self):
        """Test error handling for invalid inputs."""
        torchrl_logger.info("=== Testing error handling ===")

        # Test with non-RLvLLMEngine object
        class FakeEngine:
            pass

        with pytest.raises(TypeError, match="must implement RLvLLMEngine interface"):
            vLLMUpdaterV2(FakeEngine())

        torchrl_logger.info("✓ Error handling tests passed")


@pytest.mark.skipif(not _has_ray, reason="missing ray dependencies")
class TestVLLMUpdaterV2WithAsyncVLLM(BaseVLLMUpdaterTest):
    """Test vLLMUpdaterV2 with AsyncVLLM engines.

    All AsyncVLLM-related tests are grouped here to share the same vLLM instance
    and minimize GPU memory usage.
    """

    @pytest.fixture(scope="class", autouse=True)
    def setup_ray(self):
        """Initialize Ray for the test class."""
        if ray is not None and not ray.is_initialized():
            ray.init()
        yield
        # Note: We don't shutdown Ray here to avoid issues with other tests

    @pytest.fixture(scope="class")
    def target_vllm_engine(self, model_name):
        """Create AsyncVLLM service with low memory settings."""
        # Use minimal memory settings for testing
        service = AsyncVLLM.from_pretrained(
            model_name,
            num_devices=1,
            num_replicas=1,
            gpu_memory_utilization=0.3,  # Low GPU memory usage
            dtype="float16",
            max_model_len=512,  # Short context for minimal KV cache
            max_num_seqs=1,  # Minimal batch size
            enable_prefix_caching=False,  # Disable to save memory
        )

        torchrl_logger.info(
            f"Created AsyncVLLM service with {service.num_replicas} replicas"
        )
        yield service

        # Cleanup
        try:
            service.shutdown()
            torchrl_logger.info("AsyncVLLM service shut down successfully")
        except Exception as e:
            torchrl_logger.warning(f"Error during AsyncVLLM cleanup: {e}")
        finally:
            # Force garbage collection and CUDA memory cleanup
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def test_async_vllm_specific_features(self, target_vllm_engine):
        """Test AsyncVLLM-specific features."""
        torchrl_logger.info("=== Testing AsyncVLLM-specific features ===")

        # Test that it's actually an AsyncVLLM instance
        assert isinstance(target_vllm_engine, AsyncVLLM)

        # Test service properties
        assert target_vllm_engine.num_replicas == 1
        assert len(target_vllm_engine.actors) == 1

        # Test that actors are ready
        assert target_vllm_engine._launched is True

        torchrl_logger.info("✓ AsyncVLLM-specific tests passed")


@pytest.mark.skipif(not _has_ray, reason="missing ray dependencies")
class TestVLLMUpdaterV2WithRayWorker(BaseVLLMUpdaterTest):
    """Test vLLMUpdaterV2 with Ray worker engines.

    All Ray worker-related tests are grouped here to share the same vLLM instance.
    """

    @pytest.fixture(scope="class", autouse=True)
    def setup_ray(self):
        """Initialize Ray for the test class."""
        if ray is not None and not ray.is_initialized():
            ray.init()
        yield

    @pytest.fixture(scope="class")
    def target_vllm_engine(self, model_name):
        """Create Ray worker with low memory settings."""
        # Create Ray worker with minimal memory usage
        worker = make_vllm_worker(
            model_name=model_name,
            make_ray_worker=True,
            num_devices=1,
            gpu_memory_utilization=0.3,  # Low GPU memory usage
            dtype="float16",
            max_model_len=512,  # Short context for minimal KV cache
            max_num_seqs=1,  # Minimal batch size
        )

        torchrl_logger.info("Created Ray worker")
        yield worker

        # Cleanup
        try:
            if hasattr(worker, "ray_actor") and ray is not None:
                ray.kill(worker.ray_actor)
                torchrl_logger.info("Ray worker killed successfully")
        except Exception as e:
            torchrl_logger.warning(f"Error during Ray worker cleanup: {e}")
        finally:
            # Force garbage collection and CUDA memory cleanup
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def test_ray_worker_specific_features(self, target_vllm_engine):
        """Test Ray worker-specific features."""
        torchrl_logger.info("=== Testing Ray worker-specific features ===")

        # Test that it's actually a RayLLMWorker instance
        assert isinstance(target_vllm_engine, RayLLMWorker)

        # Test worker properties
        assert hasattr(target_vllm_engine, "ray_actor")
        assert target_vllm_engine._tensor_parallel_size == 1

        torchrl_logger.info("✓ Ray worker-specific tests passed")


class TestVLLMUpdaterV2WithLocalLLM(BaseVLLMUpdaterTest):
    """Test vLLMUpdaterV2 with local LLM engines.

    All local LLM-related tests are grouped here to share the same vLLM instance.
    """

    @pytest.fixture(scope="class")
    def target_vllm_engine(self, model_name):
        """Create local LLM with low memory settings."""
        # Create local LLM with minimal memory usage
        llm = make_vllm_worker(
            model_name=model_name,
            make_ray_worker=False,
            num_devices=1,
            gpu_memory_utilization=0.3,  # Low GPU memory usage
            dtype="float16",
            max_model_len=512,  # Short context for minimal KV cache
            max_num_seqs=1,  # Minimal batch size
        )

        torchrl_logger.info("Created local LLM")
        yield llm

        # Cleanup
        try:
            # For local LLM, we might need to explicitly delete the instance
            if hasattr(llm, "llm_instance"):
                del llm.llm_instance
                torchrl_logger.info("Local LLM instance deleted")
        except Exception as e:
            torchrl_logger.warning(f"Error during local LLM cleanup: {e}")
        finally:
            # Force garbage collection and CUDA memory cleanup
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def test_local_llm_specific_features(self, target_vllm_engine):
        """Test local LLM-specific features."""
        torchrl_logger.info("=== Testing local LLM-specific features ===")

        # Test that it's actually a LocalLLMWrapper instance
        assert isinstance(target_vllm_engine, LocalLLMWrapper)

        # Test wrapper properties
        assert hasattr(target_vllm_engine, "llm_instance")
        assert target_vllm_engine._tensor_parallel_size == 1

        torchrl_logger.info("✓ Local LLM-specific tests passed")


if __name__ == "__main__":
    # Simple smoke test
    pytest.main([__file__, "-v", "-s"])
