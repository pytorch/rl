# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import argparse
import gc
import importlib.util
import time
from abc import ABC, abstractmethod

import pytest
import torch
from torchrl._utils import _DTYPE_TO_STR_DTYPE, _STR_DTYPE_TO_DTYPE, logger

# Check for dependencies
_has_vllm = importlib.util.find_spec("vllm") is not None
_has_transformers = importlib.util.find_spec("transformers") is not None
_has_ray = importlib.util.find_spec("ray") is not None

if _has_vllm:
    from vllm import LLM, SamplingParams

    try:
        from vllm.utils import get_open_port
    except ImportError:
        # In vLLM 0.13+, get_open_port may be in a different location
        def get_open_port():
            import socket

            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("", 0))
                return s.getsockname()[1]

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

    from torchrl.testing import (
        WorkerTransformerDoubleBuffer,
        WorkerTransformerNCCL,
        WorkerVLLMDoubleBuffer,
        WorkerVLLMNCCL,
    )
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


@pytest.mark.gpu
@pytest.mark.skipif(not _has_transformers, reason="missing transformers dependencies")
@pytest.mark.skipif(not _has_vllm, reason="missing vllm dependencies")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class BaseVLLMUpdaterTest(ABC):
    """Abstract base class for vLLM updater tests.

    This class provides common test patterns and ensures proper memory management
    by having each concrete test class handle its own vLLM instance lifecycle.
    """

    @pytest.fixture(scope="class")
    def model_name(self):
        """Model name for testing - small model for faster testing."""
        return "Qwen/Qwen2.5-0.5B"

    @pytest.fixture(scope="class")
    def sampling_params(self):
        """Sampling parameters for testing."""
        if SamplingParams is not None:
            return SamplingParams(temperature=0.8, max_tokens=50)
        return None

    @pytest.fixture(scope="class")
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

        wrapper = TransformersWrapper(
            model,
            tokenizer=tokenizer,
            input_mode="text",
            generate=False,
            return_log_probs=True,
        )

        yield wrapper

        # Cleanup
        try:
            del wrapper
            del model
            logger.info("Source policy cleaned up")
        except Exception as e:
            logger.warning(f"Error during source policy cleanup: {e}")
        finally:
            # Force garbage collection and CUDA memory cleanup
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    @abstractmethod
    @pytest.fixture
    def target_vllm_engine(self, model_name):
        """Create target vLLM engine - must be implemented by subclasses."""

    def test_config_extraction(self, target_vllm_engine):
        """Test that configuration is correctly extracted from engine."""
        logger.info(
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

        logger.info(
            f"✓ Config extraction test passed for {type(target_vllm_engine).__name__}"
        )

    def test_updater_v2_creation(self, target_vllm_engine):
        """Test vLLMUpdaterV2 creation and configuration detection."""
        logger.info(
            f"=== Testing vLLMUpdaterV2 creation for {type(target_vllm_engine).__name__} ==="
        )

        # Create updater
        updater = vLLMUpdaterV2(target_vllm_engine)

        # Verify configuration is extracted
        assert updater.get_tp_size() >= 1
        assert isinstance(updater.master_address, str)
        assert isinstance(updater.master_port, int)
        assert isinstance(updater.model_metadata, dict)

        logger.info(
            f"✓ vLLMUpdaterV2 creation test passed for {type(target_vllm_engine).__name__}"
        )

    @pytest.mark.slow
    def test_weight_update_interface(self, source_policy, target_vllm_engine):
        """Test weight update using the vLLMUpdaterV2 interface."""
        logger.info(
            f"=== Testing weight update for {type(target_vllm_engine).__name__} ==="
        )

        # Create updater
        updater = vLLMUpdaterV2(target_vllm_engine)

        # Get model metadata from source policy
        model_metadata = vLLMUpdaterV2.get_model_metadata(source_policy)
        assert len(model_metadata) > 0
        logger.info(f"Found {len(model_metadata)} parameters in model metadata")

        # Initialize updater
        updater.init(model_metadata)

        # Test weight update
        updater.push_weights_from_transformers(source_policy)

        logger.info(
            f"✓ Weight update test passed for {type(target_vllm_engine).__name__}"
        )

    def test_error_handling(self):
        """Test error handling for invalid inputs."""
        logger.info("=== Testing error handling ===")

        # Test with non-RLvLLMEngine object
        class FakeEngine:
            pass

        with pytest.raises(TypeError, match="must implement RLvLLMEngine interface"):
            vLLMUpdaterV2(FakeEngine())

        logger.info("✓ Error handling tests passed")


@pytest.mark.xfail(
    reason="AsyncVLLM tests fail due to Ray placement group timeout. "
    "ray.get(pg.ready(), timeout=180) times out. See LLM_TEST_ISSUES.md for details.",
    strict=False,
)
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

        logger.info(f"Created AsyncVLLM service with {service.num_replicas} replicas")
        yield service

        # Cleanup
        try:
            service.shutdown()
            logger.info("AsyncVLLM service shut down successfully")
        except Exception as e:
            logger.warning(f"Error during AsyncVLLM cleanup: {e}")
        finally:
            # Force garbage collection and CUDA memory cleanup
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def test_async_vllm_specific_features(self, target_vllm_engine):
        """Test AsyncVLLM-specific features."""
        logger.info("=== Testing AsyncVLLM-specific features ===")

        # Test that it's actually an AsyncVLLM instance
        assert isinstance(target_vllm_engine, AsyncVLLM)

        # Test service properties
        assert target_vllm_engine.num_replicas == 1
        assert len(target_vllm_engine.actors) == 1

        # Test that actors are ready
        assert target_vllm_engine._launched is True

        logger.info("✓ AsyncVLLM-specific tests passed")


@pytest.mark.skipif(not _has_ray, reason="missing ray dependencies")
@pytest.mark.skip(reason="vLLM fixture issues in CI - needs investigation")
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
        if not _has_vllm:
            pytest.skip("vllm not installed")
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

        logger.info("Created Ray worker")
        yield worker

        # Cleanup
        try:
            if hasattr(worker, "ray_actor") and ray is not None:
                ray.kill(worker.ray_actor)
                logger.info("Ray worker killed successfully")
        except Exception as e:
            logger.warning(f"Error during Ray worker cleanup: {e}")
        finally:
            # Force garbage collection and CUDA memory cleanup
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def test_ray_worker_specific_features(self, target_vllm_engine):
        """Test Ray worker-specific features."""
        logger.info("=== Testing Ray worker-specific features ===")

        # Test that it's actually a RayLLMWorker instance
        assert isinstance(target_vllm_engine, RayLLMWorker)

        # Test worker properties
        assert hasattr(target_vllm_engine, "ray_actor")
        assert target_vllm_engine._tensor_parallel_size == 1

        logger.info("✓ Ray worker-specific tests passed")


@pytest.mark.skip(reason="vLLM fixture issues in CI - needs investigation")
class TestVLLMUpdaterV2WithLocalLLM(BaseVLLMUpdaterTest):
    """Test vLLMUpdaterV2 with local LLM engines.

    All local LLM-related tests are grouped here to share the same vLLM instance.
    """

    @pytest.fixture(scope="class")
    def target_vllm_engine(self, model_name):
        """Create local LLM with low memory settings."""
        if not _has_vllm:
            pytest.skip("vllm not installed")
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

        logger.info("Created local LLM")
        yield llm

        # Cleanup
        try:
            # For local LLM, we might need to explicitly delete the instance
            if hasattr(llm, "llm_instance"):
                del llm.llm_instance
                logger.info("Local LLM instance deleted")
        except Exception as e:
            logger.warning(f"Error during local LLM cleanup: {e}")
        finally:
            # Force garbage collection and CUDA memory cleanup
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def test_local_llm_specific_features(self, target_vllm_engine):
        """Test local LLM-specific features."""
        logger.info("=== Testing local LLM-specific features ===")

        # Test that it's actually a LocalLLMWrapper instance
        assert isinstance(target_vllm_engine, LocalLLMWrapper)

        # Test wrapper properties
        assert hasattr(target_vllm_engine, "llm_instance")
        assert target_vllm_engine._tensor_parallel_size == 1

        logger.info("✓ Local LLM-specific tests passed")


@pytest.mark.xfail(
    reason="AsyncVLLM tests fail due to Ray placement group timeout. "
    "See LLM_TEST_ISSUES.md for details.",
    strict=False,
)
@pytest.mark.gpu
@pytest.mark.skipif(not _has_ray, reason="missing ray dependencies")
@pytest.mark.skipif(not _has_vllm, reason="missing vllm dependencies")
@pytest.mark.skipif(not _has_transformers, reason="missing transformers dependencies")
@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 3,
    reason="CUDA not available or not enough GPUs (need 3: 2 for vLLM workers, 1 for trainer)",
)
class TestWeightSyncVLLMNCCL:
    """Test vLLM weight synchronization using the Sender/Receiver API.

    This test suite verifies weight synchronization between a transformer trainer
    and vLLM inference workers using collective communication (NCCL).
    """

    @staticmethod
    def serialize_metadata(metadata: dict[str, tuple[torch.dtype, torch.Size]]) -> dict:
        """Convert metadata with torch dtypes and sizes to JSON-serializable format.

        Args:
            metadata: Dict mapping parameter names to (dtype, shape) tuples

        Returns:
            JSON-serializable dict with string dtype representations
        """
        serialized = {}
        for name, (dtype, shape) in metadata.items():
            serialized[name] = {
                "dtype": _DTYPE_TO_STR_DTYPE[dtype],
                "shape": list(shape),
            }
        return serialized

    @staticmethod
    def deserialize_metadata(
        serialized: dict,
    ) -> dict[str, tuple[torch.dtype, torch.Size]]:
        """Convert JSON-serialized metadata back to torch dtypes and sizes.

        Args:
            serialized: JSON dict with string dtype representations

        Returns:
            Dict mapping parameter names to (dtype, shape) tuples
        """
        metadata = {}
        for name, info in serialized.items():
            dtype = _STR_DTYPE_TO_DTYPE[info["dtype"]]
            shape = torch.Size(info["shape"])
            metadata[name] = (dtype, shape)
        return metadata

    @staticmethod
    def _make_worker_vllm(model_name: str = "Qwen/Qwen2.5-0.5B"):
        """Create a vLLM wrapper with AsyncVLLM backend."""
        from torchrl.modules.llm.backends import AsyncVLLM
        from torchrl.modules.llm.policies import vLLMWrapper

        async_engine = AsyncVLLM.from_pretrained(
            model_name,
            num_replicas=2,  # Number of engine replicas
        )
        wrapper = vLLMWrapper(async_engine, input_mode="history")
        return wrapper

    @staticmethod
    def _make_worker_transformer(model_name: str = "Qwen/Qwen2.5-0.5B"):
        """Create a transformer model for training."""
        from transformers import AutoModelForCausalLM

        transformer = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.float16,
        )
        transformer = transformer.cuda()
        return transformer

    def test_weight_sync_vllm_collective_ray(self, request):
        """Test weight sync between transformer trainer and vLLM workers.

        Uses Ray remote calls for RPC coordination.

        This test demonstrates the simplified API using named Ray actors:
        1. Trainer is created as a named actor "Trainer"
        2. vLLM receiver discovers trainer by name to fetch metadata
        3. Both initialize simultaneously for collective handshake
        4. Weight updates can be triggered via RPC to the trainer
        """
        import ray

        if not ray.is_initialized():
            ray.init()

        # Determine model based on --runslow flag
        if request.config.getoption("--runslow"):
            model_name = "Qwen/Qwen2.5-3B"
            logger.info("Using large model (3B) for slow test")
        else:
            model_name = "Qwen/Qwen2.5-0.5B"
            logger.info("Using small model (0.5B) for fast test")

        try:
            # Create scheme configuration
            # Use a unique port for each test run to avoid conflicts
            import random

            test_port = random.randint(30000, 40000)
            scheme_config = {
                "master_address": "localhost",
                "master_port": test_port,
                "gpus_per_replica": 1,  # tp_size × dp_size × pp_size (1×1×1=1)
                "num_replicas": 2,  # Number of engine replicas
                "strategy": "state_dict",
                # device defaults to 0 - Ray sets CUDA_VISIBLE_DEVICES per actor
            }
            logger.info(f"Using NCCL port {test_port}")
            # world_size = 1 (trainer) + 2 (replicas) × 1 (gpus_per_replica) = 3

            logger.info(
                "Creating receiver actor first (vLLM workers need 2 GPUs via placement group)..."
            )
            # Create receiver actor first - it will find trainer by name
            receiver = WorkerVLLMNCCL.as_remote().remote(
                scheme_config, model_name, trainer_actor_name="Trainer"
            )

            # Set up vLLM engine (creates placement group with 2 GPUs for workers)
            logger.info("Setting up vLLM engine...")
            ray.get(receiver.setup.remote())
            logger.info("vLLM engine setup complete")

            # Now create trainer actor (needs 1 GPU for training and NCCL rank 0)
            logger.info("Creating trainer actor (needs 1 GPU)...")
            trainer = (
                WorkerTransformerNCCL.as_remote()
                .options(name="Trainer")
                .remote(scheme_config, model_name)
            )
            logger.info("Trainer actor created")

            # Sequential initialization to avoid deadlock:
            # 1. Receiver gets metadata from trainer (RPC) and completes setup
            logger.info("Step 1: Receiver fetching metadata from trainer...")
            ray.get(receiver.init_metadata.remote())

            # Get vLLM engine reference from receiver for RPC coordination
            logger.info("Getting vLLM engine reference from receiver...")
            vllm_engine = ray.get(receiver.get_engine.remote())

            # 2. Start NCCL init on both sides (parallel dispatch)
            logger.info("Step 2: Starting NCCL init on both trainer and workers...")
            # Dispatch both futures in parallel
            nccl_worker_fut = (
                receiver.init.remote()
            )  # Starts vLLM worker background threads
            nccl_trainer_fut = trainer.init.remote(
                vllm_engine=vllm_engine
            )  # Pass engine for RPC

            # Wait for trainer first - it blocks until all ranks (including worker threads) participate
            # This ensures the collective completes before we proceed
            logger.info(
                "Waiting for trainer NCCL init (blocks until all ranks ready)..."
            )
            ray.get(nccl_trainer_fut)

            # Receiver future should already be done (it just dispatched threads and waited for RPCs)
            logger.info("Waiting for receiver NCCL init...")
            ray.get(nccl_worker_fut)

            # 3. NCCL collective completes - all ranks synchronized
            logger.info("NCCL collective initialization complete!")

            # Get initial state
            initial_sum = ray.get(trainer.get_first_param_sum.remote())

            # Trigger weight update with modification
            # Trainer now handles RPC coordination internally (periodic-mono pattern)
            logger.info("=== Starting weight update ===")
            t0 = time.time()
            ray.get(trainer.update_weights.remote(modify_weights=True))
            t1 = time.time()
            update_time = t1 - t0
            logger.info(f"=== NCCL weight update completed in {update_time:.3f}s ===")

            # Verify weights changed
            updated_sum = ray.get(trainer.get_first_param_sum.remote())
            assert updated_sum != initial_sum, "Weights should have changed"

            # Verify receiver still functional
            assert ray.get(receiver.get_sample_output.remote()) == "vllm_ready"

        finally:
            if ray.is_initialized():
                ray.shutdown()


@pytest.mark.gpu
@pytest.mark.xfail(
    reason="AsyncVLLM tests fail due to Ray placement group timeout. "
    "See LLM_TEST_ISSUES.md for details.",
    strict=False,
)
@pytest.mark.skipif(not _has_ray, reason="missing ray dependencies")
@pytest.mark.skipif(not _has_vllm, reason="missing vllm dependencies")
@pytest.mark.skipif(not _has_transformers, reason="missing transformers dependencies")
@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2,
    reason="CUDA not available or not enough GPUs (need 2: 1 for vLLM, 1 for trainer)",
)
class TestWeightSyncVLLMDoubleBuffer:
    """Test vLLM weight synchronization using double-buffered shared storage.

    This test suite verifies weight synchronization between a transformer trainer
    and vLLM inference workers using memory-mapped TensorDict storage.
    """

    @staticmethod
    def _make_worker_vllm(model_name: str = "Qwen/Qwen2.5-0.5B"):
        """Create a vLLM wrapper with AsyncVLLM backend."""
        from torchrl.modules.llm.backends import AsyncVLLM
        from torchrl.modules.llm.policies import vLLMWrapper

        async_engine = AsyncVLLM.from_pretrained(
            model_name,
            num_replicas=1,  # Single replica for simplicity
        )
        wrapper = vLLMWrapper(async_engine, input_mode="history")
        return wrapper

    @staticmethod
    def _make_worker_transformer(model_name: str = "Qwen/Qwen2.5-0.5B"):
        """Create a transformer model for training."""
        from transformers import AutoModelForCausalLM

        transformer = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.float16,
        )
        transformer = transformer.cuda()
        return transformer

    def test_weight_sync_vllm_double_buffer_ray(self, tmpdir, request):
        """Test weight sync using double-buffered storage with Ray.

        This test demonstrates the simplified double-buffer API:
        1. Trainer writes weights to shared directory
        2. vLLM receiver polls and reads from shared directory
        3. No coordination needed - simple push/pull model
        """
        import ray

        if not ray.is_initialized():
            ray.init()

        # Determine model based on --runslow flag
        if request.config.getoption("--runslow"):
            model_name = "Qwen/Qwen2.5-3B"
            logger.info("Using large model (3B) for slow test")
        else:
            model_name = "Qwen/Qwen2.5-0.5B"
            logger.info("Using small model (0.5B) for fast test")

        try:
            # Create temporary directory for weight storage
            logger.info(f"Using temporary directory for weights: {tmpdir}")

            # Create scheme configuration
            scheme_config = {
                "remote_addr": str(tmpdir),
                "num_threads": 128,
                "strategy": "state_dict",
            }

            # Create trainer actor
            logger.info("Creating trainer actor...")
            trainer = WorkerTransformerDoubleBuffer.as_remote().remote(
                scheme_config, model_name
            )
            logger.info("Trainer actor created")

            # Create receiver actor
            logger.info("Creating receiver actor...")
            receiver = WorkerVLLMDoubleBuffer.as_remote().remote(
                scheme_config, model_name
            )

            # Set up vLLM engine
            logger.info("Setting up vLLM engine...")
            ray.get(receiver.setup.remote())
            logger.info("vLLM engine setup complete")

            # Get initial state
            initial_sum = ray.get(trainer.get_first_param_sum.remote())
            logger.info(f"Initial param sum: {initial_sum}")

            # Trigger weight update with modification and measure send timing
            logger.info("=== Starting weight update timing measurement ===")
            t0 = time.time()
            ray.get(trainer.update_weights.remote(modify_weights=True))
            t1 = time.time()
            send_time = t1 - t0
            logger.info(f"=== Weights written to storage in {send_time:.3f}s ===")

            # Verify weights changed on trainer side
            updated_sum = ray.get(trainer.get_first_param_sum.remote())
            assert updated_sum != initial_sum, "Weights should have changed"
            logger.info(f"Updated param sum: {updated_sum}")

            # Receiver polls and applies weights - measure receive timing
            logger.info("Receiver polling for weights...")
            t2 = time.time()
            success = ray.get(receiver.poll_and_apply_weights.remote())
            t3 = time.time()
            receive_time = t3 - t2
            total_time = t3 - t0
            assert success, "Weight application should succeed"
            logger.info(f"=== Weights received and applied in {receive_time:.3f}s ===")
            logger.info(
                f"=== Total double-buffer update time: {total_time:.3f}s (send: {send_time:.3f}s, receive: {receive_time:.3f}s) ==="
            )

            # Verify receiver is still functional
            assert ray.get(receiver.get_sample_output.remote()) == "vllm_ready"
            logger.info("Test completed successfully!")

        finally:
            if ray.is_initialized():
                ray.shutdown()


if __name__ == "__main__":
    args, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst"] + unknown)
