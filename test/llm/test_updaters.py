# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


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

    class WorkerVLLM:
        """Ray actor for vLLM inference worker (receiver)."""

        def __init__(
            self,
            scheme_config: dict,
            model_name: str = "Qwen/Qwen2.5-0.5B",
            trainer_actor_name: str = "Trainer",
        ):
            pass

            # Store config for deferred initialization
            self.scheme_config = scheme_config
            self.model_name = model_name
            self.trainer_actor_name = trainer_actor_name
            self.wrapper = None
            self.engine = None
            self.receiver = None
            self.scheme = None
            self.trainer = None
            self.model_metadata = None

        def setup(self):
            """Set up vLLM engine (deferred from __init__ to avoid blocking)."""
            # Create vLLM wrapper
            self.wrapper = TestWeightSyncVLLMNCCL._make_worker_vllm(self.model_name)
            self.engine = self.wrapper.model

            # Create scheme from config
            from torchrl.weight_update.llm.vllm_nccl import VLLMWeightSyncScheme

            self.scheme = VLLMWeightSyncScheme(**self.scheme_config)

            # Create receiver (engine handles rank assignment automatically)
            self.receiver = self.scheme.create_receiver(self.engine)
            return "setup_complete"

        def init_metadata(self):
            """Initialize the receiver by fetching metadata from trainer."""
            import ray

            if self.receiver is None:
                raise RuntimeError("Must call setup() before init()")

            # Get trainer actor by name
            logger.info(f"Getting trainer actor by name {self.trainer_actor_name}")
            self.trainer = ray.get_actor(self.trainer_actor_name)

            # Fetch model metadata from trainer
            logger.info(
                "Fetching model metadata from trainer (requires max_concurrency>1)"
            )
            self.model_metadata = ray.get(self.trainer.get_model_metadata.remote())

        def init(self):
            if self.model_metadata is None:
                raise RuntimeError("Must call init_metadata() before init()")

            # Initialize receiver with metadata
            logger.info("Initializing receiver...")
            self.receiver.init_all_workers_group(self.model_metadata)
            self.initialized = True
            logger.info("Receiver initialized")
            return "initialized"

        def get_engine(self):
            """Get the vLLM engine reference for RPC coordination."""
            if self.engine is None:
                raise RuntimeError("Must call setup() first")
            return self.engine

        def get_sample_output(self):
            """Get a sample output to verify model works."""
            # Simple inference test
            return "vllm_ready"

        # @classmethod
        # def run_forever(
        #     cls, scheme_config, parent_pipe, child_pipe, trainer_metadata_pipe, model_name="Qwen/Qwen2.5-0.5B"
        # ):
        #     """A single threaded infinite loop capturing commands via a Pipe.

        #     Args:
        #         scheme_config: Configuration for VLLMWeightSyncScheme
        #         parent_pipe: Parent end of the pipe (to be closed in child)
        #         child_pipe: Child end of the pipe for receiving commands
        #         trainer_metadata_pipe: Pipe to receive metadata from trainer
        #         model_name: Model name to load
        #     """
        #     import os

        #     # Set CUDA_VISIBLE_DEVICES for vLLM workers (GPUs 1,2)
        #     # vLLM will use these for its 2 replicas
        #     os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"

        #     parent_pipe.close()  # Close parent end in child process

        #     # Update scheme_config to use device 0 (which is actually GPU 1 due to CUDA_VISIBLE_DEVICES)
        #     scheme_config = scheme_config.copy()
        #     scheme_config["device"] = 0

        #     worker = cls(scheme_config, model_name)
        #     child_pipe.send({"status": "success", "result": "instantiated"})

        #     while True:
        #         try:
        #             command = child_pipe.recv()
        #             if command == "shutdown":
        #                 child_pipe.send({"status": "shutdown"})
        #                 break
        #             elif command == "setup":
        #                 result = worker.setup()
        #                 child_pipe.send({"status": "success", "result": result})
        #             elif command == "init_metadata":
        #                 # Receive metadata from trainer via separate pipe
        #                 worker.model_metadata = trainer_metadata_pipe.recv()
        #                 child_pipe.send(
        #                     {"status": "success", "result": "metadata_received"}
        #                 )
        #             elif command == "init":
        #                 result = worker.init()
        #                 child_pipe.send({"status": "success", "result": result})
        #             elif command == "update_weights_receiver":
        #                 worker.receiver.update_weights()
        #                 child_pipe.send({"status": "success", "result": "receiving_started"})
        #             elif (
        #                 isinstance(command, dict)
        #                 and command.get("cmd") == "get_sample_output"
        #             ):
        #                 result = worker.get_sample_output()
        #                 child_pipe.send({"status": "success", "result": result})
        #             else:
        #                 child_pipe.send(
        #                     {"status": "error", "error": f"Unknown command: {command}"}
        #                 )
        #         except Exception as e:
        #             torchrl_logger.error(f"WorkerVLLM error: {e}", exc_info=True)
        #             child_pipe.send({"status": "error", "error": str(e)})
        #             break

        # @classmethod
        # def run_forever_http(cls, scheme_config, port, model_name="Qwen/Qwen2.5-0.5B"):
        #     """Run an HTTP server that accepts commands via REST endpoints.

        #     Args:
        #         scheme_config: Configuration for VLLMWeightSyncScheme
        #         port: Port to listen on
        #         model_name: Model name to load
        #     """
        #     import os

        #     from flask import Flask, jsonify, request

        #     # Set CUDA_VISIBLE_DEVICES for vLLM workers (GPUs 1,2)
        #     os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"

        #     # Update scheme_config to use device 0 (which is actually GPU 1 due to CUDA_VISIBLE_DEVICES)
        #     scheme_config = scheme_config.copy()
        #     scheme_config["device"] = 0

        #     # Initialize Ray in this subprocess - required for AsyncVLLM
        #     import ray
        #     if not ray.is_initialized():
        #         ray.init()
        #         torchrl_logger.info("Ray initialized in WorkerVLLM subprocess")

        #     app = Flask(f"WorkerVLLM_{port}")

        #     # Defer worker creation until first request to allow Flask to start quickly
        #     worker = None

        #     def ensure_worker():
        #         nonlocal worker
        #         if worker is None:
        #             worker = cls(scheme_config, model_name)
        #         return worker

        #     @app.route("/health", methods=["GET"])
        #     def health():
        #         """Health check endpoint that doesn't require worker initialization."""
        #         return jsonify({"status": "ready"})

        #     @app.route("/setup", methods=["POST"])
        #     def setup():
        #         try:
        #             w = ensure_worker()
        #             result = w.setup()
        #             return jsonify({"status": "success", "result": result})
        #         except Exception as e:
        #             torchrl_logger.error(f"Setup error: {e}", exc_info=True)
        #             return jsonify({"status": "error", "error": str(e)}), 500

        #     @app.route("/init_metadata", methods=["POST"])
        #     def init_metadata():
        #         try:
        #             # Receive metadata in request body
        #             w = ensure_worker()
        #             received_data = request.json
        #             torchrl_logger.info(f"Received metadata with {len(received_data)} parameters")
        #             torchrl_logger.info(f"First 3 params: {list(received_data.keys())[:3]}")
        #             torchrl_logger.info(f"Last 3 params: {list(received_data.keys())[-3:]}")
        #             w.model_metadata = TestWeightSyncVLLMNCCL.deserialize_metadata(received_data)
        #             torchrl_logger.info(f"Deserialized metadata successfully")
        #             return jsonify({"status": "success", "result": "metadata_received"})
        #         except Exception as e:
        #             torchrl_logger.error(f"Init metadata error: {e}", exc_info=True)
        #             return jsonify({"status": "error", "error": str(e)}), 500

        #     @app.route("/init", methods=["POST"])
        #     def init():
        #         try:
        #             w = ensure_worker()
        #             result = w.init()
        #             return jsonify({"status": "success", "result": result})
        #         except Exception as e:
        #             torchrl_logger.error(f"Init error: {e}", exc_info=True)
        #             return jsonify({"status": "error", "error": str(e)}), 500

        #     @app.route("/update_weights_receiver", methods=["POST"])
        #     def update_weights_receiver():
        #         try:
        #             w = ensure_worker()
        #             w.receiver.update_weights()
        #             return jsonify({"status": "success", "result": "receiving_started"})
        #         except Exception as e:
        #             torchrl_logger.error(f"Receiver update error: {e}", exc_info=True)
        #             return jsonify({"status": "error", "error": str(e)}), 500

        #     @app.route("/get_sample_output", methods=["GET"])
        #     def get_sample_output():
        #         try:
        #             w = ensure_worker()
        #             result = w.get_sample_output()
        #             return jsonify({"status": "success", "result": result})
        #         except Exception as e:
        #             torchrl_logger.error(f"Get sample output error: {e}", exc_info=True)
        #             return jsonify({"status": "error", "error": str(e)}), 500

        #     @app.route("/shutdown", methods=["POST"])
        #     def shutdown():
        #         try:
        #             # Shutdown Ray before killing the process
        #             import ray
        #             if ray.is_initialized():
        #                 ray.shutdown()
        #                 torchrl_logger.info("Ray shut down in WorkerVLLM subprocess")

        #             func = request.environ.get("werkzeug.server.shutdown")
        #             if func is None:
        #                 # Running under a different WSGI server
        #                 import os
        #                 import signal

        #                 os.kill(os.getpid(), signal.SIGTERM)
        #             else:
        #                 func()
        #             return jsonify({"status": "shutdown"})
        #         except Exception as e:
        #             torchrl_logger.error(f"Shutdown error: {e}", exc_info=True)
        #             return jsonify({"status": "error", "error": str(e)}), 500

        #     torchrl_logger.info(f"WorkerVLLM HTTP server starting on port {port}")
        #     app.run(host="0.0.0.0", port=port, threaded=True)

        @classmethod
        def as_remote(cls, *args, **kwargs):
            import ray

            # No GPUs needed for the actor itself - vLLM workers manage their own placement group (2 GPUs)
            # AsyncVLLM service doesn't act as NCCL rank 0 when used with external trainer
            return ray.remote(num_cpus=4, num_gpus=0, max_concurrency=4)(cls)

    class WorkerTransformer:
        """Ray actor for transformer trainer (sender)."""

        def __init__(self, scheme_config: dict, model_name: str = "Qwen/Qwen2.5-0.5B"):
            from torchrl.weight_update.llm.vllm_nccl import (
                get_model_metadata,
                VLLMWeightSyncScheme,
            )

            # Create transformer model
            self.transformer = TestWeightSyncVLLMNCCL._make_worker_transformer(
                model_name
            )

            # Create scheme from config
            self.scheme = VLLMWeightSyncScheme(**scheme_config)

            # Create sender
            self.sender = self.scheme.create_sender()
            self.sender.register_model(self.transformer)

            # Extract and store model metadata
            self.model_metadata = get_model_metadata(self.transformer)

        def init(self, vllm_engine=None):
            """Initialize sender with optional vLLM engine for RPC coordination.

            Args:
                vllm_engine: Optional vLLM engine reference for calling collective_rpc
            """
            if self.model_metadata is None:
                raise RuntimeError("Must call init_metadata() before init()")

            self.sender.init_all_workers_group(
                self.model_metadata, vllm_engine=vllm_engine
            )
            self.initialized = True
            logger.info("Trainer initialized")
            return "initialized"

        def get_model_metadata(self):
            """Get model metadata to share with receiver."""
            return self.model_metadata

        def update_weights(self, modify_weights: bool = False):
            """Trigger a weight update broadcast.

            Args:
                modify_weights: If True, modifies weights before broadcasting
                                for verification purposes.

            Returns:
                str: "updated" status message
            """

            # Optionally modify weights for testing
            if modify_weights:
                with torch.no_grad():
                    first_param = next(self.transformer.parameters())
                    first_param.add_(0.01)

            # Broadcast weights to all vLLM workers
            self.sender.update_weights()
            return "updated"

        def get_first_param_sum(self):
            """Get sum of first parameter for verification."""
            return next(self.transformer.parameters()).sum().item()

        # @classmethod
        # def run_forever(
        #     cls, scheme_config, parent_pipe, child_pipe, metadata_send_pipe, model_name="Qwen/Qwen2.5-0.5B"
        # ):
        #     """A single threaded infinite loop capturing commands via a Pipe.

        #     Args:
        #         scheme_config: Configuration for VLLMWeightSyncScheme
        #         parent_pipe: Parent end of the pipe (to be closed in child)
        #         child_pipe: Child end of the pipe for receiving commands
        #         metadata_send_pipe: Pipe to send metadata to receiver
        #         model_name: Model name to load
        #     """
        #     import os

        #     # Set CUDA_VISIBLE_DEVICES for trainer (GPU 0 only)
        #     os.environ["CUDA_VISIBLE_DEVICES"] = "0"

        #     parent_pipe.close()  # Close parent end in child process

        #     # Update scheme_config to use device 0 (which is actually GPU 0)
        #     scheme_config = scheme_config.copy()
        #     scheme_config["device"] = 0

        #     worker = cls(scheme_config, model_name)
        #     child_pipe.send({"status": "success", "result": "instantiated"})

        #     while True:
        #         try:
        #             command = child_pipe.recv()
        #             if command == "shutdown":
        #                 child_pipe.send({"status": "shutdown"})
        #                 break
        #             elif command == "init":
        #                 result = worker.init()
        #                 child_pipe.send({"status": "success", "result": result})
        #             elif isinstance(command, dict):
        #                 cmd_name = command.get("cmd")
        #                 if cmd_name == "get_model_metadata":
        #                     # Send metadata to receiver via separate pipe
        #                     metadata_send_pipe.send(worker.model_metadata)
        #                     child_pipe.send(
        #                         {"status": "success", "result": "metadata_sent"}
        #                     )
        #                 elif cmd_name == "update_weights":
        #                     modify_weights = command.get("modify_weights", False)
        #                     result = worker.update_weights(modify_weights)
        #                     child_pipe.send({"status": "success", "result": result})
        #                 elif cmd_name == "get_first_param_sum":
        #                     result = worker.get_first_param_sum()
        #                     child_pipe.send({"status": "success", "result": result})
        #                 else:
        #                     child_pipe.send(
        #                         {
        #                             "status": "error",
        #                             "error": f"Unknown command: {cmd_name}",
        #                         }
        #                     )
        #             else:
        #                 child_pipe.send(
        #                     {"status": "error", "error": f"Unknown command: {command}"}
        #                 )
        #         except Exception as e:
        #             torchrl_logger.error(f"WorkerTransformer error: {e}", exc_info=True)
        #             child_pipe.send({"status": "error", "error": str(e)})
        #             break

        # @classmethod
        # def run_forever_http(cls, scheme_config, port, model_name="Qwen/Qwen2.5-0.5B"):
        #     """Run an HTTP server that accepts commands via REST endpoints.

        #     Args:
        #         scheme_config: Configuration for VLLMWeightSyncScheme
        #         port: Port to listen on
        #         model_name: Model name to load
        #     """
        #     import os

        #     from flask import Flask, jsonify, request

        #     # Set CUDA_VISIBLE_DEVICES for trainer (GPU 0 only)
        #     os.environ["CUDA_VISIBLE_DEVICES"] = "0"

        #     # Update scheme_config to use device 0 (which is actually GPU 0)
        #     scheme_config = scheme_config.copy()
        #     scheme_config["device"] = 0

        #     app = Flask(f"WorkerTransformer_{port}")

        #     # Defer worker creation until first request to allow Flask to start quickly
        #     worker = None

        #     def ensure_worker():
        #         nonlocal worker
        #         if worker is None:
        #             worker = cls(scheme_config, model_name)
        #         return worker

        #     @app.route("/health", methods=["GET"])
        #     def health():
        #         """Health check endpoint that doesn't require worker initialization."""
        #         return jsonify({"status": "ready"})

        #     @app.route("/init", methods=["POST"])
        #     def init():
        #         try:
        #             w = ensure_worker()
        #             result = w.init()
        #             return jsonify({"status": "success", "result": result})
        #         except Exception as e:
        #             torchrl_logger.error(f"Init error: {e}", exc_info=True)
        #             return jsonify({"status": "error", "error": str(e)}), 500

        #     @app.route("/get_model_metadata", methods=["GET"])
        #     def get_model_metadata():
        #         try:
        #             # Return metadata as JSON
        #             w = ensure_worker()
        #             serialized = TestWeightSyncVLLMNCCL.serialize_metadata(w.model_metadata)
        #             return jsonify({"status": "success", "result": serialized})
        #         except Exception as e:
        #             torchrl_logger.error(f"Get metadata error: {e}", exc_info=True)
        #             return jsonify({"status": "error", "error": str(e)}), 500

        #     @app.route("/update_weights", methods=["POST"])
        #     def update_weights():
        #         try:
        #             data = request.json or {}
        #             modify_weights = data.get("modify_weights", False)
        #             w = ensure_worker()
        #             result = w.update_weights(modify_weights)
        #             return jsonify({"status": "success", "result": result})
        #         except Exception as e:
        #             torchrl_logger.error(f"Update weights error: {e}", exc_info=True)
        #             return jsonify({"status": "error", "error": str(e)}), 500

        #     @app.route("/get_first_param_sum", methods=["GET"])
        #     def get_first_param_sum():
        #         try:
        #             w = ensure_worker()
        #             result = w.get_first_param_sum()
        #             return jsonify({"status": "success", "result": result})
        #         except Exception as e:
        #             torchrl_logger.error(f"Get param sum error: {e}", exc_info=True)
        #             return jsonify({"status": "error", "error": str(e)}), 500

        #     @app.route("/shutdown", methods=["POST"])
        #     def shutdown():
        #         try:
        #             func = request.environ.get("werkzeug.server.shutdown")
        #             if func is None:
        #                 # Running under a different WSGI server
        #                 import os
        #                 import signal

        #                 os.kill(os.getpid(), signal.SIGTERM)
        #             else:
        #                 func()
        #             return jsonify({"status": "shutdown"})
        #         except Exception as e:
        #             torchrl_logger.error(f"Shutdown error: {e}", exc_info=True)
        #             return jsonify({"status": "error", "error": str(e)}), 500

        #     torchrl_logger.info(
        #         f"WorkerTransformer HTTP server starting on port {port}"
        #     )
        #     app.run(host="0.0.0.0", port=port, threaded=True)

        @classmethod
        def as_remote(cls, *args, **kwargs):
            import ray

            return ray.remote(num_cpus=4, num_gpus=1, max_concurrency=4)(cls)

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
            receiver = TestWeightSyncVLLMNCCL.WorkerVLLM.as_remote().remote(
                scheme_config, model_name, trainer_actor_name="Trainer"
            )

            # Set up vLLM engine (creates placement group with 2 GPUs for workers)
            logger.info("Setting up vLLM engine...")
            ray.get(receiver.setup.remote())
            logger.info("vLLM engine setup complete")

            # Now create trainer actor (needs 1 GPU for training and NCCL rank 0)
            logger.info("Creating trainer actor (needs 1 GPU)...")
            trainer = (
                TestWeightSyncVLLMNCCL.WorkerTransformer.as_remote()
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

    # def test_weight_sync_vllm_collective_mp(self, request):
    #     """Test weight sync between transformer trainer and vLLM workers.

    #     Uses multiprocessing with pipes for RPC coordination instead of Ray.

    #     This test demonstrates the same behavior as test_weight_sync_vllm_collective_ray
    #     but using Python's multiprocessing:
    #     1. Trainer and receiver run in separate processes
    #     2. Main process coordinates via pipe commands
    #     3. Metadata exchange happens via separate pipes
    #     4. Both initialize simultaneously for collective handshake
    #     5. Weight updates can be triggered via pipe commands
    #     """
    #     import random

    #     # Determine model based on --runslow flag
    #     if request.config.getoption("--runslow"):
    #         model_name = "Qwen/Qwen2.5-3B"
    #         torchrl_logger.info("Using large model (3B) for slow test")
    #     else:
    #         model_name = "Qwen/Qwen2.5-0.5B"
    #         torchrl_logger.info("Using small model (0.5B) for fast test")

    #     # Create scheme configuration
    #     test_port = 10234
    #     scheme_config = {
    #         "master_address": "localhost",
    #         "master_port": test_port,
    #         "gpus_per_replica": 1,
    #         "num_replicas": 2,
    #         "strategy": "state_dict",
    #         # device will be set in each worker's run_forever based on CUDA_VISIBLE_DEVICES
    #         # Trainer: CUDA_VISIBLE_DEVICES="0" -> uses GPU 0
    #         # Receiver: CUDA_VISIBLE_DEVICES="1,2" -> vLLM uses GPUs 1,2 for its 2 replicas
    #     }
    #     torchrl_logger.info(f"Using NCCL port {test_port}")

    #     # Create pipes for communication
    #     # Pipe for trainer commands
    #     trainer_parent_pipe, trainer_child_pipe = mp.Pipe()
    #     # Pipe for receiver commands
    #     receiver_parent_pipe, receiver_child_pipe = mp.Pipe()
    #     # Pipe for metadata exchange (trainer -> receiver)
    #     metadata_send_pipe, metadata_recv_pipe = mp.Pipe()

    #     trainer_proc = None
    #     receiver_proc = None
    #     try:
    #         # Start receiver process (needs 2 GPUs for vLLM workers)
    #         torchrl_logger.info(
    #             "Starting receiver process (vLLM workers need 2 GPUs)..."
    #         )
    #         receiver_proc = mp.Process(
    #             target=TestWeightSyncVLLMNCCL.WorkerVLLM.run_forever,
    #             args=(
    #                 scheme_config,
    #                 receiver_parent_pipe,
    #                 receiver_child_pipe,
    #                 metadata_recv_pipe,
    #                 model_name,
    #             ),
    #         )
    #         receiver_proc.start()
    #         receiver_child_pipe.close()  # Close child end in parent

    #         # Start trainer process (needs 1 GPU)
    #         torchrl_logger.info("Starting trainer process (needs 1 GPU)...")
    #         trainer_proc = mp.Process(
    #             target=TestWeightSyncVLLMNCCL.WorkerTransformer.run_forever,
    #             args=(
    #                 scheme_config,
    #                 trainer_parent_pipe,
    #                 trainer_child_pipe,
    #                 metadata_send_pipe,
    #                 model_name,
    #             ),
    #         )
    #         trainer_proc.start()
    #         trainer_child_pipe.close()  # Close child end in parent

    #         # Helper to send command and wait for response
    #         def send_command(pipe, command, timeout=180.0):
    #             pipe.send(command)
    #             if pipe.poll(timeout):
    #                 response = pipe.recv()
    #                 if response.get("status") == "error":
    #                     raise RuntimeError(f"Command failed: {response.get('error')}")
    #                 return response
    #             else:
    #                 raise TimeoutError(f"Command {command} timed out")

    #         # Check for successful instantiation
    #         assert receiver_parent_pipe.recv()["status"] == "success"
    #         # Check for successful instantiation
    #         assert trainer_parent_pipe.recv()["status"] == "success"

    #         # Step 1: Setup vLLM engine
    #         torchrl_logger.info("Setting up vLLM engine...")
    #         send_command(receiver_parent_pipe, "setup")
    #         torchrl_logger.info("vLLM engine setup complete")

    #         # Step 2: Receiver gets metadata from trainer
    #         torchrl_logger.info("Step 1: Receiver fetching metadata from trainer...")
    #         # Trainer sends metadata
    #         send_command(trainer_parent_pipe, {"cmd": "get_model_metadata"})
    #         # Receiver receives metadata
    #         send_command(receiver_parent_pipe, "init_metadata")
    #         torchrl_logger.info("Metadata exchange complete")

    #         # Step 3: Start NCCL init on both sides (parallel)
    #         torchrl_logger.info(
    #             "Step 2: Starting NCCL init on both trainer and workers..."
    #         )
    #         # Send init commands to both (non-blocking on main process side)
    #         trainer_parent_pipe.send("init")
    #         receiver_parent_pipe.send("init")

    #         # Wait for both to complete
    #         torchrl_logger.info("Waiting for trainer NCCL init...")
    #         trainer_response = None
    #         if trainer_parent_pipe.poll(60.0):  # Longer timeout for NCCL
    #             trainer_response = trainer_parent_pipe.recv()
    #             if trainer_response.get("status") == "error":
    #                 raise RuntimeError(
    #                     f"Trainer init failed: {trainer_response.get('error')}"
    #                 )
    #         else:
    #             raise TimeoutError("Trainer NCCL init timed out")

    #         torchrl_logger.info("Waiting for receiver NCCL init...")
    #         receiver_response = None
    #         if receiver_parent_pipe.poll(60.0):
    #             receiver_response = receiver_parent_pipe.recv()
    #             if receiver_response.get("status") == "error":
    #                 raise RuntimeError(
    #                     f"Receiver init failed: {receiver_response.get('error')}"
    #                 )
    #         else:
    #             raise TimeoutError("Receiver NCCL init timed out")

    #         torchrl_logger.info("NCCL collective initialization complete!")

    #         # Get initial state
    #         initial_response = send_command(
    #             trainer_parent_pipe, {"cmd": "get_first_param_sum"}
    #         )
    #         initial_sum = initial_response.get("result")
    #         torchrl_logger.info(f"Initial param sum: {initial_sum}")

    #         # Trigger weight update with modification using concurrent pattern
    #         torchrl_logger.info("Triggering concurrent weight update...")

    #         # Send both commands without waiting (they'll execute concurrently)
    #         t0 = time.time()
    #         receiver_parent_pipe.send("update_weights_receiver")
    #         trainer_parent_pipe.send({"cmd": "update_weights", "modify_weights": True})

    #         # Wait for both responses
    #         if receiver_parent_pipe.poll(180.0):
    #             receiver_response = receiver_parent_pipe.recv()
    #             if receiver_response.get("status") == "error":
    #                 raise RuntimeError(f"Receiver update failed: {receiver_response.get('error')}")
    #         else:
    #             raise TimeoutError("Receiver update timed out")

    #         if trainer_parent_pipe.poll(180.0):
    #             trainer_response = trainer_parent_pipe.recv()
    #             if trainer_response.get("status") == "error":
    #                 raise RuntimeError(f"Trainer update failed: {trainer_response.get('error')}")
    #         else:
    #             raise TimeoutError("Trainer update timed out")

    #         t1 = time.time()
    #         update_time = t1 - t0
    #         torchrl_logger.info(f"=== NCCL weight update completed in {update_time:.3f}s ===")

    #         # Verify weights changed
    #         updated_response = send_command(
    #             trainer_parent_pipe, {"cmd": "get_first_param_sum"}
    #         )
    #         updated_sum = updated_response.get("result")
    #         torchrl_logger.info(f"Updated param sum: {updated_sum}")
    #         assert updated_sum != initial_sum, "Weights should have changed"

    #         # Verify receiver still functional
    #         sample_response = send_command(
    #             receiver_parent_pipe, {"cmd": "get_sample_output"}
    #         )
    #         assert sample_response.get("result") == "vllm_ready"

    #         torchrl_logger.info("Test completed successfully!")

    #     finally:
    #         # Shutdown processes
    #         torchrl_logger.info("Shutting down processes...")
    #         try:
    #             trainer_parent_pipe.send("shutdown")
    #             receiver_parent_pipe.send("shutdown")
    #         except Exception as e:
    #             torchrl_logger.warning(f"Error sending shutdown: {e}")

    #         # Wait for processes to exit
    #         if trainer_proc is not None and trainer_proc.is_alive():
    #             trainer_proc.join(timeout=5.0)
    #             if trainer_proc.is_alive():
    #                 torchrl_logger.warning(
    #                     "Trainer process did not exit, terminating..."
    #                 )
    #                 trainer_proc.terminate()
    #                 trainer_proc.join(timeout=2.0)

    #         if receiver_proc is not None and receiver_proc.is_alive():
    #             receiver_proc.join(timeout=5.0)
    #             if receiver_proc.is_alive():
    #                 torchrl_logger.warning(
    #                     "Receiver process did not exit, terminating..."
    #                 )
    #                 receiver_proc.terminate()
    #                 receiver_proc.join(timeout=2.0)

    #         # Close pipes
    #         trainer_parent_pipe.close()
    #         receiver_parent_pipe.close()
    #         metadata_send_pipe.close()
    #         metadata_recv_pipe.close()

    # def test_weight_sync_vllm_collective_http(self, request):
    #     """Test weight sync between transformer trainer and vLLM workers.

    #     Uses HTTP/REST for RPC coordination instead of Ray or pipes.

    #     This test demonstrates the same behavior using HTTP:
    #     1. Trainer and receiver run Flask servers in separate processes
    #     2. Main process coordinates via HTTP POST/GET requests
    #     3. Metadata exchange happens via REST endpoints
    #     4. Both initialize simultaneously for collective handshake
    #     5. Weight updates can be triggered via HTTP requests

    #     Benefits:
    #     - Easy to debug with curl/browser
    #     - Works across any network
    #     - Language-agnostic (workers could be in different languages)
    #     - No special dependencies beyond Flask
    #     """
    #     import random

    #     import requests

    #     # Determine model based on --runslow flag
    #     if request.config.getoption("--runslow"):
    #         model_name = "Qwen/Qwen2.5-3B"
    #         torchrl_logger.info("Using large model (3B) for slow test")
    #     else:
    #         model_name = "Qwen/Qwen2.5-0.5B"
    #         torchrl_logger.info("Using small model (0.5B) for fast test")

    #     # Create scheme configuration
    #     test_port = 10235
    #     scheme_config = {
    #         "master_address": "localhost",
    #         "master_port": test_port,
    #         "gpus_per_replica": 1,
    #         "num_replicas": 2,
    #         "strategy": "state_dict",
    #     }
    #     torchrl_logger.info(f"Using NCCL port {test_port}")

    #     # Choose random ports for HTTP servers
    #     receiver_http_port = random.randint(5000, 5100)
    #     trainer_http_port = random.randint(5100, 5200)

    #     try:
    #         # Start receiver HTTP server (needs 2 GPUs for vLLM workers)
    #         torchrl_logger.info(
    #             f"Starting receiver HTTP server on port {receiver_http_port}..."
    #         )
    #         receiver_proc = mp.Process(
    #             target=TestWeightSyncVLLMNCCL.WorkerVLLM.run_forever_http,
    #             args=(scheme_config, receiver_http_port, model_name),
    #         )
    #         receiver_proc.start()

    #         # Start trainer HTTP server (needs 1 GPU)
    #         torchrl_logger.info(
    #             f"Starting trainer HTTP server on port {trainer_http_port}..."
    #         )
    #         trainer_proc = mp.Process(
    #             target=TestWeightSyncVLLMNCCL.WorkerTransformer.run_forever_http,
    #             args=(scheme_config, trainer_http_port, model_name),
    #         )
    #         trainer_proc.start()

    #         # Wait for servers to be ready by polling health endpoints
    #         torchrl_logger.info("Waiting for HTTP servers to start...")
    #         receiver_url = f"http://localhost:{receiver_http_port}"
    #         trainer_url = f"http://localhost:{trainer_http_port}"

    #         # Poll health endpoints with timeout
    #         start_time = time.time()
    #         timeout = 180.0
    #         receiver_ready = False
    #         trainer_ready = False

    #         while time.time() - start_time < timeout:
    #             if not receiver_ready:
    #                 try:
    #                     resp = requests.get(f"{receiver_url}/health", timeout=1.0)
    #                     if resp.status_code == 200:
    #                         receiver_ready = True
    #                         torchrl_logger.info("Receiver HTTP server is ready")
    #                 except Exception:
    #                     pass  # Server not ready yet

    #             if not trainer_ready:
    #                 try:
    #                     resp = requests.get(f"{trainer_url}/health", timeout=1.0)
    #                     if resp.status_code == 200:
    #                         trainer_ready = True
    #                         torchrl_logger.info("Trainer HTTP server is ready")
    #                 except Exception:
    #                     pass  # Server not ready yet

    #             if receiver_ready and trainer_ready:
    #                 break

    #             time.sleep(0.5)

    #         if not (receiver_ready and trainer_ready):
    #             raise TimeoutError(
    #                 f"Servers did not start within {timeout}s. "
    #                 f"Receiver: {receiver_ready}, Trainer: {trainer_ready}"
    #             )

    #         # Helper to make HTTP requests
    #         def http_request(
    #             base_url, endpoint, method="POST", data=None, timeout=180.0
    #         ):
    #             url = f"{base_url}{endpoint}"
    #             try:
    #                 if method == "GET":
    #                     response = requests.get(url, timeout=timeout)
    #                 else:
    #                     response = requests.post(url, json=data, timeout=timeout)

    #                 response.raise_for_status()
    #                 result = response.json()

    #                 if result.get("status") == "error":
    #                     raise RuntimeError(f"Request failed: {result.get('error')}")
    #                 return result
    #             except requests.exceptions.RequestException as e:
    #                 raise RuntimeError(f"HTTP request to {url} failed: {e}")

    #         # Step 1: Setup vLLM engine
    #         torchrl_logger.info("Setting up vLLM engine via HTTP...")
    #         http_request(receiver_url, "/setup")
    #         torchrl_logger.info("vLLM engine setup complete")

    #         # Step 2: Receiver gets metadata from trainer
    #         torchrl_logger.info("Step 1: Fetching metadata from trainer via HTTP...")
    #         # Get metadata from trainer
    #         metadata_response = http_request(
    #             trainer_url, "/get_model_metadata", method="GET"
    #         )
    #         metadata = metadata_response.get("result")
    #         torchrl_logger.info(f"Fetched metadata with {len(metadata)} parameters")
    #         torchrl_logger.info(f"First 3 params: {list(metadata.keys())[:3]}")
    #         torchrl_logger.info(f"Last 3 params: {list(metadata.keys())[-3:]}")

    #         # Send metadata to receiver
    #         http_request(receiver_url, "/init_metadata", data=metadata)
    #         torchrl_logger.info("Metadata exchange complete")

    #         # Step 3: Start NCCL init on both sides
    #         # Note: HTTP is synchronous, so we need to do this carefully
    #         # We'll use threading to make parallel requests
    #         torchrl_logger.info(
    #             "Step 2: Starting NCCL init on both trainer and workers..."
    #         )

    #         import queue
    #         from threading import Thread

    #         # Queues to collect results
    #         trainer_queue = queue.Queue()
    #         receiver_queue = queue.Queue()

    #         def init_trainer():
    #             try:
    #                 result = http_request(trainer_url, "/init", timeout=180.0)
    #                 trainer_queue.put(("success", result))
    #             except Exception as e:
    #                 trainer_queue.put(("error", str(e)))

    #         def init_receiver():
    #             try:
    #                 result = http_request(receiver_url, "/init", timeout=180.0)
    #                 receiver_queue.put(("success", result))
    #             except Exception as e:
    #                 receiver_queue.put(("error", str(e)))

    #         # Start both initializations in parallel
    #         trainer_thread = Thread(target=init_trainer)
    #         receiver_thread = Thread(target=init_receiver)

    #         trainer_thread.start()
    #         receiver_thread.start()

    #         # Wait for both to complete
    #         torchrl_logger.info("Waiting for trainer NCCL init...")
    #         trainer_thread.join(timeout=180.0)
    #         status, result = trainer_queue.get()
    #         if status == "error":
    #             raise RuntimeError(f"Trainer init failed: {result}")

    #         torchrl_logger.info("Waiting for receiver NCCL init...")
    #         receiver_thread.join(timeout=180.0)
    #         status, result = receiver_queue.get()
    #         if status == "error":
    #             raise RuntimeError(f"Receiver init failed: {result}")

    #         torchrl_logger.info("NCCL collective initialization complete!")

    #         # Get initial state
    #         initial_response = http_request(
    #             trainer_url, "/get_first_param_sum", method="GET"
    #         )
    #         initial_sum = initial_response.get("result")
    #         torchrl_logger.info(f"Initial param sum: {initial_sum}")

    #         # Trigger weight update with modification using concurrent pattern
    #         torchrl_logger.info("Triggering concurrent weight update...")

    #         # Use threading to call both endpoints concurrently
    #         receiver_queue = queue.Queue()
    #         trainer_queue = queue.Queue()

    #         def update_receiver():
    #             try:
    #                 result = http_request(receiver_url, "/update_weights_receiver", timeout=180.0)
    #                 receiver_queue.put(("success", result))
    #             except Exception as e:
    #                 receiver_queue.put(("error", str(e)))

    #         def update_trainer():
    #             try:
    #                 result = http_request(trainer_url, "/update_weights", data={"modify_weights": True}, timeout=180.0)
    #                 trainer_queue.put(("success", result))
    #             except Exception as e:
    #                 trainer_queue.put(("error", str(e)))

    #         # Start both updates in parallel
    #         t0 = time.time()
    #         receiver_update_thread = Thread(target=update_receiver)
    #         trainer_update_thread = Thread(target=update_trainer)

    #         receiver_update_thread.start()
    #         trainer_update_thread.start()

    #         # Wait for both to complete
    #         receiver_update_thread.join(timeout=180.0)
    #         trainer_update_thread.join(timeout=180.0)

    #         # Check results
    #         status, result = receiver_queue.get()
    #         if status == "error":
    #             raise RuntimeError(f"Receiver update failed: {result}")

    #         status, result = trainer_queue.get()
    #         if status == "error":
    #             raise RuntimeError(f"Trainer update failed: {result}")

    #         t1 = time.time()
    #         update_time = t1 - t0
    #         torchrl_logger.info(f"=== NCCL weight update completed in {update_time:.3f}s ===")

    #         # Verify weights changed
    #         updated_response = http_request(
    #             trainer_url, "/get_first_param_sum", method="GET"
    #         )
    #         updated_sum = updated_response.get("result")
    #         torchrl_logger.info(f"Updated param sum: {updated_sum}")
    #         assert updated_sum != initial_sum, "Weights should have changed"

    #         # Verify receiver still functional
    #         sample_response = http_request(
    #             receiver_url, "/get_sample_output", method="GET"
    #         )
    #         assert sample_response.get("result") == "vllm_ready"

    #         torchrl_logger.info("Test completed successfully!")
    #         torchrl_logger.info(
    #             f"You can debug with: curl http://localhost:{trainer_http_port}/get_first_param_sum"
    #         )

    #     finally:
    #         # Shutdown processes
    #         torchrl_logger.info("Shutting down HTTP servers...")
    #         try:
    #             requests.post(
    #                 f"http://localhost:{trainer_http_port}/shutdown", timeout=2.0
    #             )
    #         except Exception as e:
    #             torchrl_logger.warning(f"Error shutting down trainer: {e}")

    #         try:
    #             requests.post(
    #                 f"http://localhost:{receiver_http_port}/shutdown", timeout=2.0
    #             )
    #         except Exception as e:
    #             torchrl_logger.warning(f"Error shutting down receiver: {e}")

    #         # Wait for processes to exit
    #         if trainer_proc.is_alive():
    #             trainer_proc.join(timeout=5.0)
    #             if trainer_proc.is_alive():
    #                 torchrl_logger.warning(
    #                     "Trainer process did not exit, terminating..."
    #                 )
    #                 trainer_proc.terminate()
    #                 trainer_proc.join(timeout=2.0)

    #         if receiver_proc.is_alive():
    #             receiver_proc.join(timeout=5.0)
    #             if receiver_proc.is_alive():
    #                 torchrl_logger.warning(
    #                     "Receiver process did not exit, terminating..."
    #                 )
    #                 receiver_proc.terminate()
    #                 receiver_proc.join(timeout=2.0)


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

    class WorkerVLLM:
        """Ray actor for vLLM inference worker (receiver)."""

        def __init__(self, scheme_config: dict, model_name: str = "Qwen/Qwen2.5-0.5B"):
            # Store config for deferred initialization
            self.scheme_config = scheme_config
            self.model_name = model_name
            self.wrapper = None
            self.engine = None
            self.receiver = None
            self.scheme = None

        def setup(self):
            """Set up vLLM engine and receiver."""
            # Create vLLM wrapper
            self.wrapper = TestWeightSyncVLLMDoubleBuffer._make_worker_vllm(
                self.model_name
            )
            self.engine = self.wrapper.model

            # Create scheme from config
            from torchrl.weight_update.llm.vllm_double_buffer import (
                VLLMDoubleBufferSyncScheme,
            )

            self.scheme = VLLMDoubleBufferSyncScheme(**self.scheme_config)

            # Create receiver
            self.receiver = self.scheme.create_receiver(self.engine)
            logger.info("Receiver setup complete")
            return "setup_complete"

        def poll_and_apply_weights(self):
            """Poll for new weights and apply them to the engine."""
            if self.receiver is None:
                raise RuntimeError("Must call setup() first")

            success = self.receiver.poll_and_apply()
            return success

        def get_sample_output(self):
            """Get a sample output to verify model works."""
            return "vllm_ready"

        @classmethod
        def as_remote(cls, *args, **kwargs):
            import ray

            # vLLM worker needs 1 GPU
            return ray.remote(num_cpus=2, num_gpus=1, max_concurrency=4)(cls)

    class WorkerTransformer:
        """Ray actor for transformer trainer (sender)."""

        def __init__(self, scheme_config: dict, model_name: str = "Qwen/Qwen2.5-0.5B"):
            from torchrl.weight_update.llm.vllm_double_buffer import (
                VLLMDoubleBufferSyncScheme,
            )

            # Create transformer model
            self.transformer = TestWeightSyncVLLMDoubleBuffer._make_worker_transformer(
                model_name
            )

            # Create scheme from config
            self.scheme = VLLMDoubleBufferSyncScheme(**scheme_config)

            # Create sender
            self.sender = self.scheme.create_sender()
            self.sender.register_model(self.transformer)
            logger.info("Trainer setup complete")

        def update_weights(self, modify_weights: bool = False):
            """Trigger a weight update by writing to shared storage.

            Args:
                modify_weights: If True, modifies weights before writing
                                for verification purposes.

            Returns:
                str: "updated" status message
            """
            # Optionally modify weights for testing
            if modify_weights:
                with torch.no_grad():
                    first_param = next(self.transformer.parameters())
                    first_param.add_(0.01)

            # Write weights to shared storage
            self.sender.update_weights()
            return "updated"

        def get_first_param_sum(self):
            """Get sum of first parameter for verification."""
            return next(self.transformer.parameters()).sum().item()

        @classmethod
        def as_remote(cls, *args, **kwargs):
            import ray

            return ray.remote(num_cpus=2, num_gpus=1, max_concurrency=4)(cls)

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
            trainer = (
                TestWeightSyncVLLMDoubleBuffer.WorkerTransformer.as_remote().remote(
                    scheme_config, model_name
                )
            )
            logger.info("Trainer actor created")

            # Create receiver actor
            logger.info("Creating receiver actor...")
            receiver = TestWeightSyncVLLMDoubleBuffer.WorkerVLLM.as_remote().remote(
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
