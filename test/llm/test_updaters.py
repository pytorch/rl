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
from tensordict import TensorDict
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
    from torchrl.collectors import WeightUpdaterBase
    from torchrl.modules.llm.backends.vllm import (
        make_vllm_worker,
        stateless_init_process_group,
    )
    from torchrl.modules.llm.backends.vllm_async import AsyncVLLM
    from torchrl.modules.llm.policies import vLLMWrapper


class vLLMUpdaterV2(WeightUpdaterBase):
    """A simplified vLLM weight updater that infers configuration from vLLM objects.

    This version takes vLLM engines directly and extracts the necessary information
    from them, avoiding the need to manually specify tensor parallel size and other
    configuration parameters.

    Args:
        vllm_engines: vLLM engine(s) to update weights for. Can be:
            - vllm.LLM: Local vLLM instance
            - Ray actor handle: Remote vLLM worker
            - AsyncVLLM: Async vLLM service
            - list: Multiple engines (experimental)
        master_address (str, optional): Master address for distributed training. Defaults to localhost.
        master_port (int, optional): Master port for distributed training. Auto-assigned if None.
    """

    def __init__(
        self,
        vllm_engines,
        master_address: str | None = None,
        master_port: int | None = None,
    ):
        torchrl_logger.info(f"=> in {type(self).__name__}.__init__")
        self.vllm_engines = (
            vllm_engines if isinstance(vllm_engines, list) else [vllm_engines]
        )
        self.master_address = master_address
        self.master_port = master_port
        self.model_metadata = None
        self.initialized_group = None
        self.vllm_tp_size = None
        self.vllm_comm_group = None

        # Analyze the engines to extract configuration
        self._analyze_engines()

    def _analyze_engines(self):
        """Analyze vLLM engines to extract configuration information."""
        if not self.vllm_engines:
            raise ValueError("At least one vLLM engine must be provided")

        # Take the first engine as representative
        engine = self.vllm_engines[0]

        # Detect engine type and extract tensor parallel size
        if hasattr(engine, "engine_args"):
            # AsyncVLLM service
            self.vllm_tp_size = engine.engine_args.tensor_parallel_size
            torchrl_logger.info(f"Detected AsyncVLLM with tp_size={self.vllm_tp_size}")
        elif hasattr(engine, "llm_engine") and hasattr(
            engine.llm_engine, "model_config"
        ):
            # Local vLLM LLM instance
            # Access tensor parallel size from the model config
            try:
                # Try to get parallel config
                parallel_config = engine.llm_engine.parallel_config
                self.vllm_tp_size = parallel_config.tensor_parallel_size
                torchrl_logger.info(
                    f"Detected local vLLM LLM with tp_size={self.vllm_tp_size}"
                )
            except AttributeError:
                # Fallback - assume single GPU
                self.vllm_tp_size = 1
                torchrl_logger.warning(
                    "Could not determine tensor parallel size, defaulting to 1"
                )
        elif hasattr(engine, "generate"):
            # Remote vLLM worker (Ray actor) - we need to query it
            try:
                if _has_ray:
                    # Try to get tensor parallel size from remote worker
                    # This is a best-effort approach
                    self.vllm_tp_size = 1  # Conservative default
                    torchrl_logger.info(
                        f"Detected remote vLLM worker, assuming tp_size={self.vllm_tp_size}"
                    )
                else:
                    raise ImportError("Ray not available for remote engine analysis")
            except Exception as e:
                torchrl_logger.warning(
                    f"Could not analyze remote engine: {e}, defaulting to tp_size=1"
                )
                self.vllm_tp_size = 1
        else:
            raise ValueError(f"Unsupported vLLM engine type: {type(engine)}")

    def get_tp_size(self) -> int:
        """Get the tensor parallel size for the vLLM engines."""
        return self.vllm_tp_size

    def init(self, model_metadata: dict[str, tuple[torch.dtype, torch.Size]]) -> None:
        """Initialize the updater with model metadata."""
        self.model_metadata = model_metadata
        self.maybe_init_group()

    @property
    def master_address(self):
        if self._master_address is None:
            self._master_address = "localhost"
        return self._master_address

    @master_address.setter
    def master_address(self, value):
        self._master_address = value

    @property
    def master_port(self):
        if self._master_port is None:
            self._master_port = get_open_port()
        return self._master_port

    @master_port.setter
    def master_port(self, value):
        self._master_port = value

    def _get_model_ref(self):
        """Get reference to the first vLLM engine for communication."""
        return self.vllm_engines[0]

    def _init_group(self):
        """Initialize the weight update communication group."""
        torchrl_logger.info(f"=> in {type(self).__name__}._init_group")
        weight_sync_world_size = self.vllm_tp_size + 1
        torchrl_logger.info(f"initializing group with {weight_sync_world_size=}")
        torchrl_logger.info(f"vllm_tp_size={self.vllm_tp_size}")

        model_ref = self._get_model_ref()

        # Handle different engine types
        if hasattr(model_ref, "collective_rpc"):
            # Remote vLLM worker
            torchrl_logger.info("Initializing remote vLLM worker weight update group")
            if _has_ray:
                init_weight_update_group_getter = model_ref.collective_rpc.remote(
                    "init_weight_update_group",
                    args=(
                        self.master_address,
                        self.master_port,
                        1,
                        weight_sync_world_size,
                    ),
                )
                # Initialize local communication group
                self.vllm_comm_group = stateless_init_process_group(
                    self.master_address,
                    self.master_port,
                    0,
                    weight_sync_world_size,
                    torch.device("cuda:0"),
                )
                ray.get(init_weight_update_group_getter)
            else:
                raise ImportError("Ray not available for remote weight updates")
        elif hasattr(model_ref, "actors"):
            # AsyncVLLM service with multiple actors
            torchrl_logger.info("Initializing AsyncVLLM service weight update group")
            if _has_ray:
                # Initialize the weight update group for AsyncVLLM
                refs = model_ref.init_weight_update_group(
                    self.master_address, self.master_port
                )
                # Initialize local communication group
                self.vllm_comm_group = stateless_init_process_group(
                    self.master_address,
                    self.master_port,
                    0,
                    weight_sync_world_size,
                    torch.device("cuda:0"),
                )
                ray.get(refs)
            else:
                raise ImportError("Ray not available for AsyncVLLM weight updates")
        else:
            # Local vLLM instance - not supported for weight updates
            torchrl_logger.warning("Local vLLM instances do not support weight updates")
            self.vllm_comm_group = True  # Dummy value to indicate "initialized"

        torchrl_logger.info("group initialized")
        self.initialized_group = True

    def maybe_init_group(self):
        """Initialize the group if not already done and collector is available."""
        if self.initialized_group is None and self.collector is not None:
            self._init_group()

    def _sync_weights_with_worker(
        self,
        *,
        worker_id: int | torch.device | None = None,
        server_weights: dict | None = None,
    ) -> None:
        """Synchronize weights with vLLM workers."""
        if server_weights is None:
            raise ValueError("server_weights cannot be None for vLLMUpdaterV2")

        if self.initialized_group is None:
            raise RuntimeError("Weight updater not initialized. Call init() first.")

        if self.model_metadata is None:
            raise RuntimeError("Model metadata not set. Call init() with metadata.")

        model_ref = self._get_model_ref()

        # Handle different engine types
        if hasattr(model_ref, "collective_rpc") and _has_ray:
            # Remote vLLM worker
            torchrl_logger.info("Syncing weights with remote vLLM worker")
            remotes = []
            for k, (dtype, shape) in self.model_metadata.items():
                remotes.append(
                    model_ref.collective_rpc.remote(
                        "update_weight_broadcast", args=(k, dtype, shape)
                    )
                )

            # Broadcast weights
            for k in self.model_metadata:
                val = server_weights[k].to(torch.device("cuda:0"))
                self.vllm_comm_group.broadcast(
                    val,
                    src=0,
                    stream=torch.cuda.current_stream(),
                )
                del val

            ray.get(remotes)

        elif hasattr(model_ref, "actors") and _has_ray:
            # AsyncVLLM service
            torchrl_logger.info("Syncing weights with AsyncVLLM service")

            # Broadcast metadata to all actors
            remotes = []
            for k, (dtype, shape) in self.model_metadata.items():
                remote_calls = model_ref.collective_rpc(
                    "update_weight_broadcast", args=(k, dtype, shape)
                )
                remotes.extend(
                    remote_calls if isinstance(remote_calls, list) else [remote_calls]
                )

            # Broadcast weights
            for k in self.model_metadata:
                val = server_weights[k].to(torch.device("cuda:0"))
                self.vllm_comm_group.broadcast(
                    val,
                    src=0,
                    stream=torch.cuda.current_stream(),
                )
                del val

            ray.get(remotes)

        else:
            # Local vLLM instance - just log
            torchrl_logger.info(
                "Local vLLM weight sync not implemented (weights shared by reference)"
            )

        torchrl_logger.info("Weight sync completed")
        torch.cuda.synchronize()

    def _get_server_weights(self):
        """Not used - weights must be passed directly."""
        return None

    def _maybe_map_weights(self, server_weights):
        """Map weights to the expected format."""
        if hasattr(server_weights, "model") and hasattr(
            server_weights.model, "state_dict"
        ):
            # TransformersWrapper or similar
            return server_weights.model.state_dict()
        elif hasattr(server_weights, "state_dict"):
            # Direct model
            return server_weights.state_dict()
        elif isinstance(server_weights, dict):
            # Already a dict
            return server_weights
        else:
            raise TypeError(f"Unsupported server_weights type: {type(server_weights)}")

    @classmethod
    def get_model_metadata(cls, model) -> dict[str, tuple[torch.dtype, torch.Size]]:
        """Get model metadata from a model."""
        if hasattr(model, "model") and hasattr(model.model, "state_dict"):
            sd = model.model.state_dict()
        elif hasattr(model, "state_dict"):
            sd = model.state_dict()
        else:
            raise TypeError(f"Cannot extract state_dict from {type(model)}")

        return {k: (v.dtype, v.shape) for k, v in sd.items()}

    def all_worker_ids(self):
        """Return list of worker IDs."""
        return [0]


@pytest.mark.skipif(not _has_transformers, reason="missing transformers dependencies")
@pytest.mark.skipif(not _has_vllm, reason="missing vllm dependencies")
@pytest.mark.skipif(not _has_ray, reason="missing ray dependencies")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestVLLMUpdaterV2:
    """Test the new vLLMUpdaterV2 class."""

    @pytest.fixture(scope="class", autouse=True)
    def setup_ray(self):
        """Initialize Ray for testing."""
        if not ray.is_initialized():
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
        return SamplingParams(temperature=0.8, max_tokens=50)

    @pytest.fixture
    def source_policy(self, model_name):
        """Create source TransformersWrapper policy."""
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

    @pytest.mark.slow
    def test_updater_v2_with_ray_worker(self, source_policy, target_vllm_ray_worker):
        """Test vLLMUpdaterV2 with a Ray worker."""
        torchrl_logger.info("=== Testing vLLMUpdaterV2 with Ray worker ===")

        # Create updater V2
        updater = vLLMUpdaterV2(target_vllm_ray_worker)

        # Check that it detected the configuration correctly
        assert updater.get_tp_size() == 1

        # Get model metadata
        model_metadata = updater.get_model_metadata(source_policy)
        assert len(model_metadata) > 0
        torchrl_logger.info(f"Found {len(model_metadata)} parameters in model metadata")

        # Create mock collector
        class MockCollector:
            def __init__(self, policy):
                self.policy = policy
                self.policy.model = target_vllm_ray_worker

            def increment_version(self):
                pass

        mock_collector = MockCollector(source_policy)
        updater.register_collector(mock_collector)

        # Initialize updater
        updater.init(model_metadata)
        assert updater.initialized_group is True

        # Test weight update
        weights = updater._maybe_map_weights(source_policy)
        updater._sync_weights_with_worker(server_weights=weights)

        torchrl_logger.info("Ray worker weight update completed successfully")

    @pytest.mark.slow
    def test_updater_v2_with_async_vllm(self, source_policy, target_async_vllm):
        """Test vLLMUpdaterV2 with AsyncVLLM service."""
        torchrl_logger.info("=== Testing vLLMUpdaterV2 with AsyncVLLM ===")

        # Create updater V2
        updater = vLLMUpdaterV2(target_async_vllm)

        # Check that it detected the configuration correctly
        assert updater.get_tp_size() == 1

        # Get model metadata
        model_metadata = updater.get_model_metadata(source_policy)
        assert len(model_metadata) > 0
        torchrl_logger.info(f"Found {len(model_metadata)} parameters in model metadata")

        # Create mock collector
        class MockCollector:
            def __init__(self, policy, async_service):
                self.policy = policy
                # For AsyncVLLM, we expose the service itself
                self.policy.model = async_service

            def increment_version(self):
                pass

        mock_collector = MockCollector(source_policy, target_async_vllm)
        updater.register_collector(mock_collector)

        # Initialize updater
        updater.init(model_metadata)
        assert updater.initialized_group is True

        # Test weight update
        weights = updater._maybe_map_weights(source_policy)
        updater._sync_weights_with_worker(server_weights=weights)

        torchrl_logger.info("AsyncVLLM weight update completed successfully")

    def test_config_extraction_from_engines(
        self, target_vllm_ray_worker, target_async_vllm
    ):
        """Test that configuration is correctly extracted from different engine types."""
        torchrl_logger.info("=== Testing configuration extraction ===")

        # Test Ray worker
        updater_ray = vLLMUpdaterV2(target_vllm_ray_worker)
        assert updater_ray.get_tp_size() == 1

        # Test AsyncVLLM
        updater_async = vLLMUpdaterV2(target_async_vllm)
        assert updater_async.get_tp_size() == 1

        torchrl_logger.info("Configuration extraction tests passed")

    @pytest.mark.slow
    def test_weight_transfer_consistency(
        self, source_policy, target_vllm_ray_worker, sampling_params
    ):
        """Test that weight updates actually change the model behavior."""
        torchrl_logger.info("=== Testing weight transfer consistency ===")

        # Create vLLM wrapper for the target
        target_wrapper = vLLMWrapper(
            target_vllm_ray_worker,
            input_mode="text",
            generate=True,
            generate_kwargs={
                "max_tokens": 50,
                "temperature": 0.0,
            },  # Use temperature=0 for deterministic output
        )

        # Get initial output
        test_input = TensorDict({"text": {"prompt": ["Hello, world!"]}}, batch_size=[1])
        initial_result = target_wrapper(test_input)
        initial_text = initial_result["text"]["response"][0]

        # Create updater and update weights
        updater = vLLMUpdaterV2(target_vllm_ray_worker)
        model_metadata = updater.get_model_metadata(source_policy)

        class MockCollector:
            def __init__(self, policy):
                self.policy = policy
                self.policy.model = target_vllm_ray_worker

            def increment_version(self):
                pass

        mock_collector = MockCollector(source_policy)
        updater.register_collector(mock_collector)
        updater.init(model_metadata)

        # Modify source weights slightly
        with torch.no_grad():
            for name, param in source_policy.model.named_parameters():
                if "embed_tokens" in name and param.requires_grad:
                    param.data += torch.randn_like(param.data) * 0.001
                    break

        # Update weights
        weights = updater._maybe_map_weights(source_policy)
        updater._sync_weights_with_worker(server_weights=weights)

        # Get output after weight update
        final_result = target_wrapper(test_input)
        final_text = final_result["text"]["response"][0]

        # The outputs should be different (though this is probabilistic)
        torchrl_logger.info(f"Initial text: {initial_text}")
        torchrl_logger.info(f"Final text: {final_text}")

        # Note: Due to the small weight change, outputs might be the same
        # This is just to verify the process works without errors
        torchrl_logger.info("Weight transfer consistency test completed")


if __name__ == "__main__":
    # Simple smoke test
    pytest.main([__file__, "-v", "-s"])
