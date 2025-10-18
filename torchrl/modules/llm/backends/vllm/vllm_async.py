# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Async vLLM engine implementation for efficient batching and inference.

This module provides an async vLLM engine that leverages native vLLM batching
for better performance and memory efficiency compared to the explicit batching
approach used in the legacy vLLM backend.
"""

from __future__ import annotations

import asyncio
import os
import random
import uuid
from collections.abc import Iterator, Sequence
from typing import Any, Literal, TYPE_CHECKING

import torch
from torchrl._utils import logger as torchrl_logger

# Import RLvLLMEngine and shared utilities
from .base import RLvLLMEngine
from .vllm_utils import stateless_init_process_group

try:
    import ray
    from ray.util.placement_group import placement_group, remove_placement_group
    from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
except ImportError:
    ray = None

    def placement_group(*args, **kwargs):
        """Placement group is not available when ray is not installed."""
        raise ImportError(
            "ray is not installed. Please install it with `pip install ray`."
        )

    def remove_placement_group(*args, **kwargs):
        """Remove placement group is not available when ray is not installed."""
        raise ImportError(
            "ray is not installed. Please install it with `pip install ray`."
        )

    class PlacementGroupSchedulingStrategy:
        """Placement group scheduling strategy is not available when ray is not installed."""

        def __init__(self, *args, **kwargs):
            raise ImportError(
                "ray is not installed. Please install it with `pip install ray`."
            )


if TYPE_CHECKING:
    from vllm.engine.async_llm_engine import AsyncEngineArgs
    from vllm.engine.request import RequestOutput
    from vllm.engine.sampling_params import SamplingParams

TIMEOUT_SECONDS = os.getenv("TORCHRL_VLLM_TIMEOUT_SECONDS", 300)

try:
    import vllm

    _has_vllm = True
except ImportError:
    vllm = None
    _has_vllm = False


if not _has_vllm:

    class Worker:
        """Placeholder for Worker class when vLLM is not installed."""

        def __init__(self, *args, **kwargs):
            raise ImportError(
                "vllm is not installed. Please install it with `pip install vllm`."
            )

else:
    from vllm.worker.worker import Worker


class _AsyncvLLMWorker(Worker):
    """Async vLLM worker for Ray with weight update capabilities.

    This worker extends the base vLLM Worker to support async operations
    and weight updates via NCCL communication groups.
    """

    def __init__(self, *args, **kwargs):
        torchrl_logger.info(f"=> in {type(self).__name__}.__init__")
        torchrl_logger.info(f"visible devices {os.getenv('CUDA_VISIBLE_DEVICES')}")
        torchrl_logger.info(f"device count {torch.cuda.device_count()}")
        super().__init__(*args, **kwargs)
        self.model_update_group = None

    def init_weight_update_group(
        self,
        master_address: str,
        master_port: str,
        rank_offset: int,
        world_size: int,
    ):
        """Initialize weight update group for this worker.

        Args:
            master_address (str): The master address for distributed training.
            master_port (str): The master port for distributed training.
            rank_offset (int): Rank offset for this worker in the global weight update group.
            world_size (int): Total number of processes in the weight update group.
        """
        from vllm.distributed.parallel_state import get_world_group

        torchrl_logger.info(f"=> in {type(self).__name__}.init_weight_update_group")
        if self.model_update_group is not None:
            torchrl_logger.info("Model update group already initialized")
            return

        # Get the local rank within the tensor parallel group
        tp_group = get_world_group()
        local_rank = tp_group.rank
        torchrl_logger.info(f"Local rank in tensor parallel group: {local_rank}")

        # Calculate the global rank for weight update group
        rank = local_rank + rank_offset
        torchrl_logger.info(
            f"Initializing {type(self).__name__} weight update group with "
            f"{master_address=}, {master_port=}, {rank=}, {world_size=}, device={self.device}"
        )

        # Import synchronous version for workers too
        from .vllm_utils import stateless_init_process_group

        self.model_update_group = stateless_init_process_group(
            master_address, master_port, rank, world_size, self.device
        )

        torchrl_logger.info(f"{type(self).__name__}.init_weight_update_group success")

    def update_weight(self, name: str, dtype_name: str, shape: tuple[int, ...]):
        """Update weight via broadcast from master (rank 0) - periodic-mono pattern.

        Args:
            name (str): Parameter name.
            dtype_name (str): Parameter dtype name (e.g., 'bfloat16').
            shape (tuple[int, ...]): Parameter shape.
        """
        if self.model_update_group is None:
            raise RuntimeError("Weight update group not initialized")

        # Convert dtype name to dtype (like periodic-mono)
        dtype = getattr(torch, dtype_name)

        # Workers receive broadcast from master (rank 0)
        weight = torch.empty(shape, dtype=dtype, device="cuda")
        self.model_update_group.broadcast(
            weight, src=0, stream=torch.cuda.current_stream()
        )
        self.model_runner.model.load_weights(weights=[(name, weight)])
        del weight

    def check_nccl_group_ready(self):
        """Check if NCCL group is ready for communication."""
        ready = self.model_update_group is not None
        torchrl_logger.info(f"Worker NCCL group ready: {ready}")
        return ready


class _AsyncLLMEngine:
    """Extended AsyncLLMEngine with TorchRL-specific features.

    This class wraps vLLM's AsyncLLMEngine and adds functionality needed
    for TorchRL integration, including weight updates and batch management.

    This is a private class and should not be used directly. Use the ray remote actor class :class:`AsyncLLMEngineActor` instead.

    Keyword Args:
        engine_args (AsyncEngineArgs): Arguments for creating the AsyncLLMEngine instances.
        bundle_indices (list[int], optional): Bundle indices for the engine.
        enable_prefix_caching (bool, optional): Whether to enable prefix caching.

            .. warning::
                enable_prefix_caching is set to False by default, which is recommended if prompt log probs are needed.
                Set it to True if prompt log probs are not needed.
                See `this issue <https://github.com/vllm-project/vllm/issues/8268>`_ for more details.
    """

    def __init__(
        self,
        *,
        engine_args: AsyncEngineArgs,
        bundle_indices: list[int] | None = None,
        enable_prefix_caching: bool = False,
    ):
        if not _has_vllm:
            raise ImportError(
                "vllm is not installed. Please install it with `pip install vllm`."
            )

        from vllm import AsyncLLMEngine

        worker_cls = "torchrl.modules.llm.backends.vllm.vllm_async._AsyncvLLMWorker"
        if engine_args.worker_cls != "auto":
            old_worker_cls = engine_args.worker_cls
            torchrl_logger.warning(
                f"Overriding worker_cls from {old_worker_cls} to {worker_cls}"
            )

        if bundle_indices is not None:
            os.environ["VLLM_RAY_BUNDLE_INDICES"] = ",".join(map(str, bundle_indices))

        engine_args.worker_cls = worker_cls
        engine_args.enable_prefix_caching = enable_prefix_caching

        # Create the engine directly - this is the source of the blocking ray.get issue
        # but we need to handle it differently for multiple replicas
        self.engine = AsyncLLMEngine.from_engine_args(engine_args)
        self.bundle_indices = bundle_indices

    def ready(self) -> bool:
        """Check if engine is ready for inference."""
        return True

    async def generate(
        self,
        prompts: Any = None,
        sampling_params: SamplingParams | None = None,
        *,
        prompt_token_ids: list[int] | list[list[int]] | None = None,
        use_tqdm: bool = True,
        lora_request: Any = None,
        prompt_adapter_request: Any = None,
        guided_options_request: Any = None,
        timeout_seconds: float | None = None,
    ) -> RequestOutput | list[RequestOutput]:
        """Generate text with the same interface as vLLM.LLM.generate.

        This method mirrors the interface of vLLM.LLM.generate to provide seamless
        compatibility between sync and async engines.

        Args:
            prompts: String, TokensPrompt, or list of these. Input prompts for generation.
            sampling_params: SamplingParams object for controlling generation behavior.
            prompt_token_ids: Alternative to prompts - token IDs for generation.
            use_tqdm: Whether to show progress bar (not used in async engine).
            lora_request: LoRA request for adapter-based generation.
            guided_options_request: Guided decoding options.
            timeout_seconds: Timeout for generation in seconds.

        Returns:
            RequestOutput or list of RequestOutput: Generated outputs from vLLM.
        """
        if not _has_vllm:
            raise ImportError(
                "vllm is not installed. Please install it with `pip install vllm`."
            )

        from vllm import RequestOutput, SamplingParams, TokensPrompt

        # Track whether input was originally a single prompt
        single_prompt_input = False

        # Handle prompt_token_ids if provided
        if prompt_token_ids is not None:
            if prompts is not None:
                raise ValueError("Cannot specify both prompts and prompt_token_ids")

            # Convert token IDs to TokensPrompt objects
            if not prompt_token_ids:
                raise ValueError("prompt_token_ids cannot be empty")

            # Check if it's a list of lists or a single list
            if prompt_token_ids and isinstance(prompt_token_ids[0], list):
                # List of token ID lists
                prompts = [
                    TokensPrompt(prompt_token_ids=tokens) for tokens in prompt_token_ids
                ]
            else:
                # Single token ID list - cast to ensure type compatibility
                token_list = list(prompt_token_ids) if prompt_token_ids else []
                prompts = TokensPrompt(prompt_token_ids=token_list)
                single_prompt_input = True

        elif prompts is None:
            raise ValueError("Must specify either prompts or prompt_token_ids")
        else:
            # prompts was provided directly
            if not isinstance(prompts, (list, tuple)):
                single_prompt_input = True

        # Default sampling params if not provided
        if sampling_params is None:
            sampling_params = SamplingParams()

        async def _gen_one(prompt) -> RequestOutput:
            request_id = str(uuid.uuid4())
            final = None

            # Build kwargs for engine.generate
            gen_kwargs = {
                "prompt": prompt,
                "sampling_params": sampling_params,
                "request_id": request_id,
            }

            # Add optional parameters if provided
            if lora_request is not None:
                gen_kwargs["lora_request"] = lora_request
            if prompt_adapter_request is not None:
                gen_kwargs["prompt_adapter_request"] = prompt_adapter_request
            if guided_options_request is not None:
                gen_kwargs["guided_options_request"] = guided_options_request

            async for output in self.engine.generate(**gen_kwargs):
                if output.finished:
                    final = output
            assert final is not None
            return final

        async def _run_generation():
            if single_prompt_input:
                return await _gen_one(prompts)

            # List of prompts: run concurrently
            tasks = [asyncio.create_task(_gen_one(p)) for p in prompts]
            results = await asyncio.gather(*tasks)
            return results

        try:
            if timeout_seconds is not None and timeout_seconds > 0:
                return await asyncio.wait_for(
                    _run_generation(), timeout=timeout_seconds
                )
            else:
                return await _run_generation()
        except TimeoutError:
            # Best-effort cleanup
            try:
                abort_fn = getattr(self.engine, "abort", None)
                if callable(abort_fn):
                    # We can't easily track all request IDs, so this is best-effort
                    pass
            except Exception:
                pass
            raise TimeoutError(
                f"vLLM generation timed out after {timeout_seconds} seconds"
            )

    async def get_tokenizer(self):
        """Get the tokenizer from the engine."""
        return await self.engine.get_tokenizer()

    async def collective_rpc_v1(
        self,
        method: str,
        timeout: float | None = None,
        args: tuple = (),
        kwargs: dict | None = None,
    ):
        """Perform a collective RPC call to the given method (vLLM V1).

        Args:
            method (str): Method name to call.
            timeout (float | None): Timeout for the RPC call.
            args (tuple): Arguments to pass to the method.
            kwargs (dict | None): Keyword arguments to pass to the method.
        """
        from vllm import envs

        if envs and envs.VLLM_USE_V1:
            return await self.engine.collective_rpc(method, timeout, args, kwargs)
        else:
            return self.engine.engine.collective_rpc(method, timeout, args, kwargs)

    def collective_rpc_v0(
        self,
        method: str,
        timeout: float | None = None,
        args: tuple = (),
        kwargs: dict | None = None,
    ):
        """Perform a collective RPC call to the given method (vLLM V0).

        Args:
            method (str): Method name to call.
            timeout (float | None): Timeout for the RPC call.
            args (tuple): Arguments to pass to the method.
            kwargs (dict | None): Keyword arguments to pass to the method.
        """
        return self.engine.engine.collective_rpc(method, timeout, args, kwargs)

    def get_num_unfinished_requests(self) -> int:
        """Get the number of unfinished requests in the engine.

        Returns:
            int: Number of unfinished requests.
        """
        try:
            # Try to access the method directly if available
            if hasattr(self.engine, "get_num_unfinished_requests"):
                return self.engine.get_num_unfinished_requests()
            # Fallback to accessing through engine.engine for v0
            elif hasattr(self.engine, "engine") and hasattr(
                self.engine.engine, "get_num_unfinished_requests"
            ):
                return self.engine.engine.get_num_unfinished_requests()
            else:
                # If method not available, return 0 as fallback
                torchrl_logger.warning(
                    "get_num_unfinished_requests not available, returning 0"
                )
                return 0
        except Exception as e:
            torchrl_logger.warning(f"Error getting unfinished requests count: {e}")
            return 0

    def get_cache_usage(self) -> float:
        """Get the KV cache usage as a fraction between 0 and 1.

        Returns:
            float: Cache usage fraction (0.0 = empty, 1.0 = full).
        """
        try:
            # Try to get cache usage from the engine
            if hasattr(self.engine, "engine") and hasattr(
                self.engine.engine, "cache_config"
            ):
                # Access the LLM engine's cache information
                cache_config = self.engine.engine.cache_config
                if hasattr(cache_config, "cache_usage"):
                    return cache_config.cache_usage
                elif hasattr(self.engine.engine, "scheduler"):
                    # Try to get usage from the scheduler
                    scheduler = self.engine.engine.scheduler
                    if hasattr(scheduler, "get_num_free_gpu_blocks") and hasattr(
                        scheduler, "get_num_total_gpu_blocks"
                    ):
                        free_blocks = scheduler.get_num_free_gpu_blocks()
                        total_blocks = scheduler.get_num_total_gpu_blocks()
                        if total_blocks > 0:
                            return 1.0 - (free_blocks / total_blocks)
            # Fallback: return a random value for now (this should be replaced with actual metrics)
            torchrl_logger.warning(
                "Cache usage metrics not available, returning random value"
            )
            return (
                random.random() * 0.5
            )  # Return a value between 0 and 0.5 to simulate partial usage
        except Exception as e:
            torchrl_logger.warning(f"Error getting cache usage: {e}")
            return 0.0


def _gpus_per_replica(engine_args: AsyncEngineArgs) -> int:
    """Get the number of GPUs per replica for the given engine args."""
    return (
        engine_args.tensor_parallel_size
        * getattr(engine_args, "data_parallel_size", 1)  # Default to 1 if not present
        * getattr(
            engine_args, "pipeline_parallel_size", 1
        )  # Default to 1 if not present
    )


def _get_bundle_indices(placement_group, index: int, length: int) -> list[int]:
    """Get bundle indices for a placement group.

    Address https://github.com/ray-project/ray/issues/51117
    This function is used to get the bundle indices of a placement group
    and ensure that the bundles placed on the same node are grouped together.

    Args:
        placement_group: Ray placement group.
        index (int): Index of the current replica.
        length (int): Number of bundles per replica.

    Returns:
        list[int]: Bundle indices for this replica.
    """
    if ray is None:
        raise ImportError(
            "ray is not installed. Please install it with `pip install ray`."
        )

    pg_infos = ray.util.placement_group_table(placement_group)

    node_id_to_bundles = {}
    for bundle, node_id in pg_infos["bundles_to_node_id"].items():
        node_id_to_bundles.setdefault(node_id, []).append(bundle)

    sorted_bundle_indices = sum(node_id_to_bundles.values(), [])
    return sorted_bundle_indices[index * length : (index + 1) * length]


# Create Ray remote versions
if ray is not None and _has_vllm:
    _AsyncLLMEngineActor = ray.remote(num_cpus=0, num_gpus=0)(_AsyncLLMEngine)
else:
    _AsyncLLMEngineActor = None


class AsyncVLLM(RLvLLMEngine):
    """A service that manages multiple async vLLM engine actors for distributed inference.

    This is the main entry point for async vLLM inference in TorchRL. It manages multiple
    vLLM engine replicas running as Ray actors, providing load balancing, weight updates,
    and a unified interface for text generation.

    The service automatically handles Ray actor lifecycle management, GPU allocation through
    placement groups, and provides both synchronous and asynchronous generation interfaces
    that are compatible with the standard vLLM API.

    Args:
        engine_args (AsyncEngineArgs): Configuration for the vLLM engines.
        num_replicas (int, optional): Number of engine replicas to create. Defaults to 1.
        actor_class (optional): Custom Ray actor class. Defaults to the internal actor implementation.
        enable_prefix_caching (bool, optional): Whether to enable prefix caching. Defaults to False.

            .. warning::
                enable_prefix_caching is set to False by default, which is recommended if prompt log probs are needed.
                Set it to True if prompt log probs are not needed.
                See `this issue <https://github.com/vllm-project/vllm/issues/8268>`_ for more details.

    Example:
        >>> from torchrl.modules.llm.backends.vllm_async import AsyncVLLM
        >>> from vllm import SamplingParams
        >>>
        >>> # Simple usage - single GPU, single replica
        >>> service = AsyncVLLM.from_pretrained("Qwen/Qwen2.5-3B")
        >>>
        >>> # Advanced usage - multi-GPU tensor parallel with multiple replicas
        >>> service = AsyncVLLM.from_pretrained(
        ...     "Qwen/Qwen2.5-7B",
        ...     num_devices=2,  # Use 2 GPUs for tensor parallelism
        ...     num_replicas=2,  # Create 2 replicas for higher throughput
        ...     max_model_len=4096
        ... )
        >>>
        >>> # Generate text
        >>> sampling_params = SamplingParams(temperature=0.7, max_tokens=100)
        >>> result = service.generate("Hello, world!", sampling_params)
        >>> print(result.outputs[0].text)
        >>>
        >>> # Alternative: using AsyncEngineArgs directly for advanced configuration
        >>> from vllm import AsyncEngineArgs
        >>> engine_args = AsyncEngineArgs(
        ...     model="Qwen/Qwen2.5-3B",
        ...     tensor_parallel_size=2
        ... )
        >>> service = AsyncVLLM.launch(engine_args, num_replicas=2)

    .. note::
        **Architecture and Design**

        The AsyncVLLM service implements a distributed inference architecture with the following key components:

        1. **Ray Actor Management**: Each replica runs as a separate Ray actor with dedicated GPU resources.
           The service creates a placement group to ensure optimal GPU allocation and co-location of
           tensor-parallel workers on the same node when possible.

        2. **Load Balancing**: Generation requests are distributed across replicas using random selection
           by default, or can target specific replicas using the `actor_index` parameter.

        3. **Weight Synchronization**: The service supports weight updates across all replicas through
           NCCL communication groups, enabling integration with distributed training workflows.

        4. **Resource Management**: Automatic GPU allocation and cleanup through Ray placement groups,
           with proper shutdown procedures to prevent resource leaks.

        5. **API Compatibility**: Provides the same interface as vLLM's synchronous `LLM.generate()`
           method, making it a drop-in replacement for async workloads.

        **Ray Integration**

        The service leverages Ray's actor model for distributed execution. Each replica is an independent
        Ray actor that can be scheduled on different nodes. The service handles actor lifecycle,
        monitors readiness, and provides centralized access to all replicas.

        **Performance Considerations**

        - Prefix caching is enabled by default for better performance with repeated prompts
        - Tensor parallelism is supported for large models that don't fit on single GPUs
        - Multiple replicas allow concurrent processing of different requests
        - Native vLLM batching is used within each replica for optimal throughput

        **Error Handling**

        The service includes timeout support, graceful shutdown procedures, and best-effort
        request cleanup on failures. Ray's fault tolerance mechanisms provide additional
        resilience for long-running inference workloads.
    """

    def __init__(
        self,
        engine_args: AsyncEngineArgs,
        num_replicas: int = 1,
        actor_class=None,
        enable_prefix_caching: bool = False,
    ):
        if not _has_vllm:
            raise ImportError(
                "vllm is not installed. Please install it with `pip install vllm`."
            )
        if ray is None:
            raise ImportError(
                "ray is not installed. Please install it with `pip install ray`."
            )

        # Enable prefix caching by default for better performance
        engine_args.enable_prefix_caching = enable_prefix_caching

        self.engine_args = engine_args
        self.num_replicas = num_replicas
        self.actor_class = actor_class or _AsyncLLMEngineActor
        self.actors: list = []
        self._launched = False
        self._service_id = uuid.uuid4().hex[
            :8
        ]  # Unique suffix to avoid name collisions
        self._placement_group = None
        self._load_balancer = None

    def _launch(self):
        """Launch all actor replicas."""
        if self._launched:
            torchrl_logger.warning("AsyncVLLMEngineService already launched")
            return

        # Check if CUDA is available since vLLM requires GPU
        if not torch.cuda.is_available():
            raise RuntimeError(
                "AsyncVLLM requires CUDA but no GPU devices are available. "
                "Please run on a machine with GPU support."
            )

        torchrl_logger.info(
            f"Launching {self.num_replicas} async vLLM engine actors..."
        )

        # Create placement groups - one per replica to avoid conflicts
        self._placement_groups = []

        # Create actor replicas sequentially to avoid race conditions
        for i in range(self.num_replicas):
            torchrl_logger.info(
                f"Creating async actor replica {i + 1}/{self.num_replicas} ..."
            )

            # Create individual placement group for this replica
            bundles = [
                {"GPU": 1.0, "CPU": 1.0}
                for _ in range(self.engine_args.tensor_parallel_size)
            ]
            torchrl_logger.info(
                f"Creating placement group for replica {i + 1} with {len(bundles)} bundles"
            )

            placement_group_name = f"vllm-replica-{self._service_id}-{i}"
            pg = placement_group(bundles, strategy="PACK", name=placement_group_name)
            self._placement_groups.append(pg)
            torchrl_logger.info(f"Placement group {placement_group_name} created: {pg}")

            # Wait for placement group to be ready
            ray.get(pg.ready(), timeout=180)
            torchrl_logger.info(f"Placement group {placement_group_name} ready")

            # Calculate bundle indices for tensor parallelism
            bundle_indices = None
            if self.engine_args.tensor_parallel_size > 1:
                bundle_indices = list(range(self.engine_args.tensor_parallel_size))
            bundle_index = 0  # Always use first bundle since each replica has its own placement group

            scheduling_strategy = PlacementGroupSchedulingStrategy(
                placement_group=pg,
                placement_group_capture_child_tasks=True,
                placement_group_bundle_index=bundle_index,
            )

            actor = self.actor_class.options(
                name=f"async-vllm-replica-{self._service_id}-{i}",
                namespace="torchrl_vllm",
                scheduling_strategy=scheduling_strategy,
                num_gpus=0,
                num_cpus=0,
            ).remote(
                engine_args=self.engine_args,
                bundle_indices=bundle_indices,
                enable_prefix_caching=self.engine_args.enable_prefix_caching,
            )

            self.actors.append(actor)

        torchrl_logger.info("Waiting for actors to be ready")
        # Wait for this actor to be ready before creating the next one
        ready_futures = [actor.ready.remote() for actor in self.actors]
        try:
            ray.get(
                ready_futures, timeout=TIMEOUT_SECONDS
            )  # 5 minute timeout for engine initialization
            torchrl_logger.info("✅ Actors are ready")
        except Exception as e:
            torchrl_logger.error(
                f"❌ Failed to initialize actors within {TIMEOUT_SECONDS} seconds: {e}. You can increase the timeout by setting the TORCHRL_VLLM_TIMEOUT_SECONDS environment variable."
            )
            raise

        # Store the first placement group for backward compatibility
        self._placement_group = (
            self._placement_groups[0] if self._placement_groups else None
        )

        self._launched = True
        torchrl_logger.info(
            f"✅ Successfully launched {len(self.actors)} async vLLM engine actors"
        )

    @classmethod
    def launch(
        cls,
        engine_args: AsyncEngineArgs,
        num_replicas: int = 1,
    ) -> AsyncVLLM:
        """Launch a new AsyncVLLMEngineService.

        Args:
            engine_args (AsyncEngineArgs): Arguments for creating the AsyncLLMEngine instances.
            num_replicas (int): Number of actor replicas to create.

        Returns:
            AsyncVLLMEngineService: The launched service.
        """
        service = cls(engine_args, num_replicas)
        service._launch()
        # create a default load balancer with smart routing
        service.create_load_balancer()
        return service

    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        num_devices: int | None = None,
        num_replicas: int = 1,
        verbose: bool = True,
        compile: bool = True,
        **kwargs,
    ) -> AsyncVLLM:
        """Create an AsyncVLLM instance from a pretrained model.

        This is a convenience method that combines model loading and service launching
        in a single call, similar to how other ML libraries work.

        Args:
            model_name (str): The model name to pass to vLLM.
            num_devices (int, optional): Number of devices to use, per replica.
            num_replicas (int): Number of engine replicas to create.
            verbose (bool, optional): Whether to enable verbose logging with throughput statistics. Defaults to True.
            compile (bool, optional): Whether to enable model compilation for better performance. Defaults to True.
            **kwargs: Additional arguments passed to AsyncEngineArgs.

        Returns:
            AsyncVLLM: The launched async vLLM service.

        Example:
            >>> # Simple usage with defaults
            >>> service = AsyncVLLM.from_pretrained("Qwen/Qwen2.5-3B")
            >>>
            >>> # Multi-GPU tensor parallel with multiple replicas
            >>> service = AsyncVLLM.from_pretrained(
            ...     "Qwen/Qwen2.5-7B",
            ...     num_devices=2,
            ...     num_replicas=2,
            ...     max_model_len=4096
            ... )
            >>>
            >>> # Generate text
            >>> from vllm import SamplingParams
            >>> result = service.generate("Hello, world!", SamplingParams(max_tokens=50))
        """
        return make_async_vllm_engine(
            model_name=model_name,
            num_devices=num_devices,
            num_replicas=num_replicas,
            verbose=verbose,
            compile=compile,
            **kwargs,
        )

    def _is_batch(
        self, prompts: Any, prompt_token_ids: list[int] | list[list[int]] | None = None
    ) -> bool:
        """Check if the input represents a batch of prompts.

        Args:
            prompts: Input prompts that can be string, TokensPrompt, or list of these
            prompt_token_ids: Alternative token IDs input

        Returns:
            bool: True if this represents multiple prompts, False for single prompt
        """
        # If prompts is a list, we need to determine if it's a batch or a single prompt
        if isinstance(prompts, list):
            # Empty list is not a batch
            if len(prompts) == 0:
                return False

            # If all elements are integers, it's a single prompt represented as token IDs
            # We trust that if one is an int, then all are ints.
            if any(isinstance(item, int) for item in prompts):
                return False

            # If it contains strings, TokensPrompt objects, or other non-integer types,
            # it's a batch of prompts
            return True

        # If prompt_token_ids is provided and is a list of lists, it's a batch
        if prompt_token_ids is not None and isinstance(prompt_token_ids, list):
            if len(prompt_token_ids) > 0 and isinstance(prompt_token_ids[0], list):
                return True

        return False

    def _iterate(
        self, prompts: Any, prompt_token_ids: list[int] | list[list[int]] | None = None
    ):
        """Iterate over individual prompts in a batch.

        Args:
            prompts: Input prompts that can be string, TokensPrompt, or list of these
            prompt_token_ids: Alternative token IDs input

        Yields:
            tuple: (individual_prompt, individual_prompt_token_ids) for each item
        """
        if isinstance(prompts, list):
            # Check if this is actually a single prompt represented as token IDs
            if all(isinstance(item, int) for item in prompts):
                # This is a single prompt as token IDs, not a batch
                yield prompts, prompt_token_ids
                return

            # Handle list of prompts (actual batch)
            if prompt_token_ids is None:
                for prompt in prompts:
                    yield prompt, None
            elif (
                isinstance(prompt_token_ids, list)
                and len(prompt_token_ids) > 0
                and isinstance(prompt_token_ids[0], list)
            ):
                # Both prompts and prompt_token_ids are lists
                for prompt, token_ids in zip(prompts, prompt_token_ids):
                    yield prompt, token_ids
            else:
                # prompts is list, but prompt_token_ids is single list - replicate it
                for prompt in prompts:
                    yield prompt, prompt_token_ids
        else:
            # Single prompt case
            if (
                prompt_token_ids is not None
                and isinstance(prompt_token_ids, list)
                and len(prompt_token_ids) > 0
                and isinstance(prompt_token_ids[0], list)
            ):
                # Single prompt but multiple token_ids - replicate prompt
                for token_ids in prompt_token_ids:
                    yield prompts, token_ids
            else:
                # Single prompt, single (or no) token_ids
                yield prompts, prompt_token_ids

    def _generate_impl(
        self,
        prompt: Any,
        sampling_params: SamplingParams | None = None,
        *,
        prompt_token_ids: list[int] | None = None,
        use_tqdm: bool = True,
        lora_request: Any = None,
        prompt_adapter_request: Any = None,
        guided_options_request: Any = None,
        timeout_seconds: float | None = None,
        actor_index: int | None = None,
    ):
        """Generate text for a single prompt and return a Ray future.

        This is the internal implementation that returns a future instead of the result.
        Used for batched generation to enable parallel execution.

        Args:
            prompt: Single prompt (string, TokensPrompt, etc.)
            sampling_params: SamplingParams object for controlling generation behavior
            prompt_token_ids: Token IDs for a single prompt
            use_tqdm: Whether to show progress bar (not used in async engine)
            lora_request: LoRA request for adapter-based generation
            prompt_adapter_request: Prompt adapter request
            guided_options_request: Guided decoding options
            timeout_seconds: Timeout for generation in seconds
            actor_index: Specific actor to use (random if None)

        Returns:
            Ray ObjectRef: Future that will resolve to RequestOutput
        """
        if actor_index is None:
            if len(self.actors) == 1:
                actor = self.actors[0]
            else:
                if self._load_balancer is None:
                    raise RuntimeError(
                        "LoadBalancer is not created. Create a LoadBalancer using AsyncVLLM.create_load_balancer before calling generate."
                    )
                # Extract single prompt for prefix-aware routing
                single_prompt = self._extract_single_prompt_for_routing(
                    prompt, prompt_token_ids
                )
                actor_index = self._load_balancer.select_actor(prompt=single_prompt)
                actor = self.actors[actor_index]
        else:
            actor = self.actors[actor_index]

        return actor.generate.remote(
            prompt,
            sampling_params,
            prompt_token_ids=prompt_token_ids,
            use_tqdm=use_tqdm,
            lora_request=lora_request,
            prompt_adapter_request=prompt_adapter_request,
            guided_options_request=guided_options_request,
            timeout_seconds=timeout_seconds,
        )

    def generate(
        self,
        prompts: Any = None,
        sampling_params: SamplingParams | None = None,
        *,
        prompt_token_ids: list[int] | list[list[int]] | None = None,
        use_tqdm: bool = True,
        lora_request: Any = None,
        prompt_adapter_request: Any = None,
        guided_options_request: Any = None,
        timeout_seconds: float | None = None,
        actor_index: int | None = None,
    ) -> RequestOutput | list[RequestOutput]:
        """Generate text using one of the actors with vLLM.LLM.generate interface.

        This method provides the same interface as vLLM.LLM.generate for seamless
        compatibility between sync and async engines. It can be used to generate text
        within multiple threads / actors. If `actor_index` is not provided, the load balancer
        will be used to select the actor.

        `generate` is a blocking method, so it will wait for the generation to complete.

        Args:
            prompts (String, TokensPrompt, or list of these): Input prompts for generation.
            sampling_params (SamplingParams): SamplingParams object for controlling generation behavior.
            prompt_token_ids (list[int] | list[list[int]]): Alternative to prompts - token IDs for generation.
            use_tqdm (bool): Whether to show progress bar (not used in async engine).
            lora_request (Any): LoRA request for adapter-based generation.
            prompt_adapter_request (Any): Prompt adapter request.
            guided_options_request (Any): Guided decoding options.
            timeout_seconds (float | None): Timeout for generation in seconds.
            actor_index (int | None): Specific actor to use (random if None).

        Returns:
            RequestOutput | list[RequestOutput]: Generated outputs from vLLM.
        """
        # Check if this is a batch request
        if self._is_batch(prompts, prompt_token_ids):
            # Handle batched input by unbinding and sending individual requests
            futures = []
            for prompt, prompt_token_ids_i in self._iterate(prompts, prompt_token_ids):
                future = self._generate_impl(
                    prompt,
                    sampling_params,
                    prompt_token_ids=prompt_token_ids_i,
                    use_tqdm=use_tqdm,
                    lora_request=lora_request,
                    prompt_adapter_request=prompt_adapter_request,
                    guided_options_request=guided_options_request,
                    timeout_seconds=timeout_seconds,
                    actor_index=actor_index,
                )
                futures.append(future)

            # Collect all results
            results = ray.get(futures)
            return results
        else:
            # Single prompt case - call _generate_impt and get result directly
            future = self._generate_impl(
                prompts,
                sampling_params,
                prompt_token_ids=prompt_token_ids,
                use_tqdm=use_tqdm,
                lora_request=lora_request,
                prompt_adapter_request=prompt_adapter_request,
                guided_options_request=guided_options_request,
                timeout_seconds=timeout_seconds,
                actor_index=actor_index,
            )
            result = ray.get(future)
            return result

    def get_random_actor_index(self) -> int:
        """Get a random actor index."""
        return random.randint(0, len(self.actors) - 1)

    def _init_weight_update_group_internal(self, master_address: str, master_port: str):
        """Initialize NCCL weight update group across all actors.

        Args:
            master_address (str): Master address for distributed training.
            master_port (str): Master port for distributed training.

        Returns:
            list: Ray futures for initialization calls.
        """
        gpus_per_replica = _gpus_per_replica(self.engine_args)
        weight_sync_world_size = self.num_replicas * gpus_per_replica + 1
        torchrl_logger.info(
            f"AsyncVLLMEngineService requests weight update group for {self.num_replicas} actors "
            f"with {gpus_per_replica} GPUs per replica and {weight_sync_world_size} world size"
        )

        from vllm import envs

        refs = []
        for i, actor in enumerate(self.actors):
            rank_offset = 1 + i * gpus_per_replica
            if envs and envs.VLLM_USE_V1:
                actor_collective_rpc = actor.collective_rpc_v1
            else:
                actor_collective_rpc = actor.collective_rpc_v0

            refs.append(
                actor_collective_rpc.remote(
                    "init_weight_update_group",
                    args=(
                        master_address,
                        master_port,
                        rank_offset,
                        weight_sync_world_size,
                    ),
                )
            )
            torchrl_logger.info(
                f"AsyncVLLMEngineService args: {master_address=}, {master_port=}, "
                f"{rank_offset=}, {weight_sync_world_size=}"
            )
            torchrl_logger.info(
                f"AsyncVLLMEngineService requests weight update group for actor {i} "
                f"with rank_offset {rank_offset}"
            )
        return refs

    def collective_rpc(
        self,
        method: str,
        timeout: float | None = None,
        args: tuple = (),
        kwargs: dict | None = None,
    ) -> list[Any]:
        """Forward an RPC to all actors.

        Args:
            method (str): Method name to call.
            timeout (float | None): Timeout for the RPC call.
            args (tuple): Arguments to pass to the method.
            kwargs (dict | None): Keyword arguments to pass to the method.

        Returns:
            list[Any]: Ray futures for all RPC calls.
        """
        from vllm import envs

        futures = []
        for actor in self.actors:
            if envs and envs.VLLM_USE_V1:
                actor_collective_rpc = actor.collective_rpc_v1
            else:
                actor_collective_rpc = actor.collective_rpc_v0
            futures.append(actor_collective_rpc.remote(method, timeout, args, kwargs))
        return futures

    def shutdown(self):
        """Shutdown all actors and clean up resources."""
        torchrl_logger.info(
            f"Shutting down {len(self.actors)} async vLLM engine actors..."
        )

        # Kill all actors
        for i, actor in enumerate(self.actors):
            try:
                ray.kill(actor)
                torchrl_logger.info(f"Shutdown async actor {i + 1}/{len(self.actors)}")
            except Exception as e:
                torchrl_logger.warning(f"Error shutting down async actor {i + 1}: {e}")

        # Clear the actors list
        self.actors.clear()

        # Remove placement groups if any
        if hasattr(self, "_placement_groups") and self._placement_groups:
            for i, pg in enumerate(self._placement_groups):
                try:
                    remove_placement_group(pg)
                    torchrl_logger.info(
                        f"Removed placement group {i + 1}/{len(self._placement_groups)}"
                    )
                except Exception as e:
                    torchrl_logger.warning(
                        f"Error removing placement group {i + 1}: {e}"
                    )
            self._placement_groups = []

        # Remove legacy single placement group if any
        if self._placement_group is not None:
            remove_placement_group(self._placement_group)
        self._placement_group = None
        self._launched = False
        torchrl_logger.info("AsyncVLLMEngineService shutdown complete")

    # RLvLLMEngine interface implementation
    def get_tp_size(self) -> int:
        """Get the tensor parallel size."""
        return self.engine_args.tensor_parallel_size

    def get_model_metadata(self) -> dict[str, tuple[torch.dtype, torch.Size]]:
        """Get model parameter metadata.

        Note: This requires the model to be loaded. For now, we return an empty dict
        and expect the metadata to be provided externally during weight updates.
        """
        # TODO: Implement metadata extraction from loaded model
        # This would require accessing the model from one of the actors
        torchrl_logger.warning(
            "AsyncVLLM.get_model_metadata() not yet implemented - returning empty dict"
        )
        return {}

    def get_master_address(self) -> str:
        """Get the master address for weight synchronization."""
        return "localhost"  # Default for now

    def get_master_port(self) -> int:
        """Get the master port for weight synchronization."""
        # Cache the port like V1 does to ensure consistency
        if not hasattr(self, "_cached_master_port"):
            if _has_vllm:
                try:
                    from vllm.utils import get_open_port

                    self._cached_master_port = get_open_port()
                except ImportError:
                    self._cached_master_port = 29500  # Default port if import fails
            else:
                self._cached_master_port = 29500  # Default port
        return self._cached_master_port

    def init_weight_update_group(self) -> None:
        """Initialize the weight update communication group (RLvLLMEngine interface)."""
        if not self._launched:
            raise RuntimeError(
                "AsyncVLLM service must be launched before initializing weight update group"
            )

        master_address = self.get_master_address()
        master_port = self.get_master_port()

        # Call the internal method with the auto-detected parameters (like V1)
        refs = self._init_weight_update_group_internal(master_address, master_port)

        # CRITICAL: Initialize master NCCL group immediately (like V1) - don't wait for workers
        torchrl_logger.info("Setting up master NCCL group (rank 0)...")
        self._setup_nccl_master_group()

        # Now wait for workers to complete (like V1 does)
        if ray is not None:
            ray.get(refs)

        torchrl_logger.info("AsyncVLLM weight update group initialized")

    def update_weights(self, weights: Iterator[tuple[str, torch.Tensor]]) -> None:
        """Update model weights across all replicas using NCCL broadcast.

        Args:
            weights: Iterator yielding (parameter_name, tensor) tuples
        """
        if not self._launched:
            raise RuntimeError(
                "AsyncVLLM service must be launched before updating weights"
            )

        # Convert iterator to dict for easier handling
        weights_dict = dict(weights)

        if not weights_dict:
            torchrl_logger.warning("No weights provided for update")
            return

        torchrl_logger.info(
            f"Updating {len(weights_dict)} parameters across {len(self.actors)} replicas using NCCL broadcast"
        )

        self._update_weights_with_nccl_broadcast_simple(weights_dict)

        torchrl_logger.info("AsyncVLLM NCCL weight update completed")

    def _update_weights_with_nccl_broadcast_simple(
        self, weights_dict: dict[str, torch.Tensor]
    ) -> None:
        """Update weights using simple NCCL broadcast like V1.

        This approach follows the V1 pattern:
        1. Training process (master) broadcasts as rank 0
        2. All vLLM workers receive as ranks 1, 2, 3...
        3. Simple and reliable like the working V1 implementation

        Args:
            weights_dict: Dictionary of parameter names to weight tensors
        """
        import time

        if not hasattr(self, "_nccl_master_group") or self._nccl_master_group is None:
            raise RuntimeError(
                "NCCL master group not initialized. This is a bug in the setup process."
            )

        t0 = time.time()

        # Move all weights to cuda:0 (matching NCCL communicator device)
        gpu_weights = {}
        for name, weight in weights_dict.items():
            # Ensure weight is on cuda:0 (matching NCCL communicator)
            if weight.device != torch.device("cuda:0"):
                gpu_weights[name] = weight.to("cuda:0", non_blocking=True)
            else:
                gpu_weights[name] = weight

        # Use periodic-mono pattern: individual weight updates with immediate RPC->NCCL
        torchrl_logger.info(
            f"Updating {len(gpu_weights)} weights using periodic-mono pattern..."
        )

        updated_weights = 0
        with torch.cuda.device(0):  # Ensure we're on the correct CUDA device
            for name, weight in gpu_weights.items():
                # Convert dtype to string name (like periodic-mono)
                dtype_name = str(weight.dtype).split(".")[
                    -1
                ]  # "torch.bfloat16" -> "bfloat16"

                # Step 1: Send RPC to workers for this weight
                futures = self.collective_rpc(
                    "update_weight", args=(name, dtype_name, tuple(weight.shape))
                )

                # Step 2: Immediately broadcast this weight (like periodic-mono)
                self._nccl_master_group.broadcast(
                    weight, src=0, stream=torch.cuda.current_stream()
                )

                # Step 3: Wait for workers to complete this weight
                ray.get(futures)
                updated_weights += 1

        torch.cuda.synchronize()
        t2 = time.time()
        torchrl_logger.info(
            f"Successfully updated {updated_weights}/{len(gpu_weights)} weights in {t2 - t0:.3f}s"
        )

    def _setup_nccl_master_group(self) -> None:
        """Set up NCCL communication group for the master node (rank 0)."""
        # Calculate world size (should match what workers use)
        gpus_per_replica = _gpus_per_replica(self.engine_args)
        weight_sync_world_size = self.num_replicas * gpus_per_replica + 1

        master_address = self.get_master_address()
        master_port = self.get_master_port()

        torchrl_logger.info(
            f"Setting up NCCL master group: rank=0, world_size={weight_sync_world_size}, "
            f"address={master_address}:{master_port}"
        )

        # Ensure CUDA is available and initialized
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available for NCCL communication")

        # Set CUDA device before initializing NCCL
        torch.cuda.set_device(0)

        # Initialize master as rank 0 in the NCCL group (use synchronous version)
        self._nccl_master_group = stateless_init_process_group(
            master_address=master_address,
            master_port=str(master_port),
            rank=0,  # Master is always rank 0
            world_size=weight_sync_world_size,
            device=torch.device("cuda:0"),
        )

        torchrl_logger.info("NCCL master group initialized successfully")

    def get_num_unfinished_requests(
        self, actor_index: int | None = None
    ) -> int | list[int]:
        """Get the number of unfinished requests for one or all actors.

        Args:
            actor_index (int | None): Index of specific actor, or None for all actors.

        Returns:
            int | list[int]: Number of unfinished requests for the specified actor,
                           or list of counts for all actors if actor_index is None.
        """
        if not self._launched:
            raise RuntimeError(
                "AsyncVLLM service must be launched before getting request counts"
            )

        if actor_index is not None:
            if not (0 <= actor_index < len(self.actors)):
                raise IndexError(
                    f"Actor index {actor_index} out of range [0, {len(self.actors)})"
                )

            actor = self.actors[actor_index]
            return ray.get(actor.get_num_unfinished_requests.remote())
        else:
            # Get counts from all actors
            futures = [
                actor.get_num_unfinished_requests.remote() for actor in self.actors
            ]
            return ray.get(futures)

    def get_cache_usage(self, actor_index: int | None = None) -> float | list[float]:
        """Get the KV cache usage for one or all actors.

        Args:
            actor_index (int | None): Index of specific actor, or None for all actors.

        Returns:
            float | list[float]: Cache usage fraction for the specified actor,
                               or list of usage fractions for all actors if actor_index is None.
        """
        if not self._launched:
            raise RuntimeError(
                "AsyncVLLM service must be launched before getting cache usage"
            )

        if actor_index is not None:
            if not (0 <= actor_index < len(self.actors)):
                raise IndexError(
                    f"Actor index {actor_index} out of range [0, {len(self.actors)})"
                )

            actor = self.actors[actor_index]
            return ray.get(actor.get_cache_usage.remote())
        else:
            # Get usage from all actors
            futures = [actor.get_cache_usage.remote() for actor in self.actors]
            return ray.get(futures)

    def create_load_balancer(
        self,
        strategy: Literal["requests", "kv-cache"]
        | Sequence[Literal["prefix-aware", "requests", "kv-cache", "round-robin"]]
        | None = None,
        **kwargs,
    ) -> LoadBalancer:
        """Create a load balancer for this AsyncVLLM service.

        Args:
            strategy: Load balancing strategy or sequence of strategies in fallback order.
                Default: ["prefix-aware", "requests"] - tries cache-aware routing first,
                then load balancing. Single strategies: "requests", "kv-cache"
                Strategy sequences: ["prefix-aware", "requests", "round-robin"]
            **kwargs: Additional arguments passed to LoadBalancer constructor.

        Returns:
            LoadBalancer: Configured load balancer instance. This is stored in the AsyncVLLM instance.

        Examples:
            >>> service = AsyncVLLM.from_pretrained("Qwen/Qwen2.5-3B", num_replicas=3)

            >>> # Use smart defaults (prefix-aware -> requests)
            >>> lb = service.create_load_balancer()
            >>> selected_actor_index = lb.select_actor(prompt="Hello world")

            >>> # Simple single strategy
            >>> lb = service.create_load_balancer("requests")
            >>> selected_actor_index = lb.select_actor()

            >>> # Custom strategy hierarchy
            >>> lb = service.create_load_balancer(
            ...     ["prefix-aware", "kv-cache", "round-robin"],
            ...     prefix_length=16,
            ...     overload_threshold=2.0
            ... )
            >>> selected_actor_index = lb.select_actor(prompt="Hello world")
        """
        if not self._launched:
            raise RuntimeError(
                "AsyncVLLM service must be launched before creating load balancer"
            )

        load_balancer = LoadBalancer(self, strategy, **kwargs)
        self._load_balancer = load_balancer
        return load_balancer

    def _extract_single_prompt_for_routing(
        self,
        prompts: Any = None,
        prompt_token_ids: list[int] | list[list[int]] | None = None,
    ) -> str | list[int] | None:
        """Extract a single prompt for load balancer routing, if possible.

        Args:
            prompts: The prompts argument passed to generate().
            prompt_token_ids: The prompt_token_ids argument passed to generate().

        Returns:
            str | list[int] | None: Single prompt for routing, or None if multiple prompts.
        """
        try:
            # Handle prompt_token_ids first (takes precedence over prompts)
            if prompt_token_ids is not None:
                if isinstance(prompt_token_ids, list):
                    if len(prompt_token_ids) == 0:
                        return None  # Empty list
                    elif len(prompt_token_ids) == 1:
                        # Single prompt case - could be tokens directly or nested list
                        if isinstance(prompt_token_ids[0], int):
                            # Single token sequence: [token1, token2, ...]
                            return prompt_token_ids
                        elif isinstance(prompt_token_ids[0], list):
                            # Nested list with single prompt: [[token1, token2, ...]]
                            return prompt_token_ids[0]
                        else:
                            return None
                    else:
                        # Multiple prompts: [[tokens1...], [tokens2...], ...]
                        return None
                else:
                    # Not a list, invalid format
                    return None

            # Handle prompts argument
            if prompts is None:
                return None

            # Import vLLM types for proper checking
            try:
                pass
            except ImportError:
                # Fallback if imports fail
                type(None)
                type(None)

            # Single string prompt
            if isinstance(prompts, str):
                return prompts

            # TokensPrompt object
            elif hasattr(prompts, "prompt_token_ids"):  # TokensPrompt-like object
                return prompts.prompt_token_ids

            # TextPrompt object
            elif hasattr(prompts, "prompt"):  # TextPrompt-like object
                return prompts.prompt

            # List of prompts
            elif isinstance(prompts, (list, tuple)):
                if len(prompts) == 0:
                    return None  # Empty list
                elif len(prompts) == 1:
                    # Single prompt in list - recursively extract
                    return self._extract_single_prompt_for_routing(prompts[0], None)
                else:
                    # Multiple prompts - cannot do prefix routing
                    return None

            # Other types (shouldn't happen in normal usage)
            else:
                torchrl_logger.debug(
                    f"Unknown prompt type for routing: {type(prompts)}"
                )
                return None

        except Exception as e:
            torchrl_logger.debug(f"Error extracting single prompt for routing: {e}")
            return None


class LoadBalancer:
    """Load balancer for distributing requests across AsyncVLLM actors with strategy hierarchy.

    This class implements sophisticated load balancing with multiple strategies and intelligent
    fallback mechanisms. Strategies are tried in order until one succeeds, providing robust
    request routing even when some strategies fail.

    Args:
        actors: Either a single AsyncVLLM instance or a list of Ray actors.
        strategy: Single strategy or sequence of strategies in fallback order.
            Available strategies:

            - "prefix-aware": Route based on prompt prefix for cache locality
            - "requests": Select actor with fewest pending requests
            - "kv-cache": Select actor with lowest KV cache utilization
            - "round-robin": Simple round-robin distribution

            Default: ["prefix-aware", "requests"]

        prefix_length: Number of tokens/words to use for prefix routing (default: 8).
        overload_threshold: Multiplier for average load to consider actor overloaded (default: 1.5).

    Examples:
        >>> service = AsyncVLLM.from_pretrained("Qwen/Qwen2.5-3B", num_replicas=3)

        >>> # Simple strategy
        >>> lb = LoadBalancer(service, "requests")
        >>> actor_idx = lb.select_actor()

        >>> # Strategy hierarchy: try prefix-aware first, fall back to requests, then round-robin
        >>> lb = LoadBalancer(service, ["prefix-aware", "requests", "round-robin"])
        >>> actor_idx = lb.select_actor(prompt="Hello world")  # Uses prefix routing
        >>> actor_idx = lb.select_actor()  # Falls back to requests (no prompt)

        >>> # Custom configuration
        >>> lb = LoadBalancer(
        ...     service,
        ...     ["prefix-aware", "kv-cache"],
        ...     prefix_length=16,
        ...     overload_threshold=2.0
        ... )
    """

    def __init__(
        self,
        actors: list[Any] | AsyncVLLM,
        strategy: Literal["requests", "kv-cache"]
        | Sequence[Literal["prefix-aware", "requests", "kv-cache", "round-robin"]]
        | None = None,
        prefix_length: int = 8,
        overload_threshold: float = 1.5,
    ):
        if strategy is None:
            strategy = ["prefix-aware", "requests"]
        # Handle both AsyncVLLM instances and direct actor lists
        if hasattr(actors, "actors"):  # AsyncVLLM instance
            self.actors = actors.actors
            self.async_vllm = actors
        elif isinstance(actors, list):  # Direct list of actors
            self.actors = actors
            self.async_vllm = None
        else:
            raise ValueError(
                "actors must be either an AsyncVLLM instance or a list of actors"
            )

        if not self.actors:
            raise ValueError("No actors provided")

        # Handle both single strategy and strategy hierarchy
        if isinstance(strategy, str):
            self.strategies = [strategy]
        else:
            self.strategies = list(strategy)

        # Validate strategies
        valid_strategies = {"prefix-aware", "requests", "kv-cache", "round-robin"}
        for s in self.strategies:
            if s not in valid_strategies:
                raise ValueError(
                    f"Invalid strategy '{s}'. Must be one of {valid_strategies}"
                )

        if not self.strategies:
            raise ValueError("At least one strategy must be provided")

        self.strategy = self.strategies[
            0
        ]  # Primary strategy for backward compatibility
        self.prefix_length = prefix_length
        self.overload_threshold = overload_threshold
        self._round_robin_index = 0  # For round-robin fallback

    def select_actor(
        self,
        prompt: str | list[int] | None = None,
        request_context: dict[str, Any] | None = None,
    ) -> int:
        """Select the optimal actor index based on the configured strategy hierarchy.

        Args:
            prompt: The input prompt (string or token list) for prefix-aware routing.
            request_context: Additional context for routing decisions.

        Returns:
            int: Index of the selected actor in the actors list.

        Raises:
            RuntimeError: If unable to gather metrics from actors.
            ValueError: If no actors are available.
        """
        if not self.actors:
            raise ValueError("No actors available for selection")

        # Try each strategy in order until one succeeds
        for i, strategy in enumerate(self.strategies):
            try:
                torchrl_logger.debug(
                    f"Trying strategy {i + 1}/{len(self.strategies)}: {strategy}"
                )

                if strategy == "prefix-aware":
                    if prompt is not None:
                        return self._select_by_prefix_aware(prompt)
                    else:
                        torchrl_logger.debug(
                            "No prompt provided for prefix-aware routing, trying next strategy"
                        )
                        continue

                elif strategy == "requests":
                    return self._select_by_requests()

                elif strategy == "kv-cache":
                    return self._select_by_cache_usage()

                elif strategy == "round-robin":
                    return self._select_round_robin()

                else:
                    torchrl_logger.warning(
                        f"Unknown strategy: {strategy}, trying next strategy"
                    )
                    continue

            except Exception as e:
                torchrl_logger.warning(
                    f"Strategy '{strategy}' failed with error: {e}. "
                    f"Trying next strategy..."
                )
                continue

        # All strategies failed, final fallback to random
        torchrl_logger.warning(
            f"All strategies {self.strategies} failed. Falling back to random selection."
        )
        return random.randint(0, len(self.actors) - 1)

    def _select_by_requests(self) -> int:
        """Select actor with fewest pending requests."""
        if self.async_vllm is not None:
            # Use AsyncVLLM's built-in method to get request counts
            request_counts = self.async_vllm.get_num_unfinished_requests()
        else:
            # Query actors directly
            futures = [
                actor.get_num_unfinished_requests.remote() for actor in self.actors
            ]
            request_counts = ray.get(futures)

        # Find the actor with minimum pending requests
        min_requests = min(request_counts)
        min_indices = [
            i for i, count in enumerate(request_counts) if count == min_requests
        ]

        # If multiple actors have the same minimum count, choose randomly among them
        selected_index = random.choice(min_indices)

        torchrl_logger.debug(
            f"LoadBalancer (requests): Selected actor {selected_index} "
            f"with {min_requests} pending requests. "
            f"Request counts: {request_counts}"
        )

        return selected_index

    def _select_by_cache_usage(self) -> int:
        """Select actor with lowest KV cache utilization."""
        if self.async_vllm is not None:
            # Use AsyncVLLM's built-in method to get cache usage
            cache_usages = self.async_vllm.get_cache_usage()
        else:
            # Query actors directly
            futures = [actor.get_cache_usage.remote() for actor in self.actors]
            cache_usages = ray.get(futures)

        # Find the actor with minimum cache usage
        min_usage = min(cache_usages)
        min_indices = [
            i for i, usage in enumerate(cache_usages) if abs(usage - min_usage) < 1e-6
        ]

        # If multiple actors have similar cache usage, choose randomly among them
        selected_index = random.choice(min_indices)

        torchrl_logger.debug(
            f"LoadBalancer (kv-cache): Selected actor {selected_index} "
            f"with {min_usage:.3f} cache usage. "
            f"Cache usages: {[f'{u:.3f}' for u in cache_usages]}"
        )

        return selected_index

    def _select_by_prefix_aware(self, prompt: str | list[int]) -> int:
        """Select actor based on prompt prefix for cache locality.

        Args:
            prompt: Input prompt as string or token list.

        Returns:
            int: Selected actor index.

        Raises:
            ValueError: If prefix cannot be extracted.
        """
        try:
            # Extract prefix tokens
            prefix_tokens = self._extract_prefix_tokens(prompt)
            if not prefix_tokens:
                raise ValueError("Could not extract meaningful prefix tokens")

            # Create consistent hash from prefix
            prefix_hash = hash(tuple(prefix_tokens))
            preferred_actor = prefix_hash % len(self.actors)

            # Check if preferred actor is overloaded
            if self._is_actor_overloaded(preferred_actor):
                torchrl_logger.debug(
                    f"Preferred actor {preferred_actor} is overloaded "
                    f"(threshold: {self.overload_threshold}), falling back to load-based selection"
                )
                # Fall back to requests-based selection
                return self._select_by_requests()

            torchrl_logger.debug(
                f"LoadBalancer (prefix-aware): Selected actor {preferred_actor} "
                f"for prefix hash {prefix_hash} (tokens: {prefix_tokens[:4]}...)"
            )

            return preferred_actor

        except Exception as e:
            torchrl_logger.warning(f"Prefix-aware routing failed: {e}")
            raise

    def _select_round_robin(self) -> int:
        """Select actor using round-robin strategy."""
        selected = self._round_robin_index % len(self.actors)
        self._round_robin_index = (self._round_robin_index + 1) % len(self.actors)

        torchrl_logger.debug(f"LoadBalancer (round-robin): Selected actor {selected}")
        return selected

    def _extract_prefix_tokens(self, prompt: str | list[int]) -> list[int]:
        """Extract prefix tokens from prompt (string or token list).

        Args:
            prompt: Input prompt.

        Returns:
            list[int]: Prefix tokens (up to self.prefix_length).

        Raises:
            ValueError: If tokenization fails or prompt is invalid.
        """
        if isinstance(prompt, list):
            # Already tokenized
            if not prompt:
                raise ValueError("Empty token list provided")
            return prompt[: self.prefix_length]

        elif isinstance(prompt, str):
            # Need to tokenize - this requires access to tokenizer
            if not prompt.strip():
                raise ValueError("Empty or whitespace-only string provided")

            # Try to get tokenizer from AsyncVLLM instance
            if self.async_vllm is not None:
                try:
                    # This is a simplistic approach - in practice you'd want to cache the tokenizer
                    # For now, use a simple heuristic based on string content
                    return self._simple_string_hash(prompt)
                except Exception as e:
                    torchrl_logger.warning(f"Could not tokenize string: {e}")
                    return self._simple_string_hash(prompt)
            else:
                # Fall back to simple string hashing
                return self._simple_string_hash(prompt)
        else:
            raise ValueError(f"Unsupported prompt type: {type(prompt)}")

    def _simple_string_hash(self, text: str) -> list[int]:
        """Create pseudo-tokens from string for prefix routing.

        This is a fallback when proper tokenization isn't available.
        """
        # Use words as pseudo-tokens, limited to prefix_length
        words = text.strip().split()[: self.prefix_length]
        if not words:
            raise ValueError("No words found in text")

        # Convert words to integers using hash
        pseudo_tokens = [
            abs(hash(word)) % 50000 for word in words
        ]  # Simulate vocab size
        return pseudo_tokens

    def _is_actor_overloaded(self, actor_index: int) -> bool:
        """Check if an actor is overloaded compared to average load.

        Args:
            actor_index: Index of actor to check.

        Returns:
            bool: True if actor is overloaded.
        """
        try:
            if self.async_vllm is not None:
                request_counts = self.async_vllm.get_num_unfinished_requests()
            else:
                futures = [
                    actor.get_num_unfinished_requests.remote() for actor in self.actors
                ]
                request_counts = ray.get(futures)

            if not request_counts:
                return False

            avg_requests = sum(request_counts) / len(request_counts)
            actor_requests = request_counts[actor_index]

            is_overloaded = actor_requests > avg_requests * self.overload_threshold

            torchrl_logger.debug(
                f"Actor {actor_index}: {actor_requests} requests, "
                f"avg: {avg_requests:.1f}, threshold: {avg_requests * self.overload_threshold:.1f}, "
                f"overloaded: {is_overloaded}"
            )

            return is_overloaded

        except Exception as e:
            torchrl_logger.warning(f"Could not check actor load: {e}")
            return False  # Assume not overloaded if we can't check

    def get_stats(self) -> dict[str, Any]:
        """Get current load balancing statistics for all actors.

        Returns:
            dict: Statistics including request counts and cache usage for all actors.
        """
        stats = {
            "strategies": self.strategies,
            "primary_strategy": self.strategy,  # For backward compatibility
            "num_actors": len(self.actors),
            "prefix_length": self.prefix_length,
            "overload_threshold": self.overload_threshold,
            "round_robin_index": self._round_robin_index,
            "actor_stats": [],
        }

        try:
            if self.async_vllm is not None:
                request_counts = self.async_vllm.get_num_unfinished_requests()
                cache_usages = self.async_vllm.get_cache_usage()
            else:
                request_futures = [
                    actor.get_num_unfinished_requests.remote() for actor in self.actors
                ]
                cache_futures = [
                    actor.get_cache_usage.remote() for actor in self.actors
                ]
                request_counts = ray.get(request_futures)
                cache_usages = ray.get(cache_futures)

            for i, (requests, cache_usage) in enumerate(
                zip(request_counts, cache_usages)
            ):
                stats["actor_stats"].append(
                    {
                        "actor_index": i,
                        "pending_requests": requests,
                        "cache_usage": cache_usage,
                    }
                )

        except Exception as e:
            torchrl_logger.warning(f"Error gathering load balancer stats: {e}")
            stats["error"] = str(e)

        return stats


def make_async_vllm_engine(
    model_name: str,
    num_devices: int | None = None,
    num_replicas: int = 1,
    verbose: bool = True,
    compile: bool = True,
    **kwargs,
) -> AsyncVLLM:
    """Create an async vLLM engine service.

    Args:
        model_name (str): The model name to pass to vLLM.
        num_devices (int, optional): Number of devices to use, per replica.
        num_replicas (int): Number of engine replicas to create.
        verbose (bool, optional): Whether to enable verbose logging with throughput statistics. Defaults to True.
        compile (bool, optional): Whether to enable model compilation for better performance. Defaults to True.
        **kwargs: Additional arguments passed to AsyncEngineArgs.

    Returns:
        AsyncVLLM: The launched engine service.

    Raises:
        RuntimeError: If no CUDA devices are available.
        ValueError: If invalid device configuration is provided.

    Example:
        >>> # Create a single-GPU async engine
        >>> service = make_async_vllm_engine("Qwen/Qwen2.5-3B")
        >>>
        >>> # Create a 2-GPU tensor parallel async engine with 2 replicas
        >>> service = make_async_vllm_engine("Qwen/Qwen2.5-3B", num_devices=2, num_replicas=2)
        >>> # Generate text
        >>> result = service.generate("Hello, world!", sampling_params)
    """
    if not _has_vllm:
        raise ImportError(
            "vllm is not installed. Please install it with `pip install vllm`."
        )

    from vllm import AsyncEngineArgs

    # Check if CUDA is available since vLLM requires GPU
    if not torch.cuda.is_available():
        raise RuntimeError(
            "AsyncVLLM requires CUDA but no GPU devices are available. "
            "Please run on a machine with GPU support."
        )

    # Handle device specification
    if num_devices is None:
        num_devices = 1

    # Configure verbose logging if requested
    if verbose:
        import logging

        # Enable vLLM's throughput logging by setting the appropriate log level
        logging.getLogger("vllm.engine.metrics").setLevel(logging.INFO)
        logging.getLogger("vllm").setLevel(logging.INFO)

        # vLLM logs throughput stats at INFO level every few seconds
        # The stats include: prompt throughput, generation throughput, running/pending requests, GPU KV cache usage
        torchrl_logger.info(
            "Enabled verbose vLLM logging - throughput statistics will be displayed"
        )

    # Create engine args
    kwargs.setdefault("distributed_executor_backend", "ray")
    # Don't explicitly set enable_prefix_caching to avoid conflicts
    kwargs.setdefault("enable_prefix_caching", True)

    # Set compilation flag - this controls whether vLLM will compile the model for better performance
    # Disabled by default in GRPO since it can cause issues during training
    if "compilation_config" not in kwargs:
        if compile:
            kwargs["compilation_config"] = {"enabled": True}
        else:
            kwargs["compilation_config"] = {"enabled": False}

    engine_args = AsyncEngineArgs(
        model=model_name,
        tensor_parallel_size=num_devices,
        worker_cls="torchrl.modules.llm.backends.vllm.vllm_async._AsyncvLLMWorker",
        **kwargs,
    )

    return AsyncVLLM.launch(engine_args, num_replicas)
