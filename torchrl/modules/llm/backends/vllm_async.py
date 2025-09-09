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
from typing import Any

import torch
from torchrl._utils import logger as torchrl_logger

try:
    import ray
    from ray.util.placement_group import placement_group, remove_placement_group
    from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
except ImportError:
    ray = None
    placement_group = None
    remove_placement_group = None
    PlacementGroupSchedulingStrategy = None

try:
    import vllm
    from vllm import (
        AsyncEngineArgs,
        AsyncLLMEngine,
        envs,
        RequestOutput,
        SamplingParams,
        TokensPrompt,
    )
    from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
    from vllm.distributed.utils import StatelessProcessGroup
    from vllm.utils import get_open_port
    from vllm.worker.worker import Worker

    _has_vllm = True
except ImportError:
    vllm = None
    AsyncEngineArgs = Any
    AsyncLLMEngine = Any
    RequestOutput = Any
    SamplingParams = Any
    TokensPrompt = Any
    envs = None
    get_open_port = None
    Worker = object  # Use object as base class when vLLM is not available
    PyNcclCommunicator = None
    StatelessProcessGroup = None

    _has_vllm = False


def stateless_init_process_group_async(
    master_address: str | None,
    master_port: str | None,
    rank: int,
    world_size: int,
    device,
):
    """Initializes a stateless process group for distributed communication (async version).

    Creates a `StatelessProcessGroup` instance without relying on the global
    process group in `torch.distributed`. This approach is recommended for
    initializing data-plane communication (NCCL) between external processes
    (e.g., training processes) and vLLM workers.

    Args:
        master_address (str | None): The address of the master node. Defaults to "localhost" if not specified.
        master_port (str | None): The port used by the master node. Automatically assigns an open port if not specified.
        rank (int): The rank of the current process.
        world_size (int): The total number of processes in the distributed group.
        device: The device to use for communication.

    Returns:
        PyNcclCommunicator: A PyNcclCommunicator instance initialized with the created StatelessProcessGroup.
    """
    if StatelessProcessGroup is None or PyNcclCommunicator is None:
        raise ImportError(
            "vllm is not installed. Please install it with `pip install vllm`."
        )

    if master_address is None:
        master_address = "localhost"
    if master_port is None:
        master_port = get_open_port()

    master_port_int = int(master_port) if master_port else 0
    pg = StatelessProcessGroup.create(
        host=master_address, port=master_port_int, rank=rank, world_size=world_size
    )
    pynccl = PyNcclCommunicator(pg, device=device)
    return pynccl


class _AsyncvLLMWorker(Worker):
    """Async vLLM worker for Ray with weight update capabilities.

    This worker extends the base vLLM Worker to support async operations
    and weight updates via NCCL communication groups.
    """

    def __init__(self, *args, **kwargs):
        if not _has_vllm:
            raise ImportError(
                "vllm is not installed. Please install it with `pip install vllm`."
            )

        torchrl_logger.info(f"=> in {type(self).__name__}.__init__")
        torchrl_logger.info(f"visible devices {os.getenv('CUDA_VISIBLE_DEVICES')}")
        torchrl_logger.info(f"device count {torch.cuda.device_count()}")
        super().__init__(*args, **kwargs)
        self.model_update_group = None

    def init_weight_update_group(
        self, master_address: str, master_port: str, rank_offset: int, world_size: int
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

        self.model_update_group = stateless_init_process_group_async(
            master_address, master_port, rank, world_size, self.device
        )

        torchrl_logger.info(f"{type(self).__name__}.init_weight_update_group success")

    def update_weight_broadcast(self, name: str, dtype: torch.dtype, shape: torch.Size):
        """Update weight via broadcast from rank 0.

        Args:
            name (str): Parameter name.
            dtype (torch.dtype): Parameter dtype.
            shape (torch.Size): Parameter shape.
        """
        if self.model_update_group is None:
            raise RuntimeError("Weight update group not initialized")

        weight = torch.empty(shape, dtype=dtype, device="cuda")
        self.model_update_group.broadcast(
            weight, src=0, stream=torch.cuda.current_stream()
        )

        self.model_runner.model.load_weights(weights=[(name, weight)])
        del weight

    def update_weight(self, name: str, weight: torch.Tensor):
        """Update weight directly.

        Args:
            name (str): Parameter name.
            weight (torch.Tensor): Parameter weight.
        """
        self.model_runner.model.load_weights(weights=[(name, weight)])
        del weight


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

        # Configure engine for TorchRL use
        if engine_args.distributed_executor_backend is not None:
            old_backend = engine_args.distributed_executor_backend
            torchrl_logger.warning(
                f"Overriding distributed_executor_backend from {old_backend} to ray"
            )

        worker_cls = "torchrl.modules.llm.backends.vllm_async._AsyncvLLMWorker"
        if engine_args.worker_cls != "auto":
            old_worker_cls = engine_args.worker_cls
            torchrl_logger.warning(
                f"Overriding worker_cls from {old_worker_cls} to {worker_cls}"
            )

        if bundle_indices is not None:
            os.environ["VLLM_RAY_BUNDLE_INDICES"] = ",".join(map(str, bundle_indices))

        engine_args.worker_cls = worker_cls
        engine_args.distributed_executor_backend = "ray"

        engine_args.enable_prefix_caching = enable_prefix_caching

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
            prompt_adapter_request: Prompt adapter request.
            guided_options_request: Guided decoding options.
            timeout_seconds: Timeout for generation in seconds.

        Returns:
            RequestOutput or list of RequestOutput: Generated outputs from vLLM.
        """
        if not _has_vllm:
            raise ImportError(
                "vllm is not installed. Please install it with `pip install vllm`."
            )

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


class AsyncVLLM:
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

    def _launch(self):
        """Launch all actor replicas."""
        if self._launched:
            torchrl_logger.warning("AsyncVLLMEngineService already launched")
            return

        torchrl_logger.info(
            f"Launching {self.num_replicas} async vLLM engine actors..."
        )

        # Create a placement group to ensure optimal GPU allocation
        bundles = [
            {"GPU": 1.0, "CPU": 1.0}
            for _ in range(self.num_replicas * self.engine_args.tensor_parallel_size)
        ]
        torchrl_logger.info(f"Creating placement group with {len(bundles)} bundles")
        self._placement_group = placement_group(bundles, strategy="PACK")
        torchrl_logger.info(f"Placement group created: {self._placement_group}")

        # Avoid indefinite hang if resources are not available
        ray.get(self._placement_group.ready(), timeout=180)
        torchrl_logger.info(f"Placement group ready: {self._placement_group}")

        # Create actor replicas
        for i in range(self.num_replicas):
            torchrl_logger.info(
                f"Creating async actor replica {i + 1}/{self.num_replicas} ..."
            )
            bundle_indices = None
            if self.engine_args.tensor_parallel_size > 1:
                bundle_indices = _get_bundle_indices(
                    self._placement_group, i, self.engine_args.tensor_parallel_size
                )

            scheduling_strategy = PlacementGroupSchedulingStrategy(
                placement_group=self._placement_group,
                placement_group_capture_child_tasks=True,
                placement_group_bundle_index=bundle_indices[0] if bundle_indices else i,
            )

            actor = self.actor_class.options(
                name=f"async-vllm-replica-{self._service_id}-{i}",
                namespace="torchrl_vllm",
                scheduling_strategy=scheduling_strategy,
                num_gpus=0,
                num_cpus=0,
            ).remote(
                engine_args=self.engine_args,
                bundle_indices=bundle_indices or [i],
                enable_prefix_caching=self.engine_args.enable_prefix_caching,
            )

            self.actors.append(actor)

        # Wait for all actors to be ready
        ready_futures = [actor.ready.remote() for actor in self.actors]
        ray.get(ready_futures)

        self._launched = True
        torchrl_logger.info(
            f"Successfully launched {len(self.actors)} async vLLM engine actors"
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
        return service

    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        devices: list[torch.device | int] | None = None,
        num_devices: int | None = None,
        num_replicas: int = 1,
        **kwargs,
    ) -> AsyncVLLM:
        """Create an AsyncVLLM instance from a pretrained model.

        This is a convenience method that combines model loading and service launching
        in a single call, similar to how other ML libraries work.

        Args:
            model_name (str): The model name to pass to vLLM.
            devices (list[torch.device | int], optional): List of devices to use. Exclusive with num_devices.
            num_devices (int, optional): Number of devices to use. Exclusive with devices.
            num_replicas (int): Number of engine replicas to create.
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
            devices=devices,
            num_devices=num_devices,
            num_replicas=num_replicas,
            **kwargs,
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
        compatibility between sync and async engines.

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
            RequestOutput or list of RequestOutput: Generated outputs from vLLM.
        """
        if actor_index is None:
            actor = random.choice(self.actors)
        else:
            actor = self.actors[actor_index]
        list_convert = False
        if not isinstance(prompts, list):
            list_convert = True
            prompts = [prompts]
        # prompt_token_ids is a list of lists
        if prompt_token_ids is not None:
            if not isinstance(prompt_token_ids, list):
                raise ValueError("prompt_token_ids must be a list of lists")
            if not len(prompt_token_ids):
                raise ValueError("prompt_token_ids must not be empty")
            if not isinstance(prompt_token_ids[0], list):
                list_convert = True
                prompt_token_ids = [prompt_token_ids]
        results = ray.get(
            [
                actor.generate.remote(
                    prompt,
                    sampling_params,
                    prompt_token_ids=prompt_token_ids_i,
                    use_tqdm=use_tqdm,
                    lora_request=lora_request,
                    prompt_adapter_request=prompt_adapter_request,
                    guided_options_request=guided_options_request,
                    timeout_seconds=timeout_seconds,
                )
                for prompt, prompt_token_ids_i in zip(prompts, prompt_token_ids)
            ]
        )
        if list_convert:
            return results[0]
        else:
            return results

    def get_random_actor_index(self) -> int:
        """Get a random actor index."""
        return random.randint(0, len(self.actors) - 1)

    def init_weight_update_group(self, master_address: str, master_port: str):
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

        # Remove placement group if any
        if self._placement_group is not None:
            remove_placement_group(self._placement_group)
        self._placement_group = None
        self._launched = False
        torchrl_logger.info("AsyncVLLMEngineService shutdown complete")


def make_async_vllm_engine(
    model_name: str,
    devices: list[torch.device | int] | None = None,
    num_devices: int | None = None,
    num_replicas: int = 1,
    **kwargs,
) -> AsyncVLLM:
    """Create an async vLLM engine service.

    Args:
        model_name (str): The model name to pass to vLLM.
        devices (list[torch.device | int], optional): List of devices to use. Exclusive with num_devices.
        num_devices (int, optional): Number of devices to use. Exclusive with devices.
        num_replicas (int): Number of engine replicas to create.
        **kwargs: Additional arguments passed to AsyncEngineArgs.

    Returns:
        AsyncVLLMEngineService: The launched engine service.

    Example:
        >>> # Create a 2-GPU tensor parallel async engine with 2 replicas
        >>> service = make_async_vllm_engine("Qwen/Qwen2.5-3B", num_devices=2, num_replicas=2)
        >>> # Generate text
        >>> result = service.generate_text("Hello, world!", sampling_params)
    """
    if not _has_vllm:
        raise ImportError(
            "vllm is not installed. Please install it with `pip install vllm`."
        )

    # Handle device specification
    if num_devices is not None and devices is not None:
        raise ValueError("Cannot specify both num_devices and devices")
    if num_devices is not None:
        devices = None
    elif devices is None:
        devices = [0]  # Default to first GPU
        num_devices = 1
    elif len(devices) > 1:
        num_devices = len(devices)

    # Create engine args
    engine_args = AsyncEngineArgs(
        model=model_name,
        tensor_parallel_size=num_devices,
        distributed_executor_backend="ray",
        worker_cls="torchrl.modules.llm.backends.vllm_async._AsyncvLLMWorker",
        enable_prefix_caching=True,
        **kwargs,
    )

    return AsyncVLLM.launch(engine_args, num_replicas)
