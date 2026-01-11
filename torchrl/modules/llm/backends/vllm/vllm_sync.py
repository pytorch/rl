# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Synchronous vLLM backend for TorchRL.

From https://docs.vllm.ai/en/v0.7.0/getting_started/examples/rlhf.html
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import nullcontext

import torch
from torchrl._utils import logger as torchrl_logger
from torchrl.modules.llm.utils import _cuda_visible_devices

from .base import RLvLLMEngine
from .vllm_utils import stateless_init_process_group

try:
    from vllm import LLM
    from vllm.utils import get_open_port
    from vllm.worker.worker import Worker

    _has_vllm = True
except ImportError:

    class LLM:
        """Placeholder for LLM class when vLLM is not installed."""

    class Worker:
        """Placeholder for Worker class when vLLM is not installed."""

    def get_open_port():
        """Placeholder for get_open_port function when vLLM is not installed."""
        raise ImportError(
            "vllm is not installed. Please install it with `pip install vllm`."
        )

    _has_vllm = False


class _vLLMWorker(Worker):
    """Private vLLM worker for Ray.

    vLLMParameterServer will always take rank 0 in the stateless process group
    initialized by this worker. And the tp ranks associated with the LLM class
    will be in the range [1, tp_size].
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

    def init_weight_update_group(
        self, master_address, master_port, rank_offset, world_size
    ):
        from vllm.distributed.parallel_state import get_world_group

        torchrl_logger.info(f"=> in {type(self).__name__}.init_weight_update_group")

        # Get the local rank within the tensor parallel group
        tp_group = get_world_group()
        local_rank = tp_group.rank
        torchrl_logger.info(f"Local rank in tensor parallel group: {local_rank}")

        # Calculate the global rank for weight update group
        # rank_offset is 1, so ranks will be [1, 2] for tp_size=2
        rank = local_rank + rank_offset
        torchrl_logger.info(
            f"Initializing {type(self).__name__} weight update group with "
            f"{master_address=}, {master_port=}, {rank=}, {world_size=}, device={self.device}"
        )

        self.model_update_group = stateless_init_process_group(
            master_address,
            master_port,
            rank,
            world_size,
            self.device,
        )

        torchrl_logger.info(f"{type(self).__name__}.init_weight_update_group success")

    def update_weight_broadcast(self, name, dtype, shape):
        weight = torch.empty(shape, dtype=dtype, device="cuda")
        self.model_update_group.broadcast(
            weight, src=0, stream=torch.cuda.current_stream()
        )

        self.model_runner.model.load_weights(weights=[(name, weight)])
        del weight

    def update_weight(self, name, weight):
        self.model_runner.model.load_weights(weights=[(name, weight)])
        del weight

    def check_weights_changed(self):
        """Check if the weights are updated to 0."""
        # TODO: This is a test and should be treated as such
        weights_updated = True
        for p in self.model_runner.model.parameters():
            weights_updated = weights_updated and torch.allclose(p, torch.zeros_like(p))
        return weights_updated


class _LLMOnDevice(LLM):
    """Private wrapper around `vllm.LLM` to control its placement devices."""

    def __init__(self, *args, bundle_indices: list | None = None, **kwargs):
        if not _has_vllm:
            raise ImportError(
                "vllm is not installed. Please install it with `pip install vllm`."
            )

        # Stop Ray from manipulating CUDA_VISIBLE_DEVICES at the top-level
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)

        # Configure GPU utilization for Ray workers
        if bundle_indices is not None:
            os.environ[
                "VLLM_RAY_PER_WORKER_GPUS"
            ] = "0.4"  # Allow multiple workers per GPU
            os.environ["VLLM_RAY_BUNDLE_INDICES"] = ",".join(map(str, bundle_indices))
            torchrl_logger.info(
                f"Initializing LLM with bundle_indices={bundle_indices}"
            )

        self.args = args
        self.kwargs = kwargs

    def initialize(self):
        # Let vLLM handle device placement
        super().__init__(*self.args, **self.kwargs)
        return True


class RayLLMWorker(RLvLLMEngine):
    """A wrapper for Ray-based vLLM workers that implements the RLvLLMEngine interface.

    This class wraps a Ray actor handle for a vLLM worker and provides the
    standardized interface for weight updates and configuration access.
    """

    def __init__(self, ray_actor, tensor_parallel_size: int, model_name: str):
        self.ray_actor = ray_actor
        self._tensor_parallel_size = tensor_parallel_size
        self._model_name = model_name
        self._master_address = None
        self._master_port = None

    def get_tp_size(self) -> int:
        """Get the tensor parallel size."""
        return self._tensor_parallel_size

    def get_model_metadata(self) -> dict[str, tuple[torch.dtype, torch.Size]]:
        """Get model parameter metadata.

        For Ray workers, this requires loading the model to inspect parameters.
        Currently returns empty dict - should be implemented when needed.
        """
        # TODO: Implement metadata extraction from Ray worker
        torchrl_logger.warning(
            "RayLLMWorker.get_model_metadata() not implemented - returning empty dict"
        )
        return {}

    def get_master_address(self) -> str:
        """Get the master address for weight synchronization."""
        if self._master_address is None:
            self._master_address = "localhost"
        return self._master_address

    def get_master_port(self) -> int:
        """Get the master port for weight synchronization."""
        if self._master_port is None:
            self._master_port = get_open_port() if callable(get_open_port) else 29500
        return self._master_port

    def init_weight_update_group(self) -> None:
        """Initialize the weight update communication group."""
        weight_sync_world_size = self._tensor_parallel_size + 1

        try:
            import ray

            # Initialize weight update group on the Ray worker
            ray.get(
                self.ray_actor.collective_rpc.remote(
                    "init_weight_update_group",
                    args=(
                        self.get_master_address(),
                        self.get_master_port(),
                        1,
                        weight_sync_world_size,
                    ),
                )
            )
            torchrl_logger.info("Ray worker weight update group initialized")
        except ImportError:
            raise ImportError(
                "Ray not available for weight update group initialization"
            )

    def update_weights(self, weights: Iterator[tuple[str, torch.Tensor]]) -> None:
        """Update model weights via the Ray worker.

        Args:
            weights: Iterator yielding (parameter_name, tensor) tuples
        """
        try:
            import ray

            # Convert iterator to list for Ray serialization
            weights_list = list(weights)

            if not weights_list:
                torchrl_logger.warning("No weights provided for update")
                return

            torchrl_logger.info(
                f"Updating {len(weights_list)} parameters on Ray worker"
            )

            # Send weights to the Ray worker
            remotes = []
            for name, weight in weights_list:
                remotes.append(
                    self.ray_actor.collective_rpc.remote(
                        "update_weight", args=(name, weight.to("cuda"))
                    )
                )

            ray.get(remotes)
            torchrl_logger.info("Ray worker weight update completed")

        except ImportError:
            raise ImportError("Ray not available for weight updates")

    # Delegate generation methods to the Ray actor
    def generate(self, *args, **kwargs):
        """Generate text using the Ray worker."""
        try:
            import ray

            return ray.get(self.ray_actor.generate.remote(*args, **kwargs))
        except ImportError:
            raise ImportError("Ray not available for generation")


class LocalLLMWrapper(RLvLLMEngine):
    """A wrapper for local vLLM.LLM instances that implements the RLvLLMEngine interface.

    This wrapper provides the standardized interface for local vLLM instances,
    though weight updates are not applicable since the model is in the same process.
    """

    def __init__(self, llm_instance, tensor_parallel_size: int, model_name: str):
        self.llm_instance = llm_instance
        self._tensor_parallel_size = tensor_parallel_size
        self._model_name = model_name
        self._master_address = None
        self._master_port = None

    def get_tp_size(self) -> int:
        """Get the tensor parallel size."""
        return self._tensor_parallel_size

    def get_model_metadata(self) -> dict[str, tuple[torch.dtype, torch.Size]]:
        """Get model parameter metadata.

        For local LLM instances, this would require accessing the model directly.
        Currently returns empty dict.
        """
        # TODO: Implement metadata extraction from local LLM
        torchrl_logger.warning(
            "LocalLLMWrapper.get_model_metadata() not implemented - returning empty dict"
        )
        return {}

    def get_master_address(self) -> str:
        """Get the master address for weight synchronization."""
        if self._master_address is None:
            self._master_address = "localhost"
        return self._master_address

    def get_master_port(self) -> int:
        """Get the master port for weight synchronization."""
        if self._master_port is None:
            self._master_port = get_open_port() if callable(get_open_port) else 29500
        return self._master_port

    def init_weight_update_group(self) -> None:
        """Initialize the weight update communication group."""
        torchrl_logger.info("Local LLM weight update group initialized (no-op)")

    def update_weights(self, weights: Iterator[tuple[str, torch.Tensor]]) -> None:
        """Update model weights.

        For local LLM instances, weight updates are not applicable since
        the model is in the same process space.
        """
        weights_list = list(weights)
        torchrl_logger.info(
            f"Local LLM weight update (no-op) for {len(weights_list)} parameters"
        )

    # Delegate generation methods to the local LLM
    def generate(self, *args, **kwargs):
        """Generate text using the local LLM."""
        return self.llm_instance.generate(*args, **kwargs)


def make_vllm_worker(
    *,
    model_name: str,
    devices: list[torch.device | int] | None = None,
    num_devices: int | None = None,
    make_ray_worker: bool = True,
    enforce_eager: bool = False,
    enable_fp32_output: bool = False,
    **kwargs,
) -> RayLLMWorker | LocalLLMWrapper:
    """Creates a vLLM inference engine with tensor parallelism support.

    Args:
        model_name (str): The model name to pass to vLLM.LLM.
        devices (list[torch.device | int], optional): List of devices to use. Exclusive with num_devices.
        num_devices (int, optional): Number of devices to use. Exclusive with devices.
        make_ray_worker (bool, optional): Whether to create a Ray actor. Defaults to True.
        enforce_eager (bool, optional): Whether to enforce eager execution. Defaults to `False`.
        enable_fp32_output (bool, optional): Whether to enable FP32 output for the final layer. Defaults to False.
            This can help with numerical stability for certain models. Requires model-specific support in
            torchrl.modules.llm.backends._models.
        **kwargs: Additional arguments passed to vLLM.LLM.__init__.

    Returns:
        RayLLMWorker | LocalLLMWrapper: Either a Ray worker wrapper or a local LLM wrapper, both implementing RLvLLMEngine.

    Example:
        >>> # Create a 2-GPU tensor parallel worker with Ray
        >>> worker = make_vllm_worker("Qwen/Qwen2.5-3B", num_devices=2)
        >>> # Create a local LLM instance on GPU 1
        >>> llm = make_vllm_worker("Qwen/Qwen2.5-3B", devices=[1], make_ray_worker=False)
        >>> # Create with FP32 output enabled
        >>> worker = make_vllm_worker("Qwen/Qwen2.5-3B", num_devices=2, enable_fp32_output=True)
    """
    if not _has_vllm:
        raise ImportError(
            "vllm is not installed. Please install it with `pip install vllm`."
        )

    # Set FP32 output environment variable if requested
    if enable_fp32_output:
        os.environ["VLLM_ENABLE_FP32_OUTPUT"] = "1"
        torchrl_logger.info(
            "Enabled FP32 output for vLLM (VLLM_ENABLE_FP32_OUTPUT=1). "
            "This will use FP32 for the final output layer if the model supports it."
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
        # Convert devices to indices
        devices = [
            torch.device(device).index if not isinstance(device, int) else device
            for device in devices
        ]
        num_devices = len(devices)

    # Validate devices
    if devices is not None:
        for d in devices:
            if not isinstance(d, int) or d < 0 or d >= torch.cuda.device_count():
                raise ValueError(f"Invalid device index: {d}")

    if make_ray_worker:
        import ray

        if not ray.is_initialized():
            raise RuntimeError("Ray is not initialized")

        torchrl_logger.info(
            f"Creating vLLM Ray worker with tensor_parallel_size={num_devices}"
        )

        # Configure Ray remote class with minimal resources
        # Let vLLM handle GPU allocation through environment variables
        worker_cls = ray.remote(
            num_cpus=4,  # Minimal CPU request
            num_gpus=0,  # Let vLLM handle GPU allocation
        )(_LLMOnDevice)

        # Create worker with tensor parallelism config
        worker = worker_cls.remote(
            model=model_name,
            bundle_indices=devices,  # Pass device indices to _LLMOnDevice
            tensor_parallel_size=num_devices,
            distributed_executor_backend="ray",
            enforce_eager=enforce_eager,
            worker_cls="torchrl.modules.llm.backends.vllm.vllm_sync._vLLMWorker",
            **kwargs,
        )
        ray.get(worker.initialize.remote())

        # Wrap the Ray actor in RayLLMWorker to provide RLvLLMEngine interface
        return RayLLMWorker(worker, num_devices or 1, model_name)

    else:
        # Local non-Ray mode - use LLM directly
        with _cuda_visible_devices(devices) if devices is not None else nullcontext():
            torchrl_logger.info(
                f"Creating local vLLM LLM with tensor_parallel_size={num_devices}, devices={devices}"
            )
            llm_instance = LLM(
                model=model_name,
                tensor_parallel_size=num_devices,
                enforce_eager=True,
                **kwargs,
            )

            # Wrap the local LLM to provide RLvLLMEngine interface
            return LocalLLMWrapper(llm_instance, num_devices or 1, model_name)
