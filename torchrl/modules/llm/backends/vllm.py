# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Recipes for vLLM instantiation.

From https://docs.vllm.ai/en/v0.7.0/getting_started/examples/rlhf.html
"""


from __future__ import annotations

import os

from contextlib import nullcontext

import torch
from torchrl._utils import logger as torchrl_logger

from torchrl.modules.llm.utils import _cuda_visible_devices

try:
    from vllm import LLM
    from vllm.utils import get_open_port
    from vllm.worker.worker import Worker
except ImportError:

    class LLM:  # noqa
        ...

    class Worker:  # noqa
        ...

    class get_open_port:  # noqa
        ...


def stateless_init_process_group(
    master_address: str | None, master_port: str | None, rank, world_size, device
):
    """Initializes a stateless process group for distributed communication.

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
    from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
    from vllm.distributed.utils import StatelessProcessGroup

    if master_address is None:
        master_address = "localhost"  # get_ip()
    if master_port is None:
        master_port = get_open_port()

    pg = StatelessProcessGroup.create(
        host=master_address, port=master_port, rank=rank, world_size=world_size
    )
    pynccl = PyNcclCommunicator(pg, device=device)
    return pynccl


class vLLMWorker(Worker):
    """vLLM worker for Ray.

    vLLMParameterServer will always take rank 0 in the stateless process group
    initialized by this worker. And the tp ranks associated with the LLM class
    will be in the range [1, tp_size].
    """

    def __init__(self, *args, **kwargs):
        import os

        torchrl_logger.info(f"=> in {type(self).__name__}.__init__")
        torchrl_logger.info(f"visible devices {os.getenv('CUDA_VISIBLE_DEVICES')}")
        torchrl_logger.info(f"device count {torch.cuda.device_count()}")
        super().__init__(*args, **kwargs)

    def init_weight_update_group(
        self, master_address, master_port, rank_offset, world_size
    ):
        from vllm.distributed.parallel_state import get_world_group

        torchrl_logger.info(f"=> in {type(self).__name__}.init_weight_update_group")

        # Before
        # rank = get_world_group().rank + rank_offset
        # Get the local rank within the tensor parallel group
        tp_group = get_world_group()
        local_rank = tp_group.rank
        torchrl_logger.info(f"Local rank in tensor parallel group: {local_rank}")

        # Calculate the global rank for weight update group
        # rank_offset is 1, so ranks will be [1, 2] for tp_size=2
        rank = local_rank + rank_offset
        torchrl_logger.info(
            f"Initializing {type(self).__name__} weight update group with {master_address=}, {master_port=}, {rank=}, {world_size=}, device={self.device}"
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


class LLMOnDevice(LLM):
    """A thin wrapper around `vllm.LLM` to control its placement devices."""

    def __init__(self, *args, bundle_indices: list | None = None, **kwargs):
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


def make_vllm_worker(
    *,
    model_name: str,
    devices: list[torch.device | int] | None = None,
    num_devices: int | None = None,
    make_ray_worker: bool = True,
    enforce_eager: bool = False,
    **kwargs,
) -> LLM | ray.actor.ActorClass:  # noqa
    """Creates a vLLM inference engine with tensor parallelism support.

    Args:
        model_name (str): The model name to pass to vLLM.LLM.
        devices (list[torch.device | int], optional): List of devices to use. Exclusive with num_devices.
        num_devices (int, optional): Number of devices to use. Exclusive with devices.
        make_ray_worker (bool, optional): Whether to create a Ray actor. Defaults to True.
        enforce_eager (bool, optional): Whether to enforce eager execution. Defaults to `False`.
        **kwargs: Additional arguments passed to vLLM.LLM.__init__.

    Returns:
        LLM | ray.actor.ActorClass: Either a local vLLM LLM instance or a Ray actor handle.

    Example:
        >>> # Create a 2-GPU tensor parallel worker with Ray
        >>> worker = make_vllm_worker("Qwen/Qwen2.5-3B", num_devices=2)
        >>> # Create a local LLM instance on GPU 1
        >>> llm = make_vllm_worker("Qwen/Qwen2.5-3B", devices=[1], make_ray_worker=False)
    """
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
        )(LLMOnDevice)

        # Create worker with tensor parallelism config
        worker = worker_cls.remote(
            model=model_name,
            bundle_indices=devices,  # Pass device indices to LLMOnDevice
            tensor_parallel_size=num_devices,
            distributed_executor_backend="ray",
            enforce_eager=enforce_eager,
            worker_cls="torchrl.modules.llm.backends.vllm.vLLMWorker",
            **kwargs,
        )
        ray.get(worker.initialize.remote())
        return worker

    else:
        # Local non-Ray mode - use LLM directly
        with _cuda_visible_devices(devices) if devices is not None else nullcontext():
            torchrl_logger.info(
                f"Creating local vLLM LLM with tensor_parallel_size={num_devices}, devices={devices}"
            )
            return LLM(
                model=model_name,
                tensor_parallel_size=num_devices,
                enforce_eager=True,
                **kwargs,
            )
