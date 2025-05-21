# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Recipes for vLLM instantiation.

From https://docs.vllm.ai/en/v0.7.0/getting_started/examples/rlhf.html
"""


from __future__ import annotations

import os

import torch

from torchrl._utils import logger as torchrl_logger

from torchrl.modules.llm.utils import _cuda_visible_devices

from vllm import LLM
from vllm.utils import get_open_port
from vllm.worker.worker import Worker


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

        rank = get_world_group().rank + rank_offset
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

    def __init__(self, *args, **kwargs):
        import ray

        gpu_ids = ray.get_gpu_ids()
        torchrl_logger.info(f"=> in {type(self)}.__init__: {gpu_ids=}")
        assert len(gpu_ids) > 0, "No visible cuda device"
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(
            str(int(gpu_id)) for gpu_id in gpu_ids
        )
        torch.cuda.set_device(0)  # Since only one GPU is visible, it's cuda:0
        super().__init__(*args, device="cuda:0", **kwargs)


def make_vllm_worker(
    model_name,
    devices: list[torch.device | int],
    make_ray_worker: bool = True,
    **kwargs,
) -> LLM | ray.actor.ActorClass:  # noqa
    """Launches the vLLM inference engine.

    Args:
        model_name (str): a model name to pass to `vllm.LLM`.
        devices (list[torch.device | int]): a list of devices to use.
        make_ray_worker (bool, optional): whether to use ray's worker, or just a plain
            `LLM` instance. Defaults to `True`.
        **kwargs: keyword arguments to pass to `vllm.LLM.__init__`.

    Update weights example:
        >>> # simulate training, modify the weights of the model.
        >>> for name, p in train_model.named_parameters():
        ...     p.data.zero_()
        >>>
        >>> # sync weight from the training process to the inference engine.
        >>> for name, p in train_model.named_parameters():
        ...     handle = inference_server .collective_rpc.remote("update_weight_broadcast",
        ...                                        args=(name, p.dtype, p.shape))
        ...     model_update_group.broadcast(p, src=0, stream=torch.cuda.current_stream())
        ...     ray.get(handle)
    """
    if make_ray_worker:
        devices = [
            torch.device(device).index if not isinstance(device, int) else device
            for device in devices
        ]
        for d in devices:
            assert d < torch.cuda.device_count()

        import ray
        from ray.util.placement_group import placement_group
        from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

        if not ray.is_initialized():
            ray.init()

        pg = placement_group([{"GPU": 1, "CPU": 1}] * torch.cuda.device_count())

        ray.get(pg.ready())
        scheduling_inference = PlacementGroupSchedulingStrategy(
            placement_group=pg,
            placement_group_capture_child_tasks=True,
            placement_group_bundle_index=devices[0] if devices else None,
        )
        torchrl_logger.info(
            f"Create vLLM worker with {devices=}, {scheduling_inference=}"
        )
        return ray.remote(
            num_gpus=len(devices),
            num_cpus=1,
            scheduling_strategy=scheduling_inference,
        )(LLMOnDevice).remote(
            model=model_name,
            # enforce_eager=True,
            dtype="bfloat16",
            worker_cls="torchrl.modules.llm.backends.vllm.vLLMWorker",
            tensor_parallel_size=len(devices),
            distributed_executor_backend="ray",
            enable_chunked_prefill=True,
            **kwargs,
        )
    else:
        with _cuda_visible_devices(devices):
            return LLM(
                model=model_name,
                # enforce_eager=True,
                dtype="bfloat16",
                # worker_cls="torchrl.modules.llm.backends.vllm.VLLMWorker",
                worker_cls=vLLMWorker,
                tensor_parallel_size=len(devices),
                # distributed_executor_backend="ray",
                enable_chunked_prefill=True,
                **kwargs,
            )
