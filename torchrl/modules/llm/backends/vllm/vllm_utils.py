# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Shared utilities for vLLM backends."""

from __future__ import annotations


try:
    from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
    from vllm.distributed.utils import StatelessProcessGroup
    from vllm.utils import get_open_port

    _has_vllm = True
except ImportError:
    PyNcclCommunicator = None
    StatelessProcessGroup = None
    get_open_port = None
    _has_vllm = False


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
    if not _has_vllm:
        raise ImportError(
            "vllm is not installed. Please install it with `pip install vllm`."
        )

    if StatelessProcessGroup is None or PyNcclCommunicator is None:
        raise ImportError(
            "vllm is not installed. Please install it with `pip install vllm`."
        )

    if master_address is None:
        master_address = "localhost"  # get_ip()
    if master_port is None:
        master_port = get_open_port() if callable(get_open_port) else 29500

    pg = StatelessProcessGroup.create(
        host=master_address, port=int(master_port), rank=rank, world_size=world_size
    )
    pynccl = PyNcclCommunicator(pg, device=device)
    return pynccl


async def stateless_init_process_group_async(
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
    if not _has_vllm:
        raise ImportError(
            "vllm is not installed. Please install it with `pip install vllm`."
        )

    if StatelessProcessGroup is None or PyNcclCommunicator is None:
        raise ImportError(
            "vllm is not installed. Please install it with `pip install vllm`."
        )

    if master_address is None:
        master_address = "localhost"
    if master_port is None:
        master_port = get_open_port() if callable(get_open_port) else 29500

    master_port_int = int(master_port) if master_port is not None else 0
    pg = StatelessProcessGroup.create(
        host=master_address, port=master_port_int, rank=rank, world_size=world_size
    )
    pynccl = PyNcclCommunicator(pg, device=device)
    return pynccl
