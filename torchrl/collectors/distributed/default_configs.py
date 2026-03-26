# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import os
import random
import socket
from datetime import timedelta

import torch.distributed

from torchrl._utils import logger as torchrl_logger

TCP_PORT = os.environ.get("TCP_PORT", "10003")
IDLE_TIMEOUT = os.environ.get("RCP_IDLE_TIMEOUT", 10)

MAX_TIME_TO_CONNECT = 1000

SLEEP_INTERVAL = 1e-6

DEFAULT_SLURM_CONF = {
    "timeout_min": 10,
    "slurm_partition": "train",
    "slurm_cpus_per_task": 32,
    "slurm_gpus_per_node": 0,
}  #: Default value of the SLURM jobs

DEFAULT_SLURM_CONF_MAIN = {
    "timeout_min": 10,
    "slurm_partition": "train",
    "slurm_cpus_per_task": 32,
    "slurm_gpus_per_node": 1,
}  #: Default value of the SLURM main job

DEFAULT_TENSORPIPE_OPTIONS = {
    "num_worker_threads": 16,
    "rpc_timeout": 10_000,
    "_transports": ["uv"],
}


def _find_free_port() -> int:
    """Find a free port by binding to port 0 and letting the OS choose."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(("", 0))
        return s.getsockname()[1]


def _create_tcpstore_with_retry(
    host_name: str,
    port: int | None,
    world_size: int,
    is_master: bool,
    timeout: float = 10.0,
    max_retries: int = 10,
    wait_for_workers: bool = True,
) -> tuple[torch.distributed.TCPStore, int]:
    """Create a TCPStore with retry logic for handling port conflicts.

    This function attempts to create a TCPStore, and if the port is already in use,
    it will retry with different random ports up to max_retries times.

    Args:
        host_name: The hostname for the TCPStore.
        port: The initial port to try. If None, a random port will be chosen.
        world_size: The world size for the TCPStore.
        is_master: Whether this is the master (server) process.
        timeout: Timeout in seconds for the TCPStore.
        max_retries: Maximum number of retry attempts.
        wait_for_workers: Whether the master should wait for workers.
            Only used when is_master=True.

    Returns:
        A tuple of (TCPStore, actual_port) where actual_port is the port
        that was successfully bound.

    Raises:
        RuntimeError: If unable to create a TCPStore after max_retries attempts.
    """
    last_error = None

    for attempt in range(max_retries):
        if port is None or attempt > 0:
            # For the first attempt use provided port, for retries find a new free port
            current_port = _find_free_port()
        else:
            current_port = int(port)

        try:
            if is_master:
                store = torch.distributed.TCPStore(
                    host_name=host_name,
                    port=current_port,
                    world_size=world_size,
                    is_master=True,
                    timeout=timedelta(seconds=timeout),
                    wait_for_workers=wait_for_workers,
                )
            else:
                store = torch.distributed.TCPStore(
                    host_name=host_name,
                    port=current_port,
                    is_master=False,
                    timeout=timedelta(seconds=timeout),
                )
            torchrl_logger.debug(
                f"TCPStore created successfully on {host_name}:{current_port} "
                f"(attempt {attempt + 1}/{max_retries})"
            )
            return store, current_port

        except (RuntimeError, OSError) as e:
            error_msg = str(e).lower()
            if "address already in use" in error_msg or "eaddrinuse" in error_msg:
                torchrl_logger.debug(
                    f"Port {current_port} already in use, "
                    f"retrying ({attempt + 1}/{max_retries})..."
                )
                last_error = e
                # Add small random delay to reduce collision probability
                import time

                time.sleep(random.uniform(0.01, 0.1))
                continue
            # For other errors, re-raise immediately
            raise

    raise RuntimeError(
        f"Failed to create TCPStore after {max_retries} attempts. Last error: {last_error}"
    )
