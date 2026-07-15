# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Private distributed-learning primitives used by Trainer backends."""

from __future__ import annotations

import socket
from dataclasses import dataclass
from datetime import timedelta
from typing import Literal

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel


def _local_ip_address() -> str:
    """Return an address reachable by other workers on the same cluster."""
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        try:
            sock.connect(("8.8.8.8", 80))
            return sock.getsockname()[0]
        except OSError:
            return "127.0.0.1"


def _create_tcp_store(
    *,
    host: str | None = None,
    timeout: float = 120.0,
) -> tuple[dist.TCPStore, tuple[str, int]]:
    """Create a rank-zero rendezvous store on an OS-selected free port."""
    host = _local_ip_address() if host is None else host
    store = dist.TCPStore(
        host,
        0,
        world_size=None,
        is_master=True,
        timeout=timedelta(seconds=timeout),
        wait_for_workers=False,
    )
    return store, (host, store.port)


def _connect_tcp_store(
    coordinates: tuple[str, int], *, timeout: float = 120.0
) -> dist.TCPStore:
    """Connect to a store created by :func:`_create_tcp_store`."""
    host, port = coordinates
    return dist.TCPStore(
        host,
        port,
        world_size=None,
        is_master=False,
        timeout=timedelta(seconds=timeout),
    )


@dataclass
class _DDPProcessGroup:
    """Lifecycle record for one learner-owned process group.

    Learner workers are dedicated processes, so their optimization group is
    their default process group. Inference transports and weight publication
    use standalone groups and are therefore independent of this lifecycle.
    """

    rank: int
    local_rank: int
    world_size: int
    device: torch.device
    process_group: dist.ProcessGroup
    store: dist.Store
    generation: int
    owns_process_group: bool = True

    @classmethod
    def create(
        cls,
        *,
        rank: int,
        local_rank: int,
        world_size: int,
        store: dist.Store,
        backend: Literal["gloo", "nccl"] = "gloo",
        generation: int = 0,
        timeout: float = 120.0,
    ) -> _DDPProcessGroup:
        """Create a standalone process group for one learner rank."""
        if rank < 0 or rank >= world_size:
            raise ValueError(f"rank must be in [0, {world_size}), but got rank={rank}.")
        if local_rank < 0:
            raise ValueError(f"local_rank must be non-negative, got {local_rank}.")
        if dist.is_initialized():
            raise RuntimeError(
                "A learner worker cannot replace an existing default process group."
            )
        prefix_store = dist.PrefixStore(f"torchrl-learner/{generation}", store)
        if backend == "nccl":
            if not torch.cuda.is_available():
                raise RuntimeError("The NCCL learner backend requires CUDA.")
            device = torch.device("cuda", local_rank)
            torch.cuda.set_device(device)
            dist.init_process_group(
                backend="nccl",
                store=prefix_store,
                rank=rank,
                world_size=world_size,
                timeout=timedelta(seconds=timeout),
                device_id=device,
            )
        elif backend == "gloo":
            device = torch.device("cpu")
            dist.init_process_group(
                backend="gloo",
                store=prefix_store,
                rank=rank,
                world_size=world_size,
                timeout=timedelta(seconds=timeout),
            )
        else:
            raise ValueError(
                f"Unsupported learner process-group backend {backend!r}. "
                "Expected 'gloo' or 'nccl'."
            )
        return cls(
            rank=rank,
            local_rank=local_rank,
            world_size=world_size,
            device=device,
            process_group=dist.distributed_c10d._get_default_group(),
            store=store,
            generation=generation,
        )

    def wrap(self, module: nn.Module) -> DistributedDataParallel:
        """Move a module to this rank and wrap it for gradient reduction."""
        module = module.to(self.device)
        kwargs = {
            "module": module,
            "process_group": self.process_group,
            "find_unused_parameters": True,
        }
        if self.device.type == "cuda":
            kwargs.update(device_ids=[self.local_rank], output_device=self.local_rank)
        return DistributedDataParallel(**kwargs)

    def barrier(self) -> None:
        """Wait until every learner rank reaches this point."""
        dist.barrier(group=self.process_group)

    def close(self) -> None:
        """Release references owned by this record without touching other groups."""
        if not self.owns_process_group:
            return
        self.owns_process_group = False
        dist.destroy_process_group(self.process_group)
        self.process_group = None
        self.store = None
