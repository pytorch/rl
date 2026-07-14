# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import os
from collections import defaultdict
from datetime import timedelta
from typing import Any

import torch
import torch.distributed as dist
from torch import nn
from torch.optim import Optimizer
from torchrl.data.utils import DEVICE_TYPING


class DataParallelContext:
    """Minimal process-group primitives for replicated-gradient training.

    The context can initialize a process group from the environment populated
    by ``torchrun`` or wrap an externally managed process group. It deliberately
    does not wrap a module or own the training loop: call
    :meth:`broadcast_module` once after module construction and call
    :meth:`sync_gradients` after backward, before clipping and stepping.

    Args:
        rank (int): rank within the process group. Defaults to ``0``.
        world_size (int): number of ranks. Defaults to ``1``.
        local_rank (int): rank on the local host. Defaults to ``0``.
        device (DEVICE_TYPING, optional): training device. Defaults to CPU for
            a single-process context.
        process_group (ProcessGroup, optional): process group used by
            collectives. Required when ``world_size > 1``.
        owns_process_group (bool): whether :meth:`close` should destroy the
            process group. Defaults to ``False``.

    Example:
        >>> import torch
        >>> from torchrl.distributed import DataParallelContext
        >>> module = torch.nn.Linear(3, 1)
        >>> optimizer = torch.optim.SGD(module.parameters(), lr=0.1)
        >>> with DataParallelContext() as context:
        ...     _ = context.broadcast_module(module)
        ...     module(torch.ones(2, 3)).sum().backward()
        ...     context.sync_gradients(optimizer)
        ...     optimizer.step()

    .. note::
        Phase 1 synchronizes parameters and buffers only when
        :meth:`broadcast_module` is called. Mutable forward buffers, sparse
        gradients, checkpoint redistribution, DDP wrapping, and FSDP are not
        handled by this class.
    """

    def __init__(
        self,
        *,
        rank: int = 0,
        world_size: int = 1,
        local_rank: int = 0,
        device: DEVICE_TYPING | None = None,
        process_group: dist.ProcessGroup | None = None,
        owns_process_group: bool = False,
    ) -> None:
        self._validate_rank_metadata(rank, local_rank, world_size)
        if world_size > 1 and process_group is None:
            raise ValueError(
                "process_group is required when world_size is greater than one. "
                "Use from_torchrun() or from_process_group()."
            )
        self._rank = rank
        self._local_rank = local_rank
        self._world_size = world_size
        self._device = self._resolve_device(device, local_rank)
        self._process_group = process_group
        self._owns_process_group = owns_process_group
        self._closed = False

    @classmethod
    def from_torchrun(
        cls,
        *,
        backend: str | dist.Backend | None = None,
        device: DEVICE_TYPING | None = None,
        timeout: timedelta | None = None,
        init_method: str = "env://",
    ) -> DataParallelContext:
        """Initialize or wrap the default group using torchrun environment data.

        Args:
            backend (str or Backend, optional): collective backend. Defaults to
                ``"nccl"`` for CUDA and ``"gloo"`` otherwise.
            device (DEVICE_TYPING, optional): training device. If omitted, CUDA
                uses ``LOCAL_RANK`` when available and CPU is used otherwise.
            timeout (datetime.timedelta, optional): process-group timeout.
            init_method (str): process-group initialization URL. Defaults to
                ``"env://"``.

        Returns:
            A context that owns a group initialized by this call, or a context
            that wraps an already initialized default group.
        """
        rank = cls._read_torchrun_integer("RANK")
        world_size = cls._read_torchrun_integer("WORLD_SIZE")
        local_rank = cls._read_torchrun_integer("LOCAL_RANK", default=rank)
        cls._validate_rank_metadata(rank, local_rank, world_size)
        resolved_device = cls._resolve_device(device, local_rank)
        if resolved_device.type == "cuda":
            torch.cuda.set_device(resolved_device)

        owns_process_group = False
        if dist.is_initialized():
            actual_rank = dist.get_rank()
            actual_world_size = dist.get_world_size()
            if (actual_rank, actual_world_size) != (rank, world_size):
                raise RuntimeError(
                    "The initialized default process group does not match the "
                    "torchrun environment: "
                    f"group=({actual_rank}, {actual_world_size}), "
                    f"environment=({rank}, {world_size})."
                )
        elif world_size > 1:
            if backend is None:
                backend = "nccl" if resolved_device.type == "cuda" else "gloo"
            init_kwargs: dict[str, Any] = {
                "backend": backend,
                "init_method": init_method,
                "rank": rank,
                "world_size": world_size,
            }
            if timeout is not None:
                init_kwargs["timeout"] = timeout
            dist.init_process_group(**init_kwargs)
            owns_process_group = True

        return cls(
            rank=rank,
            local_rank=local_rank,
            world_size=world_size,
            device=resolved_device,
            process_group=None if world_size == 1 else dist.group.WORLD,
            owns_process_group=owns_process_group,
        )

    @classmethod
    def from_process_group(
        cls,
        process_group: dist.ProcessGroup | None = None,
        *,
        device: DEVICE_TYPING | None = None,
        local_rank: int | None = None,
    ) -> DataParallelContext:
        """Wrap an externally owned process group.

        Args:
            process_group (ProcessGroup, optional): group to wrap. Defaults to
                the initialized default group.
            device (DEVICE_TYPING, optional): training device.
            local_rank (int, optional): local rank. Defaults to ``LOCAL_RANK``
                when present, otherwise the group rank.

        Returns:
            A context that never destroys the supplied process group.
        """
        if not dist.is_initialized():
            raise RuntimeError(
                "torch.distributed must be initialized before calling "
                "from_process_group()."
            )
        rank = dist.get_rank(process_group)
        world_size = dist.get_world_size(process_group)
        if local_rank is None:
            local_rank = cls._read_torchrun_integer("LOCAL_RANK", default=rank)
        return cls(
            rank=rank,
            local_rank=local_rank,
            world_size=world_size,
            device=device,
            process_group=process_group
            if process_group is not None
            else dist.group.WORLD,
            owns_process_group=False,
        )

    @classmethod
    def from_rendezvous(
        cls,
        *,
        rank: int,
        world_size: int,
        local_rank: int,
        device: DEVICE_TYPING,
        init_method: str,
        backend: str | dist.Backend | None = None,
        timeout: timedelta | None = None,
    ) -> DataParallelContext:
        """Initialize a process group from explicit actor rendezvous metadata.

        Unlike :meth:`from_torchrun`, this constructor does not read or mutate
        environment variables. It is intended for actor runtimes such as Ray,
        where logical local rank and the actor-visible CUDA index may differ.

        Args:
            rank (int): Global rank.
            world_size (int): Number of ranks.
            local_rank (int): Rank among actors on the same node.
            device (DEVICE_TYPING): Actual actor-visible device, commonly
                ``"cuda:0"`` for a one-GPU Ray actor.
            init_method (str): Process-group rendezvous URL.
            backend (str or Backend, optional): Collective backend. Defaults to
                NCCL for CUDA devices and Gloo otherwise.
            timeout (datetime.timedelta, optional): Process-group timeout.

        Returns:
            A context that owns the process group initialized by this call.

        Example:
            Each worker enters the same rendezvous, synchronizes the initial
            policy, and leaves the process group through the context manager:

            >>> import torch
            >>> from torchrl.distributed import DataParallelContext
            >>> def initialize_rank(rank, world_size, init_method):
            ...     policy = torch.nn.Linear(3, 1)
            ...     with DataParallelContext.from_rendezvous(
            ...         rank=rank,
            ...         world_size=world_size,
            ...         local_rank=rank,
            ...         device="cpu",
            ...         backend="gloo",
            ...         init_method=init_method,
            ...     ) as context:
            ...         context.broadcast_module(policy)
            ...         context.barrier()
            ...     return policy
        """
        cls._validate_rank_metadata(rank, local_rank, world_size)
        resolved_device = cls._resolve_device(device, local_rank)
        if resolved_device.type == "cuda":
            torch.cuda.set_device(resolved_device)

        owns_process_group = False
        if dist.is_initialized():
            actual_rank = dist.get_rank()
            actual_world_size = dist.get_world_size()
            if (actual_rank, actual_world_size) != (rank, world_size):
                raise RuntimeError(
                    "The initialized process group does not match rendezvous "
                    f"metadata: group=({actual_rank}, {actual_world_size}), "
                    f"requested=({rank}, {world_size})."
                )
        elif world_size > 1:
            if backend is None:
                backend = "nccl" if resolved_device.type == "cuda" else "gloo"
            init_kwargs: dict[str, Any] = {
                "backend": backend,
                "init_method": init_method,
                "rank": rank,
                "world_size": world_size,
            }
            if timeout is not None:
                init_kwargs["timeout"] = timeout
            dist.init_process_group(**init_kwargs)
            owns_process_group = True

        return cls(
            rank=rank,
            world_size=world_size,
            local_rank=local_rank,
            device=resolved_device,
            process_group=None if world_size == 1 else dist.group.WORLD,
            owns_process_group=owns_process_group,
        )

    @property
    def rank(self) -> int:
        """Rank within the process group."""
        return self._rank

    @property
    def local_rank(self) -> int:
        """Rank on the local host."""
        return self._local_rank

    @property
    def world_size(self) -> int:
        """Number of ranks in the process group."""
        return self._world_size

    @property
    def device(self) -> torch.device:
        """Training device associated with this rank."""
        return self._device

    @property
    def process_group(self) -> dist.ProcessGroup | None:
        """The wrapped process group, if any."""
        return self._process_group

    @property
    def is_rank_zero(self) -> bool:
        """Whether this is rank zero within the process group."""
        return self.rank == 0

    @property
    def is_closed(self) -> bool:
        """Whether this context has been closed."""
        return self._closed

    def barrier(self) -> None:
        """Wait until every rank in the context reaches this call."""
        self._ensure_open()
        if self.world_size > 1:
            dist.barrier(group=self.process_group)

    def broadcast_module(self, module: nn.Module, *, src: int = 0) -> nn.Module:
        """Broadcast initial parameters and buffers from a group rank.

        Args:
            module (nn.Module): module to synchronize.
            src (int): source rank within this context. Defaults to ``0``.

        Returns:
            The input module.
        """
        self._ensure_open()
        if src < 0 or src >= self.world_size:
            raise ValueError(
                f"src must satisfy 0 <= src < world_size, got src={src} and "
                f"world_size={self.world_size}."
            )
        if self.world_size == 1:
            return module
        global_src = dist.get_global_rank(self.process_group, src)
        seen: set[int] = set()
        with torch.no_grad():
            for value in (*module.parameters(), *module.buffers()):
                if id(value) in seen:
                    continue
                seen.add(id(value))
                dist.broadcast(value, src=global_src, group=self.process_group)
        return module

    def sync_gradients(self, optimizer: Optimizer) -> None:
        """Average unique optimizer gradients across ranks.

        Parameters used on only a subset of ranks receive zero contributions
        from the other ranks. Parameters unused everywhere keep ``grad=None``.
        Sparse and other non-strided gradients fail explicitly on every rank.

        Args:
            optimizer (Optimizer): optimizer whose unique parameters should be
                synchronized.
        """
        self._ensure_open()
        parameters = self._optimizer_parameters(optimizer)
        if self.world_size == 1:
            if any(
                parameter.grad is not None
                and parameter.grad.layout is not torch.strided
                for parameter in parameters
            ):
                raise RuntimeError(
                    "DataParallelContext does not support sparse or non-strided "
                    "gradients."
                )
            return

        buckets: dict[
            tuple[torch.device, torch.dtype], list[torch.Tensor]
        ] = defaultdict(list)
        for parameter in parameters:
            buckets[(parameter.device, parameter.dtype)].append(parameter)

        for bucket_parameters in buckets.values():
            metadata = torch.tensor(
                [
                    [
                        int(parameter.grad is not None)
                        for parameter in bucket_parameters
                    ],
                    [
                        int(
                            parameter.grad is not None
                            and parameter.grad.layout is not torch.strided
                        )
                        for parameter in bucket_parameters
                    ],
                ],
                dtype=torch.int32,
                device=bucket_parameters[0].device,
            )
            dist.all_reduce(metadata, op=dist.ReduceOp.SUM, group=self.process_group)
            metadata_cpu = metadata.cpu()
            if metadata_cpu[1].any().item():
                raise RuntimeError(
                    "DataParallelContext does not support sparse or non-strided "
                    "gradients."
                )

            gradients = [
                parameter.grad
                if parameter.grad is not None
                else torch.zeros_like(parameter, memory_format=torch.preserve_format)
                for parameter in bucket_parameters
            ]
            flat_gradient = torch.cat([gradient.reshape(-1) for gradient in gradients])
            if flat_gradient.numel():
                dist.all_reduce(
                    flat_gradient, op=dist.ReduceOp.SUM, group=self.process_group
                )
            flat_gradient.div_(self.world_size)
            offset = 0
            for index, (parameter, gradient) in enumerate(
                zip(bucket_parameters, gradients)
            ):
                numel = parameter.numel()
                reduced_gradient = flat_gradient[offset : offset + numel].view_as(
                    parameter
                )
                offset += numel
                if metadata_cpu[0, index].item():
                    if parameter.grad is None:
                        parameter.grad = reduced_gradient.clone(
                            memory_format=torch.preserve_format
                        )
                    else:
                        gradient.copy_(reduced_gradient)

    def close(self) -> None:
        """Idempotently destroy a process group owned by this context."""
        if self._closed:
            return
        try:
            if self._owns_process_group and dist.is_initialized():
                dist.destroy_process_group(self.process_group)
        finally:
            self._closed = True

    def __enter__(self) -> DataParallelContext:
        self._ensure_open()
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()

    @staticmethod
    def _optimizer_parameters(optimizer: Optimizer) -> list[torch.Tensor]:
        parameters = []
        seen: set[int] = set()
        for group in optimizer.param_groups:
            for parameter in group["params"]:
                if not isinstance(parameter, torch.Tensor):
                    raise TypeError(
                        "optimizer parameter groups must contain Tensor "
                        f"instances, got {type(parameter).__name__}."
                    )
                if not parameter.requires_grad or id(parameter) in seen:
                    continue
                seen.add(id(parameter))
                parameters.append(parameter)
        return parameters

    @staticmethod
    def _read_torchrun_integer(name: str, *, default: int | None = None) -> int:
        value = os.environ.get(name)
        if value is None:
            if default is not None:
                return default
            raise RuntimeError(
                f"{name} is not set. Launch with torchrun or initialize a process "
                "group and use DataParallelContext.from_process_group()."
            )
        try:
            return int(value)
        except ValueError as err:
            raise RuntimeError(f"{name} must be an integer, got {value!r}.") from err

    @staticmethod
    def _validate_rank_metadata(rank: int, local_rank: int, world_size: int) -> None:
        for name, value in (
            ("rank", rank),
            ("local_rank", local_rank),
            ("world_size", world_size),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(
                    f"{name} must be an integer, got {type(value).__name__}."
                )
        if world_size <= 0:
            raise ValueError(f"world_size must be positive, got {world_size}.")
        if rank < 0 or rank >= world_size:
            raise ValueError(
                f"rank must satisfy 0 <= rank < world_size, got rank={rank} and "
                f"world_size={world_size}."
            )
        if local_rank < 0:
            raise ValueError(f"local_rank must be non-negative, got {local_rank}.")

    @staticmethod
    def _resolve_device(device: DEVICE_TYPING | None, local_rank: int) -> torch.device:
        if device is None:
            if torch.cuda.is_available():
                return torch.device("cuda", local_rank)
            return torch.device("cpu")
        resolved_device = torch.device(device)
        if resolved_device.type == "cuda" and resolved_device.index is None:
            resolved_device = torch.device("cuda", local_rank)
        return resolved_device

    def _ensure_open(self) -> None:
        if self.is_closed:
            raise RuntimeError("DataParallelContext is closed.")
