# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from torchrl.trainers.algorithms.configs.common import ConfigBase


@dataclass
class WeightSyncSchemeConfig(ConfigBase):
    """Base configuration for weight synchronization schemes."""

    _target_: str = "torchrl.weight_update.WeightSyncScheme"
    _partial_: bool = False

    # Common argument for all schemes
    strategy: str = "tensordict"  # "tensordict" or "state_dict"

    def __post_init__(self) -> None:
        """Post-initialization hook for weight sync scheme configurations."""


@dataclass
class MultiProcessWeightSyncSchemeConfig(ConfigBase):
    """Configuration for MultiProcessWeightSyncScheme.

    Weight synchronization for multiprocess operations using pipes.
    This scheme creates transports that communicate via multiprocessing pipes.
    """

    _target_: str = "torchrl.weight_update.MultiProcessWeightSyncScheme"
    _partial_: bool = False

    strategy: str = "tensordict"  # "tensordict" or "state_dict"

    def __post_init__(self) -> None:
        """Post-initialization hook for multiprocess weight sync scheme configurations."""


@dataclass
class SharedMemWeightSyncSchemeConfig(ConfigBase):
    """Configuration for SharedMemWeightSyncScheme.

    Weight synchronization using shared memory for in-place weight updates.
    Workers automatically see weight updates without explicit message passing.

    By default, uses lazy registration (auto_register=True) which makes it seamless
    to use with Hydra configs - models are automatically registered on first weight send.
    """

    _target_: str = "torchrl.weight_update.SharedMemWeightSyncScheme"
    _partial_: bool = False

    policy_weights: Any = None  # dict[str, TensorDictBase] | None
    strategy: str = "tensordict"  # "tensordict" or "state_dict"
    auto_register: bool = True  # Enable lazy registration by default

    def __post_init__(self) -> None:
        """Post-initialization hook for shared memory weight sync scheme configurations."""


@dataclass
class NoWeightSyncSchemeConfig(ConfigBase):
    """Configuration for NoWeightSyncScheme.

    No-op weight synchronization scheme that disables weight synchronization entirely.
    """

    _target_: str = "torchrl.weight_update.NoWeightSyncScheme"
    _partial_: bool = False

    strategy: str = "tensordict"  # Not really used, but kept for consistency

    def __post_init__(self) -> None:
        """Post-initialization hook for no weight sync scheme configurations."""


@dataclass
class RayWeightSyncSchemeConfig(ConfigBase):
    """Configuration for RayWeightSyncScheme.

    Weight synchronization for Ray distributed computing. Uses Ray's object store
    and remote calls to synchronize weights across distributed workers (Ray actors).
    """

    _target_: str = "torchrl.weight_update.RayWeightSyncScheme"
    _partial_: bool = False

    strategy: str = "tensordict"  # "tensordict" or "state_dict"

    def __post_init__(self) -> None:
        """Post-initialization hook for Ray weight sync scheme configurations."""


@dataclass
class RayModuleTransformSchemeConfig(ConfigBase):
    """Configuration for RayModuleTransformScheme.

    Weight synchronization for RayModuleTransform actors. This scheme is designed
    specifically for updating models hosted within Ray actors.
    """

    _target_: str = "torchrl.weight_update.RayModuleTransformScheme"
    _partial_: bool = False

    strategy: str = "tensordict"  # "tensordict" or "state_dict"

    def __post_init__(self) -> None:
        """Post-initialization hook for Ray module transform scheme configurations."""


@dataclass
class RPCWeightSyncSchemeConfig(ConfigBase):
    """Configuration for RPCWeightSyncScheme.

    Weight synchronization for torch.distributed.rpc. Uses RPC calls to synchronize
    weights across distributed workers.
    """

    _target_: str = "torchrl.weight_update.RPCWeightSyncScheme"
    _partial_: bool = False

    strategy: str = "tensordict"  # "tensordict" or "state_dict"

    def __post_init__(self) -> None:
        """Post-initialization hook for RPC weight sync scheme configurations."""


@dataclass
class DistributedWeightSyncSchemeConfig(ConfigBase):
    """Configuration for DistributedWeightSyncScheme.

    Weight synchronization for torch.distributed. Uses torch.distributed primitives
    (send/recv) to synchronize weights across distributed workers.
    """

    _target_: str = "torchrl.weight_update.DistributedWeightSyncScheme"
    _partial_: bool = False

    backend: str = "gloo"  # "gloo", "nccl", etc.
    sync: bool = True
    strategy: str = "tensordict"  # "tensordict" or "state_dict"

    def __post_init__(self) -> None:
        """Post-initialization hook for distributed weight sync scheme configurations."""


@dataclass
class VLLMWeightSyncSchemeConfig(ConfigBase):
    """Configuration for VLLMWeightSyncScheme.

    Weight synchronization scheme for vLLM engines using collective communication (NCCL).
    Broadcasts weights from a trainer to vLLM inference workers with parallelism support.
    """

    _target_: str = "torchrl.weight_update.llm.VLLMWeightSyncScheme"
    _partial_: bool = False

    master_address: str | None = None  # Defaults to "localhost"
    master_port: int | None = None  # Auto-assigned if None
    gpus_per_replica: int = 1  # tp_size × dp_size × pp_size
    num_replicas: int = 1
    strategy: str = "tensordict"  # "tensordict" or "state_dict"
    device: Any = 0  # torch.device | str | int

    def __post_init__(self) -> None:
        """Post-initialization hook for vLLM weight sync scheme configurations."""


@dataclass
class VLLMDoubleBufferSyncSchemeConfig(ConfigBase):
    """Configuration for VLLMDoubleBufferSyncScheme.

    Weight synchronization scheme for vLLM using double-buffered memory-mapped storage.
    Uses TensorDict's memory-mapping capabilities to transfer weights via filesystem.
    """

    _target_: str = "torchrl.weight_update.llm.VLLMDoubleBufferSyncScheme"
    _partial_: bool = False

    remote_addr: str | None = None  # Directory path where sender writes weights
    local_addr: str | None = None  # Directory path where receiver reads weights
    num_threads: int = 1  # Number of threads for memmap operations
    strategy: str = "tensordict"  # "tensordict" or "state_dict"

    def __post_init__(self) -> None:
        """Post-initialization hook for vLLM double buffer sync scheme configurations."""
        if self.remote_addr is None:
            raise ValueError("remote_addr is required for VLLMDoubleBufferSyncScheme")
        if self.local_addr is None:
            self.local_addr = self.remote_addr
