# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from torchrl.trainers.algorithms.configs.common import ConfigBase


@dataclass
class WeightUpdaterConfig(ConfigBase):
    """Base configuration for weight updaters."""

    _target_: str = "torchrl.collectors.WeightUpdaterBase"
    _partial_: bool = True

    def __post_init__(self) -> None:
        """Post-initialization hook for weight updater configurations."""


@dataclass
class VanillaWeightUpdaterConfig(ConfigBase):
    """Configuration for VanillaWeightUpdater.

    A simple implementation for updating local policy weights by directly
    fetching them from a specified source.
    """

    _target_: str = "torchrl.collectors.VanillaWeightUpdater"
    _partial_: bool = True

    # Constructor arguments
    weight_getter: Any = None  # Callable[[], TensorDictBase] | None
    policy_weights: Any = None  # TensorDictBase

    def __post_init__(self) -> None:
        """Post-initialization hook for vanilla weight updater configurations."""


@dataclass
class MultiProcessedWeightUpdaterConfig(ConfigBase):
    """Configuration for MultiProcessedWeightUpdater.

    A remote weight updater for synchronizing policy weights across multiple
    processes or devices in a multiprocessed environment.
    """

    _target_: str = "torchrl.collectors.MultiProcessedWeightUpdater"
    _partial_: bool = True

    # Constructor arguments
    get_server_weights: Any = None  # Callable[[], TensorDictBase] | None
    policy_weights: Any = None  # dict[torch.device, TensorDictBase]

    def __post_init__(self) -> None:
        """Post-initialization hook for multiprocessed weight updater configurations."""


@dataclass
class RayWeightUpdaterConfig(ConfigBase):
    """Configuration for RayWeightUpdater.

    A remote weight updater for synchronizing policy weights across remote
    workers using Ray's distributed computing capabilities.
    """

    _target_: str = "torchrl.collectors.RayWeightUpdater"
    _partial_: bool = True

    # Constructor arguments
    policy_weights: Any = None  # TensorDictBase
    remote_collectors: Any = None  # list
    max_interval: int = 0  # int

    def __post_init__(self) -> None:
        """Post-initialization hook for Ray weight updater configurations."""


@dataclass
class RPCWeightUpdaterConfig(ConfigBase):
    """Configuration for RPCWeightUpdater.

    A remote weight updater for synchronizing policy weights across remote
    workers using RPC communication.
    """

    _target_: str = "torchrl.collectors.distributed.RPCWeightUpdater"
    _partial_: bool = True

    # Constructor arguments
    collector_infos: Any = None
    collector_class: Any = None
    collector_rrefs: Any = None
    policy_weights: Any = None  # TensorDictBase
    num_workers: int = 0

    def __post_init__(self) -> None:
        """Post-initialization hook for RPC weight updater configurations."""


@dataclass
class DistributedWeightUpdaterConfig(ConfigBase):
    """Configuration for DistributedWeightUpdater.

    A remote weight updater for synchronizing policy weights across distributed
    workers using a dictionary-like store for communication.
    """

    _target_: str = "torchrl.collectors.distributed.DistributedWeightUpdater"
    _partial_: bool = True

    # Constructor arguments
    store: Any = None  # dict[str, str]
    policy_weights: Any = None  # TensorDictBase
    num_workers: int = 0
    sync: bool = True

    def __post_init__(self) -> None:
        """Post-initialization hook for distributed weight updater configurations."""


@dataclass
class RemoteModuleWeightUpdaterConfig(ConfigBase):
    """Configuration for RemoteModuleWeightUpdater.

    A weight updater for remote nn.Modules that requires explicit weight passing.
    Used when the master collector doesn't have direct access to worker weights.
    """

    _target_: str = "torchrl.collectors.RemoteModuleWeightUpdater"
    _partial_: bool = True

    def __post_init__(self) -> None:
        """Post-initialization hook for remote module weight updater configurations."""


@dataclass
class vLLMUpdaterConfig(ConfigBase):
    """Configuration for vLLMUpdater.

    A weight updater that sends weights to vLLM workers, supporting both local
    vLLM instances and remote Ray actors for LLM inference.
    """

    _target_: str = "torchrl.collectors.llm.vLLMUpdater"
    _partial_: bool = True

    # Constructor arguments
    master_address: str | None = None
    master_port: int | None = None
    model_metadata: Any = None  # dict[str, tuple[torch.dtype, torch.Size]] | None
    vllm_tp_size: int | None = None

    def __post_init__(self) -> None:
        """Post-initialization hook for vLLM updater configurations."""
