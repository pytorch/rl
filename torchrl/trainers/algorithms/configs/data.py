# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from omegaconf import MISSING

from torchrl.trainers.algorithms.configs.common import ConfigBase


@dataclass
class WriterConfig(ConfigBase):
    """Base configuration class for replay buffer writers."""

    _target_: str = "torchrl.data.replay_buffers.Writer"

    def __post_init__(self) -> None:
        """Post-initialization hook for writer configurations."""


@dataclass
class RoundRobinWriterConfig(WriterConfig):
    """Configuration for round-robin writer that distributes data across multiple storages."""

    _target_: str = "torchrl.data.replay_buffers.RoundRobinWriter"
    compilable: bool = False

    def __post_init__(self) -> None:
        """Post-initialization hook for round-robin writer configurations."""
        super().__post_init__()


@dataclass
class SamplerConfig(ConfigBase):
    """Base configuration class for replay buffer samplers."""

    _target_: str = "torchrl.data.replay_buffers.Sampler"

    def __post_init__(self) -> None:
        """Post-initialization hook for sampler configurations."""


@dataclass
class RandomSamplerConfig(SamplerConfig):
    """Configuration for random sampling from replay buffer."""

    _target_: str = "torchrl.data.replay_buffers.RandomSampler"

    def __post_init__(self) -> None:
        """Post-initialization hook for random sampler configurations."""
        super().__post_init__()


@dataclass
class WriterEnsembleConfig(WriterConfig):
    """Configuration for ensemble writer that combines multiple writers."""

    _target_: str = "torchrl.data.replay_buffers.WriterEnsemble"
    writers: list[Any] = field(default_factory=list)
    p: Any = None


@dataclass
class TensorDictMaxValueWriterConfig(WriterConfig):
    """Configuration for TensorDict max value writer."""

    _target_: str = "torchrl.data.replay_buffers.TensorDictMaxValueWriter"
    rank_key: Any = None
    reduction: str = "sum"


@dataclass
class TensorDictRoundRobinWriterConfig(WriterConfig):
    """Configuration for TensorDict round-robin writer."""

    _target_: str = "torchrl.data.replay_buffers.TensorDictRoundRobinWriter"
    compilable: bool = False


@dataclass
class ImmutableDatasetWriterConfig(WriterConfig):
    """Configuration for immutable dataset writer."""

    _target_: str = "torchrl.data.replay_buffers.ImmutableDatasetWriter"


@dataclass
class SamplerEnsembleConfig(SamplerConfig):
    """Configuration for ensemble sampler that combines multiple samplers."""

    _target_: str = "torchrl.data.replay_buffers.SamplerEnsemble"
    samplers: list[Any] = field(default_factory=list)
    p: Any = None


@dataclass
class PrioritizedSliceSamplerConfig(SamplerConfig):
    """Configuration for prioritized slice sampling from replay buffer."""

    num_slices: int | None = None
    slice_len: int | None = None
    end_key: Any = None
    traj_key: Any = None
    ends: Any = None
    trajectories: Any = None
    cache_values: bool = False
    truncated_key: Any = ("next", "truncated")
    strict_length: bool = True
    compile: Any = False
    span: Any = False
    use_gpu: Any = False
    max_capacity: int | None = None
    alpha: float | None = None
    beta: float | None = None
    eps: float | None = None
    reduction: str | None = None
    _target_: str = "torchrl.data.replay_buffers.PrioritizedSliceSampler"


@dataclass
class SliceSamplerWithoutReplacementConfig(SamplerConfig):
    """Configuration for slice sampling without replacement."""

    _target_: str = "torchrl.data.replay_buffers.SliceSamplerWithoutReplacement"
    num_slices: int | None = None
    slice_len: int | None = None
    end_key: Any = None
    traj_key: Any = None
    ends: Any = None
    trajectories: Any = None
    cache_values: bool = False
    truncated_key: Any = ("next", "truncated")
    strict_length: bool = True
    compile: Any = False
    span: Any = False
    use_gpu: Any = False


@dataclass
class SliceSamplerConfig(SamplerConfig):
    """Configuration for slice sampling from replay buffer."""

    _target_: str = "torchrl.data.replay_buffers.SliceSampler"
    num_slices: int | None = None
    slice_len: int | None = None
    end_key: Any = None
    traj_key: Any = None
    ends: Any = None
    trajectories: Any = None
    cache_values: bool = False
    truncated_key: Any = ("next", "truncated")
    strict_length: bool = True
    compile: Any = False
    span: Any = False
    use_gpu: Any = False


@dataclass
class PrioritizedSamplerConfig(SamplerConfig):
    """Configuration for prioritized sampling from replay buffer."""

    max_capacity: int | None = None
    alpha: float | None = None
    beta: float | None = None
    eps: float | None = None
    reduction: str | None = None
    _target_: str = "torchrl.data.replay_buffers.PrioritizedSampler"


@dataclass
class SamplerWithoutReplacementConfig(SamplerConfig):
    """Configuration for sampling without replacement."""

    _target_: str = "torchrl.data.replay_buffers.SamplerWithoutReplacement"
    drop_last: bool = False
    shuffle: bool = True


@dataclass
class StorageConfig(ConfigBase):
    """Base configuration class for replay buffer storage."""

    _partial_: bool = False
    _target_: str = "torchrl.data.replay_buffers.Storage"

    def __post_init__(self) -> None:
        """Post-initialization hook for storage configurations."""


@dataclass
class TensorStorageConfig(StorageConfig):
    """Configuration for tensor-based storage in replay buffer."""

    _target_: str = "torchrl.data.replay_buffers.TensorStorage"
    max_size: int | None = None
    storage: Any = None
    device: Any = None
    ndim: int | None = None
    compilable: bool = False

    def __post_init__(self) -> None:
        """Post-initialization hook for tensor storage configurations."""
        super().__post_init__()


@dataclass
class ListStorageConfig(StorageConfig):
    """Configuration for list-based storage in replay buffer."""

    _target_: str = "torchrl.data.replay_buffers.ListStorage"
    max_size: int | None = None
    compilable: bool = False


@dataclass
class StorageEnsembleWriterConfig(StorageConfig):
    """Configuration for storage ensemble writer."""

    _target_: str = "torchrl.data.replay_buffers.StorageEnsembleWriter"
    writers: list[Any] = MISSING
    transforms: list[Any] = MISSING


@dataclass
class LazyStackStorageConfig(StorageConfig):
    """Configuration for lazy stack storage."""

    _target_: str = "torchrl.data.replay_buffers.LazyStackStorage"
    max_size: int | None = None
    compilable: bool = False
    stack_dim: int = 0


@dataclass
class StorageEnsembleConfig(StorageConfig):
    """Configuration for storage ensemble."""

    _target_: str = "torchrl.data.replay_buffers.StorageEnsemble"
    storages: list[Any] = MISSING
    transforms: list[Any] = MISSING


@dataclass
class LazyMemmapStorageConfig(StorageConfig):
    """Configuration for lazy memory-mapped storage."""

    _target_: str = "torchrl.data.replay_buffers.LazyMemmapStorage"
    max_size: int | None = None
    device: Any = None
    ndim: int = 1
    compilable: bool = False


@dataclass
class LazyTensorStorageConfig(StorageConfig):
    """Configuration for lazy tensor storage."""

    _target_: str = "torchrl.data.replay_buffers.LazyTensorStorage"
    max_size: int | None = None
    device: Any = None
    ndim: int = 1
    compilable: bool = False


@dataclass
class ReplayBufferBaseConfig(ConfigBase):
    """Base configuration class for replay buffers."""

    _partial_: bool = False

    def __post_init__(self) -> None:
        """Post-initialization hook for replay buffer configurations."""


@dataclass
class TensorDictReplayBufferConfig(ReplayBufferBaseConfig):
    """Configuration for TensorDict-based replay buffer."""

    _target_: str = "torchrl.data.replay_buffers.TensorDictReplayBuffer"
    sampler: Any = None
    storage: Any = None
    writer: Any = None
    transform: Any = None
    batch_size: int | None = None

    def __post_init__(self) -> None:
        """Post-initialization hook for TensorDict replay buffer configurations."""
        super().__post_init__()


@dataclass
class ReplayBufferConfig(ReplayBufferBaseConfig):
    """Configuration for generic replay buffer."""

    _target_: str = "torchrl.data.replay_buffers.ReplayBuffer"
    sampler: Any = None
    storage: Any = None
    writer: Any = None
    transform: Any = None
    batch_size: int | None = None
