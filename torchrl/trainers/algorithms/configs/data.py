# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from fastapi.middleware import Middleware
from omegaconf import MISSING

from torchrl import data
from torchrl.trainers.algorithms.configs.common import ConfigBase


@dataclass
class WriterConfig(ConfigBase):
    _target_: str = "torchrl.data.replay_buffers.Writer"


@dataclass
class RoundRobinWriterConfig(WriterConfig):
    _target_: str = "torchrl.data.replay_buffers.RoundRobinWriter"
    compilable: bool = False


@dataclass
class SamplerConfig(ConfigBase):
    _target_: str = "torchrl.data.replay_buffers.Sampler"


@dataclass
class RandomSamplerConfig(SamplerConfig):
    _target_: str = "torchrl.data.replay_buffers.RandomSampler"



@dataclass
class WriterEnsembleConfig(WriterConfig):
    _target_: str = "torchrl.data.replay_buffers.WriterEnsemble"
    writers: list[Any] = field(default_factory=list)
    p: Any = None


@dataclass
class TensorDictMaxValueWriterConfig(WriterConfig):
    _target_: str = "torchrl.data.replay_buffers.TensorDictMaxValueWriter"
    rank_key: Any = None
    reduction: str = "sum"


@dataclass
class TensorDictRoundRobinWriterConfig(WriterConfig):
    _target_: str = "torchrl.data.replay_buffers.TensorDictRoundRobinWriter"
    compilable: bool = False


@dataclass
class ImmutableDatasetWriterConfig(WriterConfig):
    _target_: str = "torchrl.data.replay_buffers.ImmutableDatasetWriter"


@dataclass
class SamplerEnsembleConfig(SamplerConfig):
    _target_: str = "torchrl.data.replay_buffers.SamplerEnsemble"
    samplers: list[Any] = field(default_factory=list)
    p: Any = None


@dataclass
class PrioritizedSliceSamplerConfig(SamplerConfig):
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
    max_capacity: int | None = None
    alpha: float | None = None
    beta: float | None = None
    eps: float | None = None
    reduction: str | None = None
    _target_: str = "torchrl.data.replay_buffers.PrioritizedSampler"


@dataclass
class SamplerWithoutReplacementConfig(SamplerConfig):
    _target_: str = "torchrl.data.replay_buffers.SamplerWithoutReplacement"
    drop_last: bool = False
    shuffle: bool = True


@dataclass
class StorageConfig(ConfigBase):
    _partial_: bool = False
    _target_: str = "torchrl.data.replay_buffers.Storage"

@dataclass
class TensorStorageConfig(StorageConfig):
    _target_: str = "torchrl.data.replay_buffers.TensorStorage"
    max_size: int | None = None
    storage: Any = None
    device: Any = None
    ndim: int | None = None
    compilable: bool = False


@dataclass
class ListStorageConfig(StorageConfig):
    _target_: str = "torchrl.data.replay_buffers.ListStorage"
    max_size: int | None = None
    compilable: bool = False



@dataclass
class StorageEnsembleWriterConfig(StorageConfig):
    _target_: str = "torchrl.data.replay_buffers.StorageEnsembleWriter"
    writers: list[Any] = MISSING
    transforms: list[Any] = MISSING


@dataclass
class LazyStackStorageConfig(StorageConfig):
    _target_: str = "torchrl.data.replay_buffers.LazyStackStorage"
    max_size: int | None = None
    compilable: bool = False
    stack_dim: int = 0


@dataclass
class StorageEnsembleConfig(StorageConfig):
    _target_: str = "torchrl.data.replay_buffers.StorageEnsemble"
    storages: list[Any] = MISSING
    transforms: list[Any] = MISSING


@dataclass
class LazyMemmapStorageConfig(StorageConfig):
    _target_: str = "torchrl.data.replay_buffers.LazyMemmapStorage"
    max_size: int | None = None
    device: Any = None
    ndim: int = 1
    compilable: bool = False


@dataclass
class LazyTensorStorageConfig(StorageConfig):
    _target_: str = "torchrl.data.replay_buffers.LazyTensorStorage"
    max_size: int | None = None
    device: Any = None
    ndim: int = 1
    compilable: bool = False


@dataclass
class ReplayBufferBaseConfig(ConfigBase):
    _partial_: bool = False

@dataclass
class TensorDictReplayBufferConfig(ReplayBufferBaseConfig):
    _target_: str = "torchrl.data.replay_buffers.TensorDictReplayBuffer"
    sampler: Any = MISSING
    storage: Any = MISSING
    writer: Any = MISSING
    transform: Any = None
    batch_size: int | None = None


@dataclass
class ReplayBufferConfig(ReplayBufferBaseConfig):
    _target_: str = "torchrl.data.replay_buffers.ReplayBuffer"
    sampler: Any = MISSING
    storage: Any = MISSING
    writer: Any = MISSING
    transform: Any = None
    batch_size: int | None = None

