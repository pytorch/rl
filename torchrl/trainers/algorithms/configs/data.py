# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

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
    writers: list[Any] = field(default_factory=list)
    transforms: list[Any] = field(default_factory=list)


@dataclass
class LazyStackStorageConfig(StorageConfig):
    _target_: str = "torchrl.data.replay_buffers.LazyStackStorage"
    max_size: int | None = None
    compilable: bool = False
    stack_dim: int = 0


@dataclass
class StorageEnsembleConfig(StorageConfig):
    _target_: str = "torchrl.data.replay_buffers.StorageEnsemble"
    storages: list[Any] = field(default_factory=list)
    transforms: list[Any] = field(default_factory=list)


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

    @classmethod
    def default_config(cls, **kwargs) -> "LazyTensorStorageConfig":
        """Creates a default lazy tensor storage configuration.
        
        Args:
            **kwargs: Override default values
            
        Returns:
            LazyTensorStorageConfig with default values, overridden by kwargs
        """
        defaults = {
            "max_size": 100_000,
            "device": "cpu",
            "ndim": 1,
            "compilable": False,
            "_partial_": True,
        }
        defaults.update(kwargs)
        return cls(**defaults)


@dataclass
class StorageConfig(ConfigBase):
    pass

@dataclass
class ReplayBufferBaseConfig(ConfigBase):
    _partial_: bool = False

@dataclass
class TensorDictReplayBufferConfig(ReplayBufferBaseConfig):
    _target_: str = "torchrl.data.replay_buffers.TensorDictReplayBuffer"
    sampler: Any = field(default_factory=RandomSamplerConfig)
    storage: Any = field(default_factory=TensorStorageConfig)
    writer: Any = field(default_factory=RoundRobinWriterConfig)
    transform: Any = None
    batch_size: int | None = None


@dataclass
class ReplayBufferConfig(ReplayBufferBaseConfig):
    _target_: str = "torchrl.data.replay_buffers.ReplayBuffer"
    sampler: Any = field(default_factory=RandomSamplerConfig)
    storage: Any = field(default_factory=ListStorageConfig)
    writer: Any = field(default_factory=RoundRobinWriterConfig)
    transform: Any = None
    batch_size: int | None = None

    @classmethod
    def default_config(cls, **kwargs) -> "ReplayBufferConfig":
        """Creates a default replay buffer configuration.
        
        Args:
            **kwargs: Override default values. Supports nested overrides using double underscore notation
                     (e.g., "storage__max_size": 200_000)
            
        Returns:
            ReplayBufferConfig with default values, overridden by kwargs
        """
        from tensordict import TensorDict
        
        # Unflatten the kwargs using TensorDict to understand what the user wants
        kwargs_td = TensorDict(kwargs)
        unflattened_kwargs = kwargs_td.unflatten_keys("__").to_dict()
        
        # Create configs with nested overrides applied
        sampler_overrides = unflattened_kwargs.get("sampler", {})
        storage_overrides = unflattened_kwargs.get("storage", {})
        writer_overrides = unflattened_kwargs.get("writer", {})
        
        sampler_cfg = RandomSamplerConfig(**sampler_overrides) if sampler_overrides else RandomSamplerConfig()
        storage_cfg = LazyTensorStorageConfig.default_config(**storage_overrides)
        writer_cfg = RoundRobinWriterConfig(**writer_overrides) if writer_overrides else RoundRobinWriterConfig()
        
        defaults = {
            "sampler": sampler_cfg,
            "storage": storage_cfg,
            "writer": writer_cfg,
            "transform": unflattened_kwargs.get("transform", None),
            "batch_size": unflattened_kwargs.get("batch_size", 256),
            "_partial_": True,
        }
        
        return cls(**defaults)

