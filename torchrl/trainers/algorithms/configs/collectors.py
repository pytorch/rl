# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial
from typing import Any

from omegaconf import MISSING

from torchrl.trainers.algorithms.configs.common import ConfigBase
from torchrl.trainers.algorithms.configs.envs import EnvConfig


@dataclass
class BaseCollectorConfig(ConfigBase):
    """Parent class to configure a data collector."""


@dataclass
class CollectorConfig(BaseCollectorConfig):
    """A class to configure a synchronous data collector (Collector)."""

    create_env_fn: ConfigBase = MISSING
    policy: Any = None
    policy_factory: Any = None
    frames_per_batch: int | None = None
    total_frames: int = -1
    init_random_frames: int | None = 0
    device: str | None = None
    storing_device: str | None = None
    policy_device: str | None = None
    env_device: str | None = None
    create_env_kwargs: dict | None = None
    max_frames_per_traj: int | None = None
    reset_at_each_iter: bool = False
    postproc: Any = None
    split_trajs: bool = False
    exploration_type: str = "RANDOM"
    return_same_td: bool = False
    interruptor: Any = None
    set_truncated: bool = False
    use_buffers: bool = False
    replay_buffer: Any = None
    extend_buffer: bool = False
    trust_policy: bool = True
    compile_policy: Any = None
    cudagraph_policy: Any = None
    no_cuda_sync: bool = False
    weight_updater: Any = None
    weight_sync_schemes: Any = None
    track_policy_version: bool = False
    local_init_rb: bool = False
    _target_: str = "torchrl.collectors.Collector"
    _partial_: bool = False

    def __post_init__(self):
        self.create_env_fn._partial_ = True
        if self.policy_factory is not None:
            self.policy_factory._partial_ = True
        if self.weight_updater is not None:
            self.weight_updater._partial_ = True


# Legacy alias
SyncDataCollectorConfig = CollectorConfig


@dataclass
class AsyncCollectorConfig(BaseCollectorConfig):
    """Configuration for asynchronous data collector (AsyncCollector)."""

    create_env_fn: ConfigBase = field(
        default_factory=partial(EnvConfig, _partial_=True)
    )
    policy: Any = None
    policy_factory: Any = None
    frames_per_batch: int | None = None
    init_random_frames: int | None = 0
    total_frames: int = -1
    device: str | None = None
    storing_device: str | None = None
    policy_device: str | None = None
    env_device: str | None = None
    create_env_kwargs: dict | None = None
    max_frames_per_traj: int | None = None
    reset_at_each_iter: bool = False
    postproc: ConfigBase | None = None
    split_trajs: bool = False
    exploration_type: str = "RANDOM"
    set_truncated: bool = False
    use_buffers: bool = False
    replay_buffer: ConfigBase | None = None
    extend_buffer: bool = False
    trust_policy: bool = True
    compile_policy: Any = None
    cudagraph_policy: Any = None
    no_cuda_sync: bool = False
    weight_updater: Any = None
    weight_sync_schemes: Any = None
    track_policy_version: bool = False
    local_init_rb: bool = False
    _target_: str = "torchrl.collectors.AsyncCollector"
    _partial_: bool = False

    def __post_init__(self):
        self.create_env_fn._partial_ = True
        if self.policy_factory is not None:
            self.policy_factory._partial_ = True
        if self.weight_updater is not None:
            self.weight_updater._partial_ = True


# Legacy alias
AsyncDataCollectorConfig = AsyncCollectorConfig


@dataclass
class MultiSyncCollectorConfig(BaseCollectorConfig):
    """Configuration for multi-synchronous data collector (MultiSyncCollector)."""

    create_env_fn: Any = MISSING
    num_workers: int | None = None
    policy: Any = None
    policy_factory: Any = None
    frames_per_batch: int | None = None
    init_random_frames: int | None = 0
    total_frames: int = -1
    device: str | None = None
    storing_device: str | None = None
    policy_device: str | None = None
    env_device: str | None = None
    create_env_kwargs: dict | None = None
    max_frames_per_traj: int | None = None
    reset_at_each_iter: bool = False
    postproc: ConfigBase | None = None
    split_trajs: bool = False
    exploration_type: str = "RANDOM"
    set_truncated: bool = False
    use_buffers: bool = False
    replay_buffer: ConfigBase | None = None
    extend_buffer: bool = False
    trust_policy: bool = True
    compile_policy: Any = None
    cudagraph_policy: Any = None
    no_cuda_sync: bool = False
    weight_updater: Any = None
    weight_sync_schemes: Any = None
    track_policy_version: bool = False
    local_init_rb: bool = False
    _target_: str = "torchrl.collectors.MultiSyncCollector"
    _partial_: bool = False

    def __post_init__(self):
        for env_cfg in self.create_env_fn:
            env_cfg._partial_ = True
        if self.policy_factory is not None:
            self.policy_factory._partial_ = True
        if self.weight_updater is not None:
            self.weight_updater._partial_ = True


# Legacy alias
MultiSyncCollectorConfig = MultiSyncCollectorConfig


@dataclass
class MultiAsyncCollectorConfig(BaseCollectorConfig):
    """Configuration for multi-asynchronous data collector (MultiAsyncCollector)."""

    create_env_fn: Any = MISSING
    num_workers: int | None = None
    policy: Any = None
    policy_factory: Any = None
    frames_per_batch: int | None = None
    init_random_frames: int | None = 0
    total_frames: int = -1
    device: str | None = None
    storing_device: str | None = None
    policy_device: str | None = None
    env_device: str | None = None
    create_env_kwargs: dict | None = None
    max_frames_per_traj: int | None = None
    reset_at_each_iter: bool = False
    postproc: ConfigBase | None = None
    split_trajs: bool = False
    exploration_type: str = "RANDOM"
    set_truncated: bool = False
    use_buffers: bool = False
    replay_buffer: ConfigBase | None = None
    extend_buffer: bool = False
    trust_policy: bool = True
    compile_policy: Any = None
    cudagraph_policy: Any = None
    no_cuda_sync: bool = False
    weight_updater: Any = None
    weight_sync_schemes: Any = None
    track_policy_version: bool = False
    local_init_rb: bool = False
    _target_: str = "torchrl.collectors.MultiAsyncCollector"
    _partial_: bool = False

    def __post_init__(self):
        for env_cfg in self.create_env_fn:
            env_cfg._partial_ = True
        if self.policy_factory is not None:
            self.policy_factory._partial_ = True
        if self.weight_updater is not None:
            self.weight_updater._partial_ = True


# Legacy alias
MultiAsyncCollectorConfig = MultiAsyncCollectorConfig
