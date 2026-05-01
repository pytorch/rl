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
    """Hydra configuration for :class:`~torchrl.collectors.Collector`.

    Every kwarg accepted by ``Collector.__init__`` is exposed as a field here.
    """

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
    reset_when_done: bool = True
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
    weight_recv_schemes: Any = None
    track_policy_version: bool = False
    worker_idx: int | None = None
    trajs_per_batch: int | None = None

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
    """Hydra configuration for :class:`~torchrl.collectors.AsyncCollector`.

    Every kwarg accepted by ``AsyncCollector.__init__`` is exposed as a field here.
    Fields that AsyncCollector forwards to its inner ``Collector`` via ``**kwargs``
    (replay buffer, weight sync, ...) are also exposed for convenience.
    """

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
    reset_when_done: bool = True
    update_at_each_batch: bool = False
    preemptive_threshold: float | None = None
    num_threads: int | None = None
    num_sub_threads: int = 1
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
    """Hydra configuration for :class:`~torchrl.collectors.MultiSyncCollector`.

    Every kwarg accepted by ``MultiSyncCollector.__init__`` is exposed as a field here.
    """

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
    collector_class: Any = None
    max_frames_per_traj: int | None = None
    reset_at_each_iter: bool = False
    postproc: ConfigBase | None = None
    split_trajs: bool = False
    exploration_type: str = "RANDOM"
    reset_when_done: bool = True
    update_at_each_batch: bool = False
    preemptive_threshold: float | None = None
    num_threads: int | None = None
    num_sub_threads: int = 1
    cat_results: Any = None
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
    weight_recv_schemes: Any = None
    track_policy_version: bool = False
    worker_idx: int | None = None
    trajs_per_batch: int | None = None
    init_fn: Any = None

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
    """Hydra configuration for :class:`~torchrl.collectors.MultiAsyncCollector`.

    ``MultiAsyncCollector`` shares its constructor surface with
    ``MultiSyncCollector`` (both forward to the same multi-worker base), so the
    same kwargs are exposed here.
    """

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
    collector_class: Any = None
    max_frames_per_traj: int | None = None
    reset_at_each_iter: bool = False
    postproc: ConfigBase | None = None
    split_trajs: bool = False
    exploration_type: str = "RANDOM"
    reset_when_done: bool = True
    update_at_each_batch: bool = False
    preemptive_threshold: float | None = None
    num_threads: int | None = None
    num_sub_threads: int = 1
    cat_results: Any = None
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
    weight_recv_schemes: Any = None
    track_policy_version: bool = False
    worker_idx: int | None = None
    trajs_per_batch: int | None = None
    init_fn: Any = None

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
