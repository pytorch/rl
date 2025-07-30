# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial
from typing import Any

from torchrl.trainers.algorithms.configs.common import ConfigBase
from torchrl.trainers.algorithms.configs.envs import EnvConfig


@dataclass
class DataCollectorConfig(ConfigBase):
    """Parent class to configure a data collector."""


@dataclass
class SyncDataCollectorConfig(DataCollectorConfig):
    """A class to configure a synchronous data collector."""

    create_env_fn: ConfigBase = field(
        default_factory=partial(EnvConfig, _partial_=True)
    )
    policy: Any = None
    policy_factory: Any = None
    frames_per_batch: int | None = None
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
    return_same_td: bool = False
    interruptor: ConfigBase | None = None
    set_truncated: bool = False
    use_buffers: bool = False
    replay_buffer: ConfigBase | None = None
    extend_buffer: bool = False
    trust_policy: bool = True
    compile_policy: Any = None
    cudagraph_policy: Any = None
    no_cuda_sync: bool = False
    _target_: str = "torchrl.collectors.collectors.SyncDataCollector"
    _partial_: bool = False

    def __post_init__(self):
        self.create_env_fn._partial_ = True
        if self.policy_factory is not None:
            self.policy_factory._partial_ = True

    @classmethod
    def default_config(cls, **kwargs) -> "SyncDataCollectorConfig":
        """Creates a default synchronous data collector configuration.
        
        Args:
            **kwargs: Override default values. Supports nested overrides using double underscore notation
                     (e.g., "create_env_fn__env_name": "CartPole-v1")
            
        Returns:
            SyncDataCollectorConfig with default values, overridden by kwargs
        """
        from torchrl.trainers.algorithms.configs.envs import GymEnvConfig
        from tensordict import TensorDict

        # Unflatten the kwargs using TensorDict to understand what the user wants
        kwargs_td = TensorDict(kwargs)
        unflattened_kwargs = kwargs_td.unflatten_keys("__").to_dict()

        # Create configs with nested overrides applied
        env_overrides = unflattened_kwargs.get("create_env_fn", {})
        env_cfg = GymEnvConfig.default_config(**env_overrides)

        defaults = {
            "create_env_fn": env_cfg,
            "policy": unflattened_kwargs.get("policy", None),  # Will be set when instantiating
            "policy_factory": unflattened_kwargs.get("policy_factory", None),
            "frames_per_batch": unflattened_kwargs.get("frames_per_batch", 1000),
            "total_frames": unflattened_kwargs.get("total_frames", 1_000_000),
            "device": unflattened_kwargs.get("device", None),
            "storing_device": unflattened_kwargs.get("storing_device", None),
            "policy_device": unflattened_kwargs.get("policy_device", None),
            "env_device": unflattened_kwargs.get("env_device", None),
            "create_env_kwargs": unflattened_kwargs.get("create_env_kwargs", None),
            "max_frames_per_traj": unflattened_kwargs.get("max_frames_per_traj", None),
            "reset_at_each_iter": unflattened_kwargs.get("reset_at_each_iter", False),
            "postproc": unflattened_kwargs.get("postproc", None),
            "split_trajs": unflattened_kwargs.get("split_trajs", False),
            "exploration_type": unflattened_kwargs.get("exploration_type", "RANDOM"),
            "return_same_td": unflattened_kwargs.get("return_same_td", False),
            "interruptor": unflattened_kwargs.get("interruptor", None),
            "set_truncated": unflattened_kwargs.get("set_truncated", False),
            "use_buffers": unflattened_kwargs.get("use_buffers", False),
            "replay_buffer": unflattened_kwargs.get("replay_buffer", None),
            "extend_buffer": unflattened_kwargs.get("extend_buffer", False),
            "trust_policy": unflattened_kwargs.get("trust_policy", True),
            "compile_policy": unflattened_kwargs.get("compile_policy", None),
            "cudagraph_policy": unflattened_kwargs.get("cudagraph_policy", None),
            "no_cuda_sync": unflattened_kwargs.get("no_cuda_sync", False),
            "_partial_": True,
        }
        
        return cls(**defaults)


@dataclass
class AsyncDataCollectorConfig(DataCollectorConfig):
    # Copy the args of aSyncDataCollector here
    create_env_fn: ConfigBase = field(
        default_factory=partial(EnvConfig, _partial_=True)
    )
    policy: Any = None
    policy_factory: Any = None
    frames_per_batch: int | None = None
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
    _target_: str = "torchrl.collectors.collectors.aSyncDataCollector"

    def __post_init__(self):
        self.create_env_fn._partial_ = True
        if self.policy_factory is not None:
            self.policy_factory._partial_ = True


@dataclass
class MultiSyncDataCollectorConfig(DataCollectorConfig):
    # Copy the args of _MultiDataCollector here
    create_env_fn: list[ConfigBase] | None = None
    policy: Any = None
    policy_factory: Any = None
    frames_per_batch: int | None = None
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
    _target_: str = "torchrl.collectors.collectors.MultiSyncDataCollector"

    def __post_init__(self):
        for env_cfg in self.create_env_fn:
            env_cfg._partial_ = True
        if self.policy_factory is not None:
            self.policy_factory._partial_ = True


@dataclass
class MultiaSyncDataCollectorConfig(DataCollectorConfig):
    create_env_fn: list[ConfigBase] | None = None
    policy: Any = None
    policy_factory: Any = None
    frames_per_batch: int | None = None
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
    _target_: str = "torchrl.collectors.collectors.MultiaSyncDataCollector"

    def __post_init__(self):
        for env_cfg in self.create_env_fn:
            env_cfg._partial_ = True
        if self.policy_factory is not None:
            self.policy_factory._partial_ = True
