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


def _make_sync_collector_with_cross_references(*args, **kwargs):
    """Helper function to create a SyncDataCollector with automatic cross-reference resolution.
    
    This function automatically resolves cross-references to environment and policy
    from the structured config, allowing users to write configs that automatically
    connect components without manual instantiation.
    """
    from hydra.utils import instantiate
    from torchrl.collectors.collectors import SyncDataCollector
    
    # Extract collector-specific parameters
    create_env_fn = kwargs.pop("create_env_fn", None)
    policy = kwargs.pop("policy", None)
    policy_factory = kwargs.pop("policy_factory", None)
    
    # Check if we have a parent config passed through kwargs
    parent_config = kwargs.pop("_parent_config", None)
    
    # Resolve cross-references from parent config if available
    if parent_config is not None:
        # Resolve environment if not explicitly provided
        if create_env_fn is None and hasattr(parent_config, "env"):
            create_env_fn = parent_config.env
        
        # Resolve policy if not explicitly provided
        if policy is None and hasattr(parent_config, "model"):
            policy = parent_config.model
    
    # Create a callable from the environment config if it's a config object
    if create_env_fn is not None and hasattr(create_env_fn, "_target_"):
        # Create a callable that instantiates the environment config
        env_config = create_env_fn
        def create_env_callable(**kwargs):
            return instantiate(env_config, **kwargs)
        create_env_fn = create_env_callable
    elif create_env_fn is not None and hasattr(create_env_fn, "_partial_") and create_env_fn._partial_:
        # If it's a partial config, create a callable
        env_config = create_env_fn
        def create_env_callable(**kwargs):
            return instantiate(env_config, **kwargs)
        create_env_fn = create_env_callable
    
    # Instantiate the policy if it's a config object
    if policy is not None and hasattr(policy, "_target_"):
        policy = instantiate(policy)
    elif policy is not None and hasattr(policy, "_partial_") and policy._partial_:
        # If it's a partial config, instantiate it
        policy = instantiate(policy)
    
    # Create the collector
    return SyncDataCollector(
        create_env_fn=create_env_fn,
        policy=policy,
        policy_factory=policy_factory,
        **kwargs
    )


def instantiate_collector_with_cross_references(config):
    """Utility function to instantiate a collector with automatic cross-reference resolution.
    
    This function takes a full config object and automatically resolves cross-references
    between the collector, environment, and policy components.
    
    Args:
        config: The full configuration object containing env, model, network, and collector
        
    Returns:
        An instantiated collector with properly resolved environment and policy
    """
    from hydra.utils import instantiate
    
    # Create a copy of the collector config with cross-references resolved
    collector_config = config.collector.copy()
    
    # Set the environment and policy references
    collector_config.create_env_fn = config.env
    collector_config.policy = config.model
    
    # Instantiate the collector
    return instantiate(collector_config)


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
    _target_: str = "torchrl.trainers.algorithms.configs.collectors._make_sync_collector_with_cross_references"
    _partial_: bool = False

    def __post_init__(self):
        self.create_env_fn._partial_ = True
        if self.policy_factory is not None:
            self.policy_factory._partial_ = True


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
