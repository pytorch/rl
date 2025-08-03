# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from omegaconf import MISSING
from hydra.core.config_store import ConfigStore

from torchrl.trainers.algorithms.configs.common import ConfigBase
from torchrl.trainers.algorithms.configs.collectors import SyncDataCollectorConfig
from torchrl.trainers.algorithms.configs.envs import EnvConfig, GymEnvConfig
from torchrl.trainers.algorithms.configs.modules import MLPConfig, TanhNormalModelConfig


@dataclass
class Config(ConfigBase):
    """Main configuration class that holds all components and enables cross-references.
    
    This config class allows components to reference each other automatically,
    enabling a clean API where users can write config files and directly instantiate
    objects without manual cross-referencing.
    """
    
    # Core components
    env: GymEnvConfig = field(default_factory=lambda: GymEnvConfig())
    network: MLPConfig = field(default_factory=lambda: MLPConfig())
    model: TanhNormalModelConfig = field(default_factory=lambda: TanhNormalModelConfig())
    collector: SyncDataCollectorConfig = field(default_factory=lambda: SyncDataCollectorConfig())
    
    # Optional components
    trainer: Any = None
    loss: Any = None
    replay_buffer: Any = None
    sampler: Any = None
    storage: Any = None
    writer: Any = None
    optimizer: Any = None
    logger: Any = None


# Register configurations with Hydra ConfigStore
cs = ConfigStore.instance()

# Main config
cs.store(name="config", node=Config)

# Environment configs
cs.store(group="env", name="gym", node=GymEnvConfig)
cs.store(group="env", name="batched_env", node=EnvConfig)

# Network configs
cs.store(group="network", name="mlp", node=MLPConfig)
cs.store(group="network", name="convnet", node=MLPConfig)

# Model configs
cs.store(group="network", name="tensordict_module", node=MLPConfig)
cs.store(group="model", name="tanh_normal", node=TanhNormalModelConfig)
cs.store(group="model", name="value", node=MLPConfig)

# Loss configs
cs.store(group="loss", name="base", node=ConfigBase)

# Replay buffer configs
cs.store(group="replay_buffer", name="base", node=ConfigBase)
cs.store(group="replay_buffer", name="tensordict", node=ConfigBase)

# Collector configs
cs.store(group="collector", name="sync", node=SyncDataCollectorConfig)
cs.store(group="collector", name="async", node=SyncDataCollectorConfig)
cs.store(group="collector", name="multi_sync", node=SyncDataCollectorConfig)
cs.store(group="collector", name="multi_async", node=SyncDataCollectorConfig)

# Trainer configs
cs.store(group="trainer", name="ppo", node=ConfigBase)

# Storage configs
cs.store(group="storage", name="tensor", node=ConfigBase)
cs.store(group="storage", name="list", node=ConfigBase)
cs.store(group="storage", name="lazy_tensor", node=ConfigBase)
cs.store(group="storage", name="lazy_memmap", node=ConfigBase)
cs.store(group="storage", name="lazy_stack", node=ConfigBase)

# Sampler configs
cs.store(group="sampler", name="random", node=ConfigBase)
cs.store(group="sampler", name="slice", node=ConfigBase)
cs.store(group="sampler", name="prioritized", node=ConfigBase)
cs.store(group="sampler", name="without_replacement", node=ConfigBase)

# Writer configs
cs.store(group="writer", name="tensor", node=ConfigBase)
cs.store(group="writer", name="round_robin", node=ConfigBase)

# Optimizer configs
cs.store(group="optimizer", name="adam", node=ConfigBase)

# Logger configs
cs.store(group="logger", name="csv", node=ConfigBase)
cs.store(group="logger", name="tensorboard", node=ConfigBase)
cs.store(group="logger", name="wandb", node=ConfigBase)

__all__ = [
    "Config",
    "ConfigBase",
    "SyncDataCollectorConfig",
    "EnvConfig",
    "GymEnvConfig",
    "MLPConfig",
    "TanhNormalModelConfig",
]
