# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from hydra.core.config_store import ConfigStore

from torchrl.trainers.algorithms.configs.collectors import (
    AsyncDataCollectorConfig,
    DataCollectorConfig,
    MultiaSyncDataCollectorConfig,
    MultiSyncDataCollectorConfig,
    SyncDataCollectorConfig,
)

from torchrl.trainers.algorithms.configs.common import Config, ConfigBase
from torchrl.trainers.algorithms.configs.data import (
    LazyMemmapStorageConfig,
    LazyStackStorageConfig,
    LazyTensorStorageConfig,
    ListStorageConfig,
    PrioritizedSamplerConfig,
    RandomSamplerConfig,
    ReplayBufferConfig,
    RoundRobinWriterConfig,
    SamplerWithoutReplacementConfig,
    SliceSamplerConfig,
    SliceSamplerWithoutReplacementConfig,
    StorageEnsembleConfig,
    StorageEnsembleWriterConfig,
    TensorDictReplayBufferConfig,
    TensorStorageConfig,
)
from torchrl.trainers.algorithms.configs.envs import (
    BatchedEnvConfig,
    EnvConfig,
    GymEnvConfig,
    TransformedEnvConfig,
)
from torchrl.trainers.algorithms.configs.logging import (
    CSVLoggerConfig,
    LoggerConfig,
    TensorboardLoggerConfig,
    WandbLoggerConfig,
)
from torchrl.trainers.algorithms.configs.modules import (
    ConvNetConfig,
    MLPConfig,
    ModelConfig,
    TanhNormalModelConfig,
    TensorDictModuleConfig,
    ValueModelConfig,
)
from torchrl.trainers.algorithms.configs.transforms import (
    ComposeConfig,
    NoopResetEnvConfig,
    TransformConfig,
)
from torchrl.trainers.algorithms.configs.objectives import LossConfig, PPOLossConfig
from torchrl.trainers.algorithms.configs.trainers import PPOTrainerConfig, TrainerConfig
from torchrl.trainers.algorithms.configs.utils import AdamConfig

__all__ = [
    "AsyncDataCollectorConfig",
    "BatchedEnvConfig",
    "CSVLoggerConfig",
    "LoggerConfig",
    "TensorboardLoggerConfig",
    "WandbLoggerConfig",
    "StorageEnsembleWriterConfig",
    "SamplerWithoutReplacementConfig",
    "SliceSamplerWithoutReplacementConfig",
    "ConfigBase",
    "ComposeConfig",
    "ConvNetConfig",
    "DataCollectorConfig",
    "EnvConfig",
    "GymEnvConfig",
    "LazyMemmapStorageConfig",
    "LazyStackStorageConfig",
    "LazyTensorStorageConfig",
    "ListStorageConfig",
    "LossConfig",
    "MLPConfig",
    "ModelConfig",
    "MultiSyncDataCollectorConfig",
    "MultiaSyncDataCollectorConfig",
    "NoopResetEnvConfig",
    "PPOTrainerConfig",
    "PPOLossConfig",
    "PrioritizedSamplerConfig",
    "RandomSamplerConfig",
    "ReplayBufferConfig",
    "RoundRobinWriterConfig",
    "SliceSamplerConfig",
    "StorageEnsembleConfig",
    "AdamConfig",
    "SyncDataCollectorConfig",
    "TanhNormalModelConfig",
    "TensorDictModuleConfig",
    "TensorDictReplayBufferConfig",
    "TensorStorageConfig",
    "TrainerConfig",
    "TransformConfig",
    "TransformedEnvConfig",
    "ValueModelConfig",
    "ValueModelConfig",
]

# Register configurations with Hydra ConfigStore
cs = ConfigStore.instance()

# Main config
cs.store(name="config", node=Config)

# Environment configs
cs.store(group="env", name="gym", node=GymEnvConfig)
cs.store(group="env", name="batched_env", node=BatchedEnvConfig)
cs.store(group="env", name="transformed_env", node=TransformedEnvConfig)

# Network configs
cs.store(group="network", name="mlp", node=MLPConfig)
cs.store(group="network", name="convnet", node=ConvNetConfig)

# Model configs
cs.store(group="network", name="tensordict_module", node=TensorDictModuleConfig)
cs.store(group="model", name="tanh_normal", node=TanhNormalModelConfig)
cs.store(group="model", name="value", node=ValueModelConfig)

# Transform configs
cs.store(group="transform", name="noop_reset", node=NoopResetEnvConfig)
cs.store(group="transform", name="compose", node=ComposeConfig)

# Loss configs
cs.store(group="loss", name="base", node=LossConfig)

# Replay buffer configs
cs.store(group="replay_buffer", name="base", node=ReplayBufferConfig)
cs.store(group="replay_buffer", name="tensordict", node=TensorDictReplayBufferConfig)
cs.store(group="sampler", name="random", node=RandomSamplerConfig)
cs.store(
    group="sampler", name="without_replacement", node=SamplerWithoutReplacementConfig
)
cs.store(group="sampler", name="prioritized", node=PrioritizedSamplerConfig)
cs.store(group="sampler", name="slice", node=SliceSamplerConfig)
cs.store(
    group="sampler",
    name="slice_without_replacement",
    node=SliceSamplerWithoutReplacementConfig,
)
cs.store(group="storage", name="lazy_stack", node=LazyStackStorageConfig)
cs.store(group="storage", name="list", node=ListStorageConfig)
cs.store(group="storage", name="tensor", node=TensorStorageConfig)
cs.store(group="storage", name="lazy_tensor", node=LazyTensorStorageConfig)
cs.store(group="storage", name="lazy_memmap", node=LazyMemmapStorageConfig)
cs.store(group="writer", name="round_robin", node=RoundRobinWriterConfig)

# Collector configs
cs.store(group="collector", name="sync", node=SyncDataCollectorConfig)
cs.store(group="collector", name="async", node=AsyncDataCollectorConfig)
cs.store(group="collector", name="multi_sync", node=MultiSyncDataCollectorConfig)
cs.store(group="collector", name="multi_async", node=MultiaSyncDataCollectorConfig)

# Trainer configs
cs.store(group="trainer", name="base", node=TrainerConfig)
cs.store(group="trainer", name="ppo", node=PPOTrainerConfig)

# Loss configs
cs.store(group="loss", name="ppo", node=PPOLossConfig)

# Optimizer configs
cs.store(group="optimizer", name="adam", node=AdamConfig)

# Logger configs
cs.store(group="logger", name="wandb", node=WandbLoggerConfig)
cs.store(group="logger", name="tensorboard", node=TensorboardLoggerConfig)
cs.store(group="logger", name="csv", node=CSVLoggerConfig)
cs.store(group="logger", name="base", node=LoggerConfig)
