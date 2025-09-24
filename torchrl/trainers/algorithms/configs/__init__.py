# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import sys

# Check for hydra/omegaconf availability - required for config system
try:
    import hydra  # noqa: F401
    import omegaconf  # noqa: F401
    from hydra.core.config_store import ConfigStore

    _has_hydra = True
except ImportError as e:
    raise ImportError(
        "The TorchRL configuration system requires hydra-core and omegaconf. "
        "Please install them with: pip install 'torchrl[utils]' or pip install hydra-core omegaconf"
    ) from e

from torchrl.trainers.algorithms.configs.collectors import (
    AsyncDataCollectorConfig,
    DataCollectorConfig,
    MultiaSyncDataCollectorConfig,
    MultiSyncDataCollectorConfig,
    SyncDataCollectorConfig,
)

from torchrl.trainers.algorithms.configs.common import ConfigBase
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
    TransformedEnvConfig,
)
from torchrl.trainers.algorithms.configs.envs_libs import (
    BraxEnvConfig,
    DMControlEnvConfig,
    EnvLibsConfig,
    GymEnvConfig,
    HabitatEnvConfig,
    IsaacGymEnvConfig,
    JumanjiEnvConfig,
    MeltingpotEnvConfig,
    MOGymEnvConfig,
    MultiThreadedEnvConfig,
    OpenMLEnvConfig,
    OpenSpielEnvConfig,
    PettingZooEnvConfig,
    RoboHiveEnvConfig,
    SMACv2EnvConfig,
    UnityMLAgentsEnvConfig,
    VmasEnvConfig,
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
from torchrl.trainers.algorithms.configs.objectives import (
    HardUpdateConfig,
    LossConfig,
    PPOLossConfig,
    SACLossConfig,
    SoftUpdateConfig,
)
from torchrl.trainers.algorithms.configs.trainers import (
    PPOTrainerConfig,
    SACTrainerConfig,
    TrainerConfig,
)
from torchrl.trainers.algorithms.configs.transforms import (
    ActionDiscretizerConfig,
    ActionMaskConfig,
    AutoResetTransformConfig,
    BatchSizeTransformConfig,
    BinarizeRewardConfig,
    BurnInTransformConfig,
    CatFramesConfig,
    CatTensorsConfig,
    CenterCropConfig,
    ClipTransformConfig,
    ComposeConfig,
    ConditionalPolicySwitchConfig,
    ConditionalSkipConfig,
    CropConfig,
    DeviceCastTransformConfig,
    DiscreteActionProjectionConfig,
    DoubleToFloatConfig,
    DTypeCastTransformConfig,
    EndOfLifeTransformConfig,
    ExcludeTransformConfig,
    FiniteTensorDictCheckConfig,
    FlattenObservationConfig,
    FrameSkipTransformConfig,
    GrayScaleConfig,
    HashConfig,
    InitTrackerConfig,
    KLRewardTransformConfig,
    LineariseRewardsConfig,
    MultiActionConfig,
    MultiStepTransformConfig,
    NoopResetEnvConfig,
    ObservationNormConfig,
    PermuteTransformConfig,
    PinMemoryTransformConfig,
    R3MTransformConfig,
    RandomCropTensorDictConfig,
    RemoveEmptySpecsConfig,
    RenameTransformConfig,
    ResizeConfig,
    Reward2GoTransformConfig,
    RewardClippingConfig,
    RewardScalingConfig,
    RewardSumConfig,
    SelectTransformConfig,
    SignTransformConfig,
    SqueezeTransformConfig,
    StackConfig,
    StepCounterConfig,
    TargetReturnConfig,
    TensorDictPrimerConfig,
    TimeMaxPoolConfig,
    TimerConfig,
    TokenizerConfig,
    ToTensorImageConfig,
    TrajCounterConfig,
    TransformConfig,
    UnaryTransformConfig,
    UnsqueezeTransformConfig,
    VC1TransformConfig,
    VecGymEnvTransformConfig,
    VecNormConfig,
    VecNormV2Config,
    VIPRewardTransformConfig,
    VIPTransformConfig,
)
from torchrl.trainers.algorithms.configs.utils import (
    AdadeltaConfig,
    AdagradConfig,
    AdamaxConfig,
    AdamConfig,
    AdamWConfig,
    ASGDConfig,
    LBFGSConfig,
    LionConfig,
    NAdamConfig,
    RAdamConfig,
    RMSpropConfig,
    RpropConfig,
    SGDConfig,
    SparseAdamConfig,
)
from torchrl.trainers.algorithms.configs.weight_update import (
    DistributedWeightUpdaterConfig,
    MultiProcessedWeightUpdaterConfig,
    RayWeightUpdaterConfig,
    RemoteModuleWeightUpdaterConfig,
    RPCWeightUpdaterConfig,
    VanillaWeightUpdaterConfig,
    vLLMUpdaterConfig,
    WeightUpdaterConfig,
)

__all__ = [
    # Base configuration
    "ConfigBase",
    # Optimizers
    "AdamConfig",
    "AdamWConfig",
    "AdamaxConfig",
    "AdadeltaConfig",
    "AdagradConfig",
    "ASGDConfig",
    "LBFGSConfig",
    "LionConfig",
    "NAdamConfig",
    "RAdamConfig",
    "RMSpropConfig",
    "RpropConfig",
    "SGDConfig",
    "SparseAdamConfig",
    # Collectors
    "AsyncDataCollectorConfig",
    "DataCollectorConfig",
    "MultiSyncDataCollectorConfig",
    "MultiaSyncDataCollectorConfig",
    "SyncDataCollectorConfig",
    # Environments
    "BatchedEnvConfig",
    "EnvConfig",
    "TransformedEnvConfig",
    # Environment Libs
    "BraxEnvConfig",
    "DMControlEnvConfig",
    "EnvLibsConfig",
    "GymEnvConfig",
    "HabitatEnvConfig",
    "IsaacGymEnvConfig",
    "JumanjiEnvConfig",
    "MeltingpotEnvConfig",
    "MOGymEnvConfig",
    "MultiThreadedEnvConfig",
    "OpenMLEnvConfig",
    "OpenSpielEnvConfig",
    "PettingZooEnvConfig",
    "RoboHiveEnvConfig",
    "SMACv2EnvConfig",
    "UnityMLAgentsEnvConfig",
    "VmasEnvConfig",
    # Networks and Models
    "ConvNetConfig",
    "MLPConfig",
    "ModelConfig",
    "TanhNormalModelConfig",
    "TensorDictModuleConfig",
    "ValueModelConfig",
    # Transforms - Core
    "ActionDiscretizerConfig",
    "ActionMaskConfig",
    "AutoResetTransformConfig",
    "BatchSizeTransformConfig",
    "BinarizeRewardConfig",
    "BurnInTransformConfig",
    "CatFramesConfig",
    "CatTensorsConfig",
    "CenterCropConfig",
    "ClipTransformConfig",
    "ComposeConfig",
    "ConditionalPolicySwitchConfig",
    "ConditionalSkipConfig",
    "CropConfig",
    "DeviceCastTransformConfig",
    "DiscreteActionProjectionConfig",
    "DoubleToFloatConfig",
    "DTypeCastTransformConfig",
    "EndOfLifeTransformConfig",
    "ExcludeTransformConfig",
    "FiniteTensorDictCheckConfig",
    "FlattenObservationConfig",
    "FrameSkipTransformConfig",
    "GrayScaleConfig",
    "HashConfig",
    "InitTrackerConfig",
    "KLRewardTransformConfig",
    "LineariseRewardsConfig",
    "MultiActionConfig",
    "MultiStepTransformConfig",
    "NoopResetEnvConfig",
    "ObservationNormConfig",
    "PermuteTransformConfig",
    "PinMemoryTransformConfig",
    "RandomCropTensorDictConfig",
    "RemoveEmptySpecsConfig",
    "RenameTransformConfig",
    "ResizeConfig",
    "Reward2GoTransformConfig",
    "RewardClippingConfig",
    "RewardScalingConfig",
    "RewardSumConfig",
    "R3MTransformConfig",
    "SelectTransformConfig",
    "SignTransformConfig",
    "SqueezeTransformConfig",
    "StackConfig",
    "StepCounterConfig",
    "TargetReturnConfig",
    "TensorDictPrimerConfig",
    "TimerConfig",
    "TimeMaxPoolConfig",
    "ToTensorImageConfig",
    "TokenizerConfig",
    "TrajCounterConfig",
    "TransformConfig",
    "UnaryTransformConfig",
    "UnsqueezeTransformConfig",
    "VC1TransformConfig",
    "VecGymEnvTransformConfig",
    "VecNormConfig",
    "VecNormV2Config",
    "VIPRewardTransformConfig",
    "VIPTransformConfig",
    # Storage and Replay Buffers
    "LazyMemmapStorageConfig",
    "LazyStackStorageConfig",
    "LazyTensorStorageConfig",
    "ListStorageConfig",
    "ReplayBufferConfig",
    "RoundRobinWriterConfig",
    "StorageEnsembleConfig",
    "StorageEnsembleWriterConfig",
    "TensorDictReplayBufferConfig",
    "TensorStorageConfig",
    # Samplers
    "PrioritizedSamplerConfig",
    "RandomSamplerConfig",
    "SamplerWithoutReplacementConfig",
    "SliceSamplerConfig",
    "SliceSamplerWithoutReplacementConfig",
    # Losses
    "LossConfig",
    "PPOLossConfig",
    "SACLossConfig",
    # Trainers
    "PPOTrainerConfig",
    "SACTrainerConfig",
    "TrainerConfig",
    # Loggers
    "CSVLoggerConfig",
    "LoggerConfig",
    "TensorboardLoggerConfig",
    "WandbLoggerConfig",
    # Weight Updaters
    "WeightUpdaterConfig",
    "VanillaWeightUpdaterConfig",
    "MultiProcessedWeightUpdaterConfig",
    "RayWeightUpdaterConfig",
    "RemoteModuleWeightUpdaterConfig",
    "RPCWeightUpdaterConfig",
    "DistributedWeightUpdaterConfig",
    "vLLMUpdaterConfig",
]


def _register_configs():
    """Register configurations with Hydra ConfigStore.

    This function is called lazily to avoid GlobalHydra initialization issues
    during testing. It should be called explicitly when needed.
    """
    cs = ConfigStore.instance()

    # =============================================================================
    # Environment Configurations
    # =============================================================================

    # Core environment configs
    cs.store(group="env", name="gym", node=GymEnvConfig)
    cs.store(group="env", name="batched_env", node=BatchedEnvConfig)
    cs.store(group="env", name="transformed_env", node=TransformedEnvConfig)

    # Environment libs configs
    cs.store(group="env", name="brax", node=BraxEnvConfig)
    cs.store(group="env", name="dm_control", node=DMControlEnvConfig)
    cs.store(group="env", name="habitat", node=HabitatEnvConfig)
    cs.store(group="env", name="isaac_gym", node=IsaacGymEnvConfig)
    cs.store(group="env", name="jumanji", node=JumanjiEnvConfig)
    cs.store(group="env", name="meltingpot", node=MeltingpotEnvConfig)
    cs.store(group="env", name="mo_gym", node=MOGymEnvConfig)
    cs.store(group="env", name="multi_threaded", node=MultiThreadedEnvConfig)
    cs.store(group="env", name="openml", node=OpenMLEnvConfig)
    cs.store(group="env", name="openspiel", node=OpenSpielEnvConfig)
    cs.store(group="env", name="pettingzoo", node=PettingZooEnvConfig)
    cs.store(group="env", name="robohive", node=RoboHiveEnvConfig)
    cs.store(group="env", name="smacv2", node=SMACv2EnvConfig)
    cs.store(group="env", name="unity_mlagents", node=UnityMLAgentsEnvConfig)
    cs.store(group="env", name="vmas", node=VmasEnvConfig)

    # =============================================================================
    # Network and Model Configurations
    # =============================================================================

    # Network configs
    cs.store(group="network", name="mlp", node=MLPConfig)
    cs.store(group="network", name="convnet", node=ConvNetConfig)

    # Model configs
    cs.store(group="network", name="tensordict_module", node=TensorDictModuleConfig)
    cs.store(group="model", name="tanh_normal", node=TanhNormalModelConfig)
    cs.store(group="model", name="value", node=ValueModelConfig)

    # =============================================================================
    # Transform Configurations
    # =============================================================================

    # Core transforms
    cs.store(group="transform", name="noop_reset", node=NoopResetEnvConfig)
    cs.store(group="transform", name="step_counter", node=StepCounterConfig)
    cs.store(group="transform", name="compose", node=ComposeConfig)
    cs.store(group="transform", name="double_to_float", node=DoubleToFloatConfig)
    cs.store(group="transform", name="to_tensor_image", node=ToTensorImageConfig)
    cs.store(group="transform", name="clip", node=ClipTransformConfig)
    cs.store(group="transform", name="resize", node=ResizeConfig)
    cs.store(group="transform", name="center_crop", node=CenterCropConfig)
    cs.store(group="transform", name="crop", node=CropConfig)
    cs.store(
        group="transform", name="flatten_observation", node=FlattenObservationConfig
    )
    cs.store(group="transform", name="gray_scale", node=GrayScaleConfig)
    cs.store(group="transform", name="observation_norm", node=ObservationNormConfig)
    cs.store(group="transform", name="cat_frames", node=CatFramesConfig)
    cs.store(group="transform", name="reward_clipping", node=RewardClippingConfig)
    cs.store(group="transform", name="reward_scaling", node=RewardScalingConfig)
    cs.store(group="transform", name="binarize_reward", node=BinarizeRewardConfig)
    cs.store(group="transform", name="target_return", node=TargetReturnConfig)
    cs.store(group="transform", name="vec_norm", node=VecNormConfig)
    cs.store(group="transform", name="frame_skip", node=FrameSkipTransformConfig)
    cs.store(group="transform", name="device_cast", node=DeviceCastTransformConfig)
    cs.store(group="transform", name="dtype_cast", node=DTypeCastTransformConfig)
    cs.store(group="transform", name="unsqueeze", node=UnsqueezeTransformConfig)
    cs.store(group="transform", name="squeeze", node=SqueezeTransformConfig)
    cs.store(group="transform", name="permute", node=PermuteTransformConfig)
    cs.store(group="transform", name="cat_tensors", node=CatTensorsConfig)
    cs.store(group="transform", name="stack", node=StackConfig)
    cs.store(
        group="transform",
        name="discrete_action_projection",
        node=DiscreteActionProjectionConfig,
    )
    cs.store(group="transform", name="tensordict_primer", node=TensorDictPrimerConfig)
    cs.store(group="transform", name="pin_memory", node=PinMemoryTransformConfig)
    cs.store(group="transform", name="reward_sum", node=RewardSumConfig)
    cs.store(group="transform", name="exclude", node=ExcludeTransformConfig)
    cs.store(group="transform", name="select", node=SelectTransformConfig)
    cs.store(group="transform", name="time_max_pool", node=TimeMaxPoolConfig)
    cs.store(
        group="transform",
        name="random_crop_tensordict",
        node=RandomCropTensorDictConfig,
    )
    cs.store(group="transform", name="init_tracker", node=InitTrackerConfig)
    cs.store(group="transform", name="rename", node=RenameTransformConfig)
    cs.store(group="transform", name="reward2go", node=Reward2GoTransformConfig)
    cs.store(group="transform", name="action_mask", node=ActionMaskConfig)
    cs.store(group="transform", name="vec_gym_env", node=VecGymEnvTransformConfig)
    cs.store(group="transform", name="burn_in", node=BurnInTransformConfig)
    cs.store(group="transform", name="sign", node=SignTransformConfig)
    cs.store(group="transform", name="remove_empty_specs", node=RemoveEmptySpecsConfig)
    cs.store(group="transform", name="batch_size", node=BatchSizeTransformConfig)
    cs.store(group="transform", name="auto_reset", node=AutoResetTransformConfig)
    cs.store(group="transform", name="action_discretizer", node=ActionDiscretizerConfig)
    cs.store(group="transform", name="traj_counter", node=TrajCounterConfig)
    cs.store(group="transform", name="linearise_rewards", node=LineariseRewardsConfig)
    cs.store(group="transform", name="conditional_skip", node=ConditionalSkipConfig)
    cs.store(group="transform", name="multi_action", node=MultiActionConfig)
    cs.store(group="transform", name="timer", node=TimerConfig)
    cs.store(
        group="transform",
        name="conditional_policy_switch",
        node=ConditionalPolicySwitchConfig,
    )
    cs.store(
        group="transform",
        name="finite_tensordict_check",
        node=FiniteTensorDictCheckConfig,
    )
    cs.store(group="transform", name="unary", node=UnaryTransformConfig)
    cs.store(group="transform", name="hash", node=HashConfig)
    cs.store(group="transform", name="tokenizer", node=TokenizerConfig)

    # Specialized transforms
    cs.store(group="transform", name="end_of_life", node=EndOfLifeTransformConfig)
    cs.store(group="transform", name="multi_step", node=MultiStepTransformConfig)
    cs.store(group="transform", name="kl_reward", node=KLRewardTransformConfig)
    cs.store(group="transform", name="r3m", node=R3MTransformConfig)
    cs.store(group="transform", name="vc1", node=VC1TransformConfig)
    cs.store(group="transform", name="vip", node=VIPTransformConfig)
    cs.store(group="transform", name="vip_reward", node=VIPRewardTransformConfig)
    cs.store(group="transform", name="vec_norm_v2", node=VecNormV2Config)

    # =============================================================================
    # Loss Configurations
    # =============================================================================

    cs.store(group="loss", name="base", node=LossConfig)
    cs.store(group="loss", name="ppo", node=PPOLossConfig)
    cs.store(group="loss", name="sac", node=SACLossConfig)

    cs.store(group="target_net_updater", name="soft", node=SoftUpdateConfig)
    cs.store(group="target_net_updater", name="hard", node=HardUpdateConfig)

    # =============================================================================
    # Replay Buffer Configurations
    # =============================================================================

    cs.store(group="replay_buffer", name="base", node=ReplayBufferConfig)
    cs.store(
        group="replay_buffer", name="tensordict", node=TensorDictReplayBufferConfig
    )
    cs.store(group="sampler", name="random", node=RandomSamplerConfig)
    cs.store(
        group="sampler",
        name="without_replacement",
        node=SamplerWithoutReplacementConfig,
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

    # =============================================================================
    # Collector Configurations
    # =============================================================================

    cs.store(group="collector", name="sync", node=SyncDataCollectorConfig)
    cs.store(group="collector", name="async", node=AsyncDataCollectorConfig)
    cs.store(group="collector", name="multi_sync", node=MultiSyncDataCollectorConfig)
    cs.store(group="collector", name="multi_async", node=MultiaSyncDataCollectorConfig)

    # =============================================================================
    # Trainer Configurations
    # =============================================================================

    cs.store(group="trainer", name="base", node=TrainerConfig)
    cs.store(group="trainer", name="ppo", node=PPOTrainerConfig)
    cs.store(group="trainer", name="sac", node=SACTrainerConfig)

    # =============================================================================
    # Optimizer Configurations
    # =============================================================================

    cs.store(group="optimizer", name="adam", node=AdamConfig)
    cs.store(group="optimizer", name="adamw", node=AdamWConfig)
    cs.store(group="optimizer", name="adamax", node=AdamaxConfig)
    cs.store(group="optimizer", name="adadelta", node=AdadeltaConfig)
    cs.store(group="optimizer", name="adagrad", node=AdagradConfig)
    cs.store(group="optimizer", name="asgd", node=ASGDConfig)
    cs.store(group="optimizer", name="lbfgs", node=LBFGSConfig)
    cs.store(group="optimizer", name="lion", node=LionConfig)
    cs.store(group="optimizer", name="nadam", node=NAdamConfig)
    cs.store(group="optimizer", name="radam", node=RAdamConfig)
    cs.store(group="optimizer", name="rmsprop", node=RMSpropConfig)
    cs.store(group="optimizer", name="rprop", node=RpropConfig)
    cs.store(group="optimizer", name="sgd", node=SGDConfig)
    cs.store(group="optimizer", name="sparse_adam", node=SparseAdamConfig)

    # =============================================================================
    # Logger Configurations
    # =============================================================================

    cs.store(group="logger", name="wandb", node=WandbLoggerConfig)
    cs.store(group="logger", name="tensorboard", node=TensorboardLoggerConfig)
    cs.store(group="logger", name="csv", node=CSVLoggerConfig)
    cs.store(group="logger", name="base", node=LoggerConfig)

    # =============================================================================
    # Weight Updater Configurations
    # =============================================================================

    cs.store(group="weight_updater", name="base", node=WeightUpdaterConfig)
    cs.store(group="weight_updater", name="vanilla", node=VanillaWeightUpdaterConfig)
    cs.store(
        group="weight_updater",
        name="multiprocessed",
        node=MultiProcessedWeightUpdaterConfig,
    )
    cs.store(group="weight_updater", name="ray", node=RayWeightUpdaterConfig)
    cs.store(
        group="weight_updater",
        name="remote_module",
        node=RemoteModuleWeightUpdaterConfig,
    )
    cs.store(group="weight_updater", name="rpc", node=RPCWeightUpdaterConfig)
    cs.store(
        group="weight_updater", name="distributed", node=DistributedWeightUpdaterConfig
    )
    cs.store(group="weight_updater", name="vllm", node=vLLMUpdaterConfig)


if not sys.version_info < (3, 10):  #  type: ignore # noqa
    _register_configs()
