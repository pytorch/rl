# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from .batched_envs import ParallelEnv, SerialEnv
from .common import EnvBase, EnvMetaData, make_tensordict
from .custom import ChessEnv, LLMEnv, LLMHashingEnv, PendulumEnv, TicTacToeEnv
from .env_creator import env_creator, EnvCreator, get_env_metadata
from .gym_like import default_info_dict_reader, GymLikeEnv
from .libs import (
    BraxEnv,
    BraxWrapper,
    DMControlEnv,
    DMControlWrapper,
    gym_backend,
    GymEnv,
    GymWrapper,
    HabitatEnv,
    IsaacGymEnv,
    IsaacGymWrapper,
    JumanjiEnv,
    JumanjiWrapper,
    MeltingpotEnv,
    MeltingpotWrapper,
    MOGymEnv,
    MOGymWrapper,
    MultiThreadedEnv,
    MultiThreadedEnvWrapper,
    OpenMLEnv,
    OpenSpielEnv,
    OpenSpielWrapper,
    PettingZooEnv,
    PettingZooWrapper,
    register_gym_spec_conversion,
    RoboHiveEnv,
    set_gym_backend,
    SMACv2Env,
    SMACv2Wrapper,
    UnityMLAgentsEnv,
    UnityMLAgentsWrapper,
    VmasEnv,
    VmasWrapper,
)
from .model_based import DreamerDecoder, DreamerEnv, ModelBasedEnvBase
from .transforms import (
    ActionDiscretizer,
    ActionMask,
    as_nested_tensor,
    as_padded_tensor,
    AutoResetEnv,
    AutoResetTransform,
    BatchSizeTransform,
    BinarizeReward,
    BurnInTransform,
    CatFrames,
    CatTensors,
    CenterCrop,
    ClipTransform,
    Compose,
    ConditionalSkip,
    Crop,
    DataLoadingPrimer,
    DeviceCastTransform,
    DiscreteActionProjection,
    DoubleToFloat,
    DTypeCastTransform,
    EndOfLifeTransform,
    ExcludeTransform,
    FiniteTensorDictCheck,
    FlattenObservation,
    FrameSkipTransform,
    GrayScale,
    gSDENoise,
    Hash,
    InitTracker,
    KLRewardTransform,
    LineariseRewards,
    MultiAction,
    MultiStepTransform,
    NoopResetEnv,
    ObservationNorm,
    ObservationTransform,
    PermuteTransform,
    PinMemoryTransform,
    R3MTransform,
    RandomCropTensorDict,
    RemoveEmptySpecs,
    RenameTransform,
    Resize,
    Reward2GoTransform,
    RewardClipping,
    RewardScaling,
    RewardSum,
    SelectTransform,
    SignTransform,
    SqueezeTransform,
    Stack,
    StepCounter,
    TargetReturn,
    TensorDictPrimer,
    TimeMaxPool,
    Timer,
    Tokenizer,
    ToTensorImage,
    TrajCounter,
    Transform,
    TransformedEnv,
    UnaryTransform,
    UnsqueezeTransform,
    VC1Transform,
    VecGymEnvTransform,
    VecNorm,
    VecNormV2,
    VIPRewardTransform,
    VIPTransform,
)
from .utils import (
    check_env_specs,
    check_marl_grouping,
    exploration_type,
    ExplorationType,
    get_available_libraries,
    make_composite_from_td,
    MarlGroupMapType,
    RandomPolicy,
    set_exploration_type,
    step_mdp,
    terminated_or_truncated,
)

__all__ = [
    "ActionDiscretizer",
    "ActionMask",
    "VecNormV2",
    "AutoResetEnv",
    "AutoResetTransform",
    "BatchSizeTransform",
    "BinarizeReward",
    "BraxEnv",
    "BraxWrapper",
    "BurnInTransform",
    "CatFrames",
    "CatTensors",
    "CenterCrop",
    "ChessEnv",
    "ClipTransform",
    "Compose",
    "ConditionalSkip",
    "Crop",
    "DMControlEnv",
    "DMControlWrapper",
    "DTypeCastTransform",
    "DataLoadingPrimer",
    "DeviceCastTransform",
    "DiscreteActionProjection",
    "DoubleToFloat",
    "DreamerDecoder",
    "DreamerEnv",
    "EndOfLifeTransform",
    "EnvBase",
    "EnvCreator",
    "EnvMetaData",
    "ExcludeTransform",
    "ExplorationType",
    "FiniteTensorDictCheck",
    "FlattenObservation",
    "FrameSkipTransform",
    "GrayScale",
    "GymEnv",
    "GymLikeEnv",
    "GymWrapper",
    "HabitatEnv",
    "Hash",
    "InitTracker",
    "IsaacGymEnv",
    "IsaacGymWrapper",
    "JumanjiEnv",
    "JumanjiWrapper",
    "KLRewardTransform",
    "LLMEnv",
    "LLMHashingEnv",
    "LineariseRewards",
    "MOGymEnv",
    "MOGymWrapper",
    "MarlGroupMapType",
    "MeltingpotEnv",
    "MeltingpotWrapper",
    "ModelBasedEnvBase",
    "MultiAction",
    "MultiStepTransform",
    "MultiThreadedEnv",
    "MultiThreadedEnvWrapper",
    "NoopResetEnv",
    "ObservationNorm",
    "ObservationTransform",
    "OpenMLEnv",
    "OpenSpielEnv",
    "OpenSpielWrapper",
    "ParallelEnv",
    "PendulumEnv",
    "PermuteTransform",
    "PettingZooEnv",
    "PettingZooWrapper",
    "PinMemoryTransform",
    "R3MTransform",
    "RandomCropTensorDict",
    "RandomPolicy",
    "RemoveEmptySpecs",
    "RenameTransform",
    "Resize",
    "Reward2GoTransform",
    "RewardClipping",
    "RewardScaling",
    "RewardSum",
    "RoboHiveEnv",
    "SMACv2Env",
    "SMACv2Wrapper",
    "SelectTransform",
    "SerialEnv",
    "SignTransform",
    "SqueezeTransform",
    "Stack",
    "StepCounter",
    "TargetReturn",
    "TensorDictPrimer",
    "TicTacToeEnv",
    "TimeMaxPool",
    "Timer",
    "ToTensorImage",
    "Tokenizer",
    "TrajCounter",
    "Transform",
    "TransformedEnv",
    "UnaryTransform",
    "UnityMLAgentsEnv",
    "UnityMLAgentsWrapper",
    "UnsqueezeTransform",
    "VC1Transform",
    "VIPRewardTransform",
    "VIPTransform",
    "VecGymEnvTransform",
    "VecNorm",
    "VmasEnv",
    "VmasWrapper",
    "as_nested_tensor",
    "as_padded_tensor",
    "check_env_specs",
    "check_marl_grouping",
    "default_info_dict_reader",
    "env_creator",
    "exploration_type",
    "gSDENoise",
    "get_available_libraries",
    "get_env_metadata",
    "gym_backend",
    "make_composite_from_td",
    "make_tensordict",
    "register_gym_spec_conversion",
    "set_exploration_type",
    "set_gym_backend",
    "step_mdp",
    "terminated_or_truncated",
]
