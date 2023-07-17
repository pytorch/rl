# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from .common import EnvBase, EnvMetaData, make_tensordict
from .env_creator import EnvCreator, get_env_metadata
from .gym_like import default_info_dict_reader, GymLikeEnv
from .model_based import ModelBasedEnvBase
from .transforms import (
    BinarizeReward,
    CatFrames,
    CatTensors,
    CenterCrop,
    Compose,
    DiscreteActionProjection,
    DoubleToFloat,
    ExcludeTransform,
    FiniteTensorDictCheck,
    FlattenObservation,
    FrameSkipTransform,
    GrayScale,
    gSDENoise,
    InitTracker,
    KLRewardTransform,
    NoopResetEnv,
    ObservationNorm,
    ObservationTransform,
    PinMemoryTransform,
    R3MTransform,
    RandomCropTensorDict,
    RenameTransform,
    Resize,
    Reward2GoTransform,
    RewardClipping,
    RewardScaling,
    RewardSum,
    SelectTransform,
    SqueezeTransform,
    StepCounter,
    TargetReturn,
    TensorDictPrimer,
    TimeMaxPool,
    ToTensorImage,
    Transform,
    TransformedEnv,
    UnsqueezeTransform,
    VC1Transform,
    VecNorm,
    VIPRewardTransform,
    VIPTransform,
)
from .utils import (
    check_env_specs,
    exploration_mode,
    exploration_type,
    ExplorationType,
    make_composite_from_td,
    set_exploration_mode,
    set_exploration_type,
    step_mdp,
)
from .vec_env import MultiThreadedEnv, ParallelEnv, SerialEnv
