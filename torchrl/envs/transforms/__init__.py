# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from .gym_transforms import EndOfLifeTransform
from .r3m import R3MTransform
from .rb_transforms import MultiStepTransform
from .rlhf import KLRewardTransform
from .transforms import (
    ActionDiscretizer,
    ActionMask,
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
    Crop,
    DeviceCastTransform,
    DiscreteActionProjection,
    DoubleToFloat,
    DTypeCastTransform,
    ExcludeTransform,
    FiniteTensorDictCheck,
    FlattenObservation,
    FrameSkipTransform,
    GrayScale,
    gSDENoise,
    InitTracker,
    NoopResetEnv,
    ObservationNorm,
    ObservationTransform,
    PermuteTransform,
    PinMemoryTransform,
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
    StepCounter,
    TargetReturn,
    TensorDictPrimer,
    TimeMaxPool,
    ToTensorImage,
    TrajCounter,
    Transform,
    TransformedEnv,
    UnsqueezeTransform,
    VecGymEnvTransform,
    VecNorm,
)
from .vc1 import VC1Transform
from .vip import VIPRewardTransform, VIPTransform
