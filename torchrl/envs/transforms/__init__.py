# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import importlib

from ._primitive import (
    MacroAction,
    MacroPrimitive,
    MacroPrimitiveTransform,
    TargetMacroAction,
)
from .gym_transforms import EndOfLifeTransform
from .mean_action_selector import MeanActionSelector
from .module import ModuleTransform
from .r3m import R3MTransform
from .ray_service import RayTransform
from .rb_transforms import MultiStepTransform, NextStateReconstructor
from .transforms import (
    ActionChunkExecutor,
    ActionChunkTransform,
    ActionDiscretizer,
    ActionMask,
    ActionScaling,
    ActionTokenizerTransform,
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
    ConditionalPolicySwitch,
    ConditionalSkip,
    Crop,
    DecodeVideoTransform,
    DeviceCastTransform,
    DiscreteActionProjection,
    DoubleToFloat,
    DTypeCastTransform,
    ExcludeTransform,
    ExpandAs,
    FiniteTensorDictCheck,
    FlattenAction,
    FlattenObservation,
    FrameSkipTransform,
    GrayScale,
    gSDENoise,
    Hash,
    InitTracker,
    LineariseRewards,
    MultiAction,
    NextObservationDelta,
    NoopResetEnv,
    ObservationNorm,
    ObservationTransform,
    PermuteTransform,
    PinMemoryTransform,
    RandomCropTensorDict,
    RandomTruncationTransform,
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
    SuccessReward,
    TargetReturn,
    TensorDictPrimer,
    TerminateTransform,
    TimeMaxPool,
    Timer,
    Tokenizer,
    ToTensorImage,
    TrajCounter,
    Transform,
    TransformedEnv,
    UnaryTransform,
    UnsqueezeTransform,
    VecGymEnvTransform,
    VecNorm,
)
from .vc1 import VC1Transform
from .vecnorm import VecNormV2
from .vip import VIPRewardTransform, VIPTransform

_UR_PRIMITIVE_EXPORTS = {
    "RobotMacroAction",
    "RobotMacroActionMode",
    "URScriptPrimitive",
    "URScriptPrimitiveTransform",
}
_HUMANOID_PRIMITIVE_EXPORTS = {"HumanoidMacroAction"}
_SATELLITE_PRIMITIVE_EXPORTS = {"SatelliteMacroAction", "SatelliteAttitudeTransform"}


def __getattr__(name: str):
    if name in _UR_PRIMITIVE_EXPORTS:
        module = importlib.import_module("torchrl.envs.custom.mujoco._ur_primitives")
        value = getattr(module, name)
        globals()[name] = value
        return value
    if name in _HUMANOID_PRIMITIVE_EXPORTS:
        module = importlib.import_module(
            "torchrl.envs.custom.mujoco._humanoid_primitives"
        )
        value = getattr(module, name)
        globals()[name] = value
        return value
    if name in _SATELLITE_PRIMITIVE_EXPORTS:
        module = importlib.import_module(
            "torchrl.envs.custom.mujoco._satellite_primitives"
        )
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "ActionChunkExecutor",
    "ActionChunkTransform",
    "ActionDiscretizer",
    "ActionMask",
    "ActionScaling",
    "ActionTokenizerTransform",
    "AutoResetEnv",
    "AutoResetTransform",
    "BatchSizeTransform",
    "BinarizeReward",
    "BurnInTransform",
    "CatFrames",
    "CatTensors",
    "CenterCrop",
    "ClipTransform",
    "Compose",
    "ConditionalPolicySwitch",
    "ConditionalSkip",
    "Crop",
    "DTypeCastTransform",
    "DecodeVideoTransform",
    "DeviceCastTransform",
    "DiscreteActionProjection",
    "DoubleToFloat",
    "EndOfLifeTransform",
    "ExcludeTransform",
    "ExpandAs",
    "FiniteTensorDictCheck",
    "FlattenAction",
    "FlattenObservation",
    "FrameSkipTransform",
    "GrayScale",
    "Hash",
    "HumanoidMacroAction",
    "InitTracker",
    "LineariseRewards",
    "MacroAction",
    "MacroPrimitive",
    "MacroPrimitiveTransform",
    "TargetMacroAction",
    "RobotMacroAction",
    "RobotMacroActionMode",
    "SatelliteMacroAction",
    "SatelliteAttitudeTransform",
    "MeanActionSelector",
    "ModuleTransform",
    "MultiAction",
    "MultiStepTransform",
    "NextObservationDelta",
    "NextStateReconstructor",
    "NoopResetEnv",
    "ObservationNorm",
    "ObservationTransform",
    "PermuteTransform",
    "PinMemoryTransform",
    "R3MTransform",
    "RandomCropTensorDict",
    "RandomTruncationTransform",
    "RayTransform",
    "RemoveEmptySpecs",
    "RenameTransform",
    "Resize",
    "Reward2GoTransform",
    "RewardClipping",
    "RewardScaling",
    "RewardSum",
    "SelectTransform",
    "SignTransform",
    "SqueezeTransform",
    "Stack",
    "StepCounter",
    "SuccessReward",
    "TargetReturn",
    "TensorDictPrimer",
    "TerminateTransform",
    "TimeMaxPool",
    "Timer",
    "ToTensorImage",
    "Tokenizer",
    "TrajCounter",
    "Transform",
    "TransformedEnv",
    "URScriptPrimitive",
    "URScriptPrimitiveTransform",
    "UnaryTransform",
    "UnsqueezeTransform",
    "VC1Transform",
    "VIPRewardTransform",
    "VIPTransform",
    "VecGymEnvTransform",
    "VecNorm",
    "VecNormV2",
    "gSDENoise",
]
