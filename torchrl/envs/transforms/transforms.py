# Copyright (c) Meta Plobs_dictnc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Backward-compatible re-export hub for transforms.

The implementations have been split into per-category modules
(``_base.py``, ``_observation.py``, ...). Importing from
``torchrl.envs.transforms`` (the public API) or from this module
(legacy path) both continue to work unchanged.
"""
from __future__ import annotations

from torchrl.envs.transforms._action import (
    ActionDiscretizer,
    ActionMask,
    DiscreteActionProjection,
    MultiAction,
)

from torchrl.envs.transforms._base import (  # noqa: F401  # noqa: F401  # noqa: F401  # noqa: F401  # noqa: F401  # noqa: F401
    _apply_to_composite,
    _apply_to_composite_inv,
    _CallableTransform,
    _has_tv,
    AutoResetEnv,
    Compose,
    FORWARD_NOT_IMPLEMENTED,
    IMAGE_KEYS,
    ObservationTransform,
    Transform,
    TransformedEnv,
)
from torchrl.envs.transforms._clip import ClipTransform, ExpandAs
from torchrl.envs.transforms._device import (
    DeviceCastTransform,
    DoubleToFloat,
    DTypeCastTransform,
)
from torchrl.envs.transforms._env import (
    AutoResetTransform,
    BatchSizeTransform,
    BurnInTransform,
    FrameSkipTransform,
    gSDENoise,
    InitTracker,
    NoopResetEnv,
    RandomTruncationTransform,
    StepCounter,
    TensorDictPrimer,
    TrajCounter,
)
from torchrl.envs.transforms._keys import (
    ExcludeTransform,
    FlattenTensorDict,
    RemoveEmptySpecs,
    RenameTransform,
    SelectTransform,
)
from torchrl.envs.transforms._misc import (  # noqa: F401
    _InvertTransform,
    ConditionalPolicySwitch,
    ConditionalSkip,
    FiniteTensorDictCheck,
    PinMemoryTransform,
    RandomCropTensorDict,
    TimeMaxPool,
    VecGymEnvTransform,
)
from torchrl.envs.transforms._normalization import (
    ObservationNorm,
    RewardScaling,
    VecNorm,
)
from torchrl.envs.transforms._observation import (
    CatFrames,
    CenterCrop,
    Crop,
    FlattenObservation,
    GrayScale,
    PermuteTransform,
    Resize,
    SqueezeTransform,
    ToTensorImage,
    UnsqueezeTransform,
)
from torchrl.envs.transforms._reward import (
    BinarizeReward,
    LineariseRewards,
    Reward2GoTransform,
    RewardClipping,
    RewardSum,
    SignTransform,
    TargetReturn,
)
from torchrl.envs.transforms._tensor import (
    CatTensors,
    Hash,
    Stack,
    Tokenizer,
    UnaryTransform,
)
from torchrl.envs.transforms._timer import Timer

__all__ = [
    "ActionDiscretizer",
    "ActionMask",
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
    "DeviceCastTransform",
    "DiscreteActionProjection",
    "DoubleToFloat",
    "ExcludeTransform",
    "ExpandAs",
    "FiniteTensorDictCheck",
    "FlattenObservation",
    "FlattenTensorDict",
    "FrameSkipTransform",
    "GrayScale",
    "Hash",
    "InitTracker",
    "LineariseRewards",
    "MultiAction",
    "NoopResetEnv",
    "ObservationNorm",
    "ObservationTransform",
    "PermuteTransform",
    "PinMemoryTransform",
    "RandomCropTensorDict",
    "RandomTruncationTransform",
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
    "TargetReturn",
    "TensorDictPrimer",
    "TimeMaxPool",
    "Timer",
    "ToTensorImage",
    "Tokenizer",
    "TrajCounter",
    "Transform",
    "TransformedEnv",
    "UnaryTransform",
    "UnsqueezeTransform",
    "VecGymEnvTransform",
    "VecNorm",
    "gSDENoise",
]
