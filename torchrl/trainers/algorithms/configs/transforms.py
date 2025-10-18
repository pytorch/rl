# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from torchrl.trainers.algorithms.configs.common import ConfigBase


@dataclass
class TransformConfig(ConfigBase):
    """Base configuration class for transforms."""

    def __post_init__(self) -> None:
        """Post-initialization hook for transform configurations."""


@dataclass
class NoopResetEnvConfig(TransformConfig):
    """Configuration for NoopResetEnv transform."""

    noops: int = 30
    random: bool = True
    _target_: str = "torchrl.envs.transforms.transforms.NoopResetEnv"

    def __post_init__(self) -> None:
        """Post-initialization hook for NoopResetEnv configuration."""
        super().__post_init__()


@dataclass
class StepCounterConfig(TransformConfig):
    """Configuration for StepCounter transform."""

    max_steps: int | None = None
    truncated_key: str | None = "truncated"
    step_count_key: str | None = "step_count"
    update_done: bool = True
    _target_: str = "torchrl.envs.transforms.transforms.StepCounter"

    def __post_init__(self) -> None:
        """Post-initialization hook for StepCounter configuration."""
        super().__post_init__()


@dataclass
class ComposeConfig(TransformConfig):
    """Configuration for Compose transform."""

    transforms: list[Any] | None = None
    _target_: str = "torchrl.envs.transforms.transforms.Compose"

    def __post_init__(self) -> None:
        """Post-initialization hook for Compose configuration."""
        super().__post_init__()
        if self.transforms is None:
            self.transforms = []


@dataclass
class DoubleToFloatConfig(TransformConfig):
    """Configuration for DoubleToFloat transform."""

    in_keys: list[str] | None = None
    out_keys: list[str] | None = None
    in_keys_inv: list[str] | None = None
    out_keys_inv: list[str] | None = None
    _target_: str = "torchrl.envs.transforms.transforms.DoubleToFloat"

    def __post_init__(self) -> None:
        """Post-initialization hook for DoubleToFloat configuration."""
        super().__post_init__()


@dataclass
class ToTensorImageConfig(TransformConfig):
    """Configuration for ToTensorImage transform."""

    from_int: bool | None = None
    unsqueeze: bool = False
    dtype: str | None = None
    in_keys: list[str] | None = None
    out_keys: list[str] | None = None
    shape_tolerant: bool = False
    _target_: str = "torchrl.envs.transforms.transforms.ToTensorImage"

    def __post_init__(self) -> None:
        """Post-initialization hook for ToTensorImage configuration."""
        super().__post_init__()


@dataclass
class ClipTransformConfig(TransformConfig):
    """Configuration for ClipTransform."""

    in_keys: list[str] | None = None
    out_keys: list[str] | None = None
    in_keys_inv: list[str] | None = None
    out_keys_inv: list[str] | None = None
    low: float | None = None
    high: float | None = None
    _target_: str = "torchrl.envs.transforms.transforms.ClipTransform"

    def __post_init__(self) -> None:
        """Post-initialization hook for ClipTransform configuration."""
        super().__post_init__()


@dataclass
class ResizeConfig(TransformConfig):
    """Configuration for Resize transform."""

    w: int = 84
    h: int = 84
    interpolation: str = "bilinear"
    in_keys: list[str] | None = None
    out_keys: list[str] | None = None
    _target_: str = "torchrl.envs.transforms.transforms.Resize"

    def __post_init__(self) -> None:
        """Post-initialization hook for Resize configuration."""
        super().__post_init__()


@dataclass
class CenterCropConfig(TransformConfig):
    """Configuration for CenterCrop transform."""

    height: int = 84
    width: int = 84
    in_keys: list[str] | None = None
    out_keys: list[str] | None = None
    _target_: str = "torchrl.envs.transforms.transforms.CenterCrop"

    def __post_init__(self) -> None:
        """Post-initialization hook for CenterCrop configuration."""
        super().__post_init__()


@dataclass
class FlattenObservationConfig(TransformConfig):
    """Configuration for FlattenObservation transform."""

    in_keys: list[str] | None = None
    out_keys: list[str] | None = None
    _target_: str = "torchrl.envs.transforms.transforms.FlattenObservation"

    def __post_init__(self) -> None:
        """Post-initialization hook for FlattenObservation configuration."""
        super().__post_init__()


@dataclass
class GrayScaleConfig(TransformConfig):
    """Configuration for GrayScale transform."""

    in_keys: list[str] | None = None
    out_keys: list[str] | None = None
    _target_: str = "torchrl.envs.transforms.transforms.GrayScale"

    def __post_init__(self) -> None:
        """Post-initialization hook for GrayScale configuration."""
        super().__post_init__()


@dataclass
class ObservationNormConfig(TransformConfig):
    """Configuration for ObservationNorm transform."""

    loc: float = 0.0
    scale: float = 1.0
    in_keys: list[str] | None = None
    out_keys: list[str] | None = None
    standard_normal: bool = False
    eps: float = 1e-8
    _target_: str = "torchrl.envs.transforms.transforms.ObservationNorm"

    def __post_init__(self) -> None:
        """Post-initialization hook for ObservationNorm configuration."""
        super().__post_init__()


@dataclass
class CatFramesConfig(TransformConfig):
    """Configuration for CatFrames transform."""

    N: int = 4
    dim: int = -3
    in_keys: list[str] | None = None
    out_keys: list[str] | None = None
    _target_: str = "torchrl.envs.transforms.transforms.CatFrames"

    def __post_init__(self) -> None:
        """Post-initialization hook for CatFrames configuration."""
        super().__post_init__()


@dataclass
class RewardClippingConfig(TransformConfig):
    """Configuration for RewardClipping transform."""

    clamp_min: float | None = None
    clamp_max: float | None = None
    in_keys: list[str] | None = None
    out_keys: list[str] | None = None
    _target_: str = "torchrl.envs.transforms.transforms.RewardClipping"

    def __post_init__(self) -> None:
        """Post-initialization hook for RewardClipping configuration."""
        super().__post_init__()


@dataclass
class RewardScalingConfig(TransformConfig):
    """Configuration for RewardScaling transform."""

    loc: float = 0.0
    scale: float = 1.0
    in_keys: list[str] | None = None
    out_keys: list[str] | None = None
    standard_normal: bool = False
    eps: float = 1e-8
    _target_: str = "torchrl.envs.transforms.transforms.RewardScaling"

    def __post_init__(self) -> None:
        """Post-initialization hook for RewardScaling configuration."""
        super().__post_init__()


@dataclass
class VecNormConfig(TransformConfig):
    """Configuration for VecNorm transform."""

    in_keys: list[str] | None = None
    out_keys: list[str] | None = None
    decay: float = 0.99
    eps: float = 1e-8
    _target_: str = "torchrl.envs.transforms.transforms.VecNorm"

    def __post_init__(self) -> None:
        """Post-initialization hook for VecNorm configuration."""
        super().__post_init__()


@dataclass
class FrameSkipTransformConfig(TransformConfig):
    """Configuration for FrameSkipTransform."""

    frame_skip: int = 4
    in_keys: list[str] | None = None
    out_keys: list[str] | None = None
    _target_: str = "torchrl.envs.transforms.transforms.FrameSkipTransform"

    def __post_init__(self) -> None:
        """Post-initialization hook for FrameSkipTransform configuration."""
        super().__post_init__()


@dataclass
class EndOfLifeTransformConfig(TransformConfig):
    """Configuration for EndOfLifeTransform."""

    eol_key: str = "end-of-life"
    lives_key: str = "lives"
    done_key: str = "done"
    eol_attribute: str = "unwrapped.ale.lives"
    _target_: str = "torchrl.envs.transforms.gym_transforms.EndOfLifeTransform"

    def __post_init__(self) -> None:
        """Post-initialization hook for EndOfLifeTransform configuration."""
        super().__post_init__()


@dataclass
class MultiStepTransformConfig(TransformConfig):
    """Configuration for MultiStepTransform."""

    n_steps: int = 3
    gamma: float = 0.99
    in_keys: list[str] | None = None
    out_keys: list[str] | None = None
    _target_: str = "torchrl.envs.transforms.rb_transforms.MultiStepTransform"

    def __post_init__(self) -> None:
        """Post-initialization hook for MultiStepTransform configuration."""
        super().__post_init__()


@dataclass
class TargetReturnConfig(TransformConfig):
    """Configuration for TargetReturn transform."""

    target_return: float = 10.0
    mode: str = "reduce"
    in_keys: list[str] | None = None
    out_keys: list[str] | None = None
    reset_key: str | None = None
    _target_: str = "torchrl.envs.transforms.transforms.TargetReturn"

    def __post_init__(self) -> None:
        """Post-initialization hook for TargetReturn configuration."""
        super().__post_init__()


@dataclass
class BinarizeRewardConfig(TransformConfig):
    """Configuration for BinarizeReward transform."""

    in_keys: list[str] | None = None
    out_keys: list[str] | None = None
    _target_: str = "torchrl.envs.transforms.transforms.BinarizeReward"

    def __post_init__(self) -> None:
        """Post-initialization hook for BinarizeReward configuration."""
        super().__post_init__()


@dataclass
class ActionDiscretizerConfig(TransformConfig):
    """Configuration for ActionDiscretizer transform."""

    num_intervals: int = 10
    action_key: str = "action"
    out_action_key: str | None = None
    sampling: str | None = None
    categorical: bool = True
    _target_: str = "torchrl.envs.transforms.transforms.ActionDiscretizer"

    def __post_init__(self) -> None:
        """Post-initialization hook for ActionDiscretizer configuration."""
        super().__post_init__()


@dataclass
class AutoResetTransformConfig(TransformConfig):
    """Configuration for AutoResetTransform."""

    replace: bool | None = None
    fill_float: str = "nan"
    fill_int: int = -1
    fill_bool: bool = False
    _target_: str = "torchrl.envs.transforms.transforms.AutoResetTransform"

    def __post_init__(self) -> None:
        """Post-initialization hook for AutoResetTransform configuration."""
        super().__post_init__()


@dataclass
class BatchSizeTransformConfig(TransformConfig):
    """Configuration for BatchSizeTransform."""

    batch_size: list[int] | None = None
    reshape_fn: Any = None
    reset_func: Any = None
    env_kwarg: bool = False
    _target_: str = "torchrl.envs.transforms.transforms.BatchSizeTransform"

    def __post_init__(self) -> None:
        """Post-initialization hook for BatchSizeTransform configuration."""
        super().__post_init__()


@dataclass
class DeviceCastTransformConfig(TransformConfig):
    """Configuration for DeviceCastTransform."""

    device: str = "cpu"
    in_keys: list[str] | None = None
    out_keys: list[str] | None = None
    in_keys_inv: list[str] | None = None
    out_keys_inv: list[str] | None = None
    _target_: str = "torchrl.envs.transforms.transforms.DeviceCastTransform"

    def __post_init__(self) -> None:
        """Post-initialization hook for DeviceCastTransform configuration."""
        super().__post_init__()


@dataclass
class DTypeCastTransformConfig(TransformConfig):
    """Configuration for DTypeCastTransform."""

    dtype: str = "torch.float32"
    in_keys: list[str] | None = None
    out_keys: list[str] | None = None
    in_keys_inv: list[str] | None = None
    out_keys_inv: list[str] | None = None
    _target_: str = "torchrl.envs.transforms.transforms.DTypeCastTransform"

    def __post_init__(self) -> None:
        """Post-initialization hook for DTypeCastTransform configuration."""
        super().__post_init__()


@dataclass
class UnsqueezeTransformConfig(TransformConfig):
    """Configuration for UnsqueezeTransform."""

    dim: int = 0
    in_keys: list[str] | None = None
    out_keys: list[str] | None = None
    _target_: str = "torchrl.envs.transforms.transforms.UnsqueezeTransform"

    def __post_init__(self) -> None:
        """Post-initialization hook for UnsqueezeTransform configuration."""
        super().__post_init__()


@dataclass
class SqueezeTransformConfig(TransformConfig):
    """Configuration for SqueezeTransform."""

    dim: int = 0
    in_keys: list[str] | None = None
    out_keys: list[str] | None = None
    _target_: str = "torchrl.envs.transforms.transforms.SqueezeTransform"

    def __post_init__(self) -> None:
        """Post-initialization hook for SqueezeTransform configuration."""
        super().__post_init__()


@dataclass
class PermuteTransformConfig(TransformConfig):
    """Configuration for PermuteTransform."""

    dims: list[int] | None = None
    in_keys: list[str] | None = None
    out_keys: list[str] | None = None
    _target_: str = "torchrl.envs.transforms.transforms.PermuteTransform"

    def __post_init__(self) -> None:
        """Post-initialization hook for PermuteTransform configuration."""
        super().__post_init__()
        if self.dims is None:
            self.dims = [0, 2, 1]


@dataclass
class CatTensorsConfig(TransformConfig):
    """Configuration for CatTensors transform."""

    dim: int = -1
    in_keys: list[str] | None = None
    out_keys: list[str] | None = None
    _target_: str = "torchrl.envs.transforms.transforms.CatTensors"

    def __post_init__(self) -> None:
        """Post-initialization hook for CatTensors configuration."""
        super().__post_init__()


@dataclass
class StackConfig(TransformConfig):
    """Configuration for Stack transform."""

    dim: int = 0
    in_keys: list[str] | None = None
    out_keys: list[str] | None = None
    _target_: str = "torchrl.envs.transforms.transforms.Stack"

    def __post_init__(self) -> None:
        """Post-initialization hook for Stack configuration."""
        super().__post_init__()


@dataclass
class DiscreteActionProjectionConfig(TransformConfig):
    """Configuration for DiscreteActionProjection transform."""

    num_actions: int = 4
    in_keys: list[str] | None = None
    out_keys: list[str] | None = None
    _target_: str = "torchrl.envs.transforms.transforms.DiscreteActionProjection"

    def __post_init__(self) -> None:
        """Post-initialization hook for DiscreteActionProjection configuration."""
        super().__post_init__()


@dataclass
class TensorDictPrimerConfig(TransformConfig):
    """Configuration for TensorDictPrimer transform."""

    primer_spec: Any = None
    in_keys: list[str] | None = None
    out_keys: list[str] | None = None
    _target_: str = "torchrl.envs.transforms.transforms.TensorDictPrimer"

    def __post_init__(self) -> None:
        """Post-initialization hook for TensorDictPrimer configuration."""
        super().__post_init__()


@dataclass
class PinMemoryTransformConfig(TransformConfig):
    """Configuration for PinMemoryTransform."""

    in_keys: list[str] | None = None
    out_keys: list[str] | None = None
    _target_: str = "torchrl.envs.transforms.transforms.PinMemoryTransform"

    def __post_init__(self) -> None:
        """Post-initialization hook for PinMemoryTransform configuration."""
        super().__post_init__()


@dataclass
class RewardSumConfig(TransformConfig):
    """Configuration for RewardSum transform."""

    in_keys: list[str] | None = None
    out_keys: list[str] | None = None
    _target_: str = "torchrl.envs.transforms.transforms.RewardSum"

    def __post_init__(self) -> None:
        """Post-initialization hook for RewardSum configuration."""
        super().__post_init__()


@dataclass
class ExcludeTransformConfig(TransformConfig):
    """Configuration for ExcludeTransform."""

    exclude_keys: list[str] | None = None
    _target_: str = "torchrl.envs.transforms.transforms.ExcludeTransform"

    def __post_init__(self) -> None:
        """Post-initialization hook for ExcludeTransform configuration."""
        super().__post_init__()
        if self.exclude_keys is None:
            self.exclude_keys = []


@dataclass
class SelectTransformConfig(TransformConfig):
    """Configuration for SelectTransform."""

    include_keys: list[str] | None = None
    _target_: str = "torchrl.envs.transforms.transforms.SelectTransform"

    def __post_init__(self) -> None:
        """Post-initialization hook for SelectTransform configuration."""
        super().__post_init__()
        if self.include_keys is None:
            self.include_keys = []


@dataclass
class TimeMaxPoolConfig(TransformConfig):
    """Configuration for TimeMaxPool transform."""

    dim: int = -1
    in_keys: list[str] | None = None
    out_keys: list[str] | None = None
    _target_: str = "torchrl.envs.transforms.transforms.TimeMaxPool"

    def __post_init__(self) -> None:
        """Post-initialization hook for TimeMaxPool configuration."""
        super().__post_init__()


@dataclass
class RandomCropTensorDictConfig(TransformConfig):
    """Configuration for RandomCropTensorDict transform."""

    crop_size: list[int] | None = None
    in_keys: list[str] | None = None
    out_keys: list[str] | None = None
    _target_: str = "torchrl.envs.transforms.transforms.RandomCropTensorDict"

    def __post_init__(self) -> None:
        """Post-initialization hook for RandomCropTensorDict configuration."""
        super().__post_init__()
        if self.crop_size is None:
            self.crop_size = [84, 84]


@dataclass
class InitTrackerConfig(TransformConfig):
    """Configuration for InitTracker transform."""

    in_keys: list[str] | None = None
    out_keys: list[str] | None = None
    _target_: str = "torchrl.envs.transforms.transforms.InitTracker"

    def __post_init__(self) -> None:
        """Post-initialization hook for InitTracker configuration."""
        super().__post_init__()


@dataclass
class RenameTransformConfig(TransformConfig):
    """Configuration for RenameTransform."""

    key_mapping: dict[str, str] | None = None
    _target_: str = "torchrl.envs.transforms.transforms.RenameTransform"

    def __post_init__(self) -> None:
        """Post-initialization hook for RenameTransform configuration."""
        super().__post_init__()
        if self.key_mapping is None:
            self.key_mapping = {}


@dataclass
class Reward2GoTransformConfig(TransformConfig):
    """Configuration for Reward2GoTransform."""

    gamma: float = 0.99
    in_keys: list[str] | None = None
    out_keys: list[str] | None = None
    _target_: str = "torchrl.envs.transforms.transforms.Reward2GoTransform"

    def __post_init__(self) -> None:
        """Post-initialization hook for Reward2GoTransform configuration."""
        super().__post_init__()


@dataclass
class ActionMaskConfig(TransformConfig):
    """Configuration for ActionMask transform."""

    mask_key: str = "action_mask"
    in_keys: list[str] | None = None
    out_keys: list[str] | None = None
    _target_: str = "torchrl.envs.transforms.transforms.ActionMask"

    def __post_init__(self) -> None:
        """Post-initialization hook for ActionMask configuration."""
        super().__post_init__()


@dataclass
class VecGymEnvTransformConfig(TransformConfig):
    """Configuration for VecGymEnvTransform."""

    in_keys: list[str] | None = None
    out_keys: list[str] | None = None
    _target_: str = "torchrl.envs.transforms.transforms.VecGymEnvTransform"

    def __post_init__(self) -> None:
        """Post-initialization hook for VecGymEnvTransform configuration."""
        super().__post_init__()


@dataclass
class BurnInTransformConfig(TransformConfig):
    """Configuration for BurnInTransform."""

    burn_in: int = 10
    in_keys: list[str] | None = None
    out_keys: list[str] | None = None
    _target_: str = "torchrl.envs.transforms.transforms.BurnInTransform"

    def __post_init__(self) -> None:
        """Post-initialization hook for BurnInTransform configuration."""
        super().__post_init__()


@dataclass
class SignTransformConfig(TransformConfig):
    """Configuration for SignTransform."""

    in_keys: list[str] | None = None
    out_keys: list[str] | None = None
    _target_: str = "torchrl.envs.transforms.transforms.SignTransform"

    def __post_init__(self) -> None:
        """Post-initialization hook for SignTransform configuration."""
        super().__post_init__()


@dataclass
class RemoveEmptySpecsConfig(TransformConfig):
    """Configuration for RemoveEmptySpecs transform."""

    _target_: str = "torchrl.envs.transforms.transforms.RemoveEmptySpecs"

    def __post_init__(self) -> None:
        """Post-initialization hook for RemoveEmptySpecs configuration."""
        super().__post_init__()


@dataclass
class TrajCounterConfig(TransformConfig):
    """Configuration for TrajCounter transform."""

    out_key: str = "traj_count"
    repeats: int | None = None
    _target_: str = "torchrl.envs.transforms.transforms.TrajCounter"

    def __post_init__(self) -> None:
        """Post-initialization hook for TrajCounter configuration."""
        super().__post_init__()


@dataclass
class LineariseRewardsConfig(TransformConfig):
    """Configuration for LineariseRewards transform."""

    in_keys: list[str] | None = None
    out_keys: list[str] | None = None
    weights: list[float] | None = None
    _target_: str = "torchrl.envs.transforms.transforms.LineariseRewards"

    def __post_init__(self) -> None:
        """Post-initialization hook for LineariseRewards configuration."""
        super().__post_init__()
        if self.in_keys is None:
            self.in_keys = []


@dataclass
class ConditionalSkipConfig(TransformConfig):
    """Configuration for ConditionalSkip transform."""

    cond: Any = None
    _target_: str = "torchrl.envs.transforms.transforms.ConditionalSkip"

    def __post_init__(self) -> None:
        """Post-initialization hook for ConditionalSkip configuration."""
        super().__post_init__()


@dataclass
class MultiActionConfig(TransformConfig):
    """Configuration for MultiAction transform."""

    dim: int = 1
    stack_rewards: bool = True
    stack_observations: bool = False
    _target_: str = "torchrl.envs.transforms.transforms.MultiAction"

    def __post_init__(self) -> None:
        """Post-initialization hook for MultiAction configuration."""
        super().__post_init__()


@dataclass
class TimerConfig(TransformConfig):
    """Configuration for Timer transform."""

    out_keys: list[str] | None = None
    time_key: str = "time"
    _target_: str = "torchrl.envs.transforms.transforms.Timer"

    def __post_init__(self) -> None:
        """Post-initialization hook for Timer configuration."""
        super().__post_init__()


@dataclass
class ConditionalPolicySwitchConfig(TransformConfig):
    """Configuration for ConditionalPolicySwitch transform."""

    policy: Any = None
    condition: Any = None
    _target_: str = "torchrl.envs.transforms.transforms.ConditionalPolicySwitch"

    def __post_init__(self) -> None:
        """Post-initialization hook for ConditionalPolicySwitch configuration."""
        super().__post_init__()


@dataclass
class KLRewardTransformConfig(TransformConfig):
    """Configuration for KLRewardTransform."""

    in_keys: list[str] | None = None
    out_keys: list[str] | None = None
    _target_: str = "torchrl.envs.transforms.llm.KLRewardTransform"

    def __post_init__(self) -> None:
        """Post-initialization hook for KLRewardTransform configuration."""
        super().__post_init__()


@dataclass
class R3MTransformConfig(TransformConfig):
    """Configuration for R3MTransform."""

    in_keys: list[str] | None = None
    out_keys: list[str] | None = None
    model_name: str = "resnet18"
    device: str = "cpu"
    _target_: str = "torchrl.envs.transforms.r3m.R3MTransform"

    def __post_init__(self) -> None:
        """Post-initialization hook for R3MTransform configuration."""
        super().__post_init__()


@dataclass
class VC1TransformConfig(TransformConfig):
    """Configuration for VC1Transform."""

    in_keys: list[str] | None = None
    out_keys: list[str] | None = None
    device: str = "cpu"
    _target_: str = "torchrl.envs.transforms.vc1.VC1Transform"

    def __post_init__(self) -> None:
        """Post-initialization hook for VC1Transform configuration."""
        super().__post_init__()


@dataclass
class VIPTransformConfig(TransformConfig):
    """Configuration for VIPTransform."""

    in_keys: list[str] | None = None
    out_keys: list[str] | None = None
    device: str = "cpu"
    _target_: str = "torchrl.envs.transforms.vip.VIPTransform"

    def __post_init__(self) -> None:
        """Post-initialization hook for VIPTransform configuration."""
        super().__post_init__()


@dataclass
class VIPRewardTransformConfig(TransformConfig):
    """Configuration for VIPRewardTransform."""

    in_keys: list[str] | None = None
    out_keys: list[str] | None = None
    device: str = "cpu"
    _target_: str = "torchrl.envs.transforms.vip.VIPRewardTransform"

    def __post_init__(self) -> None:
        """Post-initialization hook for VIPRewardTransform configuration."""
        super().__post_init__()


@dataclass
class VecNormV2Config(TransformConfig):
    """Configuration for VecNormV2 transform."""

    in_keys: list[str] | None = None
    out_keys: list[str] | None = None
    decay: float = 0.99
    eps: float = 1e-8
    _target_: str = "torchrl.envs.transforms.vecnorm.VecNormV2"

    def __post_init__(self) -> None:
        """Post-initialization hook for VecNormV2 configuration."""
        super().__post_init__()


@dataclass
class FiniteTensorDictCheckConfig(TransformConfig):
    """Configuration for FiniteTensorDictCheck transform."""

    in_keys: list[str] | None = None
    out_keys: list[str] | None = None
    _target_: str = "torchrl.envs.transforms.transforms.FiniteTensorDictCheck"

    def __post_init__(self) -> None:
        """Post-initialization hook for FiniteTensorDictCheck configuration."""
        super().__post_init__()


@dataclass
class UnaryTransformConfig(TransformConfig):
    """Configuration for UnaryTransform."""

    fn: Any = None
    in_keys: list[str] | None = None
    out_keys: list[str] | None = None
    _target_: str = "torchrl.envs.transforms.transforms.UnaryTransform"

    def __post_init__(self) -> None:
        """Post-initialization hook for UnaryTransform configuration."""
        super().__post_init__()


@dataclass
class HashConfig(TransformConfig):
    """Configuration for Hash transform."""

    in_keys: list[str] | None = None
    out_keys: list[str] | None = None
    _target_: str = "torchrl.envs.transforms.transforms.Hash"

    def __post_init__(self) -> None:
        """Post-initialization hook for Hash configuration."""
        super().__post_init__()


@dataclass
class TokenizerConfig(TransformConfig):
    """Configuration for Tokenizer transform."""

    vocab_size: int = 1000
    in_keys: list[str] | None = None
    out_keys: list[str] | None = None
    _target_: str = "torchrl.envs.transforms.transforms.Tokenizer"

    def __post_init__(self) -> None:
        """Post-initialization hook for Tokenizer configuration."""
        super().__post_init__()


@dataclass
class CropConfig(TransformConfig):
    """Configuration for Crop transform."""

    top: int = 0
    left: int = 0
    height: int = 84
    width: int = 84
    in_keys: list[str] | None = None
    out_keys: list[str] | None = None
    _target_: str = "torchrl.envs.transforms.transforms.Crop"

    def __post_init__(self) -> None:
        """Post-initialization hook for Crop configuration."""
        super().__post_init__()


@dataclass
class FlattenTensorDictConfig(TransformConfig):
    """Configuration for flattening TensorDict during inverse pass.

    This transform reshapes the tensordict to have a flat batch dimension
    during the inverse pass, which is useful for replay buffers that need
    to store data with a flat batch structure.
    """

    _target_: str = "torchrl.envs.transforms.transforms.FlattenTensorDict"

    def __post_init__(self) -> None:
        """Post-initialization hook for FlattenTensorDict configuration."""
        super().__post_init__()


@dataclass
class ModuleTransformConfig(TransformConfig):
    """Configuration for ModuleTransform."""

    module: Any = None
    device: Any = None
    no_grad: bool = False
    inverse: bool = False
    _target_: str = "torchrl.envs.transforms.module.ModuleTransform"
    _partial_: bool = False

    def __post_init__(self) -> None:
        """Post-initialization hook for ModuleTransform configuration."""
        super().__post_init__()
