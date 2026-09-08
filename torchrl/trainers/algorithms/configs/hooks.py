"""Hydra configuration classes for trainer hooks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from omegaconf import MISSING

from torchrl.trainers.algorithms.configs.common import ConfigBase


@dataclass
class HookConfig(ConfigBase):
    """Base configuration class for trainer hooks."""

    def __post_init__(self) -> None:
        """Post-initialization hook for hook configurations."""


@dataclass
class ClearCudaCacheConfig(HookConfig):
    """Configuration for the :class:`~torchrl.trainers.ClearCudaCache` hook.

    Examples:
        >>> from torchrl.trainers.algorithms.configs.hooks import ClearCudaCacheConfig
        >>> from hydra.utils import instantiate
        >>> hook = instantiate(ClearCudaCacheConfig(interval=100))
    """

    interval: int = MISSING
    _target_: str = "torchrl.trainers.trainers.ClearCudaCache"

    def __post_init__(self) -> None:
        super().__post_init__()


@dataclass
class CountFramesLogConfig(HookConfig):
    """Configuration for the :class:`~torchrl.trainers.CountFramesLog` hook.

    Examples:
        >>> from torchrl.trainers.algorithms.configs.hooks import CountFramesLogConfig
        >>> from hydra.utils import instantiate
        >>> hook = instantiate(CountFramesLogConfig(frame_skip=4))
    """

    frame_skip: int = 1
    log_pbar: bool = False
    _target_: str = "torchrl.trainers.trainers.CountFramesLog"

    def __post_init__(self) -> None:
        super().__post_init__()


@dataclass
class EarlyStoppingConfig(HookConfig):
    """Configuration for the :class:`~torchrl.trainers.EarlyStopping` hook.

    Examples:
        >>> from torchrl.trainers.algorithms.configs.hooks import EarlyStoppingConfig
        >>> from hydra.utils import instantiate
        >>> hook = instantiate(
        ...     EarlyStoppingConfig(monitor="r_training", patience=10_000)
        ... )
    """

    monitor: Any = "r_evaluation"
    mode: str = "max"
    min_delta: float = 0.0
    patience: int = 100_000
    wait_for: int = 1_000_000
    check_finite: bool = True
    _target_: str = "torchrl.trainers.trainers.EarlyStopping"

    def __post_init__(self) -> None:
        super().__post_init__()


@dataclass
class LogScalarConfig(HookConfig):
    """Configuration for the :class:`~torchrl.trainers.LogScalar` hook.

    Examples:
        >>> from torchrl.trainers.algorithms.configs.hooks import LogScalarConfig
        >>> from hydra.utils import instantiate
        >>> hook = instantiate(
        ...     LogScalarConfig(key=["next", "reward"], logname="train_reward")
        ... )
    """

    key: Any = ("next", "reward")
    logname: str | None = None
    log_pbar: bool = False
    include_std: bool = True
    reduction: str = "mean"
    _target_: str = "torchrl.trainers.trainers.LogScalar"

    def __post_init__(self) -> None:
        super().__post_init__()


@dataclass
class LogTimingConfig(HookConfig):
    """Configuration for the :class:`~torchrl.trainers.LogTiming` hook.

    Examples:
        >>> from torchrl.trainers.algorithms.configs.hooks import LogTimingConfig
        >>> from hydra.utils import instantiate
        >>> hook = instantiate(LogTimingConfig(prefix="time", percall=True))
    """

    prefix: str = "time"
    percall: bool = True
    erase: bool = False
    _target_: str = "torchrl.trainers.trainers.LogTiming"

    def __post_init__(self) -> None:
        super().__post_init__()


@dataclass
class SelectKeysConfig(HookConfig):
    """Configuration for the :class:`~torchrl.trainers.SelectKeys` hook.

    Examples:
        >>> from torchrl.trainers.algorithms.configs.hooks import SelectKeysConfig
        >>> from hydra.utils import instantiate
        >>> hook = instantiate(SelectKeysConfig(keys=["observation", "action"]))
    """

    keys: list[str] = field(default_factory=list)
    _target_: str = "torchrl.trainers.trainers.SelectKeys"

    def __post_init__(self) -> None:
        super().__post_init__()


@dataclass
class RewardNormalizerConfig(HookConfig):
    """Configuration for the :class:`~torchrl.trainers.RewardNormalizer` hook.

    Examples:
        >>> from torchrl.trainers.algorithms.configs.hooks import RewardNormalizerConfig
        >>> from hydra.utils import instantiate
        >>> hook = instantiate(RewardNormalizerConfig(decay=0.99, scale=1.0))
    """

    decay: float = 0.999
    scale: float = 1.0
    eps: float | None = None
    log_pbar: bool = False
    reward_key: Any = None
    _target_: str = "torchrl.trainers.trainers.RewardNormalizer"

    def __post_init__(self) -> None:
        super().__post_init__()


@dataclass
class BatchSubSamplerConfig(HookConfig):
    """Configuration for the :class:`~torchrl.trainers.BatchSubSampler` hook.

    Examples:
        >>> from torchrl.trainers.algorithms.configs.hooks import BatchSubSamplerConfig
        >>> from hydra.utils import instantiate
        >>> hook = instantiate(BatchSubSamplerConfig(batch_size=64, sub_traj_len=8))
    """

    batch_size: int = MISSING
    sub_traj_len: int = 0
    min_sub_traj_len: int = 0
    _target_: str = "torchrl.trainers.trainers.BatchSubSampler"

    def __post_init__(self) -> None:
        super().__post_init__()


@dataclass
class DreamerV3OptimizationStepperConfig(HookConfig):
    """Hydra configuration for :class:`~torchrl.trainers.algorithms.DreamerV3OptimizationStepper`.

    Examples:
        With the learner and optimizer from the public stepper example:

        >>> from hydra.utils import instantiate
        >>> from torchrl.trainers.algorithms.configs import DreamerV3OptimizationStepperConfig
        >>> configured_stepper = instantiate(
        ...     DreamerV3OptimizationStepperConfig(),
        ...     loss_module=loss_module, optimizer=optimizer,
        ...     target_updater=target_updater,
        ... )
        >>> metrics = configured_stepper.step(None, sample)
        >>> assert not sample["replay_context", "state"].requires_grad
    """

    loss_module: Any = None
    optimizer: Any = None
    target_updater: Any = None
    compile_train_step: bool = False
    compile_mode: str = "default"
    cudagraph: bool = False
    warmup_steps: int = 5
    mixed_precision: bool = False
    _target_: str = "torchrl.trainers.algorithms.DreamerV3OptimizationStepper"


@dataclass
class DreamerV3UpdateRatioConfig(ConfigBase):
    """Hydra configuration for :class:`~torchrl.trainers.algorithms.DreamerV3UpdateRatio`.

    Examples:
        >>> from hydra.utils import instantiate
        >>> from torchrl.trainers.algorithms.configs import DreamerV3UpdateRatioConfig
        >>> schedule = instantiate(DreamerV3UpdateRatioConfig(ratio=0.25))
        >>> schedule(4), schedule(8)
        (1, 1)
    """

    ratio: float = MISSING
    _target_: str = "torchrl.trainers.algorithms.DreamerV3UpdateRatio"

    def __post_init__(self) -> None:
        """Initialize the update-ratio configuration."""
