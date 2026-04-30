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
