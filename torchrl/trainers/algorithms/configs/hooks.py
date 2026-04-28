"""Hydra configuration classes for trainer hooks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from torchrl.trainers.algorithms.configs.common import ConfigBase


@dataclass
class HookConfig(ConfigBase):
    """Base configuration class for trainer hooks."""

    def __post_init__(self) -> None:
        """Post-initialization hook for hook configurations."""


@dataclass
class ClearCudaCacheConfig(HookConfig):
    """Configuration for the :class:`~torchrl.trainers.ClearCudaCache` hook."""

    interval: int = 1
    _target_: str = "torchrl.trainers.trainers.ClearCudaCache"

    def __post_init__(self) -> None:
        super().__post_init__()


@dataclass
class CountFramesLogConfig(HookConfig):
    """Configuration for the :class:`~torchrl.trainers.CountFramesLog` hook."""

    frame_skip: int = 1
    log_pbar: bool = False
    _target_: str = "torchrl.trainers.trainers.CountFramesLog"

    def __post_init__(self) -> None:
        super().__post_init__()


@dataclass
class LogScalarConfig(HookConfig):
    """Configuration for the :class:`~torchrl.trainers.LogScalar` hook."""

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
    """Configuration for the :class:`~torchrl.trainers.LogTiming` hook."""

    prefix: str = "time"
    percall: bool = True
    erase: bool = False
    _target_: str = "torchrl.trainers.trainers.LogTiming"

    def __post_init__(self) -> None:
        super().__post_init__()


@dataclass
class SelectKeysConfig(HookConfig):
    """Configuration for the :class:`~torchrl.trainers.SelectKeys` hook."""

    keys: list[str] = field(default_factory=list)
    _target_: str = "torchrl.trainers.trainers.SelectKeys"

    def __post_init__(self) -> None:
        super().__post_init__()


@dataclass
class RewardNormalizerConfig(HookConfig):
    """Configuration for the :class:`~torchrl.trainers.RewardNormalizer` hook."""

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
    """Configuration for the :class:`~torchrl.trainers.BatchSubSampler` hook."""

    batch_size: int = 1
    sub_traj_len: int = 0
    min_sub_traj_len: int = 0
    _target_: str = "torchrl.trainers.trainers.BatchSubSampler"

    def __post_init__(self) -> None:
        super().__post_init__()
