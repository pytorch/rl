# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass

from torchrl.trainers.algorithms.configs.common import ConfigBase


@dataclass
class AdamConfig(ConfigBase):
    """Configuration for Adam optimizer."""

    lr: float = 1e-3
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-4
    weight_decay: float = 0.0
    amsgrad: bool = False
    _target_: str = "torch.optim.Adam"
    _partial_: bool = True

    def __post_init__(self) -> None:
        """Post-initialization hook for Adam optimizer configurations."""


@dataclass
class AdamWConfig(ConfigBase):
    """Configuration for AdamW optimizer."""

    lr: float = 1e-3
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 1e-2
    amsgrad: bool = False
    maximize: bool = False
    foreach: bool | None = None
    capturable: bool = False
    differentiable: bool = False
    fused: bool | None = None
    _target_: str = "torch.optim.AdamW"
    _partial_: bool = True

    def __post_init__(self) -> None:
        """Post-initialization hook for AdamW optimizer configurations."""


@dataclass
class AdamaxConfig(ConfigBase):
    """Configuration for Adamax optimizer."""

    lr: float = 2e-3
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 0.0
    _target_: str = "torch.optim.Adamax"
    _partial_: bool = True

    def __post_init__(self) -> None:
        """Post-initialization hook for Adamax optimizer configurations."""


@dataclass
class SGDConfig(ConfigBase):
    """Configuration for SGD optimizer."""

    lr: float = 1e-3
    momentum: float = 0.0
    dampening: float = 0.0
    weight_decay: float = 0.0
    nesterov: bool = False
    maximize: bool = False
    foreach: bool | None = None
    differentiable: bool = False
    _target_: str = "torch.optim.SGD"
    _partial_: bool = True

    def __post_init__(self) -> None:
        """Post-initialization hook for SGD optimizer configurations."""


@dataclass
class RMSpropConfig(ConfigBase):
    """Configuration for RMSprop optimizer."""

    lr: float = 1e-2
    alpha: float = 0.99
    eps: float = 1e-8
    weight_decay: float = 0.0
    momentum: float = 0.0
    centered: bool = False
    maximize: bool = False
    foreach: bool | None = None
    differentiable: bool = False
    _target_: str = "torch.optim.RMSprop"
    _partial_: bool = True

    def __post_init__(self) -> None:
        """Post-initialization hook for RMSprop optimizer configurations."""


@dataclass
class AdagradConfig(ConfigBase):
    """Configuration for Adagrad optimizer."""

    lr: float = 1e-2
    lr_decay: float = 0.0
    weight_decay: float = 0.0
    initial_accumulator_value: float = 0.0
    eps: float = 1e-10
    maximize: bool = False
    foreach: bool | None = None
    differentiable: bool = False
    _target_: str = "torch.optim.Adagrad"
    _partial_: bool = True

    def __post_init__(self) -> None:
        """Post-initialization hook for Adagrad optimizer configurations."""


@dataclass
class AdadeltaConfig(ConfigBase):
    """Configuration for Adadelta optimizer."""

    lr: float = 1.0
    rho: float = 0.9
    eps: float = 1e-6
    weight_decay: float = 0.0
    foreach: bool | None = None
    maximize: bool = False
    differentiable: bool = False
    _target_: str = "torch.optim.Adadelta"
    _partial_: bool = True

    def __post_init__(self) -> None:
        """Post-initialization hook for Adadelta optimizer configurations."""


@dataclass
class RpropConfig(ConfigBase):
    """Configuration for Rprop optimizer."""

    lr: float = 1e-2
    etas: tuple[float, float] = (0.5, 1.2)
    step_sizes: tuple[float, float] = (1e-6, 50.0)
    foreach: bool | None = None
    maximize: bool = False
    differentiable: bool = False
    _target_: str = "torch.optim.Rprop"
    _partial_: bool = True

    def __post_init__(self) -> None:
        """Post-initialization hook for Rprop optimizer configurations."""


@dataclass
class ASGDConfig(ConfigBase):
    """Configuration for ASGD optimizer."""

    lr: float = 1e-2
    lambd: float = 1e-4
    alpha: float = 0.75
    t0: float = 1e6
    weight_decay: float = 0.0
    foreach: bool | None = None
    maximize: bool = False
    differentiable: bool = False
    _target_: str = "torch.optim.ASGD"
    _partial_: bool = True

    def __post_init__(self) -> None:
        """Post-initialization hook for ASGD optimizer configurations."""


@dataclass
class LBFGSConfig(ConfigBase):
    """Configuration for LBFGS optimizer."""

    lr: float = 1.0
    max_iter: int = 20
    max_eval: int | None = None
    tolerance_grad: float = 1e-7
    tolerance_change: float = 1e-9
    history_size: int = 100
    line_search_fn: str | None = None
    _target_: str = "torch.optim.LBFGS"
    _partial_: bool = True

    def __post_init__(self) -> None:
        """Post-initialization hook for LBFGS optimizer configurations."""


@dataclass
class RAdamConfig(ConfigBase):
    """Configuration for RAdam optimizer."""

    lr: float = 1e-3
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 0.0
    _target_: str = "torch.optim.RAdam"
    _partial_: bool = True

    def __post_init__(self) -> None:
        """Post-initialization hook for RAdam optimizer configurations."""


@dataclass
class NAdamConfig(ConfigBase):
    """Configuration for NAdam optimizer."""

    lr: float = 2e-3
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 0.0
    momentum_decay: float = 4e-3
    foreach: bool | None = None
    _target_: str = "torch.optim.NAdam"
    _partial_: bool = True

    def __post_init__(self) -> None:
        """Post-initialization hook for NAdam optimizer configurations."""


@dataclass
class SparseAdamConfig(ConfigBase):
    """Configuration for SparseAdam optimizer."""

    lr: float = 1e-3
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    _target_: str = "torch.optim.SparseAdam"
    _partial_: bool = True

    def __post_init__(self) -> None:
        """Post-initialization hook for SparseAdam optimizer configurations."""


@dataclass
class LionConfig(ConfigBase):
    """Configuration for Lion optimizer."""

    lr: float = 1e-4
    betas: tuple[float, float] = (0.9, 0.99)
    weight_decay: float = 0.0
    _target_: str = "torch.optim.Lion"
    _partial_: bool = True

    def __post_init__(self) -> None:
        """Post-initialization hook for Lion optimizer configurations."""
