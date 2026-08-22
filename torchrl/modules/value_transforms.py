# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Invertible transforms for scalar value targets and predictions."""

from __future__ import annotations

from abc import ABCMeta, abstractmethod

import torch
from torch import nn

from . import functional as F


class ValueTransform(nn.Module, metaclass=ABCMeta):
    """Abstract base class for invertible scalar value transforms.

    A value transform maps raw rewards or returns to a numerically convenient
    prediction space. :meth:`inverse` maps predictions back to the raw value
    space before they are used for bootstrapping.

    Subclasses implement :meth:`forward` and :meth:`inverse` as element-wise
    tensor operations.

    Examples:
        >>> import torch
        >>> from torchrl.modules import ValueTransform
        >>> class ScaleValueTransform(ValueTransform):
        ...     def forward(self, value):
        ...         return value * 2
        ...     def inverse(self, value):
        ...         return value / 2
        >>> transform = ScaleValueTransform()
        >>> value = torch.tensor([-2.0, 0.0, 2.0])
        >>> transformed = transform(value)
        >>> transformed
        tensor([-4.,  0.,  4.])
        >>> transform.inverse(transformed)
        tensor([-2.,  0.,  2.])

    .. seealso::
        :class:`IdentityValueTransform`, :class:`SymLogValueTransform`,
        :class:`SignedHyperbolicValueTransform`, and
        :class:`ComposeValueTransform` for concrete transforms;
        :class:`~torchrl.modules.ValueOperator` for the prediction-space
        contract; and :class:`~torchrl.objectives.value.ValueEstimatorBase`,
        :class:`~torchrl.objectives.DDPGLoss`,
        :class:`~torchrl.objectives.TD3Loss`,
        :class:`~torchrl.objectives.SACLoss`, and
        :class:`~torchrl.objectives.PPOLoss` for consumers.
    """

    @abstractmethod
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """Map a raw value tensor to the transformed prediction space."""

    @abstractmethod
    def inverse(self, value: torch.Tensor) -> torch.Tensor:
        """Map a transformed value tensor back to raw value space."""


class IdentityValueTransform(ValueTransform):
    """Leave scalar values unchanged.

    Examples:
        >>> import torch
        >>> from torchrl.modules import IdentityValueTransform
        >>> transform = IdentityValueTransform()
        >>> value = torch.tensor([-1.0, 0.0, 1.0])
        >>> transformed = transform(value)
        >>> transformed
        tensor([-1.,  0.,  1.])
        >>> transform.inverse(transformed)
        tensor([-1.,  0.,  1.])

    .. seealso::
        :class:`ValueTransform` for the interface and
        :class:`ComposeValueTransform` for composing transforms.
    """

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """Return ``value`` unchanged."""
        return value

    def inverse(self, value: torch.Tensor) -> torch.Tensor:
        """Return ``value`` unchanged."""
        return value


class SymLogValueTransform(ValueTransform):
    """Symmetric-log value transform used by DreamerV3.

    This transform applies :func:`torchrl.modules.functional.symlog` in the
    forward direction and :func:`torchrl.modules.functional.symexp` in the
    inverse direction.

    Examples:
        >>> import torch
        >>> from torchrl.modules import SymLogValueTransform
        >>> transform = SymLogValueTransform()
        >>> value = torch.tensor([-100.0, 0.0, 100.0])
        >>> transformed = transform(value)
        >>> transformed
        tensor([-4.6151,  0.0000,  4.6151])
        >>> transform.inverse(transformed)
        tensor([-100.0000,    0.0000,  100.0000])

    .. seealso::
        :func:`torchrl.modules.functional.symlog` and
        :func:`torchrl.modules.functional.symexp` for the functional form,
        and :class:`SignedHyperbolicValueTransform` for an alternative
        nonlinear transform.

    .. note::
        See `Mastering Diverse Domains through World Models
        <https://arxiv.org/abs/2301.04104>`_ (Hafner et al., 2023).
    """

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """Apply :func:`torchrl.modules.functional.symlog` to ``value``."""
        return F.symlog(value)

    def inverse(self, value: torch.Tensor) -> torch.Tensor:
        """Apply :func:`torchrl.modules.functional.symexp` to ``value``."""
        return F.symexp(value)


class SignedHyperbolicValueTransform(ValueTransform):
    """Signed-hyperbolic transform for large-magnitude value targets.

    Args:
        epsilon (float, optional): Positive linear correction that keeps the
            inverse Lipschitz continuous. Defaults to ``1e-3``.

    Examples:
        >>> import torch
        >>> from torchrl.modules import SignedHyperbolicValueTransform
        >>> transform = SignedHyperbolicValueTransform(epsilon=1e-3)
        >>> value = torch.tensor([-100.0, 0.0, 100.0])
        >>> transformed = transform(value)
        >>> transformed
        tensor([-9.1499,  0.0000,  9.1499])
        >>> transform.inverse(transformed)
        tensor([-100.0000,    0.0000,  100.0000])

    .. seealso::
        :func:`torchrl.modules.functional.signed_hyperbolic` and
        :func:`torchrl.modules.functional.signed_parabolic` for the functional
        form, and :class:`SymLogValueTransform` for an alternative nonlinear
        transform.

    .. note::
        See `Observe and Look Further: Achieving Consistent Performance on
        Atari <https://arxiv.org/abs/1805.11593>`_ (Pohlen et al., 2018).
    """

    def __init__(self, epsilon: float = 1e-3) -> None:
        super().__init__()
        if epsilon <= 0:
            raise ValueError(f"epsilon must be positive, got {epsilon}.")
        self.epsilon = epsilon

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """Apply the signed-hyperbolic transform to ``value``."""
        return F.signed_hyperbolic(value, self.epsilon)

    def inverse(self, value: torch.Tensor) -> torch.Tensor:
        """Apply the signed-parabolic inverse to ``value``."""
        return F.signed_parabolic(value, self.epsilon)


class ComposeValueTransform(ValueTransform):
    """Compose value transforms while preserving the inverse mapping.

    Forward transforms are applied in the order provided. Inverse transforms
    are applied in reverse order.

    Args:
        *transforms (ValueTransform): Transforms to compose.

    Examples:
        >>> import torch
        >>> from torchrl.modules import (
        ...     ComposeValueTransform,
        ...     SignedHyperbolicValueTransform,
        ...     SymLogValueTransform,
        ... )
        >>> transform = ComposeValueTransform(
        ...     SignedHyperbolicValueTransform(), SymLogValueTransform()
        ... )
        >>> value = torch.tensor([-100.0, 0.0, 100.0])
        >>> transformed = transform(value)
        >>> transformed
        tensor([-2.3175,  0.0000,  2.3175])
        >>> transform.inverse(transformed)
        tensor([-100.0000,    0.0000,  100.0000])

    .. seealso::
        :class:`ValueTransform` for the component interface,
        :class:`SymLogValueTransform`, and
        :class:`SignedHyperbolicValueTransform`.
    """

    def __init__(self, *transforms: ValueTransform) -> None:
        super().__init__()
        if not transforms:
            raise ValueError("ComposeValueTransform requires at least one transform.")
        if not all(isinstance(transform, ValueTransform) for transform in transforms):
            raise TypeError("All transforms must be ValueTransform instances.")
        self.transforms = nn.ModuleList(transforms)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """Apply the component transforms in order."""
        for transform in self.transforms:
            value = transform(value)
        return value

    def inverse(self, value: torch.Tensor) -> torch.Tensor:
        """Apply the component inverse transforms in reverse order."""
        for transform in reversed(self.transforms):
            value = transform.inverse(value)
        return value
