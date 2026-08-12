# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Invertible transforms for scalar value targets and predictions."""
from __future__ import annotations

from abc import ABCMeta, abstractmethod

import torch
from torch import nn


def symlog(value: torch.Tensor) -> torch.Tensor:
    """Apply the element-wise symmetric logarithm transform.

    The transform is defined as
    ``sign(value) * log(1 + abs(value))`` and compresses both positive and
    negative values while remaining approximately linear around zero.

    Args:
        value (torch.Tensor): Input tensor.

    Returns:
        A tensor with the same shape, dtype, and device as ``value``.

    Examples:
        >>> import torch
        >>> from torchrl.modules import symlog
        >>> symlog(torch.tensor([-100.0, 0.0, 100.0]))
        tensor([-4.6151,  0.0000,  4.6151])
    """
    transformed = value.sign() * value.abs().log1p()
    # ``sign`` has a zero derivative at the origin even though symlog has a
    # derivative of one there. Keep the mathematically correct local gradient.
    return torch.where(value == 0, value, transformed)


def symexp(value: torch.Tensor) -> torch.Tensor:
    """Apply the inverse symmetric exponential transform element-wise.

    Args:
        value (torch.Tensor): Input tensor in symmetric-log space.

    Returns:
        A tensor with the same shape, dtype, and device as ``value``.

    Examples:
        >>> import torch
        >>> from torchrl.modules import symexp, symlog
        >>> value = torch.tensor([-1000.0, 0.0, 1000.0])
        >>> torch.allclose(symexp(symlog(value)), value, atol=1e-4)
        True
    """
    transformed = value.sign() * value.abs().expm1()
    return torch.where(value == 0, value, transformed)


def signed_hyperbolic(value: torch.Tensor, epsilon: float = 1e-3) -> torch.Tensor:
    """Apply the signed hyperbolic value transform.

    This is the scale-compressing transform introduced by Pohlen et al. and
    used by algorithms in the MuZero and Muesli families:

    ``sign(value) * (sqrt(abs(value) + 1) - 1) + epsilon * value``.

    Args:
        value (torch.Tensor): Input tensor.
        epsilon (float, optional): Positive linear correction that keeps the
            inverse Lipschitz continuous. Defaults to ``1e-3``.

    Returns:
        A tensor with the same shape, dtype, and device as ``value``.

    Examples:
        >>> import torch
        >>> from torchrl.modules import signed_hyperbolic
        >>> signed_hyperbolic(torch.tensor([-100.0, 0.0, 100.0]))
        tensor([-9.1499,  0.0000,  9.1499])

    .. note::
        See `Observe and Look Further: Achieving Consistent Performance on
        Atari <https://arxiv.org/abs/1805.11593>`_ (Pohlen et al., 2018).
    """
    if epsilon <= 0:
        raise ValueError(f"epsilon must be positive, got {epsilon}.")
    transformed = value.sign() * (torch.sqrt(value.abs() + 1) - 1) + epsilon * value
    origin = value * (0.5 + epsilon)
    return torch.where(value == 0, origin, transformed)


def signed_parabolic(value: torch.Tensor, epsilon: float = 1e-3) -> torch.Tensor:
    """Apply the inverse of :func:`signed_hyperbolic` element-wise.

    Args:
        value (torch.Tensor): Input tensor in signed-hyperbolic space.
        epsilon (float, optional): Positive linear correction used by the
            corresponding :func:`signed_hyperbolic` call. Defaults to
            ``1e-3``.

    Returns:
        A tensor with the same shape, dtype, and device as ``value``.

    Examples:
        >>> import torch
        >>> from torchrl.modules import signed_hyperbolic, signed_parabolic
        >>> value = torch.tensor([-1000.0, 0.0, 1000.0])
        >>> torch.allclose(
        ...     signed_parabolic(signed_hyperbolic(value)), value, atol=1e-3
        ... )
        True
    """
    if epsilon <= 0:
        raise ValueError(f"epsilon must be positive, got {epsilon}.")
    magnitude = value.abs()
    discriminant = torch.sqrt(1 + 4 * epsilon * (magnitude + 1 + epsilon))
    # This rationalized form of (discriminant - 1) / (2 * epsilon)
    # avoids cancellation when epsilon is small.
    root = 2 * (magnitude + 1 + epsilon) / (discriminant + 1)
    transformed = value.sign() * (root.square() - 1)
    origin = value / (0.5 + epsilon)
    return torch.where(value == 0, origin, transformed)


class ValueTransform(nn.Module, metaclass=ABCMeta):
    """Abstract base class for invertible scalar value transforms.

    A value transform maps raw rewards or returns to a numerically convenient
    prediction space. :meth:`inverse` maps predictions back to the raw value
    space before they are used for bootstrapping.

    Subclasses implement :meth:`forward` and :meth:`inverse` as element-wise
    tensor operations.
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
        >>> torch.equal(transform(value), transform.inverse(value))
        True
    """

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """Return ``value`` unchanged."""
        return value

    def inverse(self, value: torch.Tensor) -> torch.Tensor:
        """Return ``value`` unchanged."""
        return value


class SymLogValueTransform(ValueTransform):
    """Symmetric-log value transform used by DreamerV3.

    This transform applies :func:`symlog` in the forward direction and
    :func:`symexp` in the inverse direction.

    Examples:
        >>> import torch
        >>> from torchrl.modules import SymLogValueTransform
        >>> transform = SymLogValueTransform()
        >>> value = torch.tensor([-100.0, 0.0, 100.0])
        >>> torch.allclose(transform.inverse(transform(value)), value)
        True

    .. note::
        See `Mastering Diverse Domains through World Models
        <https://arxiv.org/abs/2301.04104>`_ (Hafner et al., 2023).
    """

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """Apply :func:`symlog` to ``value``."""
        return symlog(value)

    def inverse(self, value: torch.Tensor) -> torch.Tensor:
        """Apply :func:`symexp` to ``value``."""
        return symexp(value)


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
        >>> torch.allclose(transform.inverse(transform(value)), value, atol=1e-4)
        True

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
        """Apply :func:`signed_hyperbolic` to ``value``."""
        return signed_hyperbolic(value, self.epsilon)

    def inverse(self, value: torch.Tensor) -> torch.Tensor:
        """Apply :func:`signed_parabolic` to ``value``."""
        return signed_parabolic(value, self.epsilon)


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
        >>> torch.allclose(transform.inverse(transform(value)), value, atol=1e-4)
        True
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
