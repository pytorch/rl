# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Functional operations for TorchRL modules."""
from __future__ import annotations

import torch


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
        >>> from torchrl.modules import functional as F
        >>> value = torch.tensor([-100.0, 0.0, 100.0])
        >>> F.symlog(value)
        tensor([-4.6151,  0.0000,  4.6151])

    .. seealso::
        :func:`symexp` for the inverse operation and
        :class:`~torchrl.modules.SymLogValueTransform` for the module form.
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
        >>> from torchrl.modules import functional as F
        >>> value = torch.tensor([-100.0, 0.0, 100.0])
        >>> transformed = F.symlog(value)
        >>> transformed
        tensor([-4.6151,  0.0000,  4.6151])
        >>> F.symexp(transformed)
        tensor([-100.0000,    0.0000,  100.0000])

    .. seealso::
        :func:`symlog` for the forward operation and
        :class:`~torchrl.modules.SymLogValueTransform` for the module form.
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
        >>> from torchrl.modules import functional as F
        >>> value = torch.tensor([-100.0, 0.0, 100.0])
        >>> F.signed_hyperbolic(value)
        tensor([-9.1499,  0.0000,  9.1499])

    .. seealso::
        :func:`signed_parabolic` for the inverse operation and
        :class:`~torchrl.modules.SignedHyperbolicValueTransform` for the module
        form.

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
        >>> from torchrl.modules import functional as F
        >>> value = torch.tensor([-100.0, 0.0, 100.0])
        >>> transformed = F.signed_hyperbolic(value)
        >>> transformed
        tensor([-9.1499,  0.0000,  9.1499])
        >>> F.signed_parabolic(transformed)
        tensor([-100.0000,    0.0000,  100.0000])

    .. seealso::
        :func:`signed_hyperbolic` for the forward operation and
        :class:`~torchrl.modules.SignedHyperbolicValueTransform` for the module
        form.
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


__all__ = ["signed_hyperbolic", "signed_parabolic", "symexp", "symlog"]
