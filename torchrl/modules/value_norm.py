# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Value normalisation for actor-critic algorithms.

Defines the abstract :class:`ValueNorm` interface and two concrete
implementations:

- :class:`PopArtValueNorm` — exponential-moving-average mean / mean-of-squares
  with debiasing (van Hasselt et al., *Multi-task Deep RL with PopArt*,
  AAAI 2019, https://arxiv.org/abs/1809.04474). Used by MAPPO
  (Yu et al. 2022) to stabilise the critic loss when reward scales drift.
- :class:`RunningValueNorm` — exact Welford running mean / variance with no
  decay. Cheaper and more stable when value targets are stationary; tends to
  be the better default for shorter / non-curriculum runs.

Plug any subclass into :class:`~torchrl.objectives.multiagent.MAPPOLoss` (or
your own actor-critic loss) via ``value_norm=...``.
"""
from __future__ import annotations

from abc import ABCMeta, abstractmethod

import torch
from torch import nn


class ValueNorm(nn.Module, metaclass=ABCMeta):
    """Abstract base class for value normalisers.

    A *value normaliser* keeps a running estimate of the location and scale of
    the value target seen during training. Critics use it to:

    - **normalize** the regression target before computing MSE, keeping the
      critic loss on a fixed scale across episodes / reward inflations;
    - **denormalize** the critic's output back to the real reward scale when
      forming bootstrapped value estimates inside GAE / TD.

    Subclasses must implement :meth:`update`, :meth:`normalize`, and
    :meth:`denormalize`. The convention is that all three operate on tensors
    whose trailing dims match :attr:`shape` (the per-element value shape,
    usually ``(1,)``).
    """

    shape: tuple[int, ...]

    def __init__(
        self,
        *,
        shape: int | tuple[int, ...] = 1,
        epsilon: float = 1e-5,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()
        if isinstance(shape, int):
            shape = (shape,)
        self.shape = tuple(shape)
        self.epsilon = epsilon
        self._device = device

    # ------------------------------------------------------------------ API

    @abstractmethod
    def update(self, value_target: torch.Tensor) -> None:
        """Fold a batch of value targets into the running stats."""

    @abstractmethod
    def normalize(self, value_target: torch.Tensor) -> torch.Tensor:
        """Standardise ``value_target`` using the current running stats."""

    @abstractmethod
    def denormalize(self, normalised_value: torch.Tensor) -> torch.Tensor:
        """Inverse of :meth:`normalize` — recover real-scale values."""

    # ------------------------------------------------------- shared helpers

    def _check_trailing_shape(self, value_target: torch.Tensor) -> tuple[int, ...]:
        if value_target.shape[-len(self.shape) :] != self.shape:
            raise ValueError(
                f"{type(self).__name__} was initialised with shape={self.shape} "
                f"but got a value_target with trailing shape "
                f"{tuple(value_target.shape[-len(self.shape) :])}."
            )
        return tuple(range(value_target.ndim - len(self.shape)))


class PopArtValueNorm(ValueNorm):
    """PopArt-style EMA value normaliser.

    Maintains exponentially-weighted running estimates of the value-target
    mean and mean-of-squares, with debiasing (so the early-training estimates
    are unbiased even before the EMA has had time to wash out the zero
    initialisation). Equivalent to the value-normaliser used by the reference
    MAPPO implementation.

    Keyword Args:
        shape: per-element shape of the value tensor (everything except the
            leading batch / time / agent dims that get reduced). Defaults to
            ``1``.
        beta: exponential decay for the running stats. Higher = slower
            adaptation. Defaults to ``0.99999`` (the MAPPO default).
        epsilon: numerical stabiliser added to the running variance and used
            as a floor for the debiasing term. Defaults to ``1e-5``.
        device: device for the running-stats buffers.

    Example:
        >>> vn = PopArtValueNorm(shape=1)
        >>> target = torch.randn(64, 1) * 5.0 + 2.0    # mean 2, std 5
        >>> for _ in range(100):
        ...     vn.update(target)
        >>> normed = vn.normalize(target)              # ~ N(0, 1)
        >>> recovered = vn.denormalize(normed)         # back to real scale
    """

    def __init__(
        self,
        *,
        shape: int | tuple[int, ...] = 1,
        beta: float = 0.99999,
        epsilon: float = 1e-5,
        device: torch.device | None = None,
    ) -> None:
        super().__init__(shape=shape, epsilon=epsilon, device=device)
        self.beta = beta
        # Both running buffers start at zero. The debiasing term tracks
        # \sum_{s<=t} beta^{t-s}, which also starts at zero; dividing the
        # zero-init buffers by the (clamped) debias gives an unbiased EMA.
        self.register_buffer("running_mean", torch.zeros(self.shape, device=device))
        self.register_buffer("running_mean_sq", torch.zeros(self.shape, device=device))
        self.register_buffer("debiasing_term", torch.zeros((), device=device))

    def _running_stats(self) -> tuple[torch.Tensor, torch.Tensor]:
        debias = self.debiasing_term.clamp(min=self.epsilon)
        mean = self.running_mean / debias
        mean_sq = self.running_mean_sq / debias
        var = (mean_sq - mean.pow(2)).clamp(min=self.epsilon)
        return mean, var

    @torch.no_grad()
    def update(self, value_target: torch.Tensor) -> None:
        value_target = value_target.detach()
        reduce_dims = self._check_trailing_shape(value_target)
        if reduce_dims:
            batch_mean = value_target.mean(dim=reduce_dims)
            batch_mean_sq = value_target.pow(2).mean(dim=reduce_dims)
        else:
            batch_mean = value_target
            batch_mean_sq = value_target.pow(2)

        self.running_mean.mul_(self.beta).add_(batch_mean, alpha=1.0 - self.beta)
        self.running_mean_sq.mul_(self.beta).add_(batch_mean_sq, alpha=1.0 - self.beta)
        self.debiasing_term.mul_(self.beta).add_(1.0 - self.beta)

    def normalize(self, value_target: torch.Tensor) -> torch.Tensor:
        mean, var = self._running_stats()
        return (value_target - mean) / var.sqrt()

    def denormalize(self, normalised_value: torch.Tensor) -> torch.Tensor:
        mean, var = self._running_stats()
        return normalised_value * var.sqrt() + mean


class RunningValueNorm(ValueNorm):
    """Exact running mean / variance (Welford's online algorithm).

    Unlike :class:`PopArtValueNorm`, this normaliser does not decay older
    samples — it accumulates the true sample mean and variance over every
    target it has ever seen. Useful when value targets are roughly stationary
    (no curriculum, no reward-shaping schedule), where the EMA's adaptivity
    is unnecessary and the exact running stats give a slightly tighter
    estimate.

    Keyword Args:
        shape: per-element shape of the value tensor. Defaults to ``1``.
        epsilon: numerical stabiliser added to the running variance.
            Defaults to ``1e-5``.
        device: device for the running-stats buffers.

    Example:
        >>> vn = RunningValueNorm(shape=1)
        >>> for _ in range(10):
        ...     vn.update(torch.randn(64, 1) * 3.0 + 1.0)
        >>> normed = vn.normalize(torch.randn(8, 1))
    """

    def __init__(
        self,
        *,
        shape: int | tuple[int, ...] = 1,
        epsilon: float = 1e-5,
        device: torch.device | None = None,
    ) -> None:
        super().__init__(shape=shape, epsilon=epsilon, device=device)
        self.register_buffer("mean", torch.zeros(self.shape, device=device))
        # m2 stores the running sum of squared deviations from the mean
        # (Welford's M2). var = m2 / max(count - 1, 1).
        self.register_buffer("m2", torch.zeros(self.shape, device=device))
        self.register_buffer("count", torch.zeros((), device=device))

    @torch.no_grad()
    def update(self, value_target: torch.Tensor) -> None:
        value_target = value_target.detach()
        reduce_dims = self._check_trailing_shape(value_target)
        if reduce_dims:
            batch_count = float(
                torch.tensor([value_target.shape[d] for d in reduce_dims]).prod()
            )
            batch_mean = value_target.mean(dim=reduce_dims)
            batch_var = value_target.var(dim=reduce_dims, unbiased=False)
        else:
            batch_count = 1.0
            batch_mean = value_target
            batch_var = torch.zeros_like(value_target)

        # Chan et al. parallel variance update.
        delta = batch_mean - self.mean
        total = self.count + batch_count
        new_mean = self.mean + delta * (batch_count / total)
        new_m2 = (
            self.m2
            + batch_var * batch_count
            + delta.pow(2) * (self.count * batch_count / total)
        )
        self.mean.copy_(new_mean)
        self.m2.copy_(new_m2)
        self.count.fill_(total)

    def _var(self) -> torch.Tensor:
        denom = self.count.clamp(min=1.0)
        return (self.m2 / denom).clamp(min=self.epsilon)

    def normalize(self, value_target: torch.Tensor) -> torch.Tensor:
        return (value_target - self.mean) / self._var().sqrt()

    def denormalize(self, normalised_value: torch.Tensor) -> torch.Tensor:
        return normalised_value * self._var().sqrt() + self.mean
