# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Value normalisation for actor-critic algorithms.

Implements a PopArt-style running normaliser used in MAPPO (Yu et al. 2021)
to stabilise the critic loss when reward scales drift during training.
The technique was originally introduced in
"Multi-task Deep Reinforcement Learning with PopArt"
(van Hasselt et al., AAAI 2019, https://arxiv.org/abs/1809.04474).
"""
from __future__ import annotations

import torch
from torch import nn


class ValueNorm(nn.Module):
    """Running value normaliser used by :class:`MAPPOLoss` and similar critics.

    Keeps an exponentially-weighted running estimate of the mean and variance of
    the value target, and exposes :meth:`normalize` / :meth:`denormalize` so the
    critic can be trained against a unit-variance target while still bootstrapping
    with real-scale values during advantage computation.

    The MAPPO paper (Yu et al. 2021, Table 13) credits this trick with a sizeable
    win-rate improvement on SMAC, which is why it is opt-in here rather than
    bolted onto every PPO critic.

    Keyword Args:
        shape (int or tuple of int): per-element shape of the value tensor
            (everything except the leading batch / time / agent dims that get
            reduced). Use ``1`` for scalar values, which is the common case.
        beta (float): exponential decay for the running stats. Higher = slower
            adaptation. Defaults to ``0.99999`` (the MAPPO default).
        epsilon (float): added to the running variance before taking the square
            root, for numerical stability. Defaults to ``1e-5``.
        device (torch.device, optional): device for the running stats buffers.

    Example:
        >>> vn = ValueNorm(shape=1)
        >>> target = torch.randn(64, 10) * 5.0 + 2.0   # mean 2, std 5
        >>> vn.update(target)
        >>> normed = vn.normalize(target)             # ~ N(0, 1)
        >>> recovered = vn.denormalize(normed)        # back to real scale
    """

    def __init__(
        self,
        *,
        shape: int | tuple[int, ...] = 1,
        beta: float = 0.99999,
        epsilon: float = 1e-5,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()
        if isinstance(shape, int):
            shape = (shape,)
        self.shape = tuple(shape)
        self.beta = beta
        self.epsilon = epsilon

        # Both running buffers start at zero. The debiasing term tracks
        # \sum_{s<=t} beta^{t-s}, which also starts at zero; dividing the
        # zero-init buffers by the (clamped) debias gives an unbiased EMA.
        self.register_buffer("running_mean", torch.zeros(self.shape, device=device))
        self.register_buffer("running_mean_sq", torch.zeros(self.shape, device=device))
        self.register_buffer("debiasing_term", torch.zeros((), device=device))

    def _running_stats(self) -> tuple[torch.Tensor, torch.Tensor]:
        # PopArt debiasing: at step t, the running buffer is
        #     E[x] = sum_{s<=t} beta^{t-s} x_s    (un-normalised)
        # and the debiasing term is sum_{s<=t} beta^{t-s}, so dividing by it
        # gives an unbiased estimate of the EMA.
        debias = self.debiasing_term.clamp(min=self.epsilon)
        mean = self.running_mean / debias
        mean_sq = self.running_mean_sq / debias
        var = (mean_sq - mean.pow(2)).clamp(min=self.epsilon)
        return mean, var

    @torch.no_grad()
    def update(self, value_target: torch.Tensor) -> None:
        """Update running stats from a new batch of value targets.

        All dimensions of ``value_target`` *except* the trailing dims that
        match :attr:`shape` are reduced (mean) before being mixed into the
        running buffer. The shape of ``value_target`` must therefore end with
        :attr:`shape`.
        """
        value_target = value_target.detach()
        if value_target.shape[-len(self.shape) :] != self.shape:
            raise ValueError(
                f"ValueNorm was initialised with shape={self.shape} but got a "
                f"value_target with trailing shape "
                f"{tuple(value_target.shape[-len(self.shape) :])}."
            )
        reduce_dims = tuple(range(value_target.ndim - len(self.shape)))
        batch_mean = value_target.mean(dim=reduce_dims) if reduce_dims else value_target
        batch_mean_sq = (
            value_target.pow(2).mean(dim=reduce_dims)
            if reduce_dims
            else value_target.pow(2)
        )

        self.running_mean.mul_(self.beta).add_(batch_mean, alpha=1.0 - self.beta)
        self.running_mean_sq.mul_(self.beta).add_(batch_mean_sq, alpha=1.0 - self.beta)
        self.debiasing_term.mul_(self.beta).add_(1.0 - self.beta)

    def normalize(self, value_target: torch.Tensor) -> torch.Tensor:
        """Standardise ``value_target`` using the current running stats."""
        mean, var = self._running_stats()
        return (value_target - mean) / var.sqrt()

    def denormalize(self, normalised_value: torch.Tensor) -> torch.Tensor:
        """Inverse of :meth:`normalize` — recover real-scale values."""
        mean, var = self._running_stats()
        return normalised_value * var.sqrt() + mean
