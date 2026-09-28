# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimplicialNormalization(nn.Module):
    """Apply softmax independently to fixed-size feature simplices.

    Args:
        dim (int): Number of features in each simplex. The last input
            dimension must be divisible by ``dim``.
    """

    def __init__(self, dim: int):
        super().__init__()
        if isinstance(dim, bool) or not isinstance(dim, int) or dim <= 0:
            raise ValueError(f"dim must be a positive integer, got {dim!r}.")
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize the last dimension of ``x`` in groups of ``dim``."""
        if x.shape[-1] % self.dim:
            raise ValueError(
                f"The last input dimension must be divisible by dim={self.dim}, "
                f"got {x.shape[-1]}."
            )
        shape = x.shape
        x = x.reshape(*shape[:-1], -1, self.dim)
        x = F.softmax(x, dim=-1)
        return x.reshape(shape)


def log_std(
    x: torch.Tensor,
    low: torch.Tensor | float,
    dif: torch.Tensor | float,
) -> torch.Tensor:
    """Map unconstrained values to a bounded log-standard deviation.

    Args:
        x (torch.Tensor): Unconstrained values.
        low (torch.Tensor or float): Lower bound of the output interval.
        dif (torch.Tensor or float): Width of the output interval.

    Returns:
        A tensor with values in ``[low, low + dif]``.
    """
    return low + 0.5 * dif * (torch.tanh(x) + 1)


def gaussian_logprob(eps: torch.Tensor, log_std: torch.Tensor) -> torch.Tensor:
    """Compute the log probability of diagonal Gaussian samples.

    Args:
        eps (torch.Tensor): Gaussian noise samples.
        log_std (torch.Tensor): Log standard deviations for each action.

    Returns:
        The summed log probability with a trailing singleton event dimension.
    """
    return (-0.5 * eps.pow(2) - log_std - 0.9189385175704956).sum(-1, keepdim=True)


def _squash(
    mu: torch.Tensor,
    pi: torch.Tensor,
    log_pi: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Squash policy locations and samples while correcting log probability."""
    mu = torch.tanh(mu)
    pi = torch.tanh(pi)
    squashed_pi = torch.log(F.relu(1 - pi.pow(2)) + 1e-6)
    log_pi = log_pi - squashed_pi.sum(-1, keepdim=True)
    return mu, pi, log_pi


class _TdMpc2PolicyPrior(nn.Module):
    """Sample actions from the TD-MPC2 policy prior."""

    def __init__(
        self,
        network: nn.Module,
        log_std_min: float,
        log_std_max: float,
        device: torch.device | str | None = None,
    ):
        super().__init__()
        self.network = network
        self.register_buffer("log_std_min", torch.tensor(log_std_min, device=device))
        self.register_buffer(
            "log_std_dif",
            torch.tensor(log_std_max, device=device) - self.log_std_min,
        )

    def forward(
        self, latent: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, raw_log_std = self.network(latent).chunk(2, dim=-1)
        bounded_log_std = log_std(raw_log_std, self.log_std_min, self.log_std_dif)
        eps = torch.randn_like(mean)
        log_prob = gaussian_logprob(eps, bounded_log_std)
        scaled_log_prob = log_prob * eps.shape[-1]

        action = mean + eps * bounded_log_std.exp()
        mean, action, log_prob = _squash(mean, action, log_prob)

        entropy = -log_prob
        scaled_entropy = -scaled_log_prob
        return action, mean, bounded_log_std, entropy, scaled_entropy
