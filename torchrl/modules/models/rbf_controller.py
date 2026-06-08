# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import torch
import torch.nn as nn


class RBFController(nn.Module):
    """Radial Basis Function controller for moment-matching policy search.

    Implements a policy that maps Gaussian-distributed state beliefs
    ``(mean, covariance)`` to Gaussian-distributed actions using an RBF network
    followed by a sinusoidal squashing function. The moment-matching formulas
    allow analytic gradient computation through the policy during model-based
    optimization (e.g., PILCO).

    The controller uses ``n_basis`` RBF basis functions, each parameterised
    by a centre vector and a shared diagonal lengthscale. The output is a
    weighted sum of basis activations, optionally squashed through
    :meth:`squash_sin` to enforce action bounds.

    Reference: Deisenroth & Rasmussen, "PILCO: A Model-Based and Data-Efficient
    Approach to Policy Search", ICML 2011.

    Args:
        input_dim (int): Dimensionality of the state (observation) space.
        output_dim (int): Dimensionality of the action space.
        max_action (float or Tensor): Element-wise upper bound on action
            magnitude. When provided, actions are squashed through
            :meth:`squash_sin`.
        n_basis (int, optional): Number of RBF basis functions.
            Defaults to ``10``.

    Inputs:
        mean (Tensor): State mean of shape ``(*batch, input_dim)``.
        covariance (Tensor): State covariance of shape
            ``(*batch, input_dim, input_dim)``.

    Returns:
        action_mean (Tensor): Action mean of shape ``(*batch, output_dim)``.
        action_covariance (Tensor): Action covariance of shape
            ``(*batch, output_dim, output_dim)``.
        cross_covariance (Tensor): Input–output cross-covariance of shape
            ``(*batch, input_dim, output_dim)``.

    Examples:
        >>> import torch
        >>> controller = RBFController(input_dim=4, output_dim=1, max_action=2.0, n_basis=5)
        >>> mean = torch.randn(2, 4)
        >>> covariance = torch.eye(4).unsqueeze(0).expand(2, -1, -1) * 0.1
        >>> action_mean, action_cov, cross_cov = controller(mean, covariance)
        >>> action_mean.shape
        torch.Size([2, 1])
        >>> action_cov.shape
        torch.Size([2, 1, 1])
        >>> cross_cov.shape
        torch.Size([2, 4, 1])
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        max_action: float | torch.Tensor,
        n_basis: int = 10,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.max_action = max_action
        self.n_basis = n_basis

        self.centers = nn.Parameter(torch.randn(n_basis, input_dim) * 0.5)
        self.weights = nn.Parameter(torch.randn(n_basis, output_dim) * 0.1)
        self.lengthscales = nn.Parameter(torch.ones(input_dim))
        self.variance = 1.0

    @staticmethod
    def squash_sin(
        mean: torch.Tensor,
        covariance: torch.Tensor,
        max_action: float | torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Propagates a Gaussian through an element-wise ``max_action * sin(x)`` squashing.

        Computes the exact moments of the transformed distribution using
        the moment-matching identities for sine applied to Gaussian inputs.

        Args:
            mean (Tensor): Input mean, shape ``(*batch, K)``.
            covariance (Tensor): Input covariance, shape ``(*batch, K, K)``.
            max_action (float or Tensor): Per-dimension action bound.

        Returns:
            squashed_mean (Tensor): Output mean, shape ``(*batch, K)``.
            squashed_covariance (Tensor): Output covariance, shape ``(*batch, K, K)``.
            cross_covariance (Tensor): Input–output cross-covariance, shape ``(*batch, K, K)``.
        """
        K = mean.shape[-1]
        device = mean.device
        dtype = mean.dtype

        if not isinstance(max_action, torch.Tensor):
            max_action = torch.tensor(max_action, dtype=dtype, device=device)

        max_action = max_action.view(-1)
        if max_action.shape[0] == 1 and K > 1:
            max_action = max_action.expand(K)

        diag_cov = torch.diagonal(covariance, dim1=-2, dim2=-1)

        squashed_mean = max_action * torch.exp(-diag_cov / 2.0) * torch.sin(mean)

        lq = -(diag_cov.unsqueeze(-1) + diag_cov.unsqueeze(-2)) / 2.0
        q = torch.exp(lq)

        mean_diff = mean.unsqueeze(-1) - mean.unsqueeze(-2)
        mean_sum = mean.unsqueeze(-1) + mean.unsqueeze(-2)

        squashed_covariance = (torch.exp(lq + covariance) - q) * torch.cos(
            mean_diff
        ) - (torch.exp(lq - covariance) - q) * torch.cos(mean_sum)

        outer_max = max_action.unsqueeze(-2) * max_action.unsqueeze(-1)
        squashed_covariance = outer_max * squashed_covariance / 2.0

        cross_covariance = torch.diag_embed(
            max_action * torch.exp(-diag_cov / 2.0) * torch.cos(mean)
        )

        return squashed_mean, squashed_covariance, cross_covariance

    def forward(
        self, mean: torch.Tensor, covariance: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_shape = mean.shape[:-1]
        D = mean.shape[-1]
        N = self.n_basis
        device = mean.device

        # Flatten batch dimensions for computation
        mean_flat = mean.reshape(-1, D)
        covariance_flat = covariance.reshape(-1, D, D)
        B = mean_flat.shape[0]

        inv_lengthscale = torch.diag(1.0 / self.lengthscales)
        inv_lengthscale_batch = inv_lengthscale.unsqueeze(0)

        inp = self.centers.unsqueeze(0) - mean_flat.unsqueeze(1)

        B_mat = (
            inv_lengthscale_batch @ covariance_flat @ inv_lengthscale_batch
            + torch.eye(D, device=device, dtype=mean.dtype).unsqueeze(0)
        )

        scaled_inp = inp @ inv_lengthscale

        t = torch.linalg.solve(B_mat, scaled_inp.mT).mT

        exp_term = torch.exp(-0.5 * torch.sum(scaled_inp * t, dim=-1))
        log_det_sign, log_det = torch.linalg.slogdet(B_mat)
        normalizer = self.variance * torch.exp(-0.5 * log_det)
        phi_mean = normalizer.unsqueeze(-1) * exp_term

        action_mean = phi_mean @ self.weights

        t_scaled = t @ inv_lengthscale
        cross_cov = torch.bmm(t_scaled.mT, phi_mean.unsqueeze(-1) * self.weights)

        # Pairwise basis covariance (Eq. A.42–A.45 in Deisenroth thesis)
        centers_i = self.centers.unsqueeze(1)
        centers_j = self.centers.unsqueeze(0)
        diff = centers_i - centers_j
        center_bar = (centers_i + centers_j) / 2.0

        inv_lambda = 1.0 / (self.lengthscales**2)
        exp1 = -0.25 * torch.sum((diff**2) * inv_lambda, dim=-1)

        lambda_half = torch.diag((self.lengthscales**2) / 2.0)
        B_q = covariance_flat + lambda_half.unsqueeze(0)

        z = center_bar.unsqueeze(0) - mean_flat.unsqueeze(1).unsqueeze(1)
        z_flat = z.view(B, N * N, D)

        solved_z_flat = torch.linalg.solve(B_q, z_flat.mT).mT
        exp2 = -0.5 * torch.sum(z_flat * solved_z_flat, dim=-1).view(B, N, N)

        log_det_lambda_half = torch.sum(torch.log((self.lengthscales**2) / 2.0))
        _, log_det_bq = torch.linalg.slogdet(B_q)
        c_q = torch.exp(0.5 * (log_det_lambda_half - log_det_bq))

        Q = (self.variance**2 * c_q.view(B, 1, 1)) * torch.exp(
            exp1.unsqueeze(0) + exp2
        )

        W_batch = self.weights.unsqueeze(0).expand(B, N, -1)
        action_cov = torch.bmm(W_batch.mT, torch.bmm(Q, W_batch))

        outer_mean = torch.bmm(action_mean.unsqueeze(-1), action_mean.unsqueeze(1))
        action_cov = action_cov - outer_mean

        action_cov = (action_cov + action_cov.mT) / 2.0
        action_cov = (
            action_cov
            + torch.eye(self.output_dim, device=device, dtype=mean.dtype).unsqueeze(0)
            * 1e-6
        )

        if self.max_action is not None:
            action_mean, action_cov, C = self.squash_sin(
                action_mean, action_cov, self.max_action
            )
            cross_cov = torch.bmm(cross_cov, C)

        # Reshape back to original batch shape
        action_mean = action_mean.reshape(*batch_shape, self.output_dim)
        action_cov = action_cov.reshape(*batch_shape, self.output_dim, self.output_dim)
        cross_cov = cross_cov.reshape(*batch_shape, D, self.output_dim)

        return action_mean, action_cov, cross_cov
