# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import torch
from torch import nn


class RBFController(nn.Module):
    """A Radial Basis Function (RBF) controller.

    Args:
        input_dim (int): The dimensionality of the input space.
        output_dim (int): The dimensionality of the output space.
        max_action (float or torch.Tensor): The maximum action magnitude used for the squashing function.
        n_basis (int, optional): The number of basis functions to use. Defaults to 10.
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
        m: torch.Tensor, s: torch.Tensor, max_action: float | torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Squashes the output using a sine function to keep actions within the bounded range.

        Args:
            m (torch.Tensor): The mean of the distribution.
            s (torch.Tensor): The covariance matrix of the distribution.
            max_action (float or torch.Tensor): The maximum magnitude of the action bounds.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing:
                - M (torch.Tensor): The squashed mean.
                - S (torch.Tensor): The squashed covariance.
                - C (torch.Tensor): The cross-covariance between input and output.
        """
        B, K = m.shape
        device = m.device
        dtype = m.dtype

        if not isinstance(max_action, torch.Tensor):
            max_action = torch.tensor(max_action, dtype=dtype, device=device)

        max_action = max_action.view(-1)
        if max_action.shape[0] == 1 and K > 1:
            max_action = max_action.expand(K)

        diag_s = torch.diagonal(s, dim1=-2, dim2=-1)

        M = max_action * torch.exp(-diag_s / 2.0) * torch.sin(m)

        lq = -(diag_s.unsqueeze(-1) + diag_s.unsqueeze(-2)) / 2.0
        q = torch.exp(lq)

        m_diff = m.unsqueeze(-1) - m.unsqueeze(-2)
        m_sum = m.unsqueeze(-1) + m.unsqueeze(-2)

        S = (torch.exp(lq + s) - q) * torch.cos(m_diff) - (
            torch.exp(lq - s) - q
        ) * torch.cos(m_sum)

        outer_max = max_action.unsqueeze(1) * max_action.unsqueeze(0)
        S = outer_max.unsqueeze(0) * S / 2.0

        C = torch.diag_embed(max_action * torch.exp(-diag_s / 2.0) * torch.cos(m))

        return M, S, C

    def forward(
        self, m: torch.Tensor, S: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Computes the forward pass of the RBF Controller.

        Args:
            m (torch.Tensor): The mean of the input tensor.
            S (torch.Tensor): The covariance matrix of the input tensor.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing:
                - M (torch.Tensor): The mean of the action distribution.
                - S_action (torch.Tensor): The covariance of the action distribution.
                - V (torch.Tensor): The input-output cross-covariance.
        """
        B, D = m.shape
        N = self.n_basis
        device = m.device

        iL = torch.diag(1.0 / self.lengthscales)
        iL_batch = iL.unsqueeze(0)

        inp = self.centers.unsqueeze(0) - m.unsqueeze(1)

        B_mat = iL_batch @ S @ iL_batch + torch.eye(
            D, device=device, dtype=m.dtype
        ).unsqueeze(0)

        iN = inp @ iL

        t = torch.linalg.solve(B_mat, iN.mT).mT

        exp_term = torch.exp(-0.5 * torch.sum(iN * t, dim=-1))
        detB = torch.linalg.det(B_mat)
        c = self.variance / torch.sqrt(detB)
        phi_mean = c.unsqueeze(-1) * exp_term

        M = phi_mean @ self.weights

        tiL = t @ iL
        V = torch.bmm(tiL.mT, phi_mean.unsqueeze(-1) * self.weights)

        c_i = self.centers.unsqueeze(1)
        c_j = self.centers.unsqueeze(0)
        diff = c_i - c_j
        c_bar = (c_i + c_j) / 2.0

        inv_Lambda = 1.0 / (self.lengthscales**2)
        exp1 = -0.25 * torch.sum((diff**2) * inv_Lambda, dim=-1)

        Lambda_half = torch.diag((self.lengthscales**2) / 2.0)
        B_q = S + Lambda_half.unsqueeze(0)

        z = c_bar.unsqueeze(0) - m.unsqueeze(1).unsqueeze(1)
        z_flat = z.view(B, N * N, D)

        solved_z_flat = torch.linalg.solve(B_q, z_flat.mT).mT
        exp2 = -0.5 * torch.sum(z_flat * solved_z_flat, dim=-1).view(B, N, N)

        log_det_Lambda_half = torch.sum(torch.log((self.lengthscales**2) / 2.0))
        log_det_B_q = torch.logdet(B_q)
        c_q = torch.exp(0.5 * (log_det_Lambda_half - log_det_B_q))

        Q = (self.variance**2 * c_q.view(B, 1, 1)) * torch.exp(
            exp1.unsqueeze(0) + exp2
        )

        W_batch = self.weights.unsqueeze(0).expand(B, N, -1)
        S_action = torch.bmm(W_batch.mT, torch.bmm(Q, W_batch))

        M_out = torch.bmm(M.unsqueeze(-1), M.unsqueeze(1))
        S_action = S_action - M_out

        S_action = (S_action + S_action.mT) / 2.0
        S_action = (
            S_action
            + torch.eye(self.output_dim, device=device, dtype=m.dtype).unsqueeze(0)
            * 1e-6
        )

        if self.max_action is not None:
            M, S_action, C = self.squash_sin(M, S_action, self.max_action)
            V = torch.bmm(V, C)

        return M, S_action, V
