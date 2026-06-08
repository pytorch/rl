from dataclasses import dataclass

import torch
from tensordict import TensorDict, TensorDictBase
from torchrl.objectives.common import LossModule


class ExponentialQuadraticCost(LossModule):
    """Computes the expected saturating cost for a Gaussian-distributed state.

    This serves as a smooth, unimodal approximation of a 0-1 cost over a target area,
    allowing for analytic gradient computation during policy search (e.g., PILCO).
    Calculates E_{x_t}[c(x_t)] over N(m, s) as defined in Eq. (24) and (25) of
    Deisenroth & Rasmussen (2011).

    Args:
        target (torch.Tensor, optional): The target state vector. Defaults to the origin.
        weights (torch.Tensor, optional): The precision matrix mapping state dimensions
            to the cost distance metric. Defaults to the identity matrix.
        reduction (str, optional): Specifies the reduction to apply to the output:
            'mean' | 'sum' | 'none'. Defaults to 'mean'.
    """

    @dataclass
    class _AcceptedKeys:
        """Maintains default values for configurable tensordict keys."""

        loc: str | tuple[str, ...] = ("observation", "mean")
        scale: str | tuple[str, ...] = ("observation", "var")
        loss_cost: str | tuple[str, ...] = "loss_cost"

    default_keys = _AcceptedKeys

    def __init__(
        self,
        target: torch.Tensor | None = None,
        weights: torch.Tensor | None = None,
        reduction: str = "mean",
    ):
        super().__init__()
        self._tensor_keys = self._AcceptedKeys()
        self.reduction = reduction

        self.register_buffer("target", target)
        self.register_buffer("weights", weights)

    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        m = tensordict.get(self.tensor_keys.loc)
        s = tensordict.get(self.tensor_keys.scale)

        batch_shape = m.shape[:-1]
        D = m.shape[-1]
        device = m.device
        dtype = m.dtype

        weights = (
            self.weights
            if self.weights is not None
            else torch.eye(D, device=device, dtype=dtype)
        )
        target = (
            self.target
            if self.target is not None
            else torch.zeros(D, device=device, dtype=dtype)
        )

        if target.dim() == 1:
            target_shape = (*[1] * len(batch_shape), D)
            target = target.view(*target_shape).expand(*batch_shape, D)

        eye = torch.eye(D, device=device, dtype=dtype)
        eye_batch = eye.view(*[1] * len(batch_shape), D, D)

        # diff: Distance from the current mean to the target (x - x_target)
        diff = (m - target).unsqueeze(-1)

        # L_w, V_w: Eigenvalues and eigenvectors of the precision weight matrix
        L_w, V_w = torch.linalg.eigh(weights)
        L_w = torch.clamp(L_w, min=0.0)

        # U: Scaled transformation matrix for the cost weighting
        U = V_w @ torch.diag_embed(torch.sqrt(L_w)) @ V_w.transpose(-2, -1)

        # A_sym: Covariance transformation required for computing the expected cost integral
        # U is (D, D), s is (*batch_shape, D, D)
        A_sym = eye_batch + torch.matmul(U, torch.matmul(s, U))

        jitter = 1e-5
        A_sym = A_sym + jitter * eye_batch

        # L: Cholesky decomposition of A_sym for numerical stability
        L = torch.linalg.cholesky(A_sym)

        # Determinant and exponential terms for the closed-form expected cost
        log_det = 2.0 * torch.log(torch.diagonal(L, dim1=-2, dim2=-1)).sum(-1)
        det_term = torch.exp(-0.5 * log_det)

        # Mahalanobis distance components scaled by the target weights
        # U @ diff needs broadcasting
        v = torch.matmul(U.view(*[1] * len(batch_shape), D, D), diff)
        tmp = torch.cholesky_solve(v, L)
        quad = torch.matmul(v.transpose(-2, -1), tmp)
        exp_term = (-0.5 * quad).squeeze(-1).squeeze(-1)

        # Expected cost bounded in [0, 1]
        cost = 1.0 - det_term * torch.exp(exp_term)

        if self.reduction == "mean":
            loss = cost.mean()
            out_batch_size = []
        elif self.reduction == "sum":
            loss = cost.sum()
            out_batch_size = []
        elif self.reduction == "none":
            loss = cost
            out_batch_size = batch_shape
        else:
            raise ValueError(f"Unsupported reduction: {self.reduction}")
        return TensorDict({self.tensor_keys.loss_cost: loss}, batch_size=out_batch_size)
