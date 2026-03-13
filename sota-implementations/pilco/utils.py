from collections.abc import Sequence

import torch
import torch.nn as nn

from tensordict import TensorDict, TensorDictBase
from tensordict.nn import TensorDictModule

from torchrl.envs import (
    EnvBase,
    GymEnv,
    ModelBasedEnvBase,
    RewardSum,
    StepCounter,
    TransformedEnv,
)


def make_env(
    env_name: str, device: str | torch.device, from_pixels: bool = False
) -> TransformedEnv:
    """Creates the transformed environment."""
    env = TransformedEnv(
        GymEnv(env_name, pixels_only=False, from_pixels=from_pixels, device=device)
    )
    env.append_transform(RewardSum())
    env.append_transform(StepCounter())
    return env


def pendulum_cost(
    obs: TensorDictBase,
    weights: torch.Tensor | None = None,
    target: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    obs["mean"]: [B, T, D]
    obs["var"] : [B, T, D, D]
    """
    m = obs.get("mean")
    s = obs.get("var")

    B, T, D = m.shape
    device = m.device
    dtype = m.dtype

    if weights is None:
        diag_vals = torch.tensor([1.0, 1.0, 1.0, 1.0], device=device, dtype=dtype)
        weights = torch.diag(diag_vals)

    if target is None:
        target = torch.zeros(D, device=device, dtype=dtype)

    if target.dim() == 1:
        target = target.view(1, 1, D).expand(B, T, D)

    eye = torch.eye(D, device=device, dtype=dtype).view(1, 1, D, D)
    diff = (m - target).unsqueeze(-1)  # [B, T, D, 1]

    L_w, V_w = torch.linalg.eigh(weights)
    L_w = torch.clamp(L_w, min=0.0)
    U = V_w @ torch.diag_embed(torch.sqrt(L_w)) @ V_w.transpose(-2, -1)

    A_sym = eye + torch.matmul(U, torch.matmul(s, U))

    jitter = 1e-5
    A_sym = A_sym + jitter * eye

    L = torch.linalg.cholesky(A_sym)

    log_det = 2.0 * torch.log(torch.diagonal(L, dim1=-2, dim2=-1)).sum(-1)
    det_term = torch.exp(-0.5 * log_det)

    v = torch.matmul(U, diff)
    tmp = torch.cholesky_solve(v, L)
    quad = torch.matmul(v.transpose(-2, -1), tmp)
    exp_term = (-0.5 * quad).squeeze(-1).squeeze(-1)

    return (1.0 - det_term * torch.exp(exp_term)).sum(dim=1)


class ImaginedEnv(ModelBasedEnvBase):
    def __init__(
        self,
        world_model_module: TensorDictModule,
        base_env: EnvBase,
        batch_size: int | torch.Size | Sequence[int] | None = None,
        **kwargs
    ) -> None:
        if batch_size is not None:
            self.batch_size = (
                torch.Size(batch_size)
                if not isinstance(batch_size, torch.Size)
                else batch_size
            )
        elif len(base_env.batch_size) == 0:
            self.batch_size = torch.Size([1])
        else:
            self.batch_size = base_env.batch_size

        super().__init__(
            world_model_module,
            device=base_env.device,
            batch_size=self.batch_size,
            **kwargs
        )

        self.observation_spec = base_env.observation_spec.expand(
            self.batch_size
        ).clone()
        self.action_spec = base_env.action_spec.expand(self.batch_size).clone()
        self.reward_spec = base_env.reward_spec.expand(self.batch_size).clone()
        self.done_spec = base_env.done_spec.expand(self.batch_size).clone()

    def _step(self, tensordict: TensorDictBase) -> TensorDictBase:
        tensordict = self.world_model(tensordict)

        reward = torch.zeros(*tensordict.shape, 1, device=self.device)
        done = torch.zeros(*tensordict.shape, 1, dtype=torch.bool, device=self.device)
        out = TensorDict(
            {
                "observation": tensordict.get("next_observation"),
                "reward": reward,
                "done": done,
                "terminated": done.clone(),
            },
            tensordict.shape,
        )
        return out

    def _reset(
        self, tensordict: TensorDictBase | None = None, **kwargs
    ) -> TensorDictBase:
        if tensordict is None:
            tensordict = TensorDict({}, batch_size=self.batch_size, device=self.device)

        if (
            tensordict.get(("observation", "var"), None) is not None
            and tensordict.get(("observation", "mean"), None) is not None
        ):
            return tensordict.copy()

        obs = tensordict.get("observation", None)
        if obs is None:
            obs = self.observation_spec.rand(shape=self.batch_size).get("observation")
        if obs.ndim == 1:
            obs = obs.expand(self.batch_size, -1)

        obs = obs.to(self.device)
        B, D = obs.shape

        out = TensorDict(
            {
                ("observation", "mean"): obs,
                ("observation", "var"): torch.zeros(
                    B, D, D, dtype=obs.dtype, device=self.device
                ),
            },
            batch_size=self.batch_size,
            device=self.device,
        )

        out.set("done", torch.zeros(B, 1, dtype=torch.bool, device=self.device))
        out.set("terminated", torch.zeros(B, 1, dtype=torch.bool, device=self.device))

        return out


class RBFController(nn.Module):
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
