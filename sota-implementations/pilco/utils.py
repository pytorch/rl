from collections.abc import Sequence

import torch
import torch.nn as nn
from botorch.fit import fit_gpytorch_mll

from botorch.models import ModelListGP, SingleTaskGP
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.mlls import SumMarginalLogLikelihood
from gpytorch.priors import GammaPrior

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


class BoTorchGPWorldModel(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.input_dim = obs_dim + action_dim

        self.model_list: ModelListGP | None = None

        self.register_buffer("X_train", torch.empty(0))
        self.register_buffer("lengthscales", torch.zeros(self.obs_dim, self.input_dim))
        self.register_buffer("variances", torch.zeros(self.obs_dim, 1))
        self.register_buffer("noises", torch.zeros(self.obs_dim))
        self._cached_inv_K: torch.Tensor | None = None
        self._cached_beta: torch.Tensor | None = None

    @property
    def device(self) -> torch.device:
        return self.lengthscales.device

    def fit(self, dataset: TensorDictBase) -> None:
        obs = dataset["observation"]
        action = dataset["action"]
        next_obs = dataset[("next", "observation")]

        X_train = torch.cat([obs, action], dim=-1).detach().to(self.device)
        y_train = (next_obs - obs).detach().to(self.device)
        self.X_train = X_train

        models = []
        for i in range(self.obs_dim):
            train_x = X_train
            train_y = y_train[:, i].unsqueeze(-1)

            covar_module = ScaleKernel(
                RBFKernel(
                    ard_num_dims=self.input_dim, lengthscale_prior=GammaPrior(1.1, 0.1)
                ),
                outputscale_prior=GammaPrior(1.5, 0.5),
            )

            gp = SingleTaskGP(
                train_X=train_x, train_Y=train_y, covar_module=covar_module
            )
            gp.likelihood.noise_covar.register_prior(
                "noise_prior", GammaPrior(1.2, 0.05), "noise"
            )

            models.append(gp)

        self.model_list = ModelListGP(*models).to(self.device)
        mll = SumMarginalLogLikelihood(self.model_list.likelihood, self.model_list)

        fit_gpytorch_mll(mll)
        self._extract_parameters(y_train)

    def _extract_parameters(self, y_train: torch.Tensor) -> None:
        lengthscales, variances, noises, inv_Ks, betas = [], [], [], [], []

        for i, gp in enumerate(self.model_list.models):
            gp.eval()
            gp.likelihood.eval()

            ls = gp.covar_module.base_kernel.lengthscale.squeeze().detach()
            var = gp.covar_module.outputscale.detach()
            noise = gp.likelihood.noise.squeeze().detach()

            lengthscales.append(ls)
            variances.append(var)
            noises.append(noise)

            X_scaled = self.X_train / ls
            dist = torch.cdist(X_scaled, X_scaled, p=2) ** 2
            K = var * torch.exp(-0.5 * dist)

            K_noisy = K + (noise + 1e-6) * torch.eye(
                self.X_train.size(0), device=self.device
            )

            L = torch.linalg.cholesky(K_noisy)
            eye = torch.eye(L.size(0), dtype=L.dtype, device=L.device)
            inv_K = torch.cholesky_solve(eye, L)

            y = y_train[:, i].unsqueeze(-1)
            beta = torch.cholesky_solve(y, L).squeeze(-1)

            inv_Ks.append(inv_K)
            betas.append(beta)

        self.lengthscales = torch.stack(lengthscales)
        self.variances = torch.stack(variances).unsqueeze(-1)
        self.noises = torch.stack(noises)

        self._cached_inv_K = torch.stack(inv_Ks)
        self._cached_beta = torch.stack(betas)

    def compute_factorizations(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self._cached_inv_K, self._cached_beta

    def _gather_gp_params(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.lengthscales, self.variances, self.noises

    def forward(
        self, action: TensorDictBase, observation: TensorDictBase
    ) -> tuple[torch.Tensor, torch.Tensor]:
        observation_uncertain = False
        x_var = observation.get("var")
        if x_var is not None:
            observation_uncertain = not torch.all(
                torch.isclose(x_var, torch.zeros_like(x_var))
            )
        if observation_uncertain:
            return self.uncertain_forward(action, observation)
        else:
            return self.deterministic_forward(action, observation)

    def freeze_and_detach(self) -> None:
        pass

    def uncertain_forward(
        self, action: TensorDictBase, obs: TensorDictBase
    ) -> tuple[torch.Tensor, torch.Tensor]:
        inv_K, beta = self.compute_factorizations()
        lengthscales, variances, noises = self._gather_gp_params()

        m_x, s_x = obs.get("mean"), obs.get("var")
        m_u, s_u, c_xu = (
            action.get("mean"),
            action.get("var"),
            action.get("cross_covariance"),
        )

        device, dtype = m_x.device, m_x.dtype

        joint_mean = torch.cat([m_x, m_u], dim=-1)

        s_ = s_x @ c_xu
        upper = torch.cat([s_x, s_], dim=-1)
        lower = torch.cat([s_.transpose(-1, -2), s_u], dim=-1)

        joint_var = torch.cat([upper, lower], dim=-2)

        X_train = self.X_train
        num_train_pts = X_train.shape[0]
        batch_size = joint_mean.shape[0]

        inp = X_train - joint_mean.unsqueeze(1)

        inv_L = torch.diag_embed(1.0 / lengthscales).to(dtype=dtype, device=device)
        inv_N = inp.unsqueeze(1) @ inv_L.unsqueeze(0)

        B_mat = inv_L.unsqueeze(0) @ joint_var.unsqueeze(1) @ inv_L.unsqueeze(0)
        B_mat = B_mat + torch.eye(
            self.input_dim, dtype=m_x.dtype, device=m_x.device
        ).view(1, 1, self.input_dim, self.input_dim)

        t = torch.linalg.solve(B_mat, inv_N.transpose(-2, -1)).transpose(-2, -1)

        scaled_exp = torch.exp(-torch.sum(inv_N * t, dim=-1) / 2)
        lb = scaled_exp * beta.unsqueeze(0)

        det_B = torch.linalg.det(B_mat)
        c = variances.squeeze(1).unsqueeze(0) / torch.sqrt(det_B)

        pred_mean = torch.sum(lb, dim=-1) * c.squeeze(0)

        t_inv_L = t @ inv_L.unsqueeze(0)

        cross_cov_E_D = torch.matmul(
            t_inv_L.transpose(-2, -1), lb.unsqueeze(-1)
        ).squeeze(-1) * c.unsqueeze(-1)
        cross_cov = cross_cov_E_D.transpose(-2, -1)

        pred_cov = torch.zeros(
            batch_size, self.obs_dim, self.obs_dim, dtype=m_x.dtype, device=m_x.device
        )

        X_i = X_train.unsqueeze(1)
        X_j = X_train.unsqueeze(0)
        diff = X_i - X_j
        joint_mean_flat = joint_mean.unsqueeze(1).unsqueeze(1)

        for a in range(self.obs_dim):
            for b in range(self.obs_dim):
                l2_a = lengthscales[a].to(device=device, dtype=dtype) ** 2
                l2_b = lengthscales[b].to(device=device, dtype=dtype) ** 2

                inv_L_a = 1.0 / l2_a
                inv_L_b = 1.0 / l2_b
                inv_L_sum = inv_L_a + inv_L_b
                Lambda_ab = 1.0 / inv_L_sum

                z_bar = Lambda_ab * (X_i * inv_L_a + X_j * inv_L_b)
                z = z_bar.unsqueeze(0) - joint_mean_flat

                z_flat = z.view(
                    batch_size, num_train_pts * num_train_pts, self.input_dim
                )

                R_ab = joint_var @ torch.diag(inv_L_sum) + torch.eye(
                    self.input_dim, dtype=m_x.dtype, device=m_x.device
                ).unsqueeze(0)

                inv_L_plus = 1.0 / (l2_a + l2_b)
                exp1 = -0.5 * torch.sum(diff * inv_L_plus * diff, dim=-1)

                M_ab = joint_var + torch.diag(Lambda_ab).unsqueeze(0)

                solved_z_flat = torch.linalg.solve(
                    M_ab, z_flat.transpose(-2, -1)
                ).transpose(-2, -1)
                exp2 = (-0.5 * torch.sum(z_flat * solved_z_flat, dim=-1)).view(
                    batch_size, num_train_pts, num_train_pts
                )

                det_R_ab = torch.linalg.det(R_ab)
                c_ab = variances[a] * variances[b] / torch.sqrt(det_R_ab)

                Q_ab = c_ab.view(-1, 1, 1) * torch.exp(exp1.unsqueeze(0) + exp2)

                Qb = torch.matmul(Q_ab, beta[b])
                pred_cov[:, a, b] = (
                    torch.matmul(beta[a].unsqueeze(0), Qb.unsqueeze(-1))
                    .squeeze(-1)
                    .squeeze(-1)
                )

                if a == b:
                    invK_Q = torch.matmul(inv_K[a].unsqueeze(0), Q_ab)
                    trace_val = torch.diagonal(invK_Q, dim1=-2, dim2=-1).sum(-1)

                    pred_cov[:, a, a] += variances[a] - trace_val + noises[a].item()

        outer_mean = torch.bmm(pred_mean.unsqueeze(-1), pred_mean.unsqueeze(-2))
        pred_cov = pred_cov - outer_mean

        pred_cov = (pred_cov + pred_cov.transpose(-2, -1)) / 2.0

        m_dx = pred_mean
        s_dx = pred_cov
        c_xdx = cross_cov

        cov_xf = upper @ c_xdx

        m_x = m_x + m_dx

        s_x = s_x + s_dx + cov_xf + cov_xf.transpose(-2, -1)

        s_x = (s_x + s_x.transpose(-2, -1)) / 2.0
        s_x = s_x + 1e-8 * torch.eye(self.obs_dim, device=s_x.device).expand(
            s_x.shape[0], -1, -1
        )
        return m_x, s_x

    def deterministic_forward(
        self, action: TensorDictBase, observation: TensorDictBase
    ) -> tuple[torch.Tensor, torch.Tensor]:
        observation_mean = observation.get("mean")
        action_mean = action.get("mean")

        x_flat = observation_mean.view(-1, self.obs_dim)
        u_flat = action_mean.view(-1, self.action_dim)

        X_test = torch.cat([x_flat, u_flat], dim=-1)

        means, stds = [], []

        with torch.no_grad():
            for gp in self.model_list.models:
                posterior = gp.posterior(X_test)
                means.append(posterior.mean.squeeze(-1))
                stds.append(torch.sqrt(posterior.variance).squeeze(-1))

        delta_mean_flat = torch.stack(means, dim=-1)
        delta_std_flat = torch.stack(stds, dim=-1)

        batch_shape = observation_mean.shape[:-1]
        delta_mean = delta_mean_flat.view(*batch_shape, self.obs_dim)
        delta_std = delta_std_flat.view(*batch_shape, self.obs_dim)

        return observation_mean + delta_mean, torch.diag_embed(delta_std**2)


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
