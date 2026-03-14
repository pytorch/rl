# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import importlib.util

import torch
import torch.nn as nn
from tensordict import TensorDictBase

_has_gpytorch = importlib.util.find_spec("gpytorch") is not None
_has_botorch = importlib.util.find_spec("botorch") is not None


class GPWorldModel(nn.Module):
    """Gaussian Process World Model for moment-matching model-based RL.

    Fits one independent single-task GP per observation dimension using
    BoTorch/GPyTorch. Each GP models the *transition residual*
    ``delta_i = next_obs_i - obs_i`` given the concatenated ``[obs, action]``
    input. After fitting, the model supports two forward modes:

    * **Deterministic**: point predictions via the GP posterior mean/variance.
    * **Uncertain** (moment-matching): propagates Gaussian beliefs
      ``N(m, S)`` through the GP analytically, yielding the next-state
      belief ``N(m', S')``. This is the core computation in PILCO
      (Deisenroth & Rasmussen, 2011).

    Requires ``botorch`` and ``gpytorch`` as optional dependencies.

    Args:
        obs_dim (int): The dimension of the observation space.
        action_dim (int): The dimension of the action space.
        in_keys (list[str | tuple[str, ...]] | None, optional): The keys to read from the
            input TensorDict. Defaults to ["action", "observation"].
        out_keys (list[str | tuple[str, ...]] | None, optional): The keys to write the
            predicted mean and variance to in the output TensorDict.
            Defaults to [("next", "observation"), ("next", "observation_var")].

    Examples:
        >>> import torch
        >>> from tensordict import TensorDict
        >>> model = GPWorldModel(obs_dim=4, action_dim=1)  # doctest: +SKIP
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        in_keys: list[str | tuple[str, ...]] | None = None,
        out_keys: list[str | tuple[str, ...]] | None = None,
    ) -> None:
        if not _has_botorch or not _has_gpytorch:
            raise ImportError(
                "botorch and gpytorch are required to use GPWorldModel. "
                "Please install them to proceed."
            )
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.input_dim = obs_dim + action_dim

        self.in_keys = (
            in_keys
            if in_keys is not None
            else [
                ("action", "mean"),
                ("action", "var"),
                ("action", "cross_covariance"),
                ("observation", "mean"),
                ("observation", "var"),
            ]
        )

        self.out_keys = (
            out_keys
            if out_keys is not None
            else [
                ("next", "observation", "mean"),
                ("next", "observation", "var"),
            ]
        )

        self.model_list = None

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
        """Fits the Gaussian Process model to the provided dataset.

        The dataset must contain the ``"observation"``, ``"action"``, and
        ``("next", "observation")`` keys. The model predicts the difference
        between the next observation and the current observation.

        Args:
            dataset (TensorDictBase): A dataset of collected transitions.
        """
        from botorch.fit import fit_gpytorch_mll
        from botorch.models import ModelListGP, SingleTaskGP
        from gpytorch.kernels import RBFKernel, ScaleKernel
        from gpytorch.mlls import SumMarginalLogLikelihood
        from gpytorch.priors import GammaPrior

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
        """Returns the cached kernel inverse and weight vectors.

        Returns:
            inv_K (Tensor): Inverse kernel matrices, shape ``(obs_dim, N, N)``.
            beta (Tensor): Weight vectors ``K^{-1} y``, shape ``(obs_dim, N)``.
        """
        return self._cached_inv_K, self._cached_beta

    def _gather_gp_params(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns the extracted hyperparameters of each per-dimension GP."""
        return self.lengthscales, self.variances, self.noises

    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        """Forward pass for the GPWorldModel.

        Routes the request to either the deterministic or uncertain forward pass
        depending on whether the observation input contains variance.

        Args:
            tensordict (TensorDictBase): The input tensordict containing the action and observation.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: A tuple containing the mean and
            variance tensors of the next observation.
        """
        u_mean_key, u_var_key, u_cc_key, x_mean_key, x_var_key = self.in_keys

        x_var = tensordict.get(x_var_key, None)
        observation_uncertain = False
        if x_var is not None:
            observation_uncertain = not torch.all(
                torch.isclose(x_var, torch.zeros_like(x_var))
            )

        if observation_uncertain:
            return self.uncertain_forward(tensordict)
        else:
            return self.deterministic_forward(tensordict)

    def uncertain_forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        """Calculates the forward pass when the observation has uncertainty (non-zero variance).

        Propagates uncertainty through the Gaussian Process via exact moment matching.

        Args:
            tensordict (TensorDictBase): A tensordict containing the action and observation tensors.

        Returns:
            TensorDictBase: Next observation mean and variance matrices.
        """
        inv_K, beta = self.compute_factorizations()
        lengthscales, variances, noises = self._gather_gp_params()
        u_mean_key, u_var_key, u_cc_key, x_mean_key, x_var_key = self.in_keys

        m_x, s_x = tensordict.get(x_mean_key), tensordict.get(x_var_key)
        m_u, s_u, c_xu = (
            tensordict.get(u_mean_key),
            tensordict.get(u_var_key),
            tensordict.get(u_cc_key),
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
        ).reshape(1, 1, self.input_dim, self.input_dim)

        t = torch.linalg.solve(B_mat, inv_N.transpose(-2, -1)).transpose(-2, -1)

        scaled_exp = torch.exp(-torch.sum(inv_N * t, dim=-1) / 2)
        lb = scaled_exp * beta.unsqueeze(0)

        _, log_det_B = torch.linalg.slogdet(B_mat)
        c = variances.squeeze(1).unsqueeze(0) * torch.exp(-0.5 * log_det_B)

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

                _, log_det_R_ab = torch.linalg.slogdet(R_ab)
                c_ab = variances[a] * variances[b] * torch.exp(-0.5 * log_det_R_ab)

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

                    pred_cov[:, a, a] += variances[a] - trace_val + noises[a]

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

        out_mean_key, out_var_key = self.out_keys
        tensordict.set(out_mean_key, m_x)
        tensordict.set(out_var_key, s_x)
        return tensordict

    def deterministic_forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        """Calculates the forward pass when the input observation is deterministic (no variance).

        Args:
            tensordict (TensorDictBase): A tensordict containing the action and observation tensors.

        Returns:
            TensorDictBase: Next observation mean and variance matrices.
        """
        u_mean_key, u_var_key, u_cc_key, x_mean_key, x_var_key = self.in_keys
        observation_mean = tensordict.get(x_mean_key)
        action_mean = tensordict.get(u_mean_key)

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

        m_x = observation_mean + delta_mean
        s_x = torch.diag_embed(delta_std**2)

        out_mean_key, out_var_key = self.out_keys
        tensordict.set(out_mean_key, m_x)
        tensordict.set(out_var_key, s_x)
        return tensordict
