# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# Variable naming follows Deisenroth & Rasmussen (2011), "PILCO: A Model-Based
# and Data-Efficient Approach to Policy Search" (cited inline as "Eq. N").
#
# Key symbols
# -----------
#   x̃  := [x, u]              concatenated state-action input  (Eq. 1 / Sec. 2.1)
#   Δ  := x_t - x_{t-1}       transition residual              (Sec. 2.1)
#   K_a                        Gram matrix  K_{a,ij}=k_a(x̃_i,x̃_j)  (Eq. 6)
#   β_a := (K_a + σ²_ε I)^{-1}y_a  GP weight vector           (Eq. 7)
#   q_a                        kernel-mean vector               (Eq. 15)
#   Q_{ab}                     cross-kernel matrix              (Eqs. 21-22)
#   μ̃  / Σ̃                    joint state-action mean/cov      (Sec. 2.2)
#   μ_Δ / Σ_Δ                  predictive mean/cov of Δ        (Eqs. 14, 17-23)
#   μ_t / Σ_t                  next-state mean/cov             (Eqs. 10-11)

import importlib.util

import torch
import torch.nn as nn
from tensordict import TensorDictBase

_has_gpytorch = importlib.util.find_spec("gpytorch") is not None
_has_botorch = importlib.util.find_spec("botorch") is not None


class GPWorldModel(nn.Module):
    """Gaussian Process world model with moment-matching uncertainty propagation.

    Implements the probabilistic dynamics model from PILCO
    (Deisenroth & Rasmussen, 2011). One independent GP is fit per state
    dimension, each predicting the transition residual
    ``Δ = x_t - x_{t-1}`` from the concatenated state-action input
    ``x̃ = [x, u]`` (Sec. 2.1).

    :meth:`forward` supports two modes depending on whether the input
    observation carries non-zero variance:

    - **Deterministic**: uses the GP posterior mean and variance directly
      (Eqs. 7-8).
    - **Uncertain** (moment-matching): propagates a Gaussian belief
      ``N(μ, Σ)`` through the GP analytically (Eqs. 10-23).

    .. note::
        Requires ``botorch`` and ``gpytorch`` as optional dependencies.

    Args:
        obs_dim (int): Dimension D of the observation (state) space.
        action_dim (int): Dimension F of the action (control) space.
        in_keys (list of NestedKey, optional): Keys to read from the input
            :class:`~tensordict.TensorDictBase`. Must contain five entries in
            order: action mean, action covariance, state-action
            cross-covariance, observation mean, observation covariance.
            Defaults to ``[("action", "mean"), ("action", "var"),
            ("action", "cross_covariance"), ("observation", "mean"),
            ("observation", "var")]``.
        out_keys (list of NestedKey, optional): Keys to write the predicted
            next-state mean and covariance to. Defaults to
            ``[("next", "observation", "mean"),
            ("next", "observation", "var")]``.

    Examples:
        >>> import torch
        >>> from tensordict import TensorDict
        >>> model = GPWorldModel(obs_dim=4, action_dim=1)
        >>> dataset = TensorDict(
        ...     {
        ...         "observation": torch.randn(50, 4),
        ...         "action": torch.randn(50, 1),
        ...         ("next", "observation"): torch.randn(50, 4),
        ...     },
        ...     batch_size=[50],
        ... )
        >>> model.fit(dataset)

    Reference:
        Deisenroth, M. P. & Rasmussen, C. E. (2011). PILCO: A model-based
        and data-efficient approach to policy search. *ICML*.
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
        self.obs_dim = obs_dim  # D in the paper
        self.action_dim = action_dim  # F in the paper
        self.state_action_dim = obs_dim + action_dim  # D+F, dimension of x̃ (Sec. 2.1)

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

        # X̃ = [x̃_1, ..., x̃_n] ∈ R^{n×(D+F)}  – training inputs (Sec. 2.1)
        self.register_buffer("X_tilde_train", torch.empty(0))

        # ℓ_a ∈ R^{D+F}  – ARD length-scales for each output dimension a (Eq. 6).
        # Stored as [D, D+F]; the full matrix Λ_a = diag(ℓ_a²) is never
        # materialised — ℓ_a is squared on the fly wherever needed.
        # Note: GPyTorch's .lengthscale returns ℓ directly (not ℓ²).
        self.register_buffer("ell", torch.zeros(self.obs_dim, self.state_action_dim))

        # α²_a  – signal variance for each output dimension a (Eq. 6); shape [D, 1]
        self.register_buffer("alpha_sq", torch.zeros(self.obs_dim, 1))

        # σ²_{ε_a}  – noise variance for each output dimension a (Sec. 2.1); shape [D]
        self.register_buffer("sigma_sq_eps", torch.zeros(self.obs_dim))

        # (K_a + σ²_{ε_a} I)^{-1}  – cached inverse Gram matrices (Eq. 7); shape [D, n, n].
        # Registered as buffers so they survive .to(device) and state_dict round-trips.
        self.register_buffer("_cached_inv_K_noisy", None)

        # β_a = (K_a + σ²_{ε_a} I)^{-1} y_a  – GP weight vectors (Eq. 7); shape [D, n].
        # Registered as a buffer so it survives .to(device) and state_dict round-trips.
        self.register_buffer("_cached_beta", None)

    @property
    def device(self) -> torch.device:
        return self.ell.device

    def fit(self, dataset: TensorDictBase) -> None:
        """Fit one GP per state dimension to a dataset of transitions.

        Constructs training inputs ``X̃ = [x, u]`` and targets
        ``Δ_a = x_{t,a} - x_{t-1,a}``, then maximises the marginal
        log-likelihood to learn SE kernel hyper-parameters
        (ℓ_a, α²_a, σ²_{ε_a}) for each output dimension (Sec. 2.1, Eq. 6).

        .. note::
            The dataset is expected to be flat with shape ``[n, *]``. If your
            replay buffer returns multi-dimensional batches (e.g. ``[B, T, *]``),
            call ``dataset.reshape(-1)`` before passing it here.

        Args:
            dataset (TensorDictBase): Transition dataset with keys
                ``"observation"`` of shape ``(n, D)``,
                ``"action"`` of shape ``(n, F)``, and
                ``("next", "observation")`` of shape ``(n, D)``.
        """
        from botorch.fit import fit_gpytorch_mll
        from botorch.models import ModelListGP, SingleTaskGP
        from gpytorch.kernels import RBFKernel, ScaleKernel
        from gpytorch.mlls import SumMarginalLogLikelihood
        from gpytorch.priors import GammaPrior

        x_t_minus_1 = dataset["observation"]  # x_{t-1} ∈ R^{n×D}
        u_t_minus_1 = dataset["action"]  # u_{t-1} ∈ R^{n×F}
        x_t = dataset[("next", "observation")]  # x_t     ∈ R^{n×D}

        # x̃ = [x_{t-1}, u_{t-1}] ∈ R^{n×(D+F)}  – training inputs (Sec. 2.1)
        X_tilde_train = (
            torch.cat([x_t_minus_1, u_t_minus_1], dim=-1).detach().to(self.device)
        )

        # Δ ∈ R^{n×D},  Δ_{i,a} = x_{t,a} - x_{t-1,a}  – training targets (Sec. 2.1)
        Delta_train = (x_t - x_t_minus_1).detach().to(self.device)

        self.X_tilde_train = X_tilde_train

        models = []
        for a in range(self.obs_dim):
            # Each GP_a models p(Δ_a | x̃) independently (Sec. 2.1)
            Delta_a = Delta_train[:, a].unsqueeze(-1)  # y_a ∈ R^{n×1}

            covar_module = ScaleKernel(
                # SE kernel k_a(x̃, x̃') with ARD length-scales (one ℓ_{a,i}
                # per input dimension, Eq. 6)
                RBFKernel(
                    ard_num_dims=self.state_action_dim,
                    lengthscale_prior=GammaPrior(1.1, 0.1),
                ),
                outputscale_prior=GammaPrior(1.5, 0.5),  # prior on α²_a (Eq. 6)
            )

            gp_a = SingleTaskGP(
                train_X=X_tilde_train,
                train_Y=Delta_a,
                covar_module=covar_module,
            )
            gp_a.likelihood.noise_covar.register_prior(
                "noise_prior",
                GammaPrior(1.2, 0.05),
                "noise",  # prior on σ²_{ε_a} (Sec. 2.1)
            )

            models.append(gp_a)

        self.model_list = ModelListGP(*models).to(self.device)
        mll = SumMarginalLogLikelihood(self.model_list.likelihood, self.model_list)

        fit_gpytorch_mll(mll)  # evidence maximisation (Sec. 2.1)
        self._extract_and_cache_parameters(Delta_train)

    def _extract_and_cache_parameters(self, Delta_train: torch.Tensor) -> None:
        # Extract learned hyper-parameters from each GP_a and pre-compute the
        # quantities that are fixed after fitting:
        #   ℓ_a,  α²_a,  σ²_{ε_a}               (Eq. 6 / Sec. 2.1)
        #   (K_a + σ²_{ε_a} I)^{-1}              (Eq. 7)
        #   β_a = (K_a + σ²_{ε_a} I)^{-1} y_a   (Eq. 7)
        ell_list, alpha_sq_list, sigma_sq_eps_list = [], [], []
        inv_K_noisy_list, beta_list = [], []

        n = self.X_tilde_train.shape[0]  # number of training points

        for a, gp_a in enumerate(self.model_list.models):
            gp_a.eval()
            gp_a.likelihood.eval()

            # ℓ_a ∈ R^{D+F}  – ARD length-scales for GP_a (Eq. 6).
            # GPyTorch's .lengthscale returns ℓ directly (not ℓ²).
            ell_a = gp_a.covar_module.base_kernel.lengthscale.squeeze().detach()

            # α²_a  – signal variance for GP_a (Eq. 6)
            alpha_sq_a = gp_a.covar_module.outputscale.detach()

            # σ²_{ε_a}  – noise variance for GP_a (Sec. 2.1)
            sigma_sq_eps_a = gp_a.likelihood.noise.squeeze().detach()

            ell_list.append(ell_a)
            alpha_sq_list.append(alpha_sq_a)
            sigma_sq_eps_list.append(sigma_sq_eps_a)

            # K_{a,ij} = α²_a exp(-½ (x̃_i-x̃_j)^T Λ_a^{-1} (x̃_i-x̃_j))  (Eq. 6)
            # Dividing X̃ by ℓ_a gives Λ_a^{-1/2}-scaled inputs for cdist.
            X_tilde_scaled = self.X_tilde_train / ell_a
            sq_dist = torch.cdist(X_tilde_scaled, X_tilde_scaled, p=2) ** 2
            K_a = alpha_sq_a * torch.exp(-0.5 * sq_dist)

            # K_{a,noisy} = K_a + σ²_{ε_a} I  (denominator in Eq. 7)
            K_a_noisy = K_a + (sigma_sq_eps_a + 1e-6) * torch.eye(n, device=self.device)

            L_a = torch.linalg.cholesky(K_a_noisy)
            eye_n = torch.eye(n, dtype=L_a.dtype, device=L_a.device)

            # (K_a + σ²_{ε_a} I)^{-1}  (Eq. 7)
            inv_K_a_noisy = torch.cholesky_solve(eye_n, L_a)

            # y_a = [Δ_{1,a}, ..., Δ_{n,a}]^T  – targets for GP_a (Sec. 2.1)
            y_a = Delta_train[:, a].unsqueeze(-1)

            # β_a = (K_a + σ²_{ε_a} I)^{-1} y_a  (Eq. 7)
            beta_a = torch.cholesky_solve(y_a, L_a).squeeze(-1)

            inv_K_noisy_list.append(inv_K_a_noisy)
            beta_list.append(beta_a)

        self.ell = torch.stack(ell_list)  # [D, D+F]
        self.alpha_sq = torch.stack(alpha_sq_list).unsqueeze(-1)  # [D, 1]
        self.sigma_sq_eps = torch.stack(sigma_sq_eps_list)  # [D]
        self._cached_inv_K_noisy = torch.stack(inv_K_noisy_list)  # [D, n, n]
        self._cached_beta = torch.stack(beta_list)  # [D, n]

    def compute_factorizations(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the cached kernel inverses and GP weight vectors.

        Returns:
            tuple[Tensor, Tensor]: A pair ``(inv_K_noisy, beta)`` where
            ``inv_K_noisy`` has shape ``(D, n, n)`` and contains
            ``(K_a + σ²_{ε_a} I)^{-1}`` for each output dimension (Eq. 7),
            and ``beta`` has shape ``(D, n)`` and contains
            ``β_a = (K_a + σ²_{ε_a} I)^{-1} y_a`` (Eq. 7).
        """
        return self._cached_inv_K_noisy, self._cached_beta

    def _gather_gp_hyperparams(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Returns (ell, alpha_sq, sigma_sq_eps) — the SE kernel hyper-parameters
        # for each GP_a (Eq. 6 / Sec. 2.1):
        #   ell:          ℓ_{a,i},    shape [D, D+F]  (ℓ, not ℓ²)
        #   alpha_sq:     α²_a,       shape [D, 1]
        #   sigma_sq_eps: σ²_{ε_a},   shape [D]
        return self.ell, self.alpha_sq, self.sigma_sq_eps

    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        """Predict the next-state distribution given the current state and action.

        Routes to :meth:`uncertain_forward` (moment-matching, Eqs. 10-23) when
        the input observation covariance is non-zero, and to
        :meth:`deterministic_forward` (Eqs. 7-8) otherwise.

        Args:
            tensordict (TensorDictBase): Input tensordict containing keys
                defined by ``in_keys``. Observation and action tensors may be
                unbatched ``(D,)`` / ``(F,)`` or batched ``(B, D)`` /
                ``(B, F)``; a leading batch dimension will be added and removed
                automatically for unbatched inputs. The observation covariance,
                when present, must be a full matrix of shape ``(..., D, D)``
                — per-dimension variance vectors are not accepted; use
                :func:`torch.diag_embed` to convert them first.

        Returns:
            TensorDictBase: The same tensordict, updated in-place with the
            predicted next-state mean and covariance written to ``out_keys``.
        """
        u_mean_key, u_var_key, u_cc_key, x_mean_key, x_var_key = self.in_keys

        Sigma_x = tensordict.get(x_var_key, None)
        if Sigma_x is not None and Sigma_x.dim() < 2:
            raise ValueError(
                f"Expected observation covariance to have at least 2 dimensions "
                f"(..., D, D), got shape {tuple(Sigma_x.shape)}. "
                "Convert per-dimension variances with torch.diag_embed() first."
            )

        observation_uncertain = Sigma_x is not None and not torch.all(
            torch.isclose(Sigma_x, torch.zeros_like(Sigma_x))
        )

        if observation_uncertain:
            return self.uncertain_forward(tensordict)
        else:
            return self.deterministic_forward(tensordict)

    def uncertain_forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        """Moment-matching forward pass for a Gaussian input belief (Eqs. 10-23).

        Propagates the joint Gaussian belief
        ``p(x̃_{t-1}) = N(μ̃_{t-1}, Σ̃_{t-1})`` (Sec. 2.2) through the GP
        dynamics model and returns a Gaussian approximation to ``p(x_t)``
        via exact moment matching.

        Args:
            tensordict (TensorDictBase): Input tensordict with keys defined by
                ``in_keys``. Supports unbatched ``(D,)`` inputs or batched
                inputs with a single leading batch dimension ``(B, D)``.

        Returns:
            TensorDictBase: The same tensordict updated with next-state mean
            ``μ_t`` (Eq. 10) and covariance ``Σ_t`` (Eq. 11) at ``out_keys``.
        """
        inv_K_noisy, beta = self.compute_factorizations()
        ell, alpha_sq, sigma_sq_eps = self._gather_gp_hyperparams()
        u_mean_key, u_var_key, u_cc_key, x_mean_key, x_var_key = self.in_keys

        mu_x = tensordict.get(x_mean_key)  # μ_x,  shape (B×)D
        Sigma_x = tensordict.get(x_var_key)  # Σ_x,  shape (B×)D×D
        mu_u = tensordict.get(u_mean_key)  # μ_u,  shape (B×)F
        Sigma_u = tensordict.get(u_var_key)  # Σ_u,  shape (B×)F×F
        C_xu = tensordict.get(u_cc_key)  # cov[x_{t-1}, u_{t-1}],  (B×)D×F  (Eq. 12)

        # Support unbatched inputs by temporarily adding a leading batch dimension.
        unbatched = mu_x.dim() == 1
        if unbatched:
            mu_x, Sigma_x, mu_u, Sigma_u, C_xu = (
                mu_x.unsqueeze(0),
                Sigma_x.unsqueeze(0),
                mu_u.unsqueeze(0),
                Sigma_u.unsqueeze(0),
                C_xu.unsqueeze(0),
            )

        device, dtype = mu_x.device, mu_x.dtype
        B = mu_x.shape[0]  # batch size
        n = self.X_tilde_train.shape[0]  # number of training points
        D = self.obs_dim  # state dimension
        DF = self.state_action_dim  # D+F, dimension of x̃

        # ---- Build joint state-action distribution p(x̃_{t-1}) (Sec. 2.2) ----
        # μ̃_{t-1} = [μ_x; μ_u] ∈ R^{B×(D+F)}
        mu_tilde = torch.cat([mu_x, mu_u], dim=-1)

        # Σ̃_{t-1} = [[Σ_x,        Σ_x C_xu    ],
        #             [C_xu^T Σ_x^T, Σ_u        ]]  ∈ R^{B×(D+F)×(D+F)}
        Sigma_x_C_xu = Sigma_x @ C_xu  # upper-right block [B, D, F]
        Sigma_tilde = torch.cat(
            [
                torch.cat([Sigma_x, Sigma_x_C_xu], dim=-1),
                torch.cat([Sigma_x_C_xu.transpose(-1, -2), Sigma_u], dim=-1),
            ],
            dim=-2,
        )  # [B, D+F, D+F]

        # ---- Compute q_a (mean-prediction kernel vector, Eq. 15) ----
        # ν_i = x̃_i - μ̃_{t-1}  (Eq. 16); shape [B, n, D+F]
        nu = self.X_tilde_train - mu_tilde.unsqueeze(1)

        # Λ_a^{-1} as diagonal matrices; shape [D, D+F, D+F].
        # ell stores ℓ_a (not ℓ²_a), so 1/ℓ_a gives the diagonal of Λ_a^{-1/2};
        # used here to form the full Λ_a^{-1} = diag(1/ℓ²_a) = diag(1/ℓ_a)².
        inv_Lambda_diag_mats = torch.diag_embed(1.0 / ell).to(
            device=device, dtype=dtype
        )

        # Λ_a^{-1} ν_i; shape [B, D, n, D+F]
        inv_Lambda_nu = nu.unsqueeze(1) @ inv_Lambda_diag_mats.unsqueeze(0)

        # R_a = Λ_a^{-1} Σ̃_{t-1} Λ_a^{-1} + I  – normalising matrix in Eq. 15;
        # shape [B, D, D+F, D+F]
        R_a = (
            inv_Lambda_diag_mats.unsqueeze(0)
            @ Sigma_tilde.unsqueeze(1)
            @ inv_Lambda_diag_mats.unsqueeze(0)
        )
        R_a = R_a + torch.eye(DF, device=device, dtype=dtype).view(1, 1, DF, DF)

        # Solve R_a t = (Λ_a^{-1} ν_i)^T  →  t = R_a^{-1} Λ_a^{-1} ν_i^T
        t = torch.linalg.solve(R_a, inv_Lambda_nu.transpose(-2, -1)).transpose(-2, -1)

        # exp(-½ ν_i^T (Σ̃ + Λ_a)^{-1} ν_i)  – exponent in Eq. 15; shape [B, D, n]
        scaled_exp = torch.exp(-0.5 * torch.sum(inv_Lambda_nu * t, dim=-1))

        # Scalar prefactor α²_a / sqrt(|Σ̃_{t-1} Λ_a^{-1} + I|) from Eq. 15; shape [B, D]
        det_R_a = torch.linalg.det(R_a)
        c_a = alpha_sq.squeeze(-1).unsqueeze(0) / torch.sqrt(det_R_a)

        # β_a ⊙ q_a (pointwise); shape [B, D, n]
        beta_q_a = scaled_exp * beta.unsqueeze(0)

        # μ^a_Δ = β_a^T q_a  (Eq. 14); shape [B, D]
        mu_Delta = torch.sum(beta_q_a, dim=-1) * c_a.squeeze(0)

        # ---- Cross-covariance cov[x̃_{t-1}, Δ_t]  (used in Eq. 12) ----
        # Derivative of μ_Δ w.r.t. μ̃, contracted with Σ̃ (Deisenroth 2010);
        # shape [B, D+F, D]
        t_inv_Lambda = t @ inv_Lambda_diag_mats.unsqueeze(0)
        cov_xtilde_Delta = (
            torch.matmul(
                t_inv_Lambda.transpose(-2, -1), beta_q_a.unsqueeze(-1)
            ).squeeze(-1)
            * c_a.unsqueeze(-1)
        ).transpose(-2, -1)

        # ---- Compute Q_{ab} (cross-kernel matrix, Eqs. 21-22) ----
        X_i = self.X_tilde_train.unsqueeze(1)  # [n, 1, D+F]
        X_j = self.X_tilde_train.unsqueeze(0)  # [1, n, D+F]
        diff_ij = X_i - X_j  # x̃_i - x̃_j; [n, n, D+F]  (Eq. 22)

        # ell stores ℓ_a; ℓ²_a is the diagonal of Λ_a  (Eq. 6)
        ell_sq_a = (ell**2)[:, None, :]  # [D, 1, D+F]
        ell_sq_b = (ell**2)[None, :, :]  # [1, D, D+F]

        # Λ_{ab} = (Λ_a^{-1} + Λ_b^{-1})^{-1},  diagonal entries; [D, D, D+F]
        inv_ell_sq_sum = 1.0 / ell_sq_a + 1.0 / ell_sq_b
        Lambda_ab = 1.0 / inv_ell_sq_sum

        # First exponential in Q_{ab,ij}: kernel product at training inputs (Eq. 22)
        # -½ (x̃_i - x̃_j)^T (Λ_a + Λ_b)^{-1} (x̃_i - x̃_j); shape [D, D, n, n]
        inv_ell_sq_sum_ab = 1.0 / (ell_sq_a + ell_sq_b)
        exp1 = -0.5 * torch.sum(
            diff_ij.unsqueeze(0).unsqueeze(0)
            * inv_ell_sq_sum_ab.unsqueeze(2).unsqueeze(2)
            * diff_ij.unsqueeze(0).unsqueeze(0),
            dim=-1,
        )  # [D, D, n, n]

        # z̄_{ij} = Λ_{ab} (Λ_a^{-1} x̃_i + Λ_b^{-1} x̃_j)  – midpoint (Eq. 22);
        # shape [D, D, n, n, D+F]
        z_bar = Lambda_ab.unsqueeze(2).unsqueeze(2) * (
            X_i.unsqueeze(0).unsqueeze(0) / ell_sq_a.unsqueeze(2).unsqueeze(2)
            + X_j.unsqueeze(0).unsqueeze(0) / ell_sq_b.unsqueeze(2).unsqueeze(2)
        )

        # z_{ij} = z̄_{ij} - μ̃_{t-1}; shape [B, D, D, n, n, D+F]
        z_bar = z_bar.unsqueeze(0).expand(B, -1, -1, -1, -1, -1)
        z_ij = z_bar - mu_tilde[:, None, None, None, None, :]
        z_ij_flat = z_ij.view(B, D, D, n * n, DF)

        # M_{ab} = Σ̃_{t-1} + diag(Λ_{ab})  – matrix in second exp of Eq. 22;
        # shape [B, D, D, D+F, D+F]
        M_ab = Sigma_tilde[:, None, None] + torch.diag_embed(Lambda_ab)

        # Second exponential: -½ z_{ij}^T M_{ab}^{-1} z_{ij}; shape [B, D, D, n, n]
        M_ab_solved = torch.linalg.solve(M_ab, z_ij_flat.transpose(-2, -1)).transpose(
            -2, -1
        )
        exp2 = (-0.5 * torch.sum(z_ij_flat * M_ab_solved, dim=-1)).view(B, D, D, n, n)

        # R_{ab} = Σ̃_{t-1} (Λ_a^{-1} + Λ_b^{-1}) + I  – normalising matrix (Eq. 22);
        # shape [B, D, D, D+F, D+F]
        R_ab = Sigma_tilde[:, None, None] @ torch.diag_embed(
            inv_ell_sq_sum
        ) + torch.eye(DF, device=device, dtype=dtype)
        det_R_ab = torch.linalg.det(R_ab)  # [B, D, D]

        # Scalar prefactor α²_a α²_b / sqrt(|R_{ab}|)  (Eq. 22); shape [B, D, D]
        c_ab = (alpha_sq.view(1, D, 1) * alpha_sq.view(1, 1, D)) / torch.sqrt(det_R_ab)

        # Q_{ab,ij}  (Eq. 22); shape [B, D, D, n, n]
        Q_ab = c_ab.unsqueeze(-1).unsqueeze(-1) * torch.exp(exp1.unsqueeze(0) + exp2)

        # ---- Σ_Δ = predictive covariance of Δ  (Eqs. 17-23) ----
        # Off-diagonal entries: σ²_{ab} = β_a^T Q_{ab} β_b - μ^a_Δ μ^b_Δ  (Eqs. 18, 20)
        beta_a = beta.view(1, D, 1, n)  # [1, D, 1, n]
        beta_b = beta.view(1, 1, D, n)  # [1, 1, D, n]

        Q_ab_beta_b = torch.matmul(Q_ab, beta_b.unsqueeze(-1)).squeeze(
            -1
        )  # [B, D, D, n]
        Sigma_Delta = (
            torch.matmul(beta_a.unsqueeze(-2), Q_ab_beta_b.unsqueeze(-1))
            .squeeze(-1)
            .squeeze(-1)
        )  # [B, D, D]  – β_a^T Q_{ab} β_b  (Eq. 20)

        # Diagonal correction  E_{x̃}[var_f[Δ_a | x̃]] = α²_a - tr(K_a^{-1} Q_{aa})
        # added to σ²_{aa}  (Eqs. 17, 23)
        invK_Q = torch.matmul(
            inv_K_noisy.unsqueeze(0).unsqueeze(2),  # [1, D, 1, n, n]
            Q_ab,  # [B, D, D, n, n]
        )  # [B, D, D, n, n]
        trace_invK_Q = torch.diagonal(invK_Q, dim1=-2, dim2=-1).sum(-1)  # [B, D, D]

        diag_idx = torch.arange(D, device=device)
        alpha_sq_b = alpha_sq.squeeze(-1).unsqueeze(0).expand(B, -1)  # [B, D]
        sigma_sq_eps_b = sigma_sq_eps.unsqueeze(0).expand(B, -1)  # [B, D]

        # Add α²_a - tr(K_a^{-1} Q_{aa}) + σ²_{ε_a} to the diagonal  (Eqs. 17, 23)
        Sigma_Delta[:, diag_idx, diag_idx] += (
            alpha_sq_b - trace_invK_Q[:, diag_idx, diag_idx] + sigma_sq_eps_b
        )

        # Subtract outer product of means: Σ_Δ -= μ_Δ μ_Δ^T  (Eqs. 17-18)
        Sigma_Delta = Sigma_Delta - torch.bmm(
            mu_Delta.unsqueeze(-1), mu_Delta.unsqueeze(-2)
        )
        Sigma_Delta = (
            Sigma_Delta + Sigma_Delta.transpose(-2, -1)
        ) / 2  # enforce symmetry

        # ---- Propagate to next-state belief (Eqs. 10-12) ----
        # cov[x_{t-1}, Δ_t] = cov[x_{t-1}, x̃_{t-1}] · cov_xtilde_Delta  (Eq. 12)
        # cov[x_{t-1}, x̃_{t-1}] is the top-D rows of Σ̃_{t-1}: shape [B, D, D+F].
        # Using only Sigma_x_C_xu ([B, D, F]) here would be wrong — it drops
        # the Σ_x block and produces a [B, D, F] @ [B, D+F, D] shape mismatch.
        Sigma_x_rows = Sigma_tilde[:, :D, :]  # [B, D, D+F]
        cov_x_Delta = Sigma_x_rows @ cov_xtilde_Delta  # [B, D, D]

        # μ_t = μ_{t-1} + μ_Δ  (Eq. 10)
        mu_t = mu_x + mu_Delta

        # Σ_t = Σ_{t-1} + Σ_Δ + cov[x_{t-1},Δ_t] + cov[Δ_t,x_{t-1}]  (Eq. 11)
        Sigma_t = Sigma_x + Sigma_Delta + cov_x_Delta + cov_x_Delta.transpose(-2, -1)
        Sigma_t = (Sigma_t + Sigma_t.transpose(-2, -1)) / 2  # enforce symmetry
        Sigma_t = Sigma_t + 1e-8 * torch.eye(D, device=device).expand(
            B, -1, -1
        )  # jitter

        if unbatched:
            mu_t = mu_t.squeeze(0)
            Sigma_t = Sigma_t.squeeze(0)

        out_mean_key, out_var_key = self.out_keys
        tensordict.set(out_mean_key, mu_t)
        tensordict.set(out_var_key, Sigma_t)
        return tensordict

    def deterministic_forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        """Deterministic forward pass using GP posterior mean and variance (Eqs. 7-8).

        Used when the input observation is a point estimate with no uncertainty.
        Returns the GP posterior mean ``m_f(x̃_*)`` (Eq. 7) and per-dimension
        variance ``σ²_f(x̃_*)`` (Eq. 8) for each state dimension.

        Args:
            tensordict (TensorDictBase): Input tensordict with keys defined by
                ``in_keys``. Supports arbitrary leading batch dimensions
                ``(*batch, D)`` / ``(*batch, F)``, as well as unbatched
                ``(D,)`` / ``(F,)`` inputs.

        Returns:
            TensorDictBase: The same tensordict updated with next-state mean
            ``μ_t`` and diagonal covariance ``Σ_t = diag(σ²_Δ)`` at
            ``out_keys``.
        """
        u_mean_key, u_var_key, u_cc_key, x_mean_key, x_var_key = self.in_keys
        mu_x = tensordict.get(x_mean_key)  # x_{t-1}, shape (*batch, D) or (D,)
        mu_u = tensordict.get(u_mean_key)  # u_{t-1}, shape (*batch, F) or (F,)

        batch_shape = mu_x.shape[:-1]  # leading dims; () for unbatched inputs

        # Flatten all leading batch dimensions to a single axis for the GP
        # posterior call, then restore the original shape afterwards.
        x_flat = mu_x.reshape(-1, self.obs_dim)  # [B_flat, D]
        u_flat = mu_u.reshape(-1, self.action_dim)  # [B_flat, F]

        # x̃_* = [x_{t-1}, u_{t-1}] ∈ R^{B_flat×(D+F)}  (Sec. 2.1)
        X_tilde_test = torch.cat([x_flat, u_flat], dim=-1)

        # GP posterior mean m_f(x̃_*) (Eq. 7) and std σ_f(x̃_*) (Eq. 8)
        mu_Delta_list, sigma_Delta_list = [], []

        with torch.no_grad():
            for gp_a in self.model_list.models:
                posterior_a = gp_a.posterior(X_tilde_test)
                mu_Delta_list.append(posterior_a.mean.squeeze(-1))  # m_f  (Eq. 7)
                sigma_Delta_list.append(
                    torch.sqrt(posterior_a.variance).squeeze(-1)  # σ_f  (Eq. 8)
                )

        # μ_Δ – predicted residual mean; restore original batch shape
        mu_Delta = torch.stack(mu_Delta_list, dim=-1).view(*batch_shape, self.obs_dim)

        # σ_Δ – predicted residual std; restore original batch shape
        sigma_Delta = torch.stack(sigma_Delta_list, dim=-1).view(
            *batch_shape, self.obs_dim
        )

        # μ_t = x_{t-1} + μ_Δ  (deterministic version of Eq. 10)
        mu_t = mu_x + mu_Delta

        # Σ_t = diag(σ²_Δ)  – diagonal covariance from independent GP variances (Eq. 8)
        Sigma_t = torch.diag_embed(sigma_Delta**2)

        out_mean_key, out_var_key = self.out_keys
        tensordict.set(out_mean_key, mu_t)
        tensordict.set(out_var_key, Sigma_t)
        return tensordict
