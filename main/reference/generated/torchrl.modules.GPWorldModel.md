# GPWorldModel

*class*torchrl.modules.GPWorldModel(*obs_dim: int*, *action_dim: int*, *in_keys: list[str | tuple[str, ...]] | None = None*, *out_keys: list[str | tuple[str, ...]] | None = None*)[[source]](../../_modules/torchrl/modules/models/gp.html#GPWorldModel)

Gaussian Process world model with moment-matching uncertainty propagation.

Implements the probabilistic dynamics model from PILCO
(Deisenroth & Rasmussen, 2011). One independent GP is fit per state
dimension, each predicting the transition residual
`Δ = x_t - x_{t-1}` from the concatenated state-action input
`x̃ = [x, u]` (Sec. 2.1).

`forward()` supports two modes depending on whether the input
observation carries non-zero variance:

- **Deterministic**: uses the GP posterior mean and variance directly
(Eqs. 7-8).
- **Uncertain** (moment-matching): propagates a Gaussian belief
`N(μ, Σ)` through the GP analytically (Eqs. 10-23).

Note

Requires `botorch` and `gpytorch` as optional dependencies.

Parameters:

- **obs_dim** (*int*) - Dimension D of the observation (state) space.
- **action_dim** (*int*) - Dimension F of the action (control) space.
- **in_keys** (*list**of**NestedKey**,**optional*) - Keys to read from the input
[`TensorDictBase`](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase). Must contain five entries in
order: action mean, action covariance, state-action
cross-covariance, observation mean, observation covariance.
Defaults to `[("action", "mean"), ("action", "var"),
("action", "cross_covariance"), ("observation", "mean"),
("observation", "var")]`.
- **out_keys** (*list**of**NestedKey**,**optional*) - Keys to write the predicted
next-state mean and covariance to. Defaults to
`[("next", "observation", "mean"),
("next", "observation", "var")]`.

Examples

```
>>> import torch
>>> from tensordict import TensorDict
>>> model = GPWorldModel(obs_dim=4, action_dim=1)
>>> dataset = TensorDict(
... {
... "observation": torch.randn(50, 4),
... "action": torch.randn(50, 1),
... ("next", "observation"): torch.randn(50, 4),
... },
... batch_size=[50],
... )
>>> model.fit(dataset)
```

Reference:

Deisenroth, M. P. & Rasmussen, C. E. (2011). PILCO: A model-based
and data-efficient approach to policy search. *ICML*.

compute_factorizations() → tuple[[Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor), [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)][[source]](../../_modules/torchrl/modules/models/gp.html#GPWorldModel.compute_factorizations)

Return the cached kernel inverses and GP weight vectors.

Returns:

A pair `(inv_K_noisy, beta)` where
`inv_K_noisy` has shape `(D, n, n)` and contains
`(K_a + σ²_{ε_a} I)^{-1}` for each output dimension (Eq. 7),
and `beta` has shape `(D, n)` and contains
`β_a = (K_a + σ²_{ε_a} I)^{-1} y_a` (Eq. 7).

Return type:

tuple[Tensor, Tensor]

deterministic_forward(*tensordict: [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)*) → [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)[[source]](../../_modules/torchrl/modules/models/gp.html#GPWorldModel.deterministic_forward)

Deterministic forward pass using GP posterior mean and variance (Eqs. 7-8).

Used when the input observation is a point estimate with no uncertainty.
Returns the GP posterior mean `m_f(x̃_*)` (Eq. 7) and per-dimension
variance `σ²_f(x̃_*)` (Eq. 8) for each state dimension.

Parameters:

**tensordict** (*TensorDictBase*) - Input tensordict with keys defined by
`in_keys`. Supports arbitrary leading batch dimensions
`(*batch, D)` / `(*batch, F)`, as well as unbatched
`(D,)` / `(F,)` inputs.

Returns:

The same tensordict updated with next-state mean
`μ_t` and diagonal covariance `Σ_t = diag(σ²_Δ)` at
`out_keys`.

Return type:

TensorDictBase

fit(*dataset: [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)*) → None[[source]](../../_modules/torchrl/modules/models/gp.html#GPWorldModel.fit)

Fit one GP per state dimension to a dataset of transitions.

Constructs training inputs `X̃ = [x, u]` and targets
`Δ_a = x_{t,a} - x_{t-1,a}`, then maximises the marginal
log-likelihood to learn SE kernel hyper-parameters
(ℓ_a, α²_a, σ²_{ε_a}) for each output dimension (Sec. 2.1, Eq. 6).

Note

The dataset is expected to be flat with shape `[n, *]`. If your
replay buffer returns multi-dimensional batches (e.g. `[B, T, *]`),
call `dataset.reshape(-1)` before passing it here.

Parameters:

**dataset** (*TensorDictBase*) - Transition dataset with keys
`"observation"` of shape `(n, D)`,
`"action"` of shape `(n, F)`, and
`("next", "observation")` of shape `(n, D)`.

forward(*tensordict: [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)*) → [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)[[source]](../../_modules/torchrl/modules/models/gp.html#GPWorldModel.forward)

Predict the next-state distribution given the current state and action.

Routes to `uncertain_forward()` (moment-matching, Eqs. 10-23) when
any input covariance is non-zero, and to
`deterministic_forward()` (Eqs. 7-8) otherwise.

Parameters:

**tensordict** (*TensorDictBase*) - Input tensordict containing keys
defined by `in_keys`. Observation and action tensors may be
unbatched `(D,)` / `(F,)` or batched `(B, D)` /
`(B, F)`; a leading batch dimension will be added and removed
automatically for unbatched inputs. The observation covariance,
when present, must be a full matrix of shape `(..., D, D)`
-- per-dimension variance vectors are not accepted; use
[`torch.diag_embed()`](https://docs.pytorch.org/docs/stable/generated/torch.diag_embed.html#torch.diag_embed) to convert them first.

Returns:

The same tensordict, updated in-place with the
predicted next-state mean and covariance written to `out_keys`.

Return type:

TensorDictBase

uncertain_forward(*tensordict: [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)*) → [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)[[source]](../../_modules/torchrl/modules/models/gp.html#GPWorldModel.uncertain_forward)

Moment-matching forward pass for a Gaussian input belief (Eqs. 10-23).

Propagates the joint Gaussian belief
`p(x̃_{t-1}) = N(μ̃_{t-1}, Σ̃_{t-1})` (Sec. 2.2) through the GP
dynamics model and returns a Gaussian approximation to `p(x_t)`
via exact moment matching.

Parameters:

**tensordict** (*TensorDictBase*) - Input tensordict with keys defined by
`in_keys`. Supports unbatched `(D,)` inputs or batched
inputs with a single leading batch dimension `(B, D)`.

Returns:

The same tensordict updated with next-state mean
`μ_t` (Eq. 10) and covariance `Σ_t` (Eq. 11) at `out_keys`.

Return type:

TensorDictBase