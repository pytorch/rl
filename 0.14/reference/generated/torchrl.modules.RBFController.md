# RBFController

*class*torchrl.modules.RBFController(*input_dim: int*, *output_dim: int*, *max_action: float | [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*, *n_basis: int = 10*)[[source]](../../_modules/torchrl/modules/models/rbf_controller.html#RBFController)

Radial Basis Function controller for moment-matching policy search.

Implements a policy that maps Gaussian-distributed state beliefs
`(mean, covariance)` to Gaussian-distributed actions using an RBF network
followed by a sinusoidal squashing function. The moment-matching formulas
allow analytic gradient computation through the policy during model-based
optimization (e.g., PILCO).

The controller uses `n_basis` RBF basis functions, each parameterised
by a centre vector and a shared diagonal lengthscale. The output is a
weighted sum of basis activations, optionally squashed through
`squash_sin()` to enforce action bounds.

Reference: Deisenroth & Rasmussen, "PILCO: A Model-Based and Data-Efficient
Approach to Policy Search", ICML 2011.

Parameters:

- **input_dim** (*int*) - Dimensionality of the state (observation) space.
- **output_dim** (*int*) - Dimensionality of the action space.
- **max_action** (*float**or**Tensor*) - Element-wise upper bound on action
magnitude. When provided, actions are squashed through
`squash_sin()`.
- **n_basis** (*int**,**optional*) - Number of RBF basis functions.
Defaults to `10`.

Inputs:

mean (Tensor): State mean of shape `(*batch, input_dim)`.
covariance (Tensor): State covariance of shape

> `(*batch, input_dim, input_dim)`.

Returns:

Action mean of shape `(*batch, output_dim)`.
action_covariance (Tensor): Action covariance of shape

> `(*batch, output_dim, output_dim)`.

cross_covariance (Tensor): Input-output cross-covariance of shape

`(*batch, input_dim, output_dim)`.

Return type:

action_mean (Tensor)

Examples

```
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
```

forward(*mean: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*, *covariance: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*) → tuple[[Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor), [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor), [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)][[source]](../../_modules/torchrl/modules/models/rbf_controller.html#RBFController.forward)

Define the computation performed at every call.

Should be overridden by all subclasses.

Note

Although the recipe for forward pass needs to be defined within
this function, one should call the `Module` instance afterwards
instead of this since the former takes care of running the
registered hooks while the latter silently ignores them.

*static*squash_sin(*mean: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*, *covariance: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*, *max_action: float | [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*) → tuple[[Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor), [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor), [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)][[source]](../../_modules/torchrl/modules/models/rbf_controller.html#RBFController.squash_sin)

Propagates a Gaussian through an element-wise `max_action * sin(x)` squashing.

Computes the exact moments of the transformed distribution using
the moment-matching identities for sine applied to Gaussian inputs.

Parameters:

- **mean** (*Tensor*) - Input mean, shape `(*batch, K)`.
- **covariance** (*Tensor*) - Input covariance, shape `(*batch, K, K)`.
- **max_action** (*float**or**Tensor*) - Per-dimension action bound.

Returns:

Output mean, shape `(*batch, K)`.
squashed_covariance (Tensor): Output covariance, shape `(*batch, K, K)`.
cross_covariance (Tensor): Input-output cross-covariance, shape `(*batch, K, K)`.

Return type:

squashed_mean (Tensor)