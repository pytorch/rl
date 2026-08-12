# torchrl.objectives.categorical_kl_terms

torchrl.objectives.categorical_kl_terms(*posterior_logits: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*, *prior_logits: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*, *free_nats: float = 1.0*, *unimix: float = 0.01*) → tuple[[Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor), [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)][[source]](../../_modules/torchrl/objectives/dreamer_v3.html#categorical_kl_terms)

Return DreamerV3 dynamics and representation KL losses.

The dynamics term stops gradients through the posterior and the
representation term stops gradients through the prior. KL divergence is
summed over the stochastic categoricals before applying the free-nat
threshold, matching the aggregated one-hot distribution used by the
reference DreamerV3 implementation.

Parameters:

- **posterior_logits** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) - Posterior logits with shape
`[..., num_categoricals, num_classes]`.
- **prior_logits** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) - Prior logits with the same shape.
- **free_nats** (*float**,**optional*) - Minimum aggregated KL in nats. Defaults to
`1.0`.
- **unimix** (*float**,**optional*) - Fraction of uniform probability mixed into
each categorical. Defaults to `0.01`.

Returns:

A pair containing the scalar dynamics and representation KL losses.

Examples

```
>>> import torch
>>> from torchrl.objectives import categorical_kl_terms
>>> posterior = torch.randn(2, 4, 8, requires_grad=True)
>>> prior = torch.randn(2, 4, 8, requires_grad=True)
>>> dynamics, representation = categorical_kl_terms(posterior, prior)
>>> dynamics.shape, representation.shape
(torch.Size([]), torch.Size([]))
```