# has_analytic_kl

torchrl.modules.distributions.utils.has_analytic_kl(*p: [Distribution](https://docs.pytorch.org/docs/stable/distributions.html#torch.distributions.distribution.Distribution)*, *q: [Distribution](https://docs.pytorch.org/docs/stable/distributions.html#torch.distributions.distribution.Distribution)*) → bool[[source]](../../_modules/torchrl/modules/distributions/utils.html#has_analytic_kl)

Return whether `kl_divergence(p, q)` has a registered closed form.

`Independent` and `TransformedDistribution` pairs are resolved
through their bases, matching the registered torch KL implementations
without calling them (those wrappers raise `NotImplementedError` when
the inner pair is missing). Other pairs are looked up in
`torch.distributions.kl._KL_REGISTRY`.

Parameters:

- **p** (*torch.distributions.Distribution*) - left argument of
`kl_divergence(p, q)`.
- **q** (*torch.distributions.Distribution*) - right argument of
`kl_divergence(p, q)`.

Returns:

`True` if a closed-form KL is registered for this pair.

Return type:

bool

Examples

```
>>> import torch
>>> from torch import distributions as d
>>> from torchrl.modules.distributions.utils import has_analytic_kl
>>> loc = torch.zeros(2)
>>> scale = torch.ones(2)
>>> has_analytic_kl(d.Normal(loc, scale), d.Normal(loc, scale))
True
```