# has_analytic_entropy

torchrl.modules.distributions.utils.has_analytic_entropy(*dist: [Distribution](https://docs.pytorch.org/docs/stable/distributions.html#torch.distributions.distribution.Distribution)*) → bool[[source]](../../_modules/torchrl/modules/distributions/utils.html#has_analytic_entropy)

Return whether `dist` implements a closed-form `entropy()`.

The check is class-level: `type(dist).entropy is not
torch.distributions.Distribution.entropy`. `Independent` is resolved
through its base distribution because `Independent.entropy` always
exists and only works when the base distribution implements entropy.
`CompositeDistribution` is treated as not having a closed-form
entropy: its `entropy()` may return a TensorDict and still relies on
`try/except` internally. Use [`composite_entropy()`](torchrl.modules.distributions.utils.composite_entropy.html#torchrl.modules.distributions.utils.composite_entropy) for composites.

Parameters:

**dist** (*torch.distributions.Distribution*) - distribution to inspect.

Returns:

`True` if a closed-form entropy method is available.

Return type:

bool

Examples

```
>>> import torch
>>> from torch import distributions as d
>>> from torchrl.modules.distributions.utils import has_analytic_entropy
>>> has_analytic_entropy(d.Normal(torch.zeros(2), torch.ones(2)))
True
>>> has_analytic_entropy(d.Independent(d.Normal(torch.zeros(2), torch.ones(2)), 1))
True
```