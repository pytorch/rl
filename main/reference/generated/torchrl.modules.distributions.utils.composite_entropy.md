# composite_entropy

torchrl.modules.distributions.utils.composite_entropy(*distribution: [CompositeDistribution](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.nn.distributions.CompositeDistribution.html#tensordict.nn.distributions.CompositeDistribution)*, *samples_mc: int = 1*) → [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) | [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)[[source]](../../_modules/torchrl/modules/distributions/utils.html#composite_entropy)

Compute component entropy without inverse-scoring Monte Carlo samples.

Analytic component entropies are used when available. Components without
analytic entropy are estimated from atomic reparameterized samples.

Parameters:

- **distribution** (*CompositeDistribution*) - distribution whose component
entropies are computed.
- **samples_mc** (*int**,**optional*) - number of Monte Carlo samples used for
components without analytic entropy. Defaults to `1`.

Returns:

The aggregated entropy, or a TensorDict of component entropies when
composite log-probability aggregation is disabled.