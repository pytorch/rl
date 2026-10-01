# OneHotCategorical

*class*torchrl.modules.OneHotCategorical(*logits: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) | None = None*, *probs: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) | None = None*, *grad_method: [ReparamGradientStrategy](torchrl.modules.ReparamGradientStrategy.html#torchrl.modules.ReparamGradientStrategy) = ReparamGradientStrategy.PassThrough*, ***kwargs*)[[source]](../../_modules/torchrl/modules/distributions/discrete.html#OneHotCategorical)

One-hot categorical distribution.

This class behaves exactly as torch.distributions.Categorical except that it reads and produces one-hot encodings
of the discrete tensors.

Parameters:

- **logits** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) - event log probabilities (unnormalized)
- **probs** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) - event probabilities
- **grad_method** ([*ReparamGradientStrategy*](torchrl.modules.ReparamGradientStrategy.html#torchrl.modules.ReparamGradientStrategy)*,**optional*) - strategy to gather
reparameterized samples.
`ReparamGradientStrategy.PassThrough` will compute the sample gradients
by using the softmax valued log-probability as a proxy to the
sample gradients.
`ReparamGradientStrategy.RelaxedOneHot` will use
`torch.distributions.RelaxedOneHot` to sample from the distribution.

Examples

```
>>> torch.manual_seed(0)
>>> logits = torch.randn(4)
>>> dist = OneHotCategorical(logits=logits)
>>> print(dist.rsample((3,)))
tensor([[1., 0., 0., 0.],
 [0., 0., 0., 1.],
 [1., 0., 0., 0.]])
```

entropy()[[source]](../../_modules/torchrl/modules/distributions/discrete.html#OneHotCategorical.entropy)

Returns entropy of distribution, batched over batch_shape.

Returns:

Tensor of shape batch_shape.

log_prob(*value: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*) → [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)[[source]](../../_modules/torchrl/modules/distributions/discrete.html#OneHotCategorical.log_prob)

Returns the log of the probability density/mass function evaluated at
value.

Parameters:

**value** (*Tensor*) -

*property*mode*: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*

Returns the mode of the distribution.

rsample(*sample_shape: [Size](https://docs.pytorch.org/docs/stable/size.html#torch.Size) | Sequence = None*) → [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)[[source]](../../_modules/torchrl/modules/distributions/discrete.html#OneHotCategorical.rsample)

Generates a sample_shape shaped reparameterized sample or sample_shape
shaped batch of reparameterized samples if the distribution parameters
are batched.

sample(*sample_shape: [Size](https://docs.pytorch.org/docs/stable/size.html#torch.Size) | Sequence | None = None*) → [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)[[source]](../../_modules/torchrl/modules/distributions/discrete.html#OneHotCategorical.sample)

Generates a sample_shape shaped sample or sample_shape shaped batch of
samples if the distribution parameters are batched.