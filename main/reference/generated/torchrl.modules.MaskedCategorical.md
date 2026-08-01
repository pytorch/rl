# MaskedCategorical

*class*torchrl.modules.MaskedCategorical(*logits: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) | None = None*, *probs: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) | None = None*, ***, *mask: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) | None = None*, *indices: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) | None = None*, *neg_inf: float = -inf*, *padding_value: int | None = None*, *use_cross_entropy: bool = True*, *padding_side: str = 'left'*)[[source]](../../_modules/torchrl/modules/distributions/discrete.html#MaskedCategorical)

MaskedCategorical distribution.

Reference:
[https://www.tensorflow.org/agents/api_docs/python/tf_agents/distributions/masked/MaskedCategorical](https://www.tensorflow.org/agents/api_docs/python/tf_agents/distributions/masked/MaskedCategorical)

Parameters:

- **logits** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) - event log probabilities (unnormalized)
- **probs** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) - event probabilities. If provided, the probabilities
corresponding to masked items will be zeroed and the probability
re-normalized along its last dimension.

Keyword Arguments:

- **mask** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) - A boolean mask of the same shape as `logits`/`probs`
where `False` entries are the ones to be masked. Alternatively,
if `sparse_mask` is True, it represents the list of valid indices
in the distribution. Exclusive with `indices`.
- **indices** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) - A dense index tensor representing which actions
must be taken into account. Exclusive with `mask`.
- **neg_inf** (`float`, optional) - The log-probability value allocated to
invalid (out-of-mask) indices. Defaults to -inf.
- **padding_value** - The padding value in the mask tensor. When
sparse_mask == True, the padding_value will be ignored.
- **use_cross_entropy** (*bool**,**optional*) - For faster computation of the log-probability,
the cross_entropy loss functional can be used. Defaults to `True`.
- **padding_side** (*str**,**optional*) - The side of the padding. Defaults to `"left"`.

Examples

```
>>> torch.manual_seed(0)
>>> logits = torch.randn(4) / 100 # almost equal probabilities
>>> mask = torch.tensor([True, False, True, True])
>>> dist = MaskedCategorical(logits=logits, mask=mask)
>>> sample = dist.sample((10,))
>>> print(sample) # no `1` in the sample
tensor([2, 3, 0, 2, 2, 0, 2, 0, 2, 2])
>>> print(dist.log_prob(sample))
tensor([-1.1203, -1.0928, -1.0831, -1.1203, -1.1203, -1.0831, -1.1203, -1.0831,
 -1.1203, -1.1203])
>>> print(dist.log_prob(torch.ones_like(sample)))
tensor([-inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf])
>>> # with probabilities
>>> prob = torch.ones(10)
>>> prob = prob / prob.sum()
>>> mask = torch.tensor([False] + 9 * [True]) # first outcome is masked
>>> dist = MaskedCategorical(probs=prob, mask=mask)
>>> print(dist.log_prob(torch.arange(10)))
tensor([ -inf, -2.1972, -2.1972, -2.1972, -2.1972, -2.1972, -2.1972, -2.1972,
 -2.1972, -2.1972])
```

entropy()[[source]](../../_modules/torchrl/modules/distributions/discrete.html#MaskedCategorical.entropy)

Compute the entropy of the distribution.

For masked distributions, we only consider the entropy over the valid (unmasked) outcomes.
Invalid outcomes have zero probability and don't contribute to entropy.

log_prob(*value: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*) → [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)[[source]](../../_modules/torchrl/modules/distributions/discrete.html#MaskedCategorical.log_prob)

Returns the log of the probability density/mass function evaluated at
value.

Parameters:

**value** (*Tensor*) -

*property*padding_value

Padding value of the distribution mask.

If the padding value is not set, it will be inferred from the logits.

sample(*sample_shape: [Size](https://docs.pytorch.org/docs/stable/size.html#torch.Size) | Sequence[int] | None = None*) → [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)[[source]](../../_modules/torchrl/modules/distributions/discrete.html#MaskedCategorical.sample)

Generates a sample_shape shaped sample or sample_shape shaped batch of
samples if the distribution parameters are batched.