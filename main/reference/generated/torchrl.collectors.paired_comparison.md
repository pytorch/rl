# paired_comparison

*class*torchrl.collectors.paired_comparison(*first: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*, *second: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*, *dim: int = -1*, ***, *statistic: Literal['mean', 'median', 'iqm'] = 'mean'*, *confidence: float = 0.95*, *num_resamples: int = 1000*, *generator: [Generator](https://docs.pytorch.org/docs/stable/generated/torch.Generator.html#torch.Generator) | None = None*)[[source]](../../_modules/torchrl/collectors/_evaluation_stats.html#paired_comparison)

Compare two sets of paired samples, e.g. two policies evaluated on the same seeds.

Sample `i` of `first` and `second` must come from the same
conditions, as with
[`evaluate_episodes()`](torchrl.collectors.Evaluator.html#torchrl.collectors.Evaluator.evaluate_episodes) called with the
same `seed`.

Parameters:

- **first** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) - the samples of the first candidate.
- **second** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) - the samples of the second candidate, with the
same shape as `first`.
- **dim** (*int**,**optional*) - the dimension holding the samples. Defaults to `-1`.

Keyword Arguments:

- **statistic** (*str**,**optional*) - the statistic of the paired differences
`first - second`: `"mean"`, `"median"` or `"iqm"`.
Defaults to `"mean"`.
- **confidence** ([*float*](torchrl.data.llm.TopKRewardSelector.html#torchrl.data.llm.TopKRewardSelector.float)*,**optional*) - see [`bootstrap_estimate()`](torchrl.collectors.bootstrap_estimate.html#torchrl.collectors.bootstrap_estimate).
- **num_resamples** (*int**,**optional*) - see [`bootstrap_estimate()`](torchrl.collectors.bootstrap_estimate.html#torchrl.collectors.bootstrap_estimate).
- **generator** ([*torch.Generator*](https://docs.pytorch.org/docs/stable/generated/torch.Generator.html#torch.Generator)*,**optional*) - see [`bootstrap_estimate()`](torchrl.collectors.bootstrap_estimate.html#torchrl.collectors.bootstrap_estimate).

Returns:

the output of [`bootstrap_estimate()`](torchrl.collectors.bootstrap_estimate.html#torchrl.collectors.bootstrap_estimate) on `first - second`, plus
`"probability_of_improvement"`: the fraction of pairs where
`first` is larger, ties counting as half.

Examples

```
>>> import torch
>>> from torchrl.collectors import paired_comparison
>>> first = torch.tensor([3.0, 5.0, 4.0, 6.0])
>>> result = paired_comparison(first, first - 1)
>>> result["estimate"], result["probability_of_improvement"]
(tensor(1.), tensor(1.))
```