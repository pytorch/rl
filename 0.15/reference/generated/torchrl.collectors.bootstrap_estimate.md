# bootstrap_estimate

*class*torchrl.collectors.bootstrap_estimate(*values: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*, *dim: int = -1*, ***, *statistic: Literal['mean', 'median', 'iqm'] = 'iqm'*, *confidence: float = 0.95*, *num_resamples: int = 1000*, *generator: [Generator](https://docs.pytorch.org/docs/stable/generated/torch.Generator.html#torch.Generator) | None = None*)[[source]](../../_modules/torchrl/collectors/_evaluation_stats.html#bootstrap_estimate)

Reduce `values` along `dim` to an estimate with a percentile bootstrap interval.

Parameters:

- **values** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) - the samples, e.g. per-episode returns.
- **dim** (*int**,**optional*) - the dimension holding the samples. Defaults to `-1`.

Keyword Arguments:

- **statistic** (*str**,**optional*) - `"mean"`, `"median"` or `"iqm"`
(the mean of the samples left after dropping the lowest and
highest 25%). Defaults to `"iqm"`.
- **confidence** ([*float*](torchrl.data.llm.TopKRewardSelector.html#torchrl.data.llm.TopKRewardSelector.float)*,**optional*) - the coverage of the interval, in
`(0, 1)`. Defaults to `0.95`.
- **num_resamples** (*int**,**optional*) - the number of bootstrap resamples,
each drawn with replacement. Defaults to `1000`.
- **generator** ([*torch.Generator*](https://docs.pytorch.org/docs/stable/generated/torch.Generator.html#torch.Generator)*,**optional*) - the generator used to draw
resamples, for reproducible intervals. Defaults to `None`.

Returns:

a [`TensorDict`](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDict.html#tensordict.TensorDict) with entries `"estimate"`,
`"ci_low"` and `"ci_high"` whose batch size is the shape of
`values` without `dim`.

Examples

```
>>> import torch
>>> from torchrl.collectors import bootstrap_estimate
>>> returns = torch.tensor([[0.0, 0.0, 0.0, 100.0], [5.0, 5.0, 5.0, 5.0]])
>>> summary = bootstrap_estimate(returns, statistic="iqm")
>>> summary["estimate"]
tensor([0., 5.])
>>> summary["ci_low"][1], summary["ci_high"][1]
(tensor(5.), tensor(5.))
```