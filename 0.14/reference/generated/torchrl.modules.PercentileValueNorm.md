# PercentileValueNorm

*class*torchrl.modules.PercentileValueNorm(***, *shape: int | tuple[int, ...] = 1*, *quantiles: tuple[float, float] = (0.05, 0.95)*, *rate: float = 0.01*, *min_scale: float = 1.0*, *center: bool = False*, *epsilon: float = 1e-05*, *device: [device](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device) | None = None*)[[source]](../../_modules/torchrl/modules/value_norm.html#PercentileValueNorm)

DreamerV3-style EMA percentile-range value normaliser.

Tracks exponential moving averages of a low and a high quantile of the
value targets and rescales by the span between them, clamped from below:
`scale = max(min_scale, high - low)`. Following DreamerV3 (Hafner et
al., *Mastering Diverse Domains through World Models*, 2023,
[https://arxiv.org/abs/2301.04104](https://arxiv.org/abs/2301.04104)), the clamp scales large values down
without amplifying small or noisy ones, which keeps fixed coefficients
such as an entropy bonus comparable across reward scales.

By default (`center=False`) `normalize()` only divides by the
span -- the DreamerV3 recipe for advantages, which are already centred by
the value baseline. With `center=True` the low-percentile EMA is also
subtracted, mapping the tracked percentile range onto `[0, 1]`.

Keyword Arguments:

- **shape** - per-element shape of the value tensor (everything except the
leading batch / time / agent dims that get reduced). Defaults to
`1`.
- **quantiles** - lower and upper quantiles tracked by the EMA. Defaults to
`(0.05, 0.95)`.
- **rate** - EMA update rate towards the batch quantiles; higher = faster
adaptation. Defaults to `0.01`.
- **min_scale** - lower bound of the normalisation scale. Defaults to
`1.0`.
- **center** - if `True`, subtract the low-percentile EMA in
`normalize()`. Defaults to `False`.
- **epsilon** - kept for interface parity with the other normalisers;
unused because `min_scale` already bounds the divisor.
- **device** - device for the running-stats buffers.

Example

```
>>> vn = PercentileValueNorm(shape=1, rate=1.0)
>>> returns = torch.linspace(0.0, 100.0, steps=101).unsqueeze(-1)
>>> vn.update(returns)
>>> vn.scale()
tensor([90.])
>>> vn.normalize(torch.tensor([45.0]))
tensor([0.5000])
```

denormalize(*normalised_value: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*) → [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)[[source]](../../_modules/torchrl/modules/value_norm.html#PercentileValueNorm.denormalize)

Inverse of `normalize()` -- recover real-scale values.

normalize(*value_target: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*) → [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)[[source]](../../_modules/torchrl/modules/value_norm.html#PercentileValueNorm.normalize)

Standardise `value_target` using the current running stats.

scale() → [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)[[source]](../../_modules/torchrl/modules/value_norm.html#PercentileValueNorm.scale)

Multiplicative scale currently applied by `normalize()`.

Exposed separately so consumers can rescale quantities that must not
be re-centred, e.g. advantages (already centred by the value
baseline), for which only the division by the scale applies.

Deliberately not abstract until v0.16 so that subclasses written
before it existed keep instantiating (with a `DeprecationWarning`);
this default raises `NotImplementedError` when called.

update(*value_target: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*) → None[[source]](../../_modules/torchrl/modules/value_norm.html#PercentileValueNorm.update)

Fold a batch of value targets into the running stats.