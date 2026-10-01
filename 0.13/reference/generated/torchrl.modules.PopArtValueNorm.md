# PopArtValueNorm

*class*torchrl.modules.PopArtValueNorm(***, *shape: int | tuple[int, ...] = 1*, *beta: float = 0.99999*, *epsilon: float = 1e-05*, *device: [device](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device) | None = None*)[[source]](../../_modules/torchrl/modules/value_norm.html#PopArtValueNorm)

PopArt-style EMA value normaliser.

Maintains exponentially-weighted running estimates of the value-target
mean and mean-of-squares, with debiasing (so the early-training estimates
are unbiased even before the EMA has had time to wash out the zero
initialisation). Equivalent to the value-normaliser used by the reference
MAPPO implementation.

Keyword Arguments:

- **shape** - per-element shape of the value tensor (everything except the
leading batch / time / agent dims that get reduced). Defaults to
`1`.
- **beta** - exponential decay for the running stats. Higher = slower
adaptation. Defaults to `0.99999` (the MAPPO default).
- **epsilon** - numerical stabiliser added to the running variance and used
as a floor for the debiasing term. Defaults to `1e-5`.
- **device** - device for the running-stats buffers.

Example

```
>>> vn = PopArtValueNorm(shape=1)
>>> target = torch.randn(64, 1) * 5.0 + 2.0 # mean 2, std 5
>>> for _ in range(100):
... vn.update(target)
>>> normed = vn.normalize(target) # ~ N(0, 1)
>>> recovered = vn.denormalize(normed) # back to real scale
```

denormalize(*normalised_value: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*) → [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)[[source]](../../_modules/torchrl/modules/value_norm.html#PopArtValueNorm.denormalize)

Inverse of `normalize()` -- recover real-scale values.

normalize(*value_target: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*) → [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)[[source]](../../_modules/torchrl/modules/value_norm.html#PopArtValueNorm.normalize)

Standardise `value_target` using the current running stats.

update(*value_target: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*) → None[[source]](../../_modules/torchrl/modules/value_norm.html#PopArtValueNorm.update)

Fold a batch of value targets into the running stats.