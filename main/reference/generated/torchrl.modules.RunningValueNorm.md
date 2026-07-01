# RunningValueNorm

*class*torchrl.modules.RunningValueNorm(***, *shape: int | tuple[int, ...] = 1*, *epsilon: float = 1e-05*, *device: [device](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device) | None = None*)[[source]](../../_modules/torchrl/modules/value_norm.html#RunningValueNorm)

Exact running mean / variance (Welford's online algorithm).

Unlike [`PopArtValueNorm`](torchrl.modules.PopArtValueNorm.html#torchrl.modules.PopArtValueNorm), this normaliser does not decay older
samples -- it accumulates the true sample mean and variance over every
target it has ever seen. Useful when value targets are roughly stationary
(no curriculum, no reward-shaping schedule), where the EMA's adaptivity
is unnecessary and the exact running stats give a slightly tighter
estimate.

Keyword Arguments:

- **shape** - per-element shape of the value tensor. Defaults to `1`.
- **epsilon** - numerical stabiliser added to the running variance.
Defaults to `1e-5`.
- **device** - device for the running-stats buffers.

Example

```
>>> vn = RunningValueNorm(shape=1)
>>> for _ in range(10):
... vn.update(torch.randn(64, 1) * 3.0 + 1.0)
>>> normed = vn.normalize(torch.randn(8, 1))
```

denormalize(*normalised_value: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*) → [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)[[source]](../../_modules/torchrl/modules/value_norm.html#RunningValueNorm.denormalize)

Inverse of `normalize()` -- recover real-scale values.

normalize(*value_target: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*) → [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)[[source]](../../_modules/torchrl/modules/value_norm.html#RunningValueNorm.normalize)

Standardise `value_target` using the current running stats.

update(*value_target: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*) → None[[source]](../../_modules/torchrl/modules/value_norm.html#RunningValueNorm.update)

Fold a batch of value targets into the running stats.