# RunningMeanStd

*class*torchrl.envs.transforms.RunningMeanStd(*shape: tuple = ()*, *epsilon: float = 0.0001*)[[source]](../../_modules/torchrl/envs/transforms/rnd.html#RunningMeanStd)

Tracks running mean and variance using Welford's parallel algorithm.

Buffers are registered so the statistics are included in `state_dict()`
and move correctly with `.to(device)`.

Parameters:

- **shape** (*tuple*) - feature shape to track (e.g. `(obs_dim,)` or `()` for scalars).
- **epsilon** (*float**,**optional*) - small initial count for numerical stability.
Default: `1e-4`.

Examples

```
>>> rms = RunningMeanStd(shape=(4,))
>>> rms.update(torch.randn(32, 4))
>>> normed = rms.normalize(torch.randn(8, 4))
>>> normed.shape
torch.Size([8, 4])
```

normalize(*x: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*) → [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)[[source]](../../_modules/torchrl/envs/transforms/rnd.html#RunningMeanStd.normalize)

Normalize `x` to approximately zero mean, unit variance.

update(*x: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*) → None[[source]](../../_modules/torchrl/envs/transforms/rnd.html#RunningMeanStd.update)

Update running statistics with a new batch.

Parameters:

**x** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) - batch of samples. All leading dimensions are
treated as the batch dimension; trailing dimensions must match
`self.mean.shape`.