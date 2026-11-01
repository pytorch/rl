# torchrl.objectives.symexp

torchrl.objectives.symexp(*x: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*) → [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)[[source]](../../_modules/torchrl/objectives/dreamer_v3.html#symexp)

Symmetric exponential: `sign(x) * (exp(|x|) - 1)`.

Inverse of [`symlog()`](torchrl.objectives.symlog.html#torchrl.objectives.symlog).

Reference: [https://arxiv.org/abs/2301.04104](https://arxiv.org/abs/2301.04104)

Parameters:

**x** - Input tensor.

Returns:

Tensor of the same shape as `x`.

Examples

```
>>> import torch
>>> from torchrl.objectives import symexp, symlog
>>> x = torch.tensor([-1000.0, -1.0, 0.0, 1.0, 1000.0])
>>> torch.allclose(symexp(symlog(x)), x, atol=1e-4)
True
```