# torchrl.objectives.symexp

torchrl.objectives.symexp(*x: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*) → [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)[[source]](../../_modules/torchrl/modules/models/model_based_v3.html#symexp)

Apply the inverse of [`symlog()`](torchrl.objectives.symlog.html#torchrl.objectives.symlog) element-wise.

Parameters:

**x** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) - Input tensor.

Returns:

A tensor with the same shape, dtype, and device as `x`.

Examples

```
>>> import torch
>>> from torchrl.objectives import symexp, symlog
>>> x = torch.tensor([-1000.0, 0.0, 1000.0])
>>> torch.allclose(symexp(symlog(x)), x, atol=1e-4)
True
```