# torchrl.objectives.symlog

torchrl.objectives.symlog(*x: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*) → [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)[[source]](../../_modules/torchrl/modules/models/model_based_v3.html#symlog)

Apply the element-wise symmetric logarithm transform.

Parameters:

**x** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) - Input tensor.

Returns:

A tensor with the same shape, dtype, and device as `x`.

Examples

```
>>> import torch
>>> from torchrl.objectives import symlog
>>> symlog(torch.tensor([-100.0, 0.0, 100.0]))
tensor([-4.6151, 0.0000, 4.6151])
```