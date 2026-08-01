# torchrl.objectives.symlog

torchrl.objectives.symlog(*x: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*) → [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)[[source]](../../_modules/torchrl/objectives/dreamer_v3.html#symlog)

Symmetric logarithm: `sign(x) * log(|x| + 1)`.

Used by DreamerV3 to compress the dynamic range of targets and
predictions before computing reconstruction losses.

Reference: [https://arxiv.org/abs/2301.04104](https://arxiv.org/abs/2301.04104)

Parameters:

**x** - Input tensor.

Returns:

Tensor of the same shape as `x`.

Examples

```
>>> import torch
>>> from torchrl.objectives import symlog
>>> x = torch.tensor([-100.0, 0.0, 100.0])
>>> symlog(x)
tensor([-4.6151, 0.0000, 4.6151])
```