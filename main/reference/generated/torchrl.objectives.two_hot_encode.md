# torchrl.objectives.two_hot_encode

torchrl.objectives.two_hot_encode(*x: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*, *bins: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*) → [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)[[source]](../../_modules/torchrl/modules/models/dreamer_v3.html#two_hot_encode)

Encode raw scalar values on a sorted two-hot support.

Values between adjacent support points are represented by linear
interpolation in raw value space. Values outside the support saturate at
its endpoints.

Parameters:

- **x** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) - Raw scalar targets.
- **bins** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) - One-dimensional, ascending support.

Returns:

A tensor with shape `(*x.shape, bins.numel())` on the dtype and
device of `x`.

Examples

```
>>> import torch
>>> from torchrl.objectives import two_hot_encode
>>> bins = torch.tensor([-1.0, 0.0, 1.0])
>>> two_hot_encode(torch.tensor([0.25]), bins)
tensor([[0.0000, 0.7500, 0.2500]])
```