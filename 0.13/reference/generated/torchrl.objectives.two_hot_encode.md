# torchrl.objectives.two_hot_encode

torchrl.objectives.two_hot_encode(*x: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*, *bins: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*) → [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)[[source]](../../_modules/torchrl/objectives/dreamer_v3.html#two_hot_encode)

Encode a scalar tensor as a two-hot distribution over `bins`.

The scalar is split between the two nearest bin centers proportionally so
that `E[bins] = x`.

Reference: [https://arxiv.org/abs/2301.04104](https://arxiv.org/abs/2301.04104)

Parameters:

- **x** - Values to encode, shape `[...]`.
- **bins** - Sorted bin centers, shape `[num_bins]`.

Returns:

Two-hot vectors, shape `[..., num_bins]`.

Examples

```
>>> import torch
>>> from torchrl.objectives import two_hot_encode
>>> bins = torch.linspace(-1.0, 1.0, 5)
>>> two_hot_encode(torch.tensor([0.25]), bins)
tensor([[0.0000, 0.0000, 0.5000, 0.5000, 0.0000]])
```