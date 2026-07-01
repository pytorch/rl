# torchrl.objectives.two_hot_decode

torchrl.objectives.two_hot_decode(*logits: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*, *bins: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*) → [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)[[source]](../../_modules/torchrl/objectives/dreamer_v3.html#two_hot_decode)

Decode a distribution over `bins` to a scalar expectation.

Reference: [https://arxiv.org/abs/2301.04104](https://arxiv.org/abs/2301.04104)

Parameters:

- **logits** - Raw logits, shape `[..., num_bins]`.
- **bins** - Sorted bin centers, shape `[num_bins]`.

Returns:

Scalar expected values, shape `[...]`.

Examples

```
>>> import torch
>>> from torchrl.objectives import two_hot_decode, two_hot_encode
>>> bins = torch.linspace(-1.0, 1.0, 5)
>>> probs = two_hot_encode(torch.tensor([0.25]), bins)
>>> two_hot_decode((probs + 1e-8).log(), bins)
tensor([0.2500])
```