# torchrl.objectives.two_hot_decode

torchrl.objectives.two_hot_decode(*logits: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*, *bins: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*) → [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)[[source]](../../_modules/torchrl/modules/models/model_based_v3.html#two_hot_decode)

Decode logits over a raw-value support to their scalar expectation.

Parameters:

- **logits** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) - Categorical logits whose trailing dimension
matches the support size.
- **bins** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) - One-dimensional support in raw value space.

Returns:

The softmax-weighted expectation with the trailing category dimension
removed, preserving the dtype and device of `logits`.

Examples

```
>>> import torch
>>> from torchrl.objectives import two_hot_decode, two_hot_encode
>>> bins = torch.tensor([-1.0, 0.0, 1.0])
>>> encoded = two_hot_encode(torch.tensor([0.25]), bins)
>>> two_hot_decode((encoded + 1e-8).log(), bins)
tensor([0.2500])
```