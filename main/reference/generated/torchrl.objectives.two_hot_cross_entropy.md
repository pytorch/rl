# torchrl.objectives.two_hot_cross_entropy

torchrl.objectives.two_hot_cross_entropy(*logits: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*, *target: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*, *bins: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*) → [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)[[source]](../../_modules/torchrl/modules/models/model_based.html#two_hot_cross_entropy)

Return two-hot cross entropy for raw scalar targets.

Parameters:

- **logits** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) - Categorical logits with bins in the trailing
dimension.
- **target** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) - Raw scalar targets, optionally with a trailing
singleton dimension.
- **bins** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) - One-dimensional support in raw value space.

Returns:

The unreduced cross entropy with the trailing category dimension
removed.

Examples

```
>>> import torch
>>> from torchrl.objectives import two_hot_cross_entropy
>>> logits = torch.zeros(2, 3)
>>> target = torch.tensor([-0.5, 0.5])
>>> two_hot_cross_entropy(logits, target, torch.tensor([-1.0, 0.0, 1.0]))
tensor([1.0986, 1.0986])
```