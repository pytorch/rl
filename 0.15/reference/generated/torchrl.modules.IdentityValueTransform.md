# IdentityValueTransform

*class*torchrl.modules.IdentityValueTransform(**args: Any*, ***kwargs: Any*)[[source]](../../_modules/torchrl/modules/value_transforms.html#IdentityValueTransform)

Leave scalar values unchanged.

Examples

```
>>> import torch
>>> from torchrl.modules import IdentityValueTransform
>>> transform = IdentityValueTransform()
>>> value = torch.tensor([-1.0, 0.0, 1.0])
>>> transformed = transform(value)
>>> transformed
tensor([-1., 0., 1.])
>>> transform.inverse(transformed)
tensor([-1., 0., 1.])
```

See also

[`ValueTransform`](torchrl.modules.ValueTransform.html#torchrl.modules.ValueTransform) for the interface and
[`ComposeValueTransform`](torchrl.modules.ComposeValueTransform.html#torchrl.modules.ComposeValueTransform) for composing transforms.

forward(*value: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*) → [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)[[source]](../../_modules/torchrl/modules/value_transforms.html#IdentityValueTransform.forward)

Return `value` unchanged.

inverse(*value: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*) → [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)[[source]](../../_modules/torchrl/modules/value_transforms.html#IdentityValueTransform.inverse)

Return `value` unchanged.